import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

from sam import SAM
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

# from syft.frameworks.torch.dp import pate

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.SVHN('datasets/SVHN/train/', split='train', transform=transform,
                           target_transform=None, download=True)
test_data = datasets.SVHN('datasets/SVHN/test/', split='test', transform=transform,
                          target_transform=None, download=True)

num_teachers = 100
batch_size = 50


def get_data_loaders(train_data, num_teachers):
    """ Function to create data loaders for the Teacher classifier """
    teacher_loaders = []
    data_size = len(train_data) // num_teachers

    for i in range(data_size):
        indices = list(range(i * data_size, (i + 1) * data_size))
        subset_data = Subset(train_data, indices)
        loader = torch.utils.data.DataLoader(subset_data, batch_size=batch_size)
        teacher_loaders.append(loader)

    return teacher_loaders


teacher_loaders = get_data_loaders(train_data, num_teachers)

student_train_data = Subset(test_data, list(range(9000)))
student_test_data = Subset(test_data, list(range(9000, 10000)))

student_train_loader = torch.utils.data.DataLoader(student_train_data, batch_size=batch_size)
student_test_loader = torch.utils.data.DataLoader(student_test_data, batch_size=batch_size)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Classifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(5 * 10 * 10, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), 5 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(model, trainloader, criterion, optimizer, epochs=10):
    running_loss = 0
    for e in range(epochs):
        model.train()

        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


def predict(model, dataloader):
    outputs = torch.zeros(0, dtype=torch.long)
    model.eval()

    for images, labels in dataloader:
        output = model.forward(images)
        ps = torch.argmax(torch.exp(output), dim=1)
        outputs = torch.cat((outputs, ps))

    return outputs


def train_models(num_teachers):
    models = []
    for i in range(num_teachers):
        model = Classifier()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        train(model, teacher_loaders[i], criterion, optimizer)
        models.append(model)
    return models


models = train_models(num_teachers)

import numpy as np

epsilon = 0.2


def aggregated_teacher(models, dataloader, epsilon):
    preds = torch.torch.zeros((len(models), 9000), dtype=torch.long)
    for i, model in enumerate(models):
        results = predict(model, dataloader)
        preds[i] = results

    labels = np.array([]).astype(int)
    for image_preds in np.transpose(preds):
        label_counts = np.bincount(image_preds, minlength=10)
        beta = 1 / epsilon

        for i in range(len(label_counts)):
            label_counts[i] += np.random.laplace(0, beta, 1)

        new_label = np.argmax(label_counts)
        labels = np.append(labels, new_label)

    return preds.numpy(), labels


teacher_models = models
preds, student_labels = aggregated_teacher(teacher_models, student_train_loader, epsilon)


def student_loader(student_train_loader, labels):
    for i, (data, _) in enumerate(iter(student_train_loader)):
        yield data, torch.from_numpy(labels[i * len(data): (i + 1) * len(data)])


student_model = Classifier()
criterion = nn.NLLLoss()
base_optimizer = torch.optim.Adam
optimizer = SAM(student_model.parameters(), base_optimizer,  rho=0.05, adaptive=True, lr=0.003)
epochs = 100
steps = 0
running_loss = 0
for e in range(epochs):
    student_model.train()
    train_loader = student_loader(student_train_loader, student_labels)
    for images, labels in train_loader:
        steps += 1

        optimizer.zero_grad()
        output = student_model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        criterion(student_model.forward(images), labels).backward()
        optimizer.second_step(zero_grad=True)

        running_loss += loss.item()

        if steps % 50 == 0:
            test_loss = 0
            accuracy = 0
            student_model.eval()
            with torch.no_grad():
                for images, labels in student_test_loader:
                    log_ps = student_model(images)
                    test_loss += criterion(log_ps, labels).item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            student_model.train()
            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Train Loss: {:.3f}.. ".format(running_loss / len(student_train_loader)),
                  "Test Loss: {:.3f}.. ".format(test_loss / len(student_test_loader)),
                  "Accuracy: {:.3f}".format(accuracy / len(student_test_loader)))
            running_loss = 0

# from syft.frameworks.torch.dp import pate
#
# data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=preds, indices=student_labels, noise_eps=epsilon, delta=1e-5)
# print("Data Independent Epsilon:", data_ind_eps)
# print("Data Dependent Epsilon:", data_dep_eps)