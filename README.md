# Sharpness-Aware Private Aggregation of Teacher Ensembles (Sharp-PATE)
##### Course Project: [ECSE-6962 Trustworthy Machine Learning (Spring-2022)](https://piazza.com/class/ky4olbgarmr2du)
##### Instructors: [Prof. Ali Tajer](https://www.isg-rpi.com/) and [Prof. Alex Gittens](https://www.cs.rpi.edu/~gittea/)
###### By: [Momin Abbas](https://mominabbas.github.io/)



Language: Python

API: Pytorch

# Instructions
To reproduce the experiments, simply run the following commands:

For `MNIST`, run:  

```bash
python sharp_pate_mnist.py
```
```bash
python sharp_pate_mnist_privacy.py
```

For `SVHN`, run:  

```bash
python sharp_pate_svhn.py
```
```bash
python sharp_pate_svhn_privacy.py
```

Note: To run asharp-PATE, set the value of 'adaptive' to True

*For each expermient, I take the average of 10 runs

*Hyper-parameters not specified explicitly in the papers were found using grid search

# Reference
```
@article{papernot2016semi,
  title={Semi-supervised knowledge transfer for deep learning from private training data},
  author={Papernot, Nicolas and Abadi, Mart{\'\i}n and Erlingsson, Ulfar and Goodfellow, Ian and Talwar, Kunal},
  journal={arXiv preprint arXiv:1610.05755},
  year={2016}
}

@article{foret2020sharpness,
  title={Sharpness-aware minimization for efficiently improving generalization},
  author={Foret, Pierre and Kleiner, Ariel and Mobahi, Hossein and Neyshabur, Behnam},
  journal={arXiv preprint arXiv:2010.01412},
  year={2020}
}

```

