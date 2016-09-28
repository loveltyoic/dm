__author__ = 'lizihe'
from sklearn.datasets import load_boston
boston = load_boston()
from matplotlib import pyplot as plt
plt.scatter(boston.data[:, 5], boston.target, color='r')