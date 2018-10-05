import numpy as np
import pandas as pd
from numpy.linalg import inv
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


from PolynomialRegressionLSQ import *

boston = load_boston()

data = boston.data

#we explore a relationship between NOX and DIS, that is pollution and distance from an employment center
nox = data[:,4]
dis = data[:,7]

coeff = bestFitCoeff(nox,dis,5)

xx = np.linspace(0, 1, 50)
F = polynom(xx,coeff)
plt.ylim(0,15)
plt.xlabel("nox")
plt.ylabel("dis")
plt.plot(xx,F,color='r')
plt.scatter(nox,dis)
plt.show()
