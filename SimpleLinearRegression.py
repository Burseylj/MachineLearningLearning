from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("fivethirtyeight")
# we build a simple linear least squares regression algorithm

# define test data
xs = np.arange(1,7, dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def bestFitCoeffs(xs,ys):
    # store values to avoid 
    mxs = mean(xs)
    mys = mean(ys)
    
    m = ( (mxs * mys - mean(xs*ys)) /
         (mxs**2 - mean(xs**2)) )
    b = mys - m * mxs
    return m, b

sqError = lambda ys, ysReg : sum((ysReg-ys)**2)
def coeffDet(ys, ysReg):
    mys = mean(ys)
    sqErrorYsReg = sqError(ys, ysReg)
    sqErrorMys = sqError(ys, mys)
    return 1 - sqErrorYsReg/sqErrorMys

m,b = bestFitCoeffs(xs,ys)

#show regression line
print "y = {}*x + {}".format(round(m,4),b)

#plot regression line
def getRegrF(m,b):
    def regrF(x):
        return m*x + b
    return regrF
regrF = getRegrF(m,b)

# make some prediction
predictX = 8
predictY = regrF(predictX)

rSq = coeffDet(ys,regrF(xs))
print "r squared = {}".format(rSq)

plt.scatter(xs,ys)
plt.scatter(predictX,predictY,color='red')
plt.plot(xs,regrF(xs))
plt.show()
