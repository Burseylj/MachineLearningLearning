from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random





def createDataset(hm,variance,step=2,correlation=False):
    val = 1
    ys = []
    for i in xrange(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = range(len(ys))
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def bestFitCoeffs(xs,ys):
    # store values to avoid 
    mxs = mean(xs)
    mys = mean(ys)
    
    m = ( (mxs * mys - mean(xs*ys)) /
         (mxs**2 - mean(xs**2)) )
    b = mys - m * mxs
    return m, b

#we find the coefficient of determination
def coeffDet(ys, ysReg):
    #define function for square error
    sqError = lambda ys, ysReg : sum((ysReg-ys)**2)
    mys = mean(ys)
    sqErrorYsReg = sqError(ys, ysReg)
    sqErrorMys = sqError(ys, mys)
    return 1 - sqErrorYsReg/sqErrorMys



#show regression line
print "y = {}*x + {}".format(round(m,4),b)

#plot regression line
def getRegrF(m,b):
    def regrF(x):
        return m*x + b
    return regrF


if __name__=="__main__":
    style.use("fivethirtyeight")
    xs, ys = createDataset(40,40,correlation='pos')

    #get regression line
    m,b = bestFitCoeffs(xs,ys)
    regrF = getRegrF(m,b)

    # make some prediction
    predictX = max(xs) + 1
    predictY = regrF(predictX)

    rSq = coeffDet(ys,regrF(xs))
    print "r squared = {}".format(rSq)

    plt.scatter(xs,ys)
    plt.scatter(predictX,predictY,color='red')
    plt.plot(xs,regrF(xs))
    plt.show()
