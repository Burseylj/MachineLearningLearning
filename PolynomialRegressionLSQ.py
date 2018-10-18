# We implement and explore polynomial least squares regression from a linear algebra perspective
import numpy as np
import matplotlib.pyplot as plt



def getM(x,points,order):
    M = np.zeros(shape=(points,order+1))
    for i in xrange(points):
        for j in xrange(order+1):
            M[i,j] = x[i]**j
    return M


def bestFitCoeff(x,y,order):
    points = x.shape[0]
    # We are solving for coeff = ( transpose(A)*A)^-1 * transpose(A)*y
    # where A = [1 x_1 x^2_1 ... x^n_1]
    #           [1                    ]
    #                   ...
    #           [1 x_m  ...      x^n_m]
    #           y = [y1 ...ym]
    #          and coeff is the coefficient vector

    #generate A from inputs
    
    A = getM(x,points,order)

    coeff = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y)
    return coeff


#we define a function for the best fit polynomial
def polyf(coeff):
    def polynom(val):
        output = 0
        for i in xrange(coeff.size):
            output += coeff[i]*val**i
        return output
    return polynom
    

if __name__ == "__main__":
    # Random data
    points = 100
    order =1
    data = np.random.random((points,2))
    x = data[:,0]
    y = data[:,1].T
    y = x+ (y-0.5)/5

    coeff = bestFitCoeff(x,y,order)
    polynom = polyf(coeff)
    
    # Plot data, regression line
    xx = np.linspace(0, 1, 50)
    F = polynom(xx)

    plt.figure(1)
    plt.plot(xx, F, color='b')
    plt.scatter(x, y, color='r')
    plt.show()

