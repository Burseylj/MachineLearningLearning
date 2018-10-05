# We implement and explore polynomial least squares regression from a linear algebra perspective
import numpy as np
import matplotlib.pyplot as plt


# generating data

# Random data
points = 30
order =1
data = np.random.random((points,2)) # plus 1 to get constant term 
print data

# We are solving for coeff = ( transpose(A)*A)^-1 * transpose(A)*y
# where A = [1 x_1 x^2_1 ... x^n_1]
#           [1                    ]
#                   ...
#           [1 x_m  ...      x^n_m]
#           y = [y1 ...ym]
#          and coeff is the coefficient vector

#generate A from inputs
def getM(x,points,order):
    M = np.zeros(shape=(points,order+1))
    for i in xrange(points):
        for j in xrange(order+1):
            M[i,j] = x[i]**j
    return M

x = data[:,0]
A = getM(x,points,order+1)
y = data[:,1].T

coeff = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y)

#we define a function for the best fit polynomial
def bestFit(val):
    output = 0
    for i in xrange(order+1):
        output += coeff[i]*val**i
    return output

# Plot data, regression line
xx = np.linspace(0, 1, 50)
F = bestFit(xx)

plt.figure(1)
plt.plot(xx, F, color='b')
plt.scatter(data[:,0], data[:,1], color='r')
plt.show()
