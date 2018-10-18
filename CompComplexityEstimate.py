import numpy as np
import matplotlib.pyplot as plt
import timeit


# We have some O(x^n) algorithm. We want to estimate n.
# let input size be x and run time be y.
# then y ~ a*x^n + b*x^(n-1)...
# for large x,  y ~a*x^n
# so  lg(y) ~ lg(a*x^n)
#     lg(y) ~ lg(a) + n*lg(x)
# we use least squares regression to estimate ln(a) and n

def estimateOrder(x,y):
    points = x.size
    # We are solving for coeff = ( transpose(A)*A)^-1 * transpose(A)*y )
    #                          = (a,b)
    # where A = [1 ln(x_1) ]
    #           [1  ln(x_2) ]
    #                   ...
    #           [1  ln]
    #           y = [ln(y1) ...ln(ym)]
    #          and coeff is the coefficient vector

    #generate A from inputs
    xp = np.log2(x)
    yp = np.log2(y)
    A = np.column_stack( (np.ones(points), xp ))

    a,b = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(yp)
    # return values in the form of runtime = a*n**b
    return 2**a , b

def fitFunc(a,b):
    def f(x):
        return a * x**b
    return f

def _randomDataTest():
    # Random data
    points = 100
    data = np.random.random((points,2))
    x = data[:,0] *  100 + 1
    y = data[:,1].T 
    y = 2*x**3 + (y-0.5)/5
    a,b = estimateOrder(x,y)

    xx = np.linspace(0, 100, 50)
    F = fitFunc(a,b)
    yy = F(xx)

    print "Order is {}".format(b)

    plt.figure(1)
    plt.plot(xx, yy, color='b')
    plt.scatter(x, y, color='r')
    plt.show()

# function that goes through n nested loops which each do loopSize calls to f ( y**n ops)
def nestN(n, loopSize, f):
    if n >= 1:
        for x in xrange(loopSize):
            nestN(n-1, loopSize, f)
    else:
       f()
    return None

# we define some first order functions to test with, that we expect relatively constant
# execution time from
funcs = [ lambda : 1000**10, lambda: "hello"+ "world", lambda: [0]*100]
order = 4
timeitSize = 1
        
x2 = np.arange(1,50,10)
y2 = [ timeit.timeit(lambda: nestN(order, i, funcs[1]),number=timeitSize) for i in x2]
plt.figure(2)
plt.scatter(x2,y2,color='b')
plt.show()
print estimateOrder(x2,y2)

