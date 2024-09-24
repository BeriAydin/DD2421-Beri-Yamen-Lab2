import numpy , random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt



# 1. Generating the data set

classA = numpy.concatenate (
    (numpy.random. randn(10 , 2) * 0.2 + [1.5 , 0.5] ,
     numpy.random. randn(10 , 2) * 0.2 + [ -1.5 , 0.5]))
classB = numpy.random. randn(20 , 2) * 0.2 + [0.0 , -0.5]

inputs = numpy. concatenate (( classA , classB ))
targets = numpy. concatenate (
    (numpy. ones ( classA . shape [0]) ,
     -numpy. ones ( classB . shape [0])))

N = inputs.shape [0] # Number of rows (samples)

permute=list(range (N))
random. shuffle(permute)
inputs = inputs[permute, : ]
targets = targets[permute ]





def ind(S):
    # It is used to determine the class of the input. + value means 1 class, - value means -1 class.
    return 0

def linear_kernel(x, y):
    # This kernel simply returns the scalar product between the two points. This results in a linear separation.
    return numpy.dot(x, y)

def polynomial_kernel(x, y, p):
    # This kernel allows for curved decision boundaries. The exponent p (a positive integer) controls the degree of the polynomials. p = 2 will make quadratic shapes (ellipses, parabolas, hyperbolas). Setting p = 3 or higher will result in more complex shapes.
    return (1 + numpy.dot(x, y)) ** p

def RBF_kernel(x, y, sigma=5.0):
    # radial basis function kernel. This kernel uses the explicit euclidian distance between the two datapoints, and often results in very good boundaries. The parameter σ is used to control the smoothness of the boundary
    return math.exp(-numpy.linalg.norm(x-y)**2/(2*sigma))

def calculate_matrix(kernel_func, parameter):
    P = numpy.zeros((N , N))
    for i in range(N):
        for j in range(N):
            P[i,j] = targets(i)*targets(j)*kernel_func(input(i),input(j), **parameter)
    return P

def objective (alpha):
    # returns a scalar value, effectively implementing the expression should be minimized in equation (4)
    return 0

def zerofun(alpha):
    # This function should implement the equality constraint of (10). Also here, you can make use of numpy.dot to be efficient.
    return 0
