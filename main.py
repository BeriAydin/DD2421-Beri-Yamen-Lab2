import numpy , random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt



# 1. Generating the data set

classA = numpy. concatenate (
(numpy.random. randn(10 , 2) ∗ 0.2 + [1.5 , 0.5] ,
numpy.random. randn(10 , 2) ∗ 0.2 + [ −1.5 , 0.5]))
classB = numpy.random. randn(20 , 2) ∗ 0.2 + [0.0 , −0.5]
inputs = numpy. concatenate (( classA , classB ))
targets = numpy. concatenate (
(numpy. ones ( classA . shape [0]) ,
−numpy. ones ( classB . shape [0])))
N = inputs . shape [0] # Number o f rows ( s a m p l e s )
permute=l i s t (range (N))
random. shuffle (permute)
inputs = inputs [ permute , : ]

def ind(S)
    

def linear_kernel(x, y):

    return numpy.dot(x, y)


def objective (alpha):
    # returns a scalar value, effectively implementing the expression should be minimized in equation (4)
    return 0