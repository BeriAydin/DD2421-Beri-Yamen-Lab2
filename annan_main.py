#from ctypes import _NamedFuncPointer
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np


def kernel(x,y,kType):
    p=10
    sigma=1
    if kType=="linear":
        val= np.dot(x,y)
    elif kType=="polynomial":
        val=(np.dot(x,y)+1)**p
    elif kType=="RBF":
        val=math.exp(-math.pow(np.linalg.norm(np.subtract(x, y)), 2)/(2 * math.pow(sigma,2)))
    else:
        return ValueError("Nan")
    return val


def objective(alpha):
    return (1/2)*np.dot(alpha, np.dot(alpha, P)) - np.sum(alpha)


def zerofun(alpha):
    summ = np.dot(alpha,t)
    return summ


def b(alpha,kType):
    bsum = 0
    for value in nonzero:
        bsum += value[0] * value[2] * kernel(value[1], nonzero[0][1], kType)
    return bsum - nonzero[0][2]


def ind(alpha,x, y,kType, b):
    totsum = 0
    for value in nonzero:
        totsum += value[0] * value[2] * kernel([x, y], value[1],kType)
    return totsum - b



# generate test data
np.random.seed(100)
classA = np.concatenate((
    np.random.randn(10, 2) * 0.4 + [1.5, 0.5],
    np.random.randn(10, 2) * 0.4 + [-1.5, 0.5],))

classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

N = inputs.shape[0]
permute = list(range(N))
random.shuffle(permute)
x = inputs = inputs[permute, :]
t = targets = targets[permute]

# plot points from test data
plt.plot([p[0] for p in classA] ,[p[1] for p in classA] ,'b .')
plt.plot([p[0] for p in classB] ,[p[1] for p in classB] ,'r.' )
plt.axis('equal') # Force same scale on both axes


# create start- and P-matrix
start = np.zeros(N)
P = np.zeros((N,N))
# specify kernel-method
Ktype="linear"

for i in range(N):
    for j in range(N):
        P[i,j]=t[i]*t[j]*kernel(x[i],x[j],Ktype)

# DONT KNOW
C = 1

# set lower and upper bounds for alpha
B = [(0,C) for i in range(N)]
XC = {'type':'eq', 'fun':zerofun}


ret = minimize(objective, start, bounds = B, constraints = XC)
alpha = ret['x']
# round to 1e-5
alpha = np.array([round(i, 5) for i in alpha])


nonzero = [(alpha[i], inputs[i], targets[i]) for i in range(N) if abs(alpha[i]) > 10e-5]

# print support vector in different color
#plt.plot([((p)[1])[0] for p in nonzero], [((p)[1])[1] for p in nonzero] ,'go' )


xgrid = np.linspace(-5,5)
ygrid = np.linspace(-4,4)
b = b(alpha, Ktype)

grid = np.array([[ind(alpha,x,y,Ktype,b) for x in xgrid ] for y in ygrid ] )
plt.contour(xgrid, ygrid, grid, (-1,0,1), colors=('red' ,'black', 'blue'), linewidths=(1 , 3 , 1))

plt.show() # Show the plot on the screen
#plt.savefig('svmplot.pdf') # Save a copy in a file