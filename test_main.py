import numpy as np , random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


np.random.seed(100)
# 5. Generating the data set
classA = np.concatenate (
    (np.random. randn(10 , 2) * 0.2 + [1.5 , 0.5] ,
     np.random. randn(10 , 2) * 0.2 + [ -1.5 , 0.5]))
classB = np.random. randn(20 , 2) * 0.2 + [0.0 , -0.5]

inputs = np. concatenate (( classA , classB ))
targets = np. concatenate (
    (np. ones ( classA . shape [0]) ,
     -np. ones ( classB . shape [0])))

N = inputs.shape [0] # Number of rows (samples)

permute=list(range (N))
random. shuffle(permute)
inputs = inputs[permute, : ]
targets = targets[permute ]
 
k_type = 'polynomial'

def K(x,y, kernel_type):
    p = 2
    sigma = 1
    if kernel_type == 'linear':
        return np.dot(x, y)

    elif kernel_type == 'polynomial':
        return (1 + np.dot(x, y)) ** p

    elif kernel_type == 'RBF':
        return math.exp(-np.linalg.norm(x-y)**2/(2*sigma))

def calculate_matrix(X, t):
    P = np.zeros((N , N))
    for i in range(N):
        for j in range(N):
            P[i,j] = t[i]*t[j]*K(X[i], X[j], k_type)
    return P

P = calculate_matrix(inputs, targets)


def objective(alpha):
    # returns a scalar value, effectively implementing the expression should be minimized in equation (4)
    return (1/2)*np.dot(alpha, np.dot(alpha, P)) - np.sum(alpha)

def zerofun(alpha):
    # This function should implement the equality constraint of (10). Also here, you can make use of np.dot to be efficient.
    return np.dot(alpha,targets)

constraint={'type':'eq', 'fun':zerofun}
start = np.zeros(N)
C = 1

ret = minimize( objective , start , bounds=[(0, C) for b in range(N)], constraints={'type':'eq', 'fun':zerofun} )
alpha = ret ['x']

# extract the non-zero α values that are greater than 10^-5 and save them as a np array along with their corresponding input and target values.
alpha_nonzeros = [(alpha[i], inputs[i], targets[i]) for i in range(N) if abs(alpha[i]) > 10e-5]


print("Alpha values:", alpha)
print("Optimization success:", ret['success'])
print("Message:", ret['message'])

# calculate the bias term b using equation (7) for each of the support vectors.
def calculate_b():
    bsum = 0
    for value in alpha_nonzeros:
        bsum += value[0] * value[2] * K(value[1], alpha_nonzeros[0][1], k_type)
    return bsum - alpha_nonzeros[0][2]

b = calculate_b()
print(b)


def ind(S, K):
    # It is used to determine the class of the input. + value means 1 class, - value means -1 class.
    totsum = 0
    for value in alpha_nonzeros:
        totsum += value[0] * value[2] * K(S, value[1],k_type)
    return totsum - b

# 6. Plotting the data set
plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.')
plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.')
#plt.axis('equal') # Force same scale on both axes
#plt.savefig('svmplot.pdf') # Save a copy in a file 
#plt.show() # Show the plot on the screen


# 7. Plotting the decision boundary
xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
grid = np.array([[ind(np.array([x, y]), K) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), 
            colors=('red', 'black', 'blue'), 
            linewidths=(1, 3, 1 ))

plt.plot([p[0] for p in classA],
            [p[1] for p in classA],
            'b.')
plt.plot([p[0] for p in classB],
            [p[1] for p in classB],
            'r.')
plt.axis('equal') # Force same scale on both axes
plt.savefig('svmplot.pdf') # Save a copy in a file
plt.show() # Show the plot on the screen


print(ind(classA[0],K), 'A')
print(ind(classB[0],K), 'B')