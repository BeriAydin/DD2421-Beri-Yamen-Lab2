import numpy as np , random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
 
def linear_kernel(x, y):
    # This kernel simply returns the scalar product between the two points. This results in a linear separation.
    return np.dot(x, y)

def polynomial_kernel(x, y, p):
    # This kernel allows for curved decision boundaries. The exponent p (a positive integer) controls the degree of the polynomials. p = 2 will make quadratic shapes (ellipses, parabolas, hyperbolas). Setting p = 3 or higher will result in more complex shapes.
    return (1 + np.dot(x, y)) ** p

def RBF_kernel(x, y, sigma=5.0):
    # radial basis function kernel. This kernel uses the explicit euclidian distance between the two datapoints, and often results in very good boundaries. The parameter σ is used to control the smoothness of the boundary
    return math.exp(-np.linalg.norm(x-y)**2/(2*sigma))
K = linear_kernel       ######################################### Change the kernel here ############################################

def calculate_matrix(X, t, K, parameter = None):
    P = np.zeros((N , N))
    if parameter is None:
        for i in range(N):
            for j in range(N):
                P[i,j] = t[i]*t[j]*K(X[i], X[j])
    else:
        for i in range(N):
            for j in range(N):
                P[i,j] = t[i]*t[j]*K(X[i], X[j], parameter)
    return P
P = calculate_matrix(inputs, targets, K)


def objective (alpha):
    # returns a scalar value, effectively implementing the expression should be minimized in equation (4)
    return 0.5* np.dot(alpha, np.dot(P, alpha)) + np.sum(alpha)

def zerofun(alpha):
    # This function should implement the equality constraint of (10). Also here, you can make use of np.dot to be efficient.
    return np.dot(alpha,targets)

constraint={'type':'eq', 'fun':zerofun}
start = np.random.rand(N) * 0.1
C = 1000000000

ret = minimize( objective , start , bounds=[(0, C) for b in range(N)], constraints={'type':'eq', 'fun':zerofun} )
foundOpt = ret['success']
alpha = ret ['x']

# extract the non-zero α values that are greater than 10^-5 and save them as a np array along with their corresponding input and target values.
alpha_nonzeros = []
alpha_threshold = 0
for i in range(N):
    if alpha[i] > alpha_threshold:
        alpha_nonzeros.append((alpha[i], inputs[i], targets[i]))



print("Alpha values:", alpha)
print("Optimization success:", ret['success'])
print("Message:", ret['message'])

# calculate the bias term b using equation (7) for each of the support vectors.
def calculate_b(K, parameter = None):
    s = alpha_nonzeros[0] # choose the first support vector
    temp_b = 0
    for i in range(N):
        temp_b += alpha[i]*targets[i]*K(inputs[i], s[1])
    temp_b -= s[2]
    return temp_b
b = calculate_b(K)



def ind(S, K, parameter = None):
    # It is used to determine the class of the input. + value means 1 class, - value means -1 class.
    temp_class = 0
    if parameter is None:
        for i in range(N):
            temp_class += alpha[i]*targets[i]*K(S, inputs[i])
        temp_class -= calculate_b(K)
    else:
        for i in range(N):
            temp_class += alpha[i]*targets[i]*K(S, inputs[i], parameter)
        temp_class -= calculate_b(K, parameter)
    # log the class of the input and the clas
    print("Classs:", np.sign(temp_class))
    return np.sign(temp_class)

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






