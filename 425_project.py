import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import time

m = 28
m_test = 28

# establish training data
X, Y = loadlocal_mnist(images_path='train-images.idx3-ubyte', labels_path='train-labels.idx1-ubyte')
print('MNIST X:', X.shape, 'MNIST Y:', Y.shape)
x_temp = []
y_temp = []
for i in range(m):  # replace with len(Y) to get entire set or m to get select points
    if Y[i] == 0:
        y_temp.append(1)
    if Y[i] == 5:
        y_temp.append(-1)
    if Y[i] == 0 or Y[i] == 5:
        x_temp.append(X[i])

y = np.asarray(y_temp)
x = np.asarray(x_temp)
print('x:', x.shape, 'y:', y.shape)

# establish testing data
x_temp = []
y_temp = []
X_test, Y_test = loadlocal_mnist(images_path='t10k-images.idx3-ubyte', labels_path='t10k-labels.idx1-ubyte')
print('MNIST X_test:', X_test.shape, 'MNIST Y_test:', Y_test.shape)
for i in range(m_test):
    if Y_test[i] == 0:
        y_temp.append(1)
    if Y_test[i] == 5:
        y_temp.append(-1)
    if Y_test[i] == 0 or Y_test[i] == 5:
        x_temp.append(X_test[i])

y_test = np.asarray(y_temp)
x_test = np.asarray(x_temp)
print('x_test:', x_test.shape, 'y_test:', y_test.shape)

# SVM method (add graph?)
# sklearn method
start_time = time.time()
svc = LinearSVC()  # svm.SVC(kernel='linear', degree=5)
svc.fit(x, y)
# print('x:', x.shape, 'y:', y.shape)

# now predict the classification of test data
y_pred = svc.predict(x_test)
# print('x_test:', x_test.shape, 'y_test:', y_test.shape)
print('Confusion matrix: \n', confusion_matrix(y_test, y_pred))
print('Score:', svc.score(x_test, y_test, sample_weight=None))
print("--- %s seconds ---" % (time.time() - start_time))

m, n = x.shape
# PCA
start_time = time.time()
mu_hat = np.zeros((m, 1))
for i in range(m):
    mu_hat[i] = np.average(x[i])
Z = x - mu_hat
U_full, S_full, V_full = np.linalg.svd(Z, full_matrices=True)

error_PCA = np.zeros(m)  # change if switching r value
for r in range(m):
    V = V_full[:, 0:r + 1]
    B = x @ V
    ones = np.ones((m, 1))
    B_hat = np.hstack((ones, B))
    theta_v_hat = np.dot(np.linalg.pinv(B_hat), y)
    c_hat = theta_v_hat[0]
    theta_v = theta_v_hat[1:r + 2]
    y_test_hat = np.matmul(x_test, (np.matmul(V, theta_v))) + c_hat
    error_PCA[r] = np.square(np.linalg.norm(y_test - y_test_hat)) / np.square(np.linalg.norm(y_test))
# print(error_PCA.shape)
best_r_PCA = np.where(error_PCA == error_PCA.min())
print('PCA best r:', best_r_PCA)
print("--- %s seconds ---" % (time.time() - start_time))

# plot = plt.gca()
# # plotting PCA
# for r in range(m):
#     plot.set_title('PCA')
#     plot.set_xlabel('r')
#     plot.set_ylabel('error_PCA')
#     plot.scatter(r, error_PCA[r], color='g')
# plt.show()

# GDA model
start_time = time.time()
phi = np.mean(y)

mu0 = 0
mu1 = 0
y_0 = 0
y_1 = 0

for i in range(m):
    if y[i] == 1:
        mu0 = x[i] + mu0
        y_0 += 1
    else:
        mu1 = x[i] + mu1
        y_1 += 1
mu0 = mu0 // y_0
mu1 = mu1 // y_1

cov = np.zeros((n, n))
for i in range(m):
    if y[i] == 1:
        cov[i, i] = np.mean(np.square(x[i] - mu0[i]))
    else:
        cov[i, i] = np.mean(np.square(x[i] - mu1[i]))

m_test, n = x_test.shape
accuracy = 0
for i in range(m_test):
    if y_test[i] == 1:
        y_hat = max(map(max, norm.pdf(x_test, mu0, cov[i, i]))) * (1 - phi)
        accuracy = np.abs(y_test[i] - y_hat) + accuracy
    else:
        y_hat = max(map(max, norm.pdf(x_test, mu1, cov[i, i]))) * phi
        accuracy = np.abs(y_test[i] - y_hat) + accuracy
accuracy = accuracy / m_test
print('GDA error:', accuracy)
print("--- %s seconds ---" % (time.time() - start_time))

# logistic regression
start_time = time.time()


# establish theta_hat_gd, from ipython notebook optimization_intro.ipynb
def cost_gradient(theta, x, y_gd):
    yhat_gd = np.squeeze(x).dot(np.squeeze(theta))
    return 2 * (yhat_gd - y_gd) * x.T


theta_init = np.random.randn(n, 1)
theta_list = [theta_init]
max_iter = 1000
step_size = 1e-2
X_gd = np.random.randn(m, n)  # features
noise_var = 1e-6
noise = np.sqrt(noise_var) * np.random.randn(m, 1)
y_gd = np.dot(X_gd, theta_init) + noise
m, n = X_gd.shape
for _ in range(max_iter):
    grad = 0
    for ii in range(m):
        grad += cost_gradient(theta_init, X_gd[ii:ii + 1, :], y_gd[ii])
    theta_new = theta_init - step_size * grad / m
    theta_init = theta_new
    theta_list.append(theta_new)
theta_hat_gd = np.squeeze(np.array(theta_list))

# y_hat comparison
y_hat_LR = 0
accuracy2 = 0
for i in range(m_test):
    exponential = math.exp(np.dot(np.transpose(theta_hat_gd[i]), x_test[i]))
    g = 1 / (1 + exponential)
    if (g > .5):
        y_hat_LR = 1
    else:
        y_hat_LR = 0
    accuracy2 = np.abs(y_test[i] - y_hat_LR) + accuracy2
accuracy2 = accuracy2 / m_test
print('Logistic Regression error:', accuracy2)
print("--- %s seconds ---" % (time.time() - start_time))

# plot logistic regression and GDA
plot = plt.gca()
plot.scatter(m, accuracy, color='b', label='GDA-' + str(m))
plot.scatter(m, accuracy2, color='g', label='logistic regression-' + str(m))
plt.legend(loc="lower left")
plt.ylim(0, 1)
plt.show()
