import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def sigmoid(X, theta):
    z = np.dot(X, theta[1:]) + theta[0]
    return 1.0 / ( 1.0 + np.exp(-z))

def binary_cross_entropy(y, hx):
    return -y.dot(np.log(hx)) - ((1 - y).dot(np.log(1-hx)))

def error_and_lost(x, theta, y):
    hx = sigmoid(x,theta)
    lost = binary_cross_entropy(y, hx)
    error = hx - y
    return error, lost

def SGD(x, y, theta, alpha, i):
    losts = []
    for i in range(i):
        error, lost = error_and_lost(x, theta, y)
        losts.append(lost)
        grad = x.T.dot(error)
        theta[0] = theta[0] - alpha * error.sum()
        theta[1:] = theta[1:] - alpha * grad
    return losts, theta

def predict(x, theta):
    return np.where(sigmoid(x, theta) >= 0.5, 1, 0)

def test(x,theta,y):
    correct = 0
    pred = predict(x, theta)
    for i in range(len(x)):
        if y[i] == pred[i]:
            correct += 1
    return correct/len(y)

def plot_decision_boundry(test_x,test_y,X_std,theta, y, classifier, h=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # decision surface
    x1_min, x1_max = X_std[:, 0].min() - 1, X_std[:, 0].max() + 1
    x2_min, x2_max = X_std[:, 1].min() - 1, X_std[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))
    Z = classifier(np.array([xx1.ravel(), xx2.ravel()]).T,theta)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(test_y)):
        plt.scatter(x=test_x[test_y == cl, 0], y=test_x[test_y == cl, 1],
                    alpha=0.9, c=cmap(idx),
                    marker=markers[idx], label=cl)

## prepare data
iris = load_iris()
y=(iris.target == 2)*1
x=iris.data[:, [1,2]]

x_nor = np.copy(x)
x_nor[:,0] = (x_nor[:,0] - x_nor[:,0].mean()) / x_nor[:,0].std()
x_nor[:,1] = (x_nor[:,1] - x_nor[:,1].mean()) / x_nor[:,1].std()

x_train,x_test,y_train,y_test = train_test_split(x_nor,y,test_size = 0.2)


## train and test
theta = np.zeros(3)
alpha = 0.01
i = 5000

lost, Theta = SGD(x_train,y_train, theta, alpha, i)
accuracy = test(x_test, Theta, y_test)

print ('\n Coefficients :', theta[0], theta[1], theta[2])
print ("\n Accuracy: ", accuracy)

plt.plot(range(1, i+1), lost)
plt.xlabel('Iterations')
plt.ylabel('lost')
plt.title('Logistic Regression')

plot_decision_boundry(x_test, y_test, x_train, theta, y_train, classifier=predict)
# # plt.title('Logistic Regression on Test Data')
# # plt.xlabel('sepal length ')
# # plt.ylabel('sepal width ')
# # plt.legend(loc='upper right')
# # plt.tight_layout()
#