import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def plot(data, name):
    x_axis = range(len(data))
    plt.plot(x_axis, data)
    plt.xlabel("Iteration")
    plt.ylabel(name)
    plt.show()


class NeuralNetwork:
    def __init__(self, inSize, hSize, outSize, useSigmoid = True):
        self.inSize = inSize + 1
        self.hSize = hSize
        self.outSize = outSize
        self.useSigmoid = useSigmoid

        self.inNode = [1.0] * self.inSize
        self.hNode = [1.0] * self.hSize
        self.outNode = [1.0] * self.outSize

        # # initiale weight
        self.wi = np.random.randn(self.inSize, self.hSize) * 0.2
        self.wo = np.random.randn(self.hSize, self.outSize) * 0.2

        # initiate bias
        self.bi = np.zeros((self.inSize, self.hSize))
        self.bo = np.zeros((self.hSize, self.outSize))

    def feedforward(self, inputs):
        # input layer
        for i in range(self.inSize - 1):
            self.inNode[i] = inputs[i]

        # hidden layer
        for j in range(self.hSize):
            sum = 0.0
            for i in range(self.inSize):
                sum = sum + self.inNode[i] * self.wi[i][j]
            self.hNode[j] = sigmoid(sum)


        # output layer
        for k in range(self.outSize):
            sum = 0.0
            for j in range(self.hSize):
                sum = sum + self.hNode[j] * self.wo[j][k]
            self.outNode[k] = sigmoid(sum)

        return self.outNode[:]


    def backPropagate(self, targets, learningRate, momentum):
        # calculate output layer error
        delta = [0.0] * self.outSize
        for k in range(self.outSize):
            error = targets[k]-self.outNode[k]
            delta[k] = dsigmoid(self.outNode[k]) * error

        # calculate hidden layer error
        hidden_deltas = [0.0] * self.hSize
        for j in range(self.hSize):
            error = 0.0
            for k in range(self.outSize):
                error = error + delta[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.hNode[j]) * error

        # calculate input layer error
        for j in range(self.hSize):
            for k in range(self.outSize):
                change = delta[k]*self.hNode[j]
                self.wo[j][k] = self.wo[j][k] + learningRate*change + momentum*self.bo[j][k]
                self.bo[j][k] = change

        # update weight
        for i in range(self.inSize):
            for j in range(self.hSize):
                change = hidden_deltas[j]*self.inNode[i]
                self.wi[i][j] = self.wi[i][j] + learningRate*change + momentum*self.bi[i][j]
                self.bi[i][j] = change

        # calculate error
        error = 0.0
        error += 0.5 * (targets[2] - self.outNode[2]) ** 2
        return error


    def test(self, Xtest, Ytest):
        count = 0
        for i in range(len(Xtest)):
            target = np.where(Ytest[i] == 1)
            result = self.feedforward(Xtest[i])
            count += (target == np.where(result == max(result)))
        accuracy = float(count/len(Xtest))
        print('Trainning Accuracy: {}'.format(accuracy))


    def train(self, Xtrain, Ytrain, iterations, learningRate=0.1, momentum=0.01):
        accuracy = []
        losses = []
        for i in range(iterations):
            loss = 0.0
            count = 0
            for j in range(len(Xtrain)):
                target = Ytrain[j]
                result = self.feedforward(Xtrain[j])
                count += (np.where(target == 1) == np.where(result == max(result)))
                loss += self.backPropagate(target, learningRate, momentum)
            losses.append(loss)
            accuracy.append(count/len(Xtrain))
            if i%100 == 0:
                print('Training loss: {}, Trainning Accuracy: {}'.format(loss, count/len(Xtrain)))

        plot(accuracy, "Accuracy")
        plot(losses, "loss")


if __name__ == '__main__':
    # load data
    inputs = pd.read_csv('iris.csv')
    values = inputs.values
    features = values[0:, 0:4]

    # In order to shuffle data, I put them in a list.
    data = []
    for i in range(len(features)):
        sample = []
        sample.append(list(features[i]))
        if values[i][4] == 'Iris-setosa':
            sample.append([1, 0, 0])
        elif values[i][4] == 'Iris-versicolor':
            sample.append([0, 1, 0])
        else:
            sample.append([0, 0, 1])
        data.append(sample)

    # shuffle
    random.shuffle(data)

    # split data into train and test
    X = []
    Y = []
    for sample in data:
        X.append(sample[0])
        Y.append(sample[1])
    Xtrain = np.array(X[0:100])
    Xtest = np.array(X[101:])
    Ytrain = np.array(Y[0:100])
    Ytest = np.array(Y[101:])

    ############## normal training and test (2 hidden neurons) ##############
    nn = NeuralNetwork(4, 4, 3)
    nn.train(Xtrain, Ytrain, iterations=2000)
    nn.test(Xtest, Ytest)

    ############## 2 neurons in hidden layer ##############
    nn = NeuralNetwork(4, 2, 3)
    nn.train(Xtrain, Ytrain, iterations=2000)
    nn.test(Xtest, Ytest)

    ############## 8 neurons in hidden layer ##############
    nn = NeuralNetwork(4, 8, 3)
    nn.train(Xtrain, Ytrain, iterations=2000)
    nn.test(Xtest, Ytest)