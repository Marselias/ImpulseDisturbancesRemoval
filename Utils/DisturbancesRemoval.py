from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf


file = os.path.join('SampleFiles', '02.wav')

data, samplerate = sf.read(file)
data = data[1:]
n = data.shape[0]
y_hat = np.ndarray((n,))
order = 4
c = 1000
theta = np.zeros((order, 1))
regression_vector = np.zeros((order, 1))
regression_vector2 = np.zeros((order, 1))
P = np.eye(order)
P = c*P
lambda_ = 0.99
dev = 1
prev_error = 0
summary = 0
mean = 0
how_many = 0
errors = []
x = 0
theta1 = []
theta2 = []
theta3 = []
theta4 = []

for i in range(order):
    regression_vector[i] = data[i]

for i in range(4, n-5):
    if x > 0:
        regression_vector[0] = data[i-1]
        regression_vector[1] = data[i - 2]
        regression_vector[2] = data[i - 3]
        regression_vector[3] = data[i - 4]
        x = x-1
    else:
        prediction = np.matmul(np.transpose(regression_vector), theta)
        error = data[i] - prediction
        if abs(error) > 5*dev:

            summary = summary + prev_error**2
            mean = summary/i
            dev = sqrt(mean)
            prediction = np.matmul(np.transpose(regression_vector), theta)
            error = data[i + 1] - prediction
            x = 1
            errors.append(i)
            theta1.append(theta[0])
            theta2.append(theta[1])
            theta3.append(theta[2])
            theta4.append(theta[3])
            if abs(error) > 5*dev:
                summary = summary + prev_error ** 2
                mean = summary / i
                dev = sqrt(mean)
                prediction = np.matmul(np.transpose(regression_vector), theta)
                error = data[i + 2] - prediction
                x = 2
                errors.append(i+1)
                theta1.append(theta[0])
                theta2.append(theta[1])
                theta3.append(theta[2])
                theta4.append(theta[3])
                if abs(error) > 5 * dev:
                    summary = summary + prev_error ** 2
                    mean = summary / i
                    dev = sqrt(mean)
                    prediction = np.matmul(np.transpose(regression_vector), theta)
                    error = data[i + 3] - prediction
                    x = 3
                    errors.append(i+2)
                    theta1.append(theta[0])
                    theta2.append(theta[1])
                    theta3.append(theta[2])
                    theta4.append(theta[3])
                    if abs(error) > 5 * dev:
                        summary = summary + prev_error ** 2
                        mean = summary / i
                        dev = sqrt(mean)
                        x = 4
                        errors.append(i + 3)
                        theta1.append(theta[0])
                        theta2.append(theta[1])
                        theta3.append(theta[2])
                        theta4.append(theta[3])

        else:

            how_many += 1
            summary = summary + error ** 2
            mean = summary / i
            dev = sqrt(mean)
            buf = np.matmul(np.transpose(regression_vector), P)
            buf = np.matmul(buf, regression_vector)
            k = np.matmul(P, regression_vector) / (lambda_ + buf)

            theta = theta + k * error
            buf2 = np.matmul(P, regression_vector)
            buf2 = np.matmul(buf2, np.transpose(regression_vector))
            buf2 = np.matmul(buf2, P)
            P = (P - buf2 / (lambda_ + buf)) / lambda_
            prev_error = error
            regression_vector[3] = regression_vector[2]
            regression_vector[2] = regression_vector[1]
            regression_vector[1] = regression_vector[0]
            regression_vector[0] = data[i]
            theta1.append(theta[0])
            theta2.append(theta[1])
            theta3.append(theta[2])
            theta4.append(theta[3])

        if x == 1:
            data[i] = (data[i+1]+data[i-1])/2

        if x == 2:
            delta = (data[i+2]-data[i-1])/3
            data[i] = data[i-1] + delta
            data[i+1] = data[i - 1] + 2*delta

        if x == 3:
            delta = (data[i + 3] - data[i - 1]) / 4
            data[i] = data[i-1] + delta
            data[i+1] = data[i - 1] + 2*delta
            data[i+2] = data[i-1] + 3*delta

        if x == 4:
            delta = (data[i + 4] - data[i - 1]) / 5
            data[i] = data[i - 1] + delta
            data[i + 1] = data[i - 1] + 2 * delta
            data[i + 2] = data[i - 1] + 3 * delta
            data[i+3] = data[i-1] + 4*delta
print(errors)
plt.subplot(211)
plt.plot(theta1)
plt.subplot(212)
plt.plot(theta2)
plt.show()
plt.subplot(211)
plt.plot(theta3)
plt.subplot(212)
plt.plot(theta4)
plt.show()

plt.show()
sf.write('new16.wav', data, samplerate)


