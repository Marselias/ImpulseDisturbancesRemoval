import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
file = r'02.wav'
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
odchylenie = 1
blad_poprzedni = 0
suma = 0
srednia = 0
ile = 0
errors = []
x=0
theta1 = []
theta2 = []
theta3 = []
theta4 = []

for i in range(order):
    regression_vector[i] = data[i]

    #theta[i] = 0.01
for i in range(4,n-5):
    if x>0:
        regression_vector[0] = data[i-1]
        regression_vector[1] = data[i - 2]
        regression_vector[2] = data[i - 3]
        regression_vector[3] = data[i - 4]
        x=x-1
    else:
        predykcja = np.matmul(np.transpose(regression_vector),theta)
        blad = data[i] - predykcja
        if abs(blad)> 5*odchylenie:

            suma = suma + blad_poprzedni **2
            srednia = suma/i
            odchylenie = sqrt(srednia)
                    # regression_vector[3] = regression_vector[2]
                    # regression_vector[2] = regression_vector[1]
                    # regression_vector[1] = regression_vector[0]
                    # regression_vector[0] = predykcja
                    # buf = np.matmul(np.transpose(regression_vector), P)
                    # buf = np.matmul(buf, regression_vector)
                    # k = np.matmul(P, regression_vector) / (lambda_ + buf)
                    # buf2 = np.matmul(P, regression_vector)
                    # buf2 = np.matmul(buf2, np.transpose(regression_vector))
                    # buf2 = np.matmul(buf2, P)
                    # P = (P - buf2 / (lambda_ + buf)) / lambda_
            predykcja = np.matmul(np.transpose(regression_vector), theta)
            blad = data[i + 1] - predykcja
            x=1
            errors.append(i)
            theta1.append(theta[0])
            theta2.append(theta[1])
            theta3.append(theta[2])
            theta4.append(theta[3])
            if abs(blad)>5*odchylenie:
                suma = suma + blad_poprzedni **2
                srednia = suma/(i+1)
                odchylenie = sqrt(srednia)
                        # regression_vector[3] = regression_vector[2]
                        # regression_vector[2] = regression_vector[1]
                        # regression_vector[1] = regression_vector[0]
                        # regression_vector[0] = predykcja
                        # buf = np.matmul(np.transpose(regression_vector), P)
                        # buf = np.matmul(buf, regression_vector)
                        # k = np.matmul(P, regression_vector) / (lambda_ + buf)
                        # buf2 = np.matmul(P, regression_vector)
                        # buf2 = np.matmul(buf2, np.transpose(regression_vector))
                        # buf2 = np.matmul(buf2, P)
                        # P = (P - buf2 / (lambda_ + buf)) / lambda_
                predykcja = np.matmul(np.transpose(regression_vector), theta)
                blad = data[i + 2] - predykcja
                x=2
                errors.append(i+1)
                theta1.append(theta[0])
                theta2.append(theta[1])
                theta3.append(theta[2])
                theta4.append(theta[3])
                if abs(blad) > 5 * odchylenie:
                    suma = suma + blad_poprzedni ** 2
                    srednia = suma / (i + 2)
                    odchylenie = sqrt(srednia)
                            # regression_vector[3] = regression_vector[2]
                            # regression_vector[2] = regression_vector[1]
                            # regression_vector[1] = regression_vector[0]
                            # regression_vector[0] = predykcja
                            # buf = np.matmul(np.transpose(regression_vector), P)
                            # buf = np.matmul(buf, regression_vector)
                            # k = np.matmul(P, regression_vector) / (lambda_ + buf)
                            # buf2 = np.matmul(P, regression_vector)
                            # buf2 = np.matmul(buf2, np.transpose(regression_vector))
                            # buf2 = np.matmul(buf2, P)
                            # P = (P - buf2 / (lambda_ + buf)) / lambda_
                    predykcja = np.matmul(np.transpose(regression_vector), theta)
                    blad = data[i + 3] - predykcja
                    x=3
                    errors.append(i+2)
                    theta1.append(theta[0])
                    theta2.append(theta[1])
                    theta3.append(theta[2])
                    theta4.append(theta[3])
                    if abs(blad) > 5* odchylenie:
                        suma = suma + blad_poprzedni ** 2
                        srednia = suma / (i + 3)
                        odchylenie = sqrt(srednia)
                                # regression_vector[3] = regression_vector[2]
                                # regression_vector[2] = regression_vector[1]
                                # regression_vector[1] = regression_vector[0]
                                # regression_vector[0] = predykcja
                                # buf = np.matmul(np.transpose(regression_vector), P)
                                # buf = np.matmul(buf, regression_vector)
                                # k = np.matmul(P, regression_vector) / (lambda_ + buf)
                                # buf2 = np.matmul(P, regression_vector)
                                # buf2 = np.matmul(buf2, np.transpose(regression_vector))
                                # buf2 = np.matmul(buf2, P)
                                # P = (P - buf2 / (lambda_ + buf)) / lambda_
                        x=4
                        errors.append(i + 3)
                        theta1.append(theta[0])
                        theta2.append(theta[1])
                        theta3.append(theta[2])
                        theta4.append(theta[3])


        else:

            ile +=1
            suma = suma + blad**2
            srednia = srednia/i
            odchylenie = sqrt(srednia)
            buf = np.matmul(np.transpose(regression_vector), P)
            buf = np.matmul(buf, regression_vector)
            k = np.matmul(P, regression_vector) / (lambda_ + buf)

            theta = theta + k * blad
            buf2 = np.matmul(P, regression_vector)
            buf2 = np.matmul(buf2, np.transpose(regression_vector))
            buf2 = np.matmul(buf2, P)
            P = (P - buf2 / (lambda_ + buf)) / lambda_
            blad_poprzedni = blad
            regression_vector[3] = regression_vector[2]
            regression_vector[2] = regression_vector[1]
            regression_vector[1] = regression_vector[0]
            regression_vector[0] = data[i]
            theta1.append(theta[0])
            theta2.append(theta[1])
            theta3.append(theta[2])
            theta4.append(theta[3])

        if x==1:
            data[i] = (data[i+1]+data[i-1])/2

        if x==2:
            delta = (data[i+2]-data[i-1])/3
            data[i] = data[i-1] + delta
            data[i+1] = data[i - 1] + 2*delta

        if x==3:
            delta = (data[i + 3] - data[i - 1]) / 4
            data[i] = data[i-1] + delta
            data[i+1] = data[i - 1] + 2*delta
            data[i+2] = data[i-1] + 3*delta

        if x==4:
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
sf.write('new16.wav',data,samplerate)


