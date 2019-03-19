import matplotlib.pyplot as plt
import numpy as np

#step function 0 or 1
data = [0 for a in range(-10, 0)]
data.extend([1 for a in range(0, 10)])

plt.step(range(-10, 10), data)
plt.xlabel('a')
plt.ylabel('step(a)')
plt.xlim(-12, 12)
plt.ylim(-0.1, 1.1)

plt.savefig("step.png")
plt.show()

#Tanh function
data = [2 / (1 + np.exp(-2 * a)) -1 for a in range(-10, 10, 1)]

plt.plot(range(-10, 10), data)
plt.xlabel('a')
plt.ylabel('tanh(a)')
plt.xlim(-12, 12)
plt.ylim(-1.1, 1.1)

plt.savefig("Tanh.png")
plt.show()

#SIGMOID function
data = [1 / (1 + np.exp(-a)) for a in range(-10, 10, 1)]

plt.plot(range(-10, 10), data)
plt.xlabel('a')
plt.ylabel('sigmoid(a)')
plt.xlim(-12, 12)
plt.ylim(-1.1, 1.1)

plt.savefig("sigmoid.png")
plt.show()

#RELU function (rectified linear unit)
data = [max(0, a) for a in range(-10, 10, 1)]

plt.plot(range(-10, 10), data)
plt.xlabel('a')
plt.ylabel('relu(a)')
plt.xlim(-12, 12)
plt.ylim(-1.1, 1.1)

plt.savefig("relu.png")
plt.show()