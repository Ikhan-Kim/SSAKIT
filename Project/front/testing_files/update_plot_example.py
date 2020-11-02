import matplotlib.pyplot as plt
import numpy as np

plt.ion()
# for i in range(50):
#     y = np.random.random([10, 1])
#     plt.plot(y)
#     plt.draw()
#     plt.pause(0.3)
#     plt.clf()


def test():
    y = np.random.random([10, 1])
    plt.plot(y)
    plt.draw()
    plt.pause(0.3)
    plt.clf()


for i in range(50):
    test()
