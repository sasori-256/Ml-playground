# /src/test.py

import marimo
import numpy as np
import matplotlib.pyplot as plt

num: np.ndarray = np.array([1, 2, 3])
den: np.ndarray = np.array([0, 0, 1])
sys = marimo.tf(num, den)
t = np.linspace(0, 10, 100)
t, y = marimo.step(sys, t)
plt.plot(t, y)
plt.title("Step Response")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
print("Test completed successfully.")
