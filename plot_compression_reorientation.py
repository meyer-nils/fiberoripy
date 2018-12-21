"""Testing re-orientation in shearflow."""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from fiberpy.orientation import folgar_tucker_ode


h0 = 0.0062
hf = 0.002
v = 0.001
Ci = 0.0001

T = (h0 - hf) / v
t = np.linspace(0, T, 500)


def D(t):
    """Symmetric strain rate tensor."""
    return np.array([[v / (h0 - v * t), 0, 0],
                     [0, 0, 0],
                     [0, 0, -v / (h0 - v * t)]])


def W(t):
    """Skew-symmetric strain rate tensor."""
    return np.zeros((3, 3))


A0 = np.array([[0.5, 0.0, 0.0],
               [0.0, 0.5, 0.0],
               [0.0, 0.0, 0.0]])

A = odeint(folgar_tucker_ode, A0.ravel(), t, args=(25.0, D, W, Ci))

plt.subplot(331)
plt.plot(t, A[:, 0])
plt.xlabel("Time in s")
plt.ylabel("A11")

plt.subplot(332)
plt.plot(t, A[:, 1])
plt.xlabel("Time in s")
plt.ylabel("A12")

plt.subplot(333)
plt.plot(t, A[:, 2])
plt.xlabel("Time in s")
plt.ylabel("A13")

plt.subplot(334)
plt.plot(t, A[:, 3])
plt.xlabel("Time in s")
plt.ylabel("A21")

plt.subplot(335)
plt.plot(t, A[:, 4])
plt.xlabel("Time in s")
plt.ylabel("A22")

plt.subplot(336)
plt.plot(t, A[:, 5])
plt.xlabel("Time in s")
plt.ylabel("A23")

plt.subplot(337)
plt.plot(t, A[:, 6])
plt.xlabel("Time in s")
plt.ylabel("A31")

plt.subplot(338)
plt.plot(t, A[:, 7])
plt.xlabel("Time in s")
plt.ylabel("A32")

plt.subplot(339)
plt.plot(t, A[:, 8])
plt.xlabel("Time in s")
plt.ylabel("A33")


plt.suptitle("Folgar Tucker and IBOF Closure")
plt.show()
