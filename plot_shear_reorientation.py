"""Testing re-orientation in shearflow."""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from fiberpy.orientation import folgar_tucker_ode, get_equivalent_aspect_ratio

# shear rate
gamma = 1.0
# aspect ratio
ar = 13.0
# equivalent aspect ratio
are = get_equivalent_aspect_ratio(ar)
print ("Equivalent aspect ratio is %f" % are)
xi = (are**2 - 1) / (are**2 + 1)
# period fo rotation
T = 0.5*np.pi/gamma*(are + 1.0/are)
print ("Period for a quarter rotation is %f" % T)
# time steps
t = np.linspace(0, T, 500)

# load simulation data
sim_data = np.loadtxt("data/orientations.csv", delimiter=',', skiprows=1)


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, gamma, 0.0],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


A0 = np.array([[0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0]])

# computed solution
N = odeint(folgar_tucker_ode, A0.ravel(), t, args=(xi, L, 0.0))

# plots
labels = ["$N_{11}$", "$N_{12}$", "$N_{13}$",
          "$N_{21}$", "$N_{22}$", "$N_{23}$",
          "$N_{31}$", "$N_{32}$", "$N_{33}$"]

subplots = [0, 1, 2, 4, 5, 8]

legend_list = ["Jeffery", "SPH simulation"]

plt.figure()
for i in subplots:
    plt.subplot("33"+str(i+1))
    p = plt.plot(t, N[:, i], sim_data[:, 0], sim_data[:, i+1])
    plt.xlabel("Time $t$ in s")
    plt.ylabel(labels[i])
    plt.ylim([-1, 1])
plt.legend(legend_list, bbox_to_anchor=(-2.0, 0.4))
plt.tight_layout()
plt.show()
