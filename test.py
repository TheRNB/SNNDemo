import numpy as np
import matplotlib.pyplot as plt

def STDP_delta_function(t):
    if t > 0:
        return 0.8 * np.exp(-t/2) # 0.8,5
    if t < 0:
        return -0.28 * np.exp(t/8) #Equilibrium at -0.703490 # 0.28,20
    else:
        return 0

def flatSTDP_delta_function(t):
    if t > 0:
        return 0.1
    if t < 0:
        return -0.1
    else:
        return 0


vfun = np.vectorize(flatSTDP_delta_function)

x = np.linspace(-60,60)
y = vfun(x)

plt.axis((-60.0,60.0,-1.0,1.0))
plt.grid(which="both", axis="both", color="GRAY")
plt.plot(x,y,'-')
plt.savefig("/Users/aaron/Downloads/Figure_1_FSTDP_ALow.jpg", dpi=600)