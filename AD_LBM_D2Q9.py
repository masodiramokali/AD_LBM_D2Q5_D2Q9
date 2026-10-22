import numpy as np
import matplotlib.pyplot as plt

# 1. LATTICE TYPE

lattice = "D2Q9"

# 2. LATTICE VELOCITIES AND WEIGHTS

ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# 3. GRID SIZE AND TIME STEPS

Nx = 101  
Ny = 101  
Nt = 200  

x = np.arange(Nx)
y = np.arange(Ny)

X, Y = np.meshgrid(x, y)

# 4. INITIAL CONDITIONS (GAUSSIAN PULSE)

x0 = 30 
y0 = 50
sigma = 7

phi = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

# 5. LATTICE BOLTZMANN METHOD PARAMETERS

Q =9  
tau = 0.8
cs2 = 1/3 
D = cs2 * (tau - 0.5) 

# 6. ADVECTION VELOCITY

ux = 0.05
uy = 0.0

# 7. LATTICE BOLTZMANN METHOD IMPLEMENTATION

f = np.zeros((Q, Nx, Ny))
for i in range(Q):
    f[i] = w[i] * phi * (1 + (ex[i] * ux + ey[i] * uy) / cs2)

# 8. SAVE THE INITIAL STATE

phi_initial = np.sum(f, axis=0)

# 9. LATTICE BOLTZMANN TIME EVOLUTION

for t in range(Nt):

    # Macroscopic Scalar Field
    phi = np.sum(f, axis=0)

    # Collision step
    for i in range(Q):
        feq = w[i] * phi * (1 + (ex[i] * ux + ey[i] * uy) / cs2)
        f[i] = f[i]-(f[i] - feq) / tau

    # Streaming step
    for i in range(Q):
        f[i] = np.roll(f[i], shift=(ey[i], ex[i]), axis=(0, 1))

# 10. FINAL FIELD
phi_final = np.sum(f, axis=0)

# 10. ANALYTICAL SOLUTION (Gaussian: centre moves with u, variance grows by 2Dt)
# The domain is periodic, so sum the neighbouring images to include the wrapped tails.

sigma2_t = sigma**2 + 2 * D * Nt
xc = x0 + ux * Nt
yc = y0 + uy * Nt
phi_exact = np.zeros((Ny, Nx))
for mx in (-1, 0, 1):
    for my in (-1, 0, 1):
        phi_exact += (sigma**2 / sigma2_t) * np.exp(
            -((X - xc + mx * Nx)**2 + (Y - yc + my * Ny)**2) / (2 * sigma2_t)
        )


# 11. PLOTTING RESULTS
plt.figure()
plt.plot(x, phi_initial[Ny//2], label='Initial')
plt.plot(x, phi_final[Ny//2], label='Final')
plt.plot(x, phi_exact[Ny//2], '--', label='Final (analytical)')
plt.title('Advection of a Gaussian Pulse using LBM (D2Q9)')
plt.xlabel('x')
plt.ylabel('phi')
plt.legend()
plt.grid()
plt.show()
