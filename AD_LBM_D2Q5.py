import numpy as np
import matplotlib.pyplot as plt

# 1. LATTICE VELOCITIES AND WEIGHTS (D2Q5)

ex = np.array([0, 1, 0, -1, 0])
ey = np.array([0, 0, 1, 0, -1])
w = np.array([1/3, 1/6, 1/6, 1/6, 1/6])

# 2. GRID SIZE AND TIME STEPS

Nx = 101
Ny = 101
Nt = 1000

x = np.arange(Nx)
y = np.arange(Ny)

X, Y = np.meshgrid(x, y)  # shape (Ny, Nx): axis 0 is y, axis 1 is x

# 3. INITIAL CONDITIONS (GAUSSIAN PULSE)

x0 = 30
y0 = 50
sigma = 7

phi = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

# 4. LATTICE BOLTZMANN METHOD PARAMETERS

Q = 5
tau = 0.8
cs2 = 1/3
D = cs2 * (tau - 0.5)

# 5. ADVECTION VELOCITY

ux = 0.05
uy = 0.0

# 6. LATTICE BOLTZMANN METHOD IMPLEMENTATION

f = np.zeros((Q, Ny, Nx))
for i in range(Q):
    f[i] = w[i] * phi * (1 + (ex[i] * ux + ey[i] * uy) / cs2)

# 7. SAVE THE INITIAL STATE

phi_initial = np.sum(f, axis=0)

# 8. LATTICE BOLTZMANN TIME EVOLUTION

for t in range(Nt):

    # Macroscopic Scalar Field
    phi = np.sum(f, axis=0)

    # Collision step
    for i in range(Q):
        feq = w[i] * phi * (1 + (ex[i] * ux + ey[i] * uy) / cs2)
        f[i] = f[i] - (f[i] - feq) / tau

    # Streaming step
    for i in range(Q):
        f[i] = np.roll(f[i], shift=(ey[i], ex[i]), axis=(0, 1))

# 9. FINAL FIELD

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
plt.plot(x, phi_final[Ny//2], label='Final (LBM)')
plt.plot(x, phi_exact[Ny//2], '--', label='Final (analytical)')
plt.title('Advection-Diffusion of a Gaussian Pulse using LBM (D2Q5)')
plt.xlabel('x')
plt.ylabel('phi')
plt.legend()
plt.grid()
plt.show()
