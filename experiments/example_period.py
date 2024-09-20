import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system of differential equations
def system(t, z):
    x, y = z
    dxdt = -(-y - (x**4 / 4 - x**2 / 2 + y**2 / 2) * (x**3 - x))
    dydt = -(x**3 - x - (x**4 / 4 - x**2 / 2 + y**2 / 2) * y)
    return [dxdt, dydt]
# Latex form
"""
\[
x' = -\left( -y - \left( \frac{x^4}{4} - \frac{x^2}{2} + \frac{y^2}{2} \right)(x^3 - x) \right)
\]
\[
y' = -\left( x^3 - x - \left( \frac{x^4}{4} - \frac{x^2}{2} + \frac{y^2}{2} \right) y \right)
\]
"""

# Time span for the simulation
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# # Generate a grid of initial conditions for x and y near to the saddle
# x_lenght = 0.2
# y_lenght = 0.11
# x_vals = np.linspace(-x_lenght, x_lenght, 15)
# y_vals = np.linspace(-y_lenght, y_lenght, 5)

# Generate a grid of initial conditions for x and y
x_lenght = 1
y_lenght = 0.5
x_vals = np.linspace(-x_lenght, x_lenght, 10)
y_vals = np.linspace(-y_lenght, y_lenght, 10)

# Create a figure for plotting
plt.figure(figsize=(8, 8))

# Plot trajectories for each pair of initial conditions
for x0 in x_vals:
    for y0 in y_vals:
        sol = solve_ivp(system, t_span, [x0, y0], t_eval=t_eval)
        plt.plot(sol.y[0], sol.y[1], 'b-', lw=0.5)

# Add labels and title
plt.title("Flow of the system of differential equations")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
