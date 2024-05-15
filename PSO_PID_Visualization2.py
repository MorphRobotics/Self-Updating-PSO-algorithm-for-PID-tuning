import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Objective function (example: Sphere function in 2D)
def objective_function(x, y):
    return x**2 + y**2

# Particle Swarm Optimization
class PSO:
    def __init__(self, objective_function, num_particles, num_dimensions, max_iterations):
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.max_iterations = max_iterations
        self.global_best_position = np.random.uniform(-5, 5, (1, num_dimensions))
        self.global_best_value = float('inf')
        self.particles = np.random.uniform(-5, 5, (num_particles, num_dimensions))
        self.velocities = np.zeros((num_particles, num_dimensions))
        self.history = []

    def optimize(self):
        for _ in range(self.max_iterations):
            for i in range(self.num_particles):
                current_position = self.particles[i]
                current_value = self.objective_function(current_position[0], current_position[1])

                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = current_position

                # Update velocities
                inertia_weight = 0.5
                cognitive_weight = 1.5
                social_weight = 1.5
                r1 = np.random.uniform(0, 1, self.num_dimensions)
                r2 = np.random.uniform(0, 1, self.num_dimensions)

                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      cognitive_weight * r1 * (self.global_best_position - current_position) +
                                      social_weight * r2 * (self.global_best_position - current_position))

                # Update positions
                self.particles[i] += self.velocities[i]

            # Save the global best position in history for visualization
            self.history.append(self.global_best_position.copy())

            # Dynamically reduce number of particles based on distance from optimum
            distance_to_optimum = np.linalg.norm(self.global_best_position)
            if distance_to_optimum < 0.01 * (self.max_iterations - _) and self.num_particles > 5:
                self.num_particles -= 2

# Animation function
def update(num, data, sc, pso):
    sc.set_offsets(data[:, :num])
    pso.optimize()
    return sc,

# Visualization function
def visualize_pso(history, objective_function):
    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_function(X, Y)

    ax.contour(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 100), cmap='jet')

    data = np.array(history)
    sc = ax.scatter(data[:, 0], data[:, 1], color='red')

    ani = FuncAnimation(fig, update, frames=len(history), fargs=(data, sc, pso), interval=200, blit=True)
    plt.show()

# Parameters
num_particles = 50
num_dimensions = 2
max_iterations = 100

# Run PSO
pso = PSO(objective_function, num_particles, num_dimensions, max_iterations)
pso.optimize()

# Visualize PSO
visualize_pso(pso.history, objective_function)

# Output number of particles at the start and end
print("Number of particles at the start:", num_particles)
print("Number of particles at the end:", pso.num_particles)



