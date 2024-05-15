import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Objective function (example: Sphere function in 3D)
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

# Visualization function
def visualize_pso(history, objective_function):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_function(X, Y)

    ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.5)

    for i in range(len(history)):
        ax.scatter(history[i][0], history[i][1], objective_function(history[i][0], history[i][1]), color='red', s=30)

        if i > 0:
            ax.plot([history[i - 1][0], history[i][0]],
                    [history[i - 1][1], history[i][1]],
                    [objective_function(history[i - 1][0], history[i - 1][1]),
                     objective_function(history[i][0], history[i][1])], color='black', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Objective Function Value')
    ax.set_title('Particle Swarm Optimization')

    plt.show()

# Parameters
num_particles = 10
num_dimensions = 2
max_iterations = 50

# Run PSO
pso = PSO(objective_function, num_particles, num_dimensions, max_iterations)
pso.optimize()

# Visualize PSO
visualize_pso(pso.history, objective_function)



