import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Objective function: Proportional Integral Derivative (PID) controller equation
def pid_controller_itse(Kp, Ki, Kd):
    # Sample PID controller equation
    time = np.linspace(0, 10, 100)  # Time vector
    setpoint = 1.0  # Setpoint value
    dt = time[1] - time[0]  # Time step
    integral = 0
    derivative = 0
    prev_error = 0
    time_weighted_squared_error_integral = 0  # Initialize integral of time-weighted squared error
    for t in time:
        error = setpoint - (Kp + Ki * integral + Kd * derivative)
        time_weighted_squared_error_integral += t * error**2 * dt  # Integrate time-weighted squared error
        integral += error * dt
        derivative = (error - prev_error) / dt
        prev_error = error
    return time_weighted_squared_error_integral  # Return ITSE as the objective value


# Particle Swarm Optimization
class PSO:
    def __init__(self, objective_function, num_particles, num_dimensions, max_iterations):
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.max_iterations = max_iterations
        self.global_best_position = np.random.uniform(0, 1, (1, num_dimensions))  # Initial PID parameters
        self.global_best_value = float('inf')
        self.particles = np.random.uniform(0, 1, (num_particles, num_dimensions))  # Initial particle positions
        self.velocities = np.zeros((num_particles, num_dimensions))
        self.history = []

    def optimize(self):
        num_particles_start = self.num_particles  # Record initial number of particles
        for _ in range(self.max_iterations):
            for i in range(self.num_particles):
                current_position = self.particles[i]
                current_value = self.objective_function(*current_position)

                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = current_position

                # Update velocities
                inertia_weight = 0.5
                cognitive_weight = 1.5
                social_weight = 1.5
                r1 = np.random.uniform(0, 0.1, self.num_dimensions)  # Adjusting the range for random velocities
                r2 = np.random.uniform(0, 0.1, self.num_dimensions)
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      cognitive_weight * r1 * (self.global_best_position - current_position) +
                                      social_weight * r2 * (self.global_best_position - current_position))

                # Update positions
                self.particles[i] += self.velocities[i]

            # Dynamically reduce number of particles based on distance from optimum
            distance_to_optimum = np.linalg.norm(self.global_best_position)
            if distance_to_optimum < 0.02 * (self.max_iterations - _) and self.num_particles > 5:
                self.num_particles -= 1
                self.particles = self.particles[:self.num_particles]
                self.velocities = self.velocities[:self.num_particles]

            # Save the global best position in history for visualization
            self.history.append(self.global_best_position.copy())

        return self.global_best_position, num_particles_start, self.num_particles

# Plot 3D visualization of particle swarm at the end of simulation
def plot_particle_swarm(history, global_best_position):
    history = np.array(history)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(history[:, 0], history[:, 1], history[:, 2], c='b', label='Particles')
    ax.scatter(global_best_position[0], global_best_position[1], global_best_position[2], c='r', marker='x', label='Global Best')
    ax.set_xlabel('Kp')
    ax.set_ylabel('Ki')
    ax.set_zlabel('Kd')
    ax.set_title('Particle Swarm Optimization, ITSE')
    plt.legend()
    plt.show()

# Parameters
num_particles = 50
num_dimensions = 3  # Kp, Ki, Kd
max_iterations = 100

# Run PSO
pso = PSO(pid_controller_itse, num_particles, num_dimensions, max_iterations)
best_position, num_particles_start, num_particles_end = pso.optimize()

# Output initial and final number of particles
print("Initial number of particles:", num_particles_start)
print("Final number of particles:", num_particles_end)

# Visualize the particle swarm at the end of simulation
plot_particle_swarm(pso.history, best_position)

