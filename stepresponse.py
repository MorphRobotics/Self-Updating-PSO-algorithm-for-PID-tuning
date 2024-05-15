import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Define the plant transfer function
num = [1]
den = [1, 2, 1]  # Second-order system: s^2 + 2s + 1
plant = ctrl.TransferFunction(num, den)

# Define PID controller parameters
Kp = 0.99707
Ki = 0.6117
Kd = -0.0486
pid_controller = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])

# Create the closed-loop system
closed_loop_system = ctrl.feedback(pid_controller * plant)

# Time vector for simulation
t = np.linspace(0, 10, 1000)

# Step response of the closed-loop system
t, y = ctrl.step_response(closed_loop_system, T=t)

# Plot the step response
plt.plot(t, y)
plt.title('Step Response of PSO-PID Controller using ITSE')
plt.xlabel('Time')
plt.ylabel('Output')
plt.grid(True)
plt.show()





