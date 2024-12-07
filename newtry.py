import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def create_constants():
    """Create dictionary of all constant parameters"""
    constants = {
        # Vehicle parameters
        'L': 1.5,               # Length
        'm': 500.0,            # Mass
        'lf': 0.7,             # Front length
        'lr': 0.8,             # Rear length
        'mu': 1.0,             # Friction coefficient
        'maxv': 15.0 * 5.0/18.0, # Max velocity
        'minTurnRadius': 3.0,   # Min turning radius
        'rw': 0.2,             # Wheel radius

        # Electric drive parameters
        'LMax': 275.0,         # Max power
        'Jm': 0.2,             # Motor inertia
        'Rm': 0.35,            # Motor resistance
        'Km': 1.24,            # Motor constant
        'drive_eff': 0.95,     # Drive efficiency

        # Battery parameters
        'cell_capacity': 1.0,
        'cell_voltage': 3.8,
        'module_series': 10.0,
        'module_parallel': 25.0,
        'pack_series': 1.0,
        'soc_full': 0.8,
        'soc_empty': 0.3,
        'battery_eff': 0.96
    }

    # Derived battery parameters
    constants['voltage_multiplier'] = constants['pack_series'] * constants['module_series']
    constants['pack_capacity'] = constants['module_parallel'] * constants['cell_capacity']
    constants['nom_voltage'] = constants['voltage_multiplier'] * constants['cell_voltage']
    constants['total_eff'] = constants['drive_eff'] * constants['battery_eff']

    return constants

def system_dynamics(state, control, constants):
    """System dynamics equations"""
    X, Y, psi, vx, soc = state[0], state[1], state[2], state[3], state[4]
    E, delta, brake = control[0], control[1], control[2]

    # Motor and battery calculations
    motor_current = (E - constants['Km'] * vx / constants['rw']) / constants['Rm']
    motor_torque = constants['Km'] * motor_current
    battery_voltage = soc2voltage(soc, constants['voltage_multiplier'])

    # State derivatives
    Xdot = vx * ca.cos(psi)
    Ydot = vx * ca.sin(psi)
    psidot = vx / constants['L'] * ca.tan(delta)

    # Velocity and SOC derivatives
    vxdot = (motor_torque - brake) / (constants['rw'] * constants['m']) / (1 + constants['Jm']/(constants['rw']**2 * constants['m']))
    motor_power = E * (E - constants['Km'] * vx / constants['rw']) / constants['Rm']
    battery_current = motor_power / battery_voltage
    socdot = -battery_current / constants['pack_capacity'] / 3600.0

    return ca.vertcat(Xdot, Ydot, psidot, vxdot, socdot)

def soc2voltage(soc, multiplier):
    """Battery SOC to voltage conversion"""
    a1, b1 = 3.679, -0.1011
    a2, b2 = -0.2528, -6.829
    c = 0.9386
    return (a1 * ca.exp(b1 * soc) + a2 * ca.exp(b2 * soc) + c * soc * soc) * multiplier

def compute_cost(state, control, constants):
    """Modified cost function with better numerical stability"""
    vx, soc = state[3], state[4]
    E, delta = control[0], control[1]

    # More numerically stable power calculation
    motor_power = E * (E - constants['Km'] * vx / constants['rw']) / constants['Rm']
    psidot = vx / constants['L'] * ca.tan(delta)

    # Smoother sigmoid for better numerical properties
    power_term = ca.log(1 + ca.exp(motor_power)) * motor_power / constants['total_eff']

    return (1 * power_term +
            1 * (constants['soc_full'] - soc)**2 +
            0.1 * psidot**2)

def optimize_path(waypoints, arrival_times, constants):
    """Main optimization function with fixed arrival times"""
    opti = ca.Opti()
    N = len(arrival_times) - 1  # Number of segments
    points_per_segment = 10     # discretization points between waypoints
    # Total points
    total_points = (N * points_per_segment) + 1
    
    X = opti.variable(5, total_points)  # States: [x, y, psi, vx, soc]
    X = opti.variable(5, total_points)  # states: [x, y, psi, vx, soc]
    U = opti.variable(3, total_points-1)  # Controls: [voltage, steering, brake]
    
    # Calculate time steps for each segment
    segment_times = np.diff(arrival_times)
    dt = segment_times[:, None] / points_per_segment  # Dt for each segment
    # Turning radius constraint based on minimum turning radius
    max_steer_angle = np.arctan(constants['L'] / constants['minTurnRadius'])
    # Basic constraints
    opti.subject_to(opti.bounded(0.1, X[4, :], 1))  # SOC with safety margins
    opti.subject_to(opti.bounded(0, X[3, :],  constants['maxv']))  # velocity
    opti.subject_to(opti.bounded(0, U[0, :],  constants['nom_voltage']))  # voltage
    opti.subject_to(opti.bounded(-max_steer_angle, U[1, :], max_steer_angle))  # steering
    opti.subject_to(opti.bounded(0, U[2, :],
                                 constants['mu'] * constants['m'] * 9.81 * constants['rw']))  # brake

    # Dynamics constraints for each segment
    for segment in range(N):
        start_idx = segment * points_per_segment
        end_idx = (segment + 1) * points_per_segment
        
        for k in range(points_per_segment):
            idx = start_idx + k
            derivatives = system_dynamics(X[:, idx], U[:, idx], constants)
            x_next = X[:, idx] + dt[segment] * derivatives
            opti.subject_to(X[:, idx+1] == x_next)

    # Waypoint constraints
    for i in range(len(waypoints)):
        idx = i * points_per_segment
        opti.subject_to(X[0, idx] == waypoints[i, 0])  # X position
        opti.subject_to(X[1, idx] == waypoints[i, 1])  # Y position
        opti.subject_to(X[3, idx] == 0)  # velocity = 0 at waypoints
        
        if i == 0:  # Initial conditions
            opti.subject_to(X[2, 0] == 0)  # psi = 0
            opti.subject_to(X[4, 0] == constants['soc_full'])  # SOC_max
        elif i == len(waypoints)-1:  # Final conditions
            opti.subject_to(X[2, -1] == 0)  # psi = 0

    # Objective function
    total_cost = 0
    for segment in range(N):
        start_idx = segment * points_per_segment
        end_idx = (segment + 1) * points_per_segment
        
        for k in range(points_per_segment):
            idx = start_idx + k
            total_cost += compute_cost(X[:, idx], U[:, idx], constants) * dt[segment]
    
    opti.minimize(total_cost)

    # Initial guess
    for i in range(len(waypoints)-1):
        start_idx = i * points_per_segment
        end_idx = (i + 1) * points_per_segment + 1

        # Linear interpolation for position
        for j, k in enumerate(range(start_idx, end_idx)):
            alpha = j / points_per_segment
            pos = waypoints[i] + alpha * (waypoints[i+1] - waypoints[i])
            opti.set_initial(X[0, k], pos[0])
            opti.set_initial(X[1, k], pos[1])

 

    # Solver options
    opts = {
        'ipopt.max_iter': 1000,
        'ipopt.print_level': 5,  # Increase verbosity
        'print_time': True,
        'verbose': True
    }
    opti.solver('ipopt', opts)
    sol = opti.solve()

    return arrival_times[-1], sol.value(X), sol.value(U)

def plot_results(t, X, U, waypoints, arrival_times):
    """Plot optimization results with arrival times"""
    plt.figure(figsize=(10, 6))
    plt.plot(X[0, :], X[1, :], 'b-', label='Path')
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro', label='Waypoints')
    
    # Add arrival times annotation
    for i, time in enumerate(arrival_times):
        plt.annotate(f't={time:.1f}s', 
                    (waypoints[i, 0], waypoints[i, 1]),
                    xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Optimal Path with Arrival Times')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # Create time vector for plotting
    t_dense = np.linspace(0, arrival_times[-1], X.shape[1])
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t_dense, X[3, :])
    plt.ylabel('Velocity (m/s)')
    plt.subplot(3, 1, 2)
    plt.plot(t_dense[:-1], U[0, :])
    plt.ylabel('Voltage (V)')
    plt.subplot(3, 1, 3)
    plt.plot(t_dense[:-1], U[1, :])
    plt.ylabel('Steering Angle (rad)')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

def main():
    waypoints = np.array([
        (0, 0), (13.53, 11.43), (28.53, 20.77), (40.14, 17.1), (51.96, 14.45),
        (60.91, 14.24), (71.67, 17.33), (77.67, 26.31), (86.14, 38.84), (85.53, 50.45),
        (97.84, 61.4), (87.97, 77.19), (73.25, 80.76), (63.43, 81.46), (47.96, 81.5),
        (41.2, 82.16), (23.75, 89.1), (16.39, 77.98), (8.68, 63.72), (18.25, 50.11),
        (4.98, 36.12), (0, 0)
    ])

    arrival_times = np.array([
        0.00, 11.88, 23.80, 33.03, 42.06, 49.45, 58.03, 66.57, 77.16, 86.00,
        97.46, 109.84, 120.43, 128.49, 139.37, 145.91, 158.53, 168.34, 179.57,
        191.10, 203.80, 222.03
    ])

    constants = create_constants()
    T, X, U = optimize_path(waypoints, arrival_times, constants)
    plot_results(T, X, U, waypoints, arrival_times)
    
if __name__ == "__main__":
    main()