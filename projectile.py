import numpy as np
import random
import plotly.graph_objects as go

def generate_test_variables():

    gravity = 9.81 + random.uniform(-0.3, 0.3)
    air_density = 1.225 + random.uniform(-0.05, 0.05)
    wind = (
        random.uniform(-2.0, 2.0),
        random.uniform(-2.0, 2.0),
        random.uniform(-0.5, 0.5)
    )

    position = (0.0, 0.0, 0.0)
    velocity = (
        random.uniform(5.0, 20.0),
        random.uniform(5.0, 20.0),
        random.uniform(0.0, 5.0)
    )
    mass = random.uniform(0.5, 5.0)
    drag_coeff = random.uniform(0.1, 1.0)
    cross_area = random.uniform(0.005, 0.05)
    
    total_time = random.uniform(4.0, 7.0)
    

    reference_end = np.array([0.0, 0.0, 0.0])
    
    env_params = {
        'gravity': gravity,
        'air_density': air_density,
        'wind': wind
    }
    proj_params = {
        'position': position,
        'velocity': velocity,
        'mass': mass,
        'drag_coeff': drag_coeff,
        'cross_area': cross_area
    }
    
    return env_params, proj_params, total_time, reference_end


class Environment:
    def __init__(self, gravity=9.81, air_density=1.225, wind=(0.0, 0.0, 0.0)):
        self.g = gravity
        self.rho = air_density
        self.wind = np.array(wind, dtype=float)


class Projectile:
    def __init__(self,
                 position=(0.0, 0.0, 0.0),
                 velocity=(10.0, 10.0, 0.0),
                 mass=1.0,
                 drag_coeff=0.47,
                 cross_area=0.01):
        self.pos = np.array(position, dtype=float)
        self.vel = np.array(velocity, dtype=float)
        self.m   = mass
        self.Cd  = drag_coeff
        self.A   = cross_area


def derivatives(state, environment, projectile):

    x, y, z, vx, vy, vz = state
    vel = np.array([vx, vy, vz])
    

    rel_vel = vel - environment.wind
    speed = np.linalg.norm(rel_vel)
    if speed != 0:
        drag_mag = 0.5 * environment.rho * projectile.Cd * projectile.A * speed**2
        drag = -drag_mag * (rel_vel / speed)
    else:
        drag = np.zeros(3)
    

    gravity_force = np.array([0, 0, -environment.g * projectile.m])
    
    total_force = drag + gravity_force
    acceleration = total_force / projectile.m
    return np.concatenate((vel, acceleration))


def rk4_step(state, environment, projectile, dt):

    k1 = derivatives(state, environment, projectile)
    k2 = derivatives(state + 0.5 * dt * k1, environment, projectile)
    k3 = derivatives(state + 0.5 * dt * k2, environment, projectile)
    k4 = derivatives(state + dt * k3, environment, projectile)
    
    return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_flight_rk4(projectile, environment, total_time, dt=0.01):

    steps = int(total_time / dt)
    state = np.concatenate((projectile.pos, projectile.vel))
    positions = []
    velocities = []
    time_points = []
    t = 0.0
    
    for _ in range(steps + 1):
        positions.append(state[:3].copy())
        velocities.append(state[3:].copy())
        time_points.append(t)
        state = rk4_step(state, environment, projectile, dt)
        t += dt
        
        if state[2] < 0:
            break
    return time_points, positions, velocities


def find_single_velocity_correction(current_pos, current_vel, environment, projectile,
                                    remain_time, dt, target_end, max_iters=30):

    best_vel = current_vel.copy()
    best_dist = float('inf')
    
    for _ in range(max_iters):
        candidate_vel = best_vel + np.array([
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(-2, 2)
        ])
        test_proj = Projectile(position=current_pos, velocity=candidate_vel,
                               mass=projectile.m, drag_coeff=projectile.Cd, cross_area=projectile.A)
        _, pos_traj, _ = simulate_flight_rk4(test_proj, environment, remain_time, dt)
        final_pos = pos_traj[-1]
        dist = np.linalg.norm(final_pos - target_end)
        if dist < best_dist:
            best_dist = dist
            best_vel = candidate_vel.copy()
    
    return best_vel


def main():
    
    env_params, proj_params, total_time, _ = generate_test_variables()
    environment = Environment(gravity=env_params['gravity'],
                              air_density=env_params['air_density'],
                              wind=env_params['wind'])
    
    baseline_projectile = Projectile(position=proj_params['position'],
                                     velocity=proj_params['velocity'],
                                     mass=proj_params['mass'],
                                     drag_coeff=proj_params['drag_coeff'],
                                     cross_area=proj_params['cross_area'])
    
    dt = 0.01
    
    times_base, pos_base, vel_base = simulate_flight_rk4(baseline_projectile, environment, total_time, dt)
    reference_end = pos_base[-1]
    
    
    t_sway_fraction = 0.3  
    t_sway = t_sway_fraction * total_time
    steps_sway = int(t_sway / dt)
    
    sway_projectile = Projectile(position=proj_params['position'],
                                 velocity=proj_params['velocity'],
                                 mass=proj_params['mass'],
                                 drag_coeff=proj_params['drag_coeff'],
                                 cross_area=proj_params['cross_area'])
    

    state = np.concatenate((sway_projectile.pos, sway_projectile.vel))
    pos_sway_part = []
    vel_sway_part = []
    time_sway_part = []
    t = 0.0
    for _ in range(steps_sway + 1):
        pos_sway_part.append(state[:3].copy())
        vel_sway_part.append(state[3:].copy())
        time_sway_part.append(t)
        state = rk4_step(state, environment, sway_projectile, dt)
        t += dt
        if state[2] < 0:
            break
    

    sway_projectile.pos = state[:3]
    sway_projectile.vel = state[3:]
    

    sway_vector = np.array([
        random.uniform(-5.0, 5.0),
        random.uniform(-5.0, 5.0),
        random.uniform(-1.0, 2.0)
    ])
    sway_projectile.vel += sway_vector
    

    remain_time = total_time - t_sway
    

    corrected_vel = find_single_velocity_correction(
        current_pos=sway_projectile.pos.copy(),
        current_vel=sway_projectile.vel.copy(),
        environment=environment,
        projectile=sway_projectile,
        remain_time=remain_time,
        dt=dt,
        target_end=reference_end,
        max_iters=50
    )
    

    sway_projectile.vel = corrected_vel
    

    pos_sway_rest = []
    vel_sway_rest = []
    time_sway_rest = []
    state_rest = np.concatenate((sway_projectile.pos, sway_projectile.vel))
    t_rest = t_sway
    steps_rest = int(remain_time / dt)
    for _ in range(steps_rest + 1):
        pos_sway_rest.append(state_rest[:3].copy())
        vel_sway_rest.append(state_rest[3:].copy())
        time_sway_rest.append(t_rest)
        state_rest = rk4_step(state_rest, environment, sway_projectile, dt)
        t_rest += dt
        if state_rest[2] < 0:
            break
    

    pos_sway_full = pos_sway_part[:-1] + pos_sway_rest
    vel_sway_full = vel_sway_part[:-1] + vel_sway_rest
    time_sway_full = time_sway_part[:-1] + time_sway_rest
    

    baseline_arr = np.array(pos_base)
    sway_arr = np.array(pos_sway_full)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=baseline_arr[:, 0],
        y=baseline_arr[:, 1],
        z=baseline_arr[:, 2],
        mode='lines',
        line=dict(color='blue', width=4),
        name='Baseline (No Sway)'
    ))
    

    fig.add_trace(go.Scatter3d(
        x=sway_arr[:, 0],
        y=sway_arr[:, 1],
        z=sway_arr[:, 2],
        mode='lines',
        line=dict(color='red', width=4, dash='dash'),
        name='Swayed Flight (Corrected)'
    ))
    
    fig.update_layout(
        title="3D Projectile Simulation: Baseline vs. Sway Correction",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=600
    )
    
    fig.show()
    
    final_pos_no_sway = baseline_arr[-1]
    final_pos_sway = sway_arr[-1]
    diff = np.linalg.norm(final_pos_sway - final_pos_no_sway)
    
    print("\n=== SUMMARY ===")
    print(f"Total flight time: {total_time:.2f} s")
    print(f"Sway occurred at:  t_sway = {t_sway:.2f} s")
    print(f"Sway vector:       {sway_vector}")
    print(f"Corrected velocity:{corrected_vel}")
    print(f"Baseline final pos (no sway): {final_pos_no_sway}")
    print(f"Sway final pos (with fix):    {final_pos_sway}")
    print(f"Difference between final positions: {diff:.3f} m")



executemain: float = 1.0

if executemain == 1.0:
    main()