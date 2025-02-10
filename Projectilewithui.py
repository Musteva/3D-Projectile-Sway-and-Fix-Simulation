import numpy as np
import random
import plotly.graph_objects as go

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

    # Relative velocity w.r.t. wind
    rel_vel = vel - environment.wind
    speed = np.linalg.norm(rel_vel)

    # Drag force
    if speed != 0:
        drag_mag = 0.5 * environment.rho * projectile.Cd * projectile.A * speed**2
        drag = -drag_mag * (rel_vel / speed)
    else:
        drag = np.zeros(3)

    # Gravity
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
    
    for i in range(steps + 1):
        positions.append(state[:3].copy())
        velocities.append(state[3:].copy())
        time_points.append(t)
        
       
        state = rk4_step(state, environment, projectile, dt)
        t += dt
        
      
        if state[2] < 0 and i > 0:
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
   
    print("Enter the simulation parameters (press Enter to use suggested defaults):")

    gravity = input("Gravity [default=9.81 m/s^2]: ")
    gravity = float(gravity) if gravity.strip() else 9.81
    
    air_density = input("Air Density [default=1.225 kg/m^3]: ")
    air_density = float(air_density) if air_density.strip() else 1.225

    wind_x = input("Wind X component [default=0.0 m/s]: ")
    wind_x = float(wind_x) if wind_x.strip() else 0.0
    wind_y = input("Wind Y component [default=0.0 m/s]: ")
    wind_y = float(wind_y) if wind_y.strip() else 0.0
    wind_z = input("Wind Z component [default=0.0 m/s]: ")
    wind_z = float(wind_z) if wind_z.strip() else 0.0
    
    pos_x = input("Initial Projectile X position [default=0.0]: ")
    pos_x = float(pos_x) if pos_x.strip() else 0.0
    pos_y = input("Initial Projectile Y position [default=0.0]: ")
    pos_y = float(pos_y) if pos_y.strip() else 0.0
    pos_z = input("Initial Projectile Z position [default=0.0]: ")
    pos_z = float(pos_z) if pos_z.strip() else 0.0
    
    vel_x = input("Initial Projectile X velocity [default=10.0 m/s]: ")
    vel_x = float(vel_x) if vel_x.strip() else 10.0
    vel_y = input("Initial Projectile Y velocity [default=10.0 m/s]: ")
    vel_y = float(vel_y) if vel_y.strip() else 10.0
    vel_z = input("Initial Projectile Z velocity [default=0.0 m/s]: ")
    vel_z = float(vel_z) if vel_z.strip() else 0.0
    
    mass = input("Projectile mass [default=1.0 kg]: ")
    mass = float(mass) if mass.strip() else 1.0
    
    drag_coeff = input("Drag coefficient [default=0.47]: ")
    drag_coeff = float(drag_coeff) if drag_coeff.strip() else 0.47
    
    cross_area = input("Projectile cross-sectional area [default=0.01 m^2]: ")
    cross_area = float(cross_area) if cross_area.strip() else 0.01

    total_time = input("Total simulation time [default=5.0 s]: ")
    total_time = float(total_time) if total_time.strip() else 5.0

    dt = input("Time step (dt) [default=0.01 s]: ")
    dt = float(dt) if dt.strip() else 0.01

    
    environment = Environment(
        gravity=gravity,
        air_density=air_density,
        wind=(wind_x, wind_y, wind_z)
    )

    baseline_projectile = Projectile(
        position=(pos_x, pos_y, pos_z),
        velocity=(vel_x, vel_y, vel_z),
        mass=mass,
        drag_coeff=drag_coeff,
        cross_area=cross_area
    )

    
    times_base, pos_base, vel_base = simulate_flight_rk4(baseline_projectile, environment, total_time, dt)
    baseline_arr = np.array(pos_base)
    
   
    if len(baseline_arr) <= 1:
        print("\n[Warning] The baseline flight may not have propagated much. "
              "Try increasing initial Z velocity or starting above the ground.\n")
    
   
    reference_end = baseline_arr[-1] if len(baseline_arr) > 0 else np.array([0, 0, 0])

    
    t_sway_fraction = 0.3  
    t_sway = t_sway_fraction * total_time
    steps_sway = int(t_sway / dt)

   
    sway_projectile = Projectile(
        position=(pos_x, pos_y, pos_z),
        velocity=(vel_x, vel_y, vel_z),
        mass=mass,
        drag_coeff=drag_coeff,
        cross_area=cross_area
    )

    
    state = np.concatenate((sway_projectile.pos, sway_projectile.vel))
    pos_sway_part = []
    vel_sway_part = []
    time_sway_part = []
    t = 0.0
    for i in range(steps_sway + 1):
        pos_sway_part.append(state[:3].copy())
        vel_sway_part.append(state[3:].copy())
        time_sway_part.append(t)
        
        state = rk4_step(state, environment, sway_projectile, dt)
        t += dt
        
        
        if state[2] < 0 and i > 0:
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

    for i in range(steps_rest + 1):
        pos_sway_rest.append(state_rest[:3].copy())
        vel_sway_rest.append(state_rest[3:].copy())
        time_sway_rest.append(t_rest)
        
        state_rest = rk4_step(state_rest, environment, sway_projectile, dt)
        t_rest += dt
        
        if state_rest[2] < 0 and i > 0:
            break

   
    if len(pos_sway_part) > 0:
        pos_sway_full = pos_sway_part[:-1] + pos_sway_rest
        vel_sway_full = vel_sway_part[:-1] + vel_sway_rest
        time_sway_full = time_sway_part[:-1] + time_sway_rest
    else:
        
        pos_sway_full = pos_sway_part + pos_sway_rest
        vel_sway_full = vel_sway_part + vel_sway_rest
        time_sway_full = time_sway_part + time_sway_rest

    sway_arr = np.array(pos_sway_full)

  
    fig = go.Figure()


    if len(baseline_arr) > 1:
        fig.add_trace(go.Scatter3d(
            x=baseline_arr[:, 0],
            y=baseline_arr[:, 1],
            z=baseline_arr[:, 2],
            mode='lines',
            line=dict(color='blue', width=4),
            name='Baseline (No Sway)'
        ))
    else:
        print("\n[Info] Baseline data insufficient or trivial; skipping baseline plot.\n")


    if len(sway_arr) > 1:
        fig.add_trace(go.Scatter3d(
            x=sway_arr[:, 0],
            y=sway_arr[:, 1],
            z=sway_arr[:, 2],
            mode='lines',
            line=dict(color='red', width=4, dash='dash'),
            name='Swayed Flight (Corrected)'
        ))
    else:
        print("\n[Info] Sway data insufficient or trivial; skipping sway plot.\n")

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


    if len(baseline_arr) > 0:
        final_pos_no_sway = baseline_arr[-1]
    else:
        final_pos_no_sway = np.array([np.nan, np.nan, np.nan])

    if len(sway_arr) > 0:
        final_pos_sway = sway_arr[-1]
        diff = np.linalg.norm(final_pos_sway - final_pos_no_sway)
    else:
        final_pos_sway = np.array([np.nan, np.nan, np.nan])
        diff = np.nan

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
