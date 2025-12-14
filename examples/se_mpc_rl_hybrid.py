#!/usr/bin/env python3
"""
Hybrid SE-MPC + RL for Multi-Agent Flow Navigation
Architecture:
  1. FlowFieldPatch (RL): Centralized coordinator - learns flow field + topology
  2. SE-MPC (Fixed): Decentralized executor - local obstacles + formation

"Robots flow like fluid" - hierarchical multi-agent system
"""

import time
import gymnasium as gym
import numpy as np
import math
import casadi as ca
from collections import deque
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for real-time plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.patches as mpatches


# ============================================================================
# 1. FLOW FIELD PATCH (RL learns this)
# ============================================================================

class FlowFieldPatch:
    """
    Centralized coordinator that generates flow field and topology.
    This is what RL learns to optimize.
    
    NO obstacle knowledge! Only provides guidance.
    """
    
    def __init__(self, num_agents, team_goal):
        """
        Args:
            num_agents: Number of robots in team
            team_goal: Global goal position [x, y]
        """
        self.num_agents = num_agents
        self.team_goal = np.array(team_goal)
        
        # Flow field parameters (what RL learns to adjust)
        self.flow_speed_scale = 2.0  # Speed scaling (increased for faster movement)
        self.flow_spread = 2.0       # How spread out is flow
        self.formation_distance = 1.0  # Inter-robot distance
        
        # DEFORMATION parameters (RL LEARNS when to compress/expand!)
        self.compression_factor = 1.0  # 0.5 = squeezed, 1.0 = normal, 2.0 = expanded
        self.patch_radius_scale = 2.5  # RL LEARNS THIS TOO! (multiplier for patch radius)
        self.adaptive_mode = True      # Whether to adapt to sensor data
        
        # Sensor data (aggregated from all agents)
        self.aggregated_lidar = None   # Average LIDAR from all agents
        self.obstacle_density = 0.0    # How cluttered is environment
        
        # Topology matrix (which robots coordinate)
        # T[i,j] = 1 means robots i,j should maintain formation
        self.topology = np.eye(num_agents)  # Start independent
        
        # Formation type
        self.formation_type = 'line'  # 'line', 'diamond', 'column'
        
    def update_from_rl_action(self, action):
        """
        Update patch parameters from RL policy action.
        
        Args:
            action: Dictionary with RL policy outputs
        """
        # Flow parameters
        if 'flow_speed_scale' in action:
            self.flow_speed_scale = np.clip(action['flow_speed_scale'], 0.5, 2.0)
        if 'flow_spread' in action:
            self.flow_spread = np.clip(action['flow_spread'], 1.0, 5.0)
        if 'formation_distance' in action:
            self.formation_distance = np.clip(action['formation_distance'], 0.5, 3.0)
        
        # DEFORMATION parameters (RL LEARNS this!)
        if 'compression_factor' in action:
            self.compression_factor = np.clip(action['compression_factor'], 0.3, 2.0)
        if 'patch_radius_scale' in action:
            self.patch_radius_scale = np.clip(action['patch_radius_scale'], 1.5, 4.0)
        
        # Topology decisions
        if 'topology' in action:
            self.topology = action['topology']
        
        # Formation type
        if 'formation_type' in action:
            self.formation_type = action['formation_type']
    
    def update_sensor_data(self, lidar_scans):
        """
        Aggregate sensor data from all agents.
        This gives the patch "eyes" to see the environment!
        
        Args:
            lidar_scans: List of LIDAR scans from all agents
        """
        if len(lidar_scans) == 0:
            return
        
        # Average LIDAR across all agents
        self.aggregated_lidar = np.mean(lidar_scans, axis=0)
        
        # Estimate obstacle density (how cluttered)
        # Count how many beams see obstacles within 3m
        close_obstacles = np.sum(self.aggregated_lidar < 3.0)
        total_beams = len(self.aggregated_lidar)
        self.obstacle_density = close_obstacles / total_beams
    
    def get_flow_at_position(self, position, agent_idx, all_positions):
        """
        Get desired flow velocity at a given position.
        This is what SE-MPC should try to follow.
        
        Args:
            position: [x, y] current position
            agent_idx: Index of this agent
            all_positions: List of all agent positions (for formation)
        
        Returns:
            v_desired: [vx, vy] desired velocity vector
        """
        # Basic flow: towards goal
        dx = self.team_goal[0] - position[0]
        dy = self.team_goal[1] - position[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < 0.1:
            return np.array([0.0, 0.0])
        
        # Base flow direction (towards goal)
        flow_dir = np.array([dx / dist, dy / dist])
        
        # Apply formation offset based on agent index
        if self.formation_type == 'line':
            # Line formation perpendicular to flow
            lateral_offset = (agent_idx - self.num_agents/2) * self.formation_distance
            flow_dir_perp = np.array([-flow_dir[1], flow_dir[0]])  # Perpendicular
            target_pos = position + flow_dir_perp * lateral_offset
        elif self.formation_type == 'diamond':
            # Diamond formation
            # (simplified: agents arranged in diamond around center)
            pass
        else:  # column
            # Column formation along flow
            pass
        
        # Scale by speed factor
        base_speed = 8.0  # Increased base speed
        v_desired = flow_dir * self.flow_speed_scale * base_speed
        
        return v_desired
    
    def get_formation_neighbors(self, agent_idx):
        """
        Get list of neighbor indices that this agent should coordinate with.
        
        Args:
            agent_idx: Index of this agent
        
        Returns:
            neighbors: List of neighbor indices
        """
        neighbors = []
        for j in range(self.num_agents):
            if j != agent_idx and self.topology[agent_idx, j] > 0.5:
                neighbors.append(j)
        return neighbors
    
    def should_split_formation(self, agent_positions, corridor_width):
        """
        Decide if formation should split (e.g., in narrow corridor).
        RL learns when to make this decision.
        
        Args:
            agent_positions: List of all agent [x,y] positions
            corridor_width: Estimated corridor width from LIDAR
        
        Returns:
            bool: True if should split
        """
        # Simple heuristic (RL will learn better policy)
        if corridor_width < self.formation_distance * self.num_agents:
            return True
        return False


# ============================================================================
# 2. LIDAR PROCESSING
# ============================================================================

def process_lidar_scan(scan, num_beams=1080, fov=4.7, max_range=30.0, 
                       sample_beams=20, min_dist_threshold=0.5):
    """
    Convert LIDAR scan to local obstacle points for SE-MPC.
    
    Args:
        scan: Array of distance measurements (1080 values)
        num_beams: Number of beams in scan
        fov: Field of view (radians)
        max_range: Maximum range
        sample_beams: Sample every Nth beam (reduce constraint count)
        min_dist_threshold: Ignore obstacles closer than this
    
    Returns:
        local_obstacles: List of (x, y, distance) in robot frame
    """
    angles = np.linspace(-fov/2, fov/2, num_beams)
    
    local_obstacles = []
    for i in range(0, num_beams, num_beams // sample_beams):
        dist = scan[i]
        angle = angles[i]
        
        # Only include real obstacles (not max range)
        if dist < max_range - 0.1 and dist > min_dist_threshold:
            # Convert to Cartesian (robot frame)
            x_local = dist * np.cos(angle)
            y_local = dist * np.sin(angle)
            local_obstacles.append((x_local, y_local, dist))
    
    return local_obstacles


def estimate_corridor_width(scan, num_beams=1080, fov=4.7):
    """
    Estimate corridor width from LIDAR scan.
    Used for split/merge decisions.
    
    Args:
        scan: LIDAR scan array
    
    Returns:
        width: Estimated corridor width (meters)
    """
    # Find left and right closest obstacles
    mid = num_beams // 2
    quarter = num_beams // 4
    
    left_min = np.min(scan[:quarter]) if len(scan[:quarter]) > 0 else 30.0
    right_min = np.min(scan[-quarter:]) if len(scan[-quarter:]) > 0 else 30.0
    
    width = left_min + right_min
    return width


# ============================================================================
# 3. SE-MPC WITH FLOW TRACKING (Fixed, not learned)
# ============================================================================

def setup_se_mpc_with_flow(robot_radius, wheelbase):
    """
    Setup SE-MPC that:
    1. Follows flow field from patch (via cost function)
    2. Avoids local obstacles from LIDAR (via constraints)
    3. Maintains formation with neighbors (via constraints)
    
    NO GLOBAL OBSTACLES! Only local sensing.
    
    Args:
        robot_radius: Robot radius
        wheelbase: Car wheelbase
    
    Returns:
        Solver function: solve_mpc(state, v_desired, local_obstacles, neighbors, neighbor_states)
    """
    
    # MPC Parameters
    T_p = 2.0   # Shorter horizon for faster reaction
    N = 10      # Fewer steps
    dt = T_p / N
    
    opti = ca.Opti()
    
    # --- VARIABLES ---
    X = opti.variable(4, N + 1)  # State: [x, y, theta, v]
    x, y, theta, v = X[0, :], X[1, :], X[2, :], X[3, :]
    
    U = opti.variable(2, N)  # Control: [acceleration, steering_angle]
    accel, delta = U[0, :], U[1, :]
    
    # --- PARAMETERS (set at runtime) ---
    x0 = opti.parameter(4, 1)           # Initial state
    v_flow = opti.parameter(2, 1)       # Desired flow velocity [vx_des, vy_des]
    
    # Local obstacles (dynamic parameters)
    max_local_obstacles = 10
    num_obstacles = opti.parameter(1, 1)  # Actual number of obstacles
    obstacle_positions = opti.parameter(2, max_local_obstacles)  # [x, y] for each
    
    # Formation neighbors (dynamic parameters)
    max_neighbors = 3
    num_neighbors = opti.parameter(1, 1)
    neighbor_positions = opti.parameter(2, max_neighbors)
    formation_distance = opti.parameter(1, 1)
    
    # PATCH BOUNDARY (agents must stay inside!)
    patch_center = opti.parameter(2, 1)  # [x_center, y_center]
    patch_radius = opti.parameter(1, 1)  # Maximum distance from center
    
    # --- DYNAMICS (Bicycle Model) ---
    for k in range(N):
        x_next = x[k] + v[k] * ca.cos(theta[k]) * dt
        y_next = y[k] + v[k] * ca.sin(theta[k]) * dt
        theta_next = theta[k] + (v[k] / wheelbase) * ca.tan(delta[k]) * dt
        v_next = v[k] + accel[k] * dt
        
        opti.subject_to(x[k+1] == x_next)
        opti.subject_to(y[k+1] == y_next)
        opti.subject_to(theta[k+1] == theta_next)
        opti.subject_to(v[k+1] == v_next)
    
    # Initial condition
    opti.subject_to(X[:, 0] == x0)
    
    # --- CONSTRAINTS ---
    
    # 1. LOCAL obstacle avoidance (from LIDAR)
    safety_radius = robot_radius + 0.3  # Safety margin
    
    for k in range(N + 1):
        for i in range(max_local_obstacles):
            # Only apply constraint if obstacle i exists
            # Use soft constraint to avoid infeasibility
            obs_x = obstacle_positions[0, i]
            obs_y = obstacle_positions[1, i]
            
            dist_to_obs = ca.sqrt((x[k] - obs_x)**2 + (y[k] - obs_y)**2)
            
            # Conditional constraint: only active if i < num_obstacles
            # Use slack variable approach or barrier function
            # For simplicity: add to cost with high penalty
            obstacle_penalty = ca.fmax(0, safety_radius - dist_to_obs)
            
    # 2. Formation constraints (from topology)
    for k in range(N + 1):
        for i in range(max_neighbors):
            # Maintain distance to active neighbors
            nb_x = neighbor_positions[0, i]
            nb_y = neighbor_positions[1, i]
            
            dist_to_neighbor = ca.sqrt((x[k] - nb_x)**2 + (y[k] - nb_y)**2)
            
            # Keep distance within range [d_min, d_max]
            # Add to cost instead of hard constraint for flexibility
    
    # 3. Velocity bounds
    v_max = 30.0  # Increased for faster movement
    v_min = 0.1   # Small minimum to avoid division by zero
    
    for k in range(N + 1):
        opti.subject_to(v[k] <= v_max)
        opti.subject_to(v[k] >= v_min)
    
    # 4. Control bounds
    accel_max = 8.0
    delta_max = 0.4
    
    for k in range(N):
        opti.subject_to(accel[k] <= accel_max)
        opti.subject_to(accel[k] >= -accel_max)
        opti.subject_to(delta[k] <= delta_max)
        opti.subject_to(delta[k] >= -delta_max)
    
    # --- OBJECTIVE ---
    
    # 1. FLOW TRACKING (follow patch guidance)
    W_flow = 50.0
    J_flow = 0.0
    for k in range(N + 1):
        # Current velocity vector
        vx_current = v[k] * ca.cos(theta[k])
        vy_current = v[k] * ca.sin(theta[k])
        
        # Error from desired flow (with numerical stability)
        flow_error_x = vx_current - v_flow[0]
        flow_error_y = vy_current - v_flow[1]
        
        # Add small epsilon for numerical stability
        J_flow += W_flow * (flow_error_x**2 + flow_error_y**2 + 1e-6)
    
    # 2. Local obstacle avoidance (soft penalty)
    W_obstacle = 1000.0
    J_obstacle = 0.0
    for k in range(N + 1):
        for i in range(max_local_obstacles):
            obs_x = obstacle_positions[0, i]
            obs_y = obstacle_positions[1, i]
            # Add epsilon for numerical stability
            dist_to_obs = ca.sqrt((x[k] - obs_x)**2 + (y[k] - obs_y)**2 + 1e-8)
            
            # Barrier-like penalty: high cost when close
            penalty = ca.fmax(0, safety_radius - dist_to_obs)
            J_obstacle += W_obstacle * (penalty**2 + 1e-8)
    
    # 3. Formation maintenance (soft penalty)
    # W_formation = 20.0
    # J_formation = 0.0
    # for k in range(N + 1):
    #     for i in range(max_neighbors):
    #         nb_x = neighbor_positions[0, i]
    #         nb_y = neighbor_positions[1, i]
    #         # Add epsilon for numerical stability
    #         dist_to_nb = ca.sqrt((x[k] - nb_x)**2 + (y[k] - nb_y)**2 + 1e-8)
            
    #         # Penalty for deviation from desired distance
    #         formation_error = dist_to_nb - formation_distance
    #         J_formation += W_formation * (formation_error**2 + 1e-8)

    #Spring-damper approach to formation maintenance
    # 3. Spring-Damper formation (PROPER physics model!)
    W_spring = 50.0    # Spring stiffness
    W_damper = 10.0    # Damping coefficient
    d_desired = formation_distance  # Equilibrium distance
    d_min = robot_radius * 3.0  # Minimum safe distance (hard boundary)

    J_formation = 0.0
    for k in range(N + 1):
        for i in range(max_neighbors):
            nb_x = neighbor_positions[0, i]
            nb_y = neighbor_positions[1, i]
            
            # Distance to neighbor
            dx = x[k] - nb_x
            dy = y[k] - nb_y
            dist = ca.sqrt(dx**2 + dy**2 + 1e-8)
            
            # Relative velocity (approximate damping)
            if k > 0:
                dx_prev = x[k-1] - nb_x
                dy_prev = y[k-1] - nb_y
                dist_prev = ca.sqrt(dx_prev**2 + dy_prev**2 + 1e-8)
                velocity_rel = (dist - dist_prev) / dt  # Rate of change
            else:
                velocity_rel = 0.0
            
            # SPRING FORCE: F_spring = k * (dist - d_equilibrium)
            spring_error = dist - d_desired
            J_spring = W_spring * spring_error**2
            
            # DAMPER FORCE: F_damper = c * velocity_relative
            J_damper = W_damper * velocity_rel**2
            
            # COLLISION BARRIER: Very high penalty if too close
            collision_barrier = ca.fmax(0, d_min - dist)
            J_collision = 10000.0 * collision_barrier**2  # VERY high weight!
            
            J_formation += J_spring + J_damper + J_collision
    
    # 4. Speed maximization (flow like fluid!)
    W_speed = 5.0
    J_speed = 0.0
    for k in range(N + 1):
        J_speed -= W_speed * v[k]  # Negative = maximize
    
    # 5. Smoothness
    W_smooth = 0.1
    J_smooth = 0.0
    for k in range(N - 1):
        J_smooth += W_smooth * (accel[k+1] - accel[k])**2
        J_smooth += W_smooth * (delta[k+1] - delta[k])**2
    
    # 6. PATCH BOUNDARY (agents know they must stay inside!)
    W_patch = 5000.0  # High weight - IMPORTANT to stay in patch!
    J_patch = 0.0
    for k in range(N + 1):
        # Distance from patch center
        dx_patch = x[k] - patch_center[0]
        dy_patch = y[k] - patch_center[1]
        dist_from_center = ca.sqrt(dx_patch**2 + dy_patch**2 + 1e-8)
        
        # Penalty if outside patch
        violation = ca.fmax(0, dist_from_center - patch_radius)
        J_patch += W_patch * (violation**2 + 1e-8)
    
    # Total objective
    opti.minimize(J_flow + J_obstacle + J_formation + J_speed + J_smooth + J_patch)
    
    # --- Solver Setup ---
    p_opts = {"expand": True, "print_time": False, "verbose": False}
    s_opts = {
        "max_iter": 500,  # Fewer iterations for speed
        "tol": 1e-3,
        "acceptable_tol": 1e-2,
        "print_level": 0
    }
    opti.solver('ipopt', p_opts, s_opts)
    
    opti.set_initial(X, 0.0)
    opti.set_initial(U, 0.0)
    
    # Warm start storage
    prev_X_sol = None
    prev_U_sol = None
    
    def solve_mpc(state, v_desired, local_obs, neighbors, neighbor_states, formation_dist,
                  patch_cent=None, patch_rad=None):
        """
        Solve SE-MPC with flow tracking and local obstacles.
        
        Args:
            state: [x, y, theta, v]
            v_desired: [vx, vy] from flow field
            local_obs: List of (x, y, dist) local obstacles
            neighbors: List of neighbor indices
            neighbor_states: Dict of neighbor states
            formation_dist: Desired formation distance
            patch_cent: [x, y] patch center (agents know this!)
            patch_rad: Patch radius (agents know this!)
        
        Returns:
            u_opt: [accel, steering] or None if failed
        """
        nonlocal prev_X_sol, prev_U_sol
        
        try:
            # Set initial state
            opti.set_value(x0, state.reshape(4, 1))
            
            # Set flow guidance
            opti.set_value(v_flow, np.array(v_desired).reshape(2, 1))
            
            # Set local obstacles
            n_obs = min(len(local_obs), max_local_obstacles)
            opti.set_value(num_obstacles, n_obs)
            
            obs_array = np.zeros((2, max_local_obstacles))
            for i, (ox, oy, _) in enumerate(local_obs[:max_local_obstacles]):
                obs_array[0, i] = ox
                obs_array[1, i] = oy
            opti.set_value(obstacle_positions, obs_array)
            
            # Set formation neighbors
            n_nb = min(len(neighbors), max_neighbors)
            opti.set_value(num_neighbors, n_nb)
            
            nb_array = np.zeros((2, max_neighbors))
            for i, nb_idx in enumerate(neighbors[:max_neighbors]):
                if nb_idx in neighbor_states:
                    nb_state = neighbor_states[nb_idx]
                    nb_array[0, i] = nb_state[0]  # x
                    nb_array[1, i] = nb_state[1]  # y
            opti.set_value(neighbor_positions, nb_array)
            opti.set_value(formation_distance, formation_dist)
            
            # Set PATCH BOUNDARY (agents know they must stay inside!)
            if patch_cent is not None and patch_rad is not None:
                opti.set_value(patch_center, np.array(patch_cent).reshape(2, 1))
                opti.set_value(patch_radius, patch_rad)
            else:
                # Default: large patch (no constraint)
                opti.set_value(patch_center, state[:2].reshape(2, 1))
                opti.set_value(patch_radius, 100.0)
            
            # Warm start
            if prev_X_sol is not None:
                try:
                    opti.set_initial(X, prev_X_sol)
                    opti.set_initial(U, prev_U_sol)
                except:
                    pass
            
            # Solve
            sol = opti.solve()
            
            if sol.stats()['success']:
                prev_X_sol = sol.value(X)
                prev_U_sol = sol.value(U)
                
                u_opt = sol.value(U[:, 0])
                return u_opt
            else:
                prev_X_sol = None
                prev_U_sol = None
                return None
                
        except Exception as e:
            prev_X_sol = None
            prev_U_sol = None
            return None
    
    return solve_mpc


# ============================================================================
# 4. RL POLICY (learns patch parameters)
# ============================================================================

class PatchRLPolicy:
    """
    RL Policy that learns optimal flow field parameters.
    This is what you'll train!
    
    For now: simple heuristic policy
    Later: replace with neural network (PPO, SAC, etc.)
    """
    
    def __init__(self, num_agents):
        self.num_agents = num_agents
        
        # For training: you'll add neural network here
        # self.network = PolicyNetwork(state_dim, action_dim)
    
    def get_action(self, team_state):
        """
        Given team state, output patch parameters.
        
        Args:
            team_state: Dict with team information
        
        Returns:
            action: Dict with patch parameters
        """
        # Simple heuristic for now
        # RL will learn better policy!
        
        positions = team_state['positions']
        velocities = team_state['velocities']
        goal = team_state['goal']
        corridor_width = team_state['corridor_width']
        obstacle_density = team_state.get('obstacle_density', 0.0)
        
        # Compute current team spread (to determine if patch needs to be larger)
        team_centroid = np.mean(positions, axis=0)
        max_dist_from_centroid = max([np.linalg.norm(np.array(p) - team_centroid) 
                                     for p in positions])
        
        # Heuristic decisions (RL will learn better!)
        # Key insight: compression_factor depends on SENSOR DATA!
        
        # When obstacle density is high → compress patch (squeeze)
        # When corridor is narrow → compress patch (squeeze)
        # When wide and clear → expand patch
        
        if corridor_width < 2.5 or obstacle_density > 0.4:
            # SQUEEZE: Narrow or cluttered
            compression_factor = 0.5  # Compressed
            flow_speed_scale = 1.0    # Slow down
            formation_distance = 0.7  # Tight formation
            patch_radius_scale = 2.5  # Standard radius
        elif corridor_width > 4.0 and obstacle_density < 0.2:
            # EXPAND: Wide and clear
            compression_factor = 1.8  # Expanded
            flow_speed_scale = 3.0    # Speed up
            formation_distance = 1.8  # Spread out
            patch_radius_scale = 3.5  # Larger radius for spread
        else:
            # NORMAL: Middle ground
            compression_factor = 1.0  # Normal
            flow_speed_scale = 1.5    # Moderate speed
            formation_distance = 1.2  # Normal spacing
            patch_radius_scale = 3.0  # Medium radius
        
        # Adaptive radius based on team spread (IMPORTANT!)
        # If agents are spreading, increase patch radius
        if max_dist_from_centroid > formation_distance * 2.0:
            patch_radius_scale *= 1.3  # Increase radius to contain agents
        
        action = {
            'flow_speed_scale': flow_speed_scale,
            'flow_spread': 2.0,
            'formation_distance': formation_distance,
            'compression_factor': compression_factor,  # RL LEARNS THIS!
            'patch_radius_scale': patch_radius_scale,  # RL LEARNS THIS TOO!
            'topology': np.eye(self.num_agents),
            'formation_type': 'line'
        }
        
        return action
    
    def update(self, reward):
        """
        Update policy based on reward.
        This is where RL training happens.
        
        Args:
            reward: Reward signal from environment
        """
        # TODO: Implement RL update (PPO, SAC, etc.)
        pass


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

# Global matplotlib figure for real-time patch visualization
_patch_fig = None
_patch_ax = None

def init_patch_plot():
    """Initialize matplotlib figure for patch visualization."""
    global _patch_fig, _patch_ax
    plt.ion()  # Interactive mode
    _patch_fig, _patch_ax = plt.subplots(figsize=(10, 8))
    _patch_ax.set_aspect('equal')
    _patch_ax.grid(True, alpha=0.3)
    _patch_ax.set_xlabel('X (m)')
    _patch_ax.set_ylabel('Y (m)')
    _patch_ax.set_title('🎯 PATCH VISUALIZATION - Deformable Multi-Agent Container', 
                       fontsize=14, fontweight='bold')
    return _patch_fig, _patch_ax

def plot_patch_live(agent_positions, patch_radius, patch_width, obstacles=None, goal=None):
    """
    Plot patch as a deformable shape in separate matplotlib window.
    This GUARANTEES the patch is visible!
    
    Args:
        agent_positions: List of [x, y] positions for all agents
        patch_radius: Base radius (from formation_distance)
        patch_width: Actual width (RL-controlled via compression_factor)
        obstacles: Optional list of obstacle positions
        goal: Optional goal position
    """
    global _patch_fig, _patch_ax
    
    if _patch_fig is None:
        init_patch_plot()
    
    _patch_ax.clear()
    _patch_ax.set_aspect('equal')
    _patch_ax.grid(True, alpha=0.3)
    num_agents = len(agent_positions)
    _patch_ax.set_title(f'🎯 PATCH VISUALIZATION - {num_agents} Agents (RL-Controlled Deformation)', 
                       fontsize=14, fontweight='bold')
    
    # Compute team centroid
    team_centroid = np.mean(agent_positions, axis=0)
    
    # Patch dimensions (controlled by RL!)
    patch_length = patch_radius * 1.5
    
    # Draw PATCH as filled ellipse (deformable container)
    patch_ellipse = Ellipse(
        xy=team_centroid,
        width=patch_length * 2,
        height=patch_width * 2,
        angle=0,
        facecolor='cyan',
        edgecolor='darkblue',
        alpha=0.3,
        linewidth=4,
        label='Patch Boundary'
    )
    _patch_ax.add_patch(patch_ellipse)
    
    # Draw AGENTS as dots
    for i, pos in enumerate(agent_positions):
        _patch_ax.plot(pos[0], pos[1], 'ro', markersize=15, 
                      markeredgecolor='black', markeredgewidth=2,
                      label=f'Agent {i}' if i == 0 else '')
        _patch_ax.text(pos[0] + 0.3, pos[1] + 0.3, f'R{i}', 
                      fontsize=12, fontweight='bold')
    
    # Draw FORMATION CONNECTIONS
    if len(agent_positions) > 1:
        for i in range(len(agent_positions)):
            for j in range(i+1, len(agent_positions)):
                _patch_ax.plot([agent_positions[i][0], agent_positions[j][0]],
                             [agent_positions[i][1], agent_positions[j][1]],
                             'y-', linewidth=3, alpha=0.7)
    
    # Draw GOAL
    if goal is not None:
        _patch_ax.plot(goal[0], goal[1], 'g*', markersize=20, 
                      markeredgecolor='black', markeredgewidth=2,
                      label='Goal')
    
    # Draw OBSTACLES (if provided)
    if obstacles is not None and len(obstacles) > 0:
        obs_x = [o[0] for o in obstacles]
        obs_y = [o[1] for o in obstacles]
        _patch_ax.plot(obs_x, obs_y, 'kx', markersize=8, alpha=0.5)
    
    # Set axis limits (follow agents)
    x_min = min([p[0] for p in agent_positions]) - patch_length - 2
    x_max = max([p[0] for p in agent_positions]) + patch_length + 2
    y_min = min([p[1] for p in agent_positions]) - patch_width - 2
    y_max = max([p[1] for p in agent_positions]) + patch_width + 2
    _patch_ax.set_xlim(x_min, x_max)
    _patch_ax.set_ylim(y_min, y_max)
    
    # Add legend
    _patch_ax.legend(loc='upper right')
    
    # Add text info - show RL-controlled parameters
    info_text = f'Patch Radius: {patch_radius:.1f}m | Width: {patch_width:.1f}m'
    info_text += f'\n{num_agents} Agents | RL-Controlled Deformation'
    _patch_ax.text(0.02, 0.98, info_text, transform=_patch_ax.transAxes,
                  fontsize=11, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7))
    
    # Refresh
    plt.pause(0.001)

def visualize_patch_around_agents(env, agent_positions, patch_radius=3.0, corridor_width=5.0):
    """
    Visualize the flow field patch around agents as a DEFORMABLE SHAPE.
    The patch expands/squeezes based on environment like a fluid container.
    
    Args:
        env: F1TENTH environment
        agent_positions: List of [x, y] positions
        patch_radius: Radius of patch visualization
        corridor_width: Width of navigable corridor
    """
    try:
        import pygame
        
        # Get the pygame screen if available
        if hasattr(env.unwrapped, 'renderer') and env.unwrapped.renderer is not None:
            renderer = env.unwrapped.renderer
            
            # Check if we have access to the screen
            if hasattr(renderer, 'screen') and renderer.screen is not None:
                screen = renderer.screen
                
                # Compute team centroid
                team_centroid = np.mean(agent_positions, axis=0)
                
                # Convert world coordinates to screen coordinates
                # F1TENTH uses resolution and origin from track
                resolution = renderer.map_resolution if hasattr(renderer, 'map_resolution') else 0.05
                origin = renderer.map_origin if hasattr(renderer, 'map_origin') else [0, 0]
                
                # World to screen
                def world_to_screen(x, y):
                    screen_x = int((x - origin[0]) / resolution)
                    screen_y = int((y - origin[1]) / resolution)
                    # Flip y for pygame
                    if hasattr(renderer, 'map_height'):
                        screen_y = renderer.map_height - screen_y
                    return (screen_x, screen_y)
                
                # === DRAW DEFORMABLE PATCH SHAPE ===
                
                # Patch adapts to corridor width
                patch_width = min(patch_radius * 2, corridor_width * 0.8)
                patch_length = patch_radius * 1.5
                
                # Compute patch corners (elongated ellipse/rectangle)
                centroid_screen = world_to_screen(team_centroid[0], team_centroid[1])
                
                # Create patch polygon (deformable shape)
                # Ellipse approximation with multiple points
                num_points = 32
                patch_points = []
                for i in range(num_points):
                    angle = 2 * np.pi * i / num_points
                    # Deformed ellipse (adapts to corridor)
                    dx = (patch_length / resolution) * np.cos(angle)
                    dy = (patch_width / resolution) * np.sin(angle)
                    px = centroid_screen[0] + int(dx)
                    py = centroid_screen[1] + int(dy)
                    patch_points.append((px, py))
                
                # Draw filled patch with transparency (if supported)
                # Create a surface for transparency
                patch_surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
                
                # Draw filled patch (cyan with alpha)
                pygame.draw.polygon(patch_surface, (0, 255, 255, 80), patch_points)  # Semi-transparent
                
                # Draw thick patch boundary (cyan)
                pygame.draw.polygon(patch_surface, (0, 255, 255, 255), patch_points, 5)  # Thick border
                
                # Blit to screen
                screen.blit(patch_surface, (0, 0))
                
                # === DRAW AGENT MARKERS ===
                for pos in agent_positions:
                    pos_screen = world_to_screen(pos[0], pos[1])
                    # Draw bright dot to show agent inside patch
                    pygame.draw.circle(screen, (255, 255, 0), pos_screen, 8, 0)  # Yellow filled circle
                    pygame.draw.circle(screen, (0, 0, 0), pos_screen, 8, 2)  # Black outline
                
                # === DRAW FORMATION CONNECTIONS ===
                if len(agent_positions) > 1:
                    for i in range(len(agent_positions)):
                        for j in range(i+1, len(agent_positions)):
                            pos1_screen = world_to_screen(agent_positions[i][0], agent_positions[i][1])
                            pos2_screen = world_to_screen(agent_positions[j][0], agent_positions[j][1])
                            pygame.draw.line(screen, (255, 100, 0), pos1_screen, pos2_screen, 3)  # Thick orange line
                
                # === DRAW PATCH INFO TEXT ===
                font = pygame.font.SysFont('monospace', 16, bold=True)
                text = f"PATCH: {patch_radius:.1f}m | Width: {corridor_width:.1f}m"
                text_surface = font.render(text, True, (0, 255, 255))
                screen.blit(text_surface, (10, 10))
                
    except Exception as e:
        # Print error for debugging
        print(f"Visualization error: {e}")
        pass


# ============================================================================
# 6. MAIN TRAINING LOOP
# ============================================================================

def run_hybrid_se_mpc_rl(num_episodes=10, max_steps_per_episode=2000):
    """
    Main function: Hybrid RL + SE-MPC system with multi-episode training.
    
    Args:
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
    """
    
    print("=" * 60)
    print("Hybrid SE-MPC + RL Training")
    print("Architecture: Flow Field Patch (RL) + SE-MPC (Fixed)")
    print(f"Episodes: {num_episodes}, Max steps/episode: {max_steps_per_episode}")
    print("=" * 60)
    
    # Environment setup
    num_agents = 4  # Increased from 2 to 4 agents
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": num_agents,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "model": "st",
            "observation_config": {"type": "original"},  # Need LIDAR scans!
            "params": {"mu": 1.0},
            "reset_config": {"type": "rl_random_static"},
        },
        render_mode="human",
    )
    
    # Robot parameters
    robot_radius = 0.15
    wheelbase = 0.33
    
    # Team goal (end of track)
    team_goal = [50.0, 0.0]
    
    # Initialize components
    patch = FlowFieldPatch(num_agents=num_agents, team_goal=team_goal)
    rl_policy = PatchRLPolicy(num_agents=num_agents)
    
    # Setup SE-MPC solvers (one per agent)
    se_mpc_solvers = [setup_se_mpc_with_flow(robot_radius, wheelbase) 
                      for _ in range(num_agents)]
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print("\nStarting multi-episode training...")
    print(f"Num agents: {num_agents}")
    print(f"Team goal: {team_goal}")
    print("\n" + "="*60)
    print("🎯 PATCH VISUALIZATION ENABLED!")
    print("  → Look for MATPLOTLIB WINDOW showing deformable patch")
    print("  → Cyan ellipse = patch boundary (expands/squeezes)")
    print(f"  → Red dots = ALL {num_agents} AGENTS (should stay inside patch)")
    print("  → RL LEARNS when to compress/expand based on sensor data!")
    print("="*60 + "\n")
    
    # ============================================
    # MULTI-EPISODE TRAINING LOOP
    # ============================================
    for episode in range(num_episodes):
        # Reset environment
        obs, info = env.reset()
        done = False
        
        step_count = 0
        episode_start_time = time.time()
        episode_reward = 0.0
        
        # Storage for agent states
        agent_states = {}
        agent_positions = []
        agent_velocities = []
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        while not done and step_count < max_steps_per_episode:
            # ========================================
            # CENTRALIZED: Gather team state
            # ========================================
            
            agent_positions = []
            agent_velocities = []
            corridor_widths = []
            
            # With "original" observation type, obs is a single dict with arrays
            for i in range(num_agents):
                # Position and velocity from arrays
                x = obs["poses_x"][i]
                y = obs["poses_y"][i]
                theta = obs["poses_theta"][i]
                vx = obs["linear_vels_x"][i]
                vy = obs["linear_vels_y"][i]
                v = math.sqrt(vx**2 + vy**2)
                
                agent_positions.append([x, y])
                agent_velocities.append([vx, vy])
                
                # Store full state
                agent_states[i] = np.array([x, y, theta, v])
                
                # Estimate corridor width from LIDAR
                scan = obs["scans"][i]
                width = estimate_corridor_width(scan)
                corridor_widths.append(width)
            
            avg_corridor_width = np.mean(corridor_widths)
            
            # ========================================
            # AGGREGATE SENSOR DATA for Patch
            # ========================================
            # Give the patch "eyes" to see the environment!
            all_lidar_scans = [obs["scans"][i] for i in range(num_agents)]
            patch.update_sensor_data(all_lidar_scans)
            
            team_state = {
                'positions': agent_positions,
                'velocities': agent_velocities,
                'goal': team_goal,
                'corridor_width': avg_corridor_width,
                'obstacle_density': patch.obstacle_density,  # From aggregated LIDAR
                'aggregated_lidar': patch.aggregated_lidar   # Full LIDAR data
            }
            
            # ========================================
            # RL POLICY: Update patch parameters
            # ========================================
            
            rl_action = rl_policy.get_action(team_state)
            patch.update_from_rl_action(rl_action)
            
            # ========================================
            # COMPUTE PATCH BOUNDARY (agents need to know this!)
            # ========================================
            team_centroid = np.mean(agent_positions, axis=0)
            patch_radius = patch.formation_distance * patch.patch_radius_scale
            
            # ========================================
            # DECENTRALIZED: Each agent runs SE-MPC
            # ========================================
            
            actions = np.zeros((num_agents, 2))
            
            for i in range(num_agents):
                # Get my state
                state = agent_states[i]
                position = agent_positions[i]
                
                # Get flow guidance from patch
                v_desired = patch.get_flow_at_position(position, i, agent_positions)
                
                # Get formation neighbors from patch
                neighbors = patch.get_formation_neighbors(i)
                
                # Process LIDAR for local obstacles
                scan = obs["scans"][i]
                local_obstacles = process_lidar_scan(scan, sample_beams=20)
                
                # Solve SE-MPC (agents KNOW patch boundary!)
                u_opt = se_mpc_solvers[i](
                    state=state,
                    v_desired=v_desired,
                    local_obs=local_obstacles,
                    neighbors=neighbors,
                    neighbor_states=agent_states,
                    formation_dist=patch.formation_distance,
                    patch_cent=team_centroid,  # Agents know patch center!
                    patch_rad=patch_radius      # Agents know patch radius!
                )
                
                if u_opt is not None:
                    # Convert to f1tenth action
                    accel_opt = u_opt[0]
                    steering_opt = u_opt[1]
                    
                    # Integration for speed
                    v_current = state[3]
                    v_next = v_current + accel_opt * 0.05
                    v_next = np.clip(v_next, 0.1, 10.0)  # Match MPC bounds
                    
                    steering_opt = np.clip(steering_opt, -0.4, 0.4)
                    
                    actions[i] = [steering_opt, v_next]
                else:
                    # Fallback - move forward with moderate speed
                    actions[i] = [0.0, 3.0]
            
            # Step environment
            obs, step_reward, done, truncated, info = env.step(actions)
            
            # Visualize patch on map (deformable shape)
            # Patch radius and width NOW CONTROLLED BY RL!
            # (patch_radius already computed above for SE-MPC)
            
            # Patch width NOW CONTROLLED BY RL! (compression_factor)
            # This is what RL LEARNS!
            base_width = patch_radius * 2
            patch_width = base_width * patch.compression_factor  # RL-learned deformation!
            
            # Constrain by physical corridor (can't be wider than corridor)
            patch_width = min(patch_width, avg_corridor_width * 0.9)
            
            # Try pygame overlay
            visualize_patch_around_agents(env, agent_positions, patch_radius, patch_width)
            
            # MATPLOTLIB REAL-TIME VISUALIZATION (guaranteed to show patch!)
            if step_count % 10 == 0:  # Update every 10 steps for performance
                plot_patch_live(agent_positions, patch_radius, patch_width, 
                              obstacles=None, goal=team_goal)
            
            # Console logging with patch shape info
            if step_count % 100 == 0:
                compression_pct = patch.compression_factor * 100
                print(f"  [PATCH VISIBLE] Radius: {patch_radius:.1f}m, Width: {patch_width:.1f}m, Speed: {patch.flow_speed_scale:.1f}")
                print(f"  [RL COMPRESSION] Factor: {patch.compression_factor:.2f} ({compression_pct:.0f}%), Obstacle density: {patch.obstacle_density:.2f}")
                if patch.compression_factor < 0.7:
                    print(f"  [PATCH SHAPE] 🔴 SQUEEZED (RL decision based on sensors!)")
                elif patch.compression_factor > 1.3:
                    print(f"  [PATCH SHAPE] 🟢 EXPANDED (RL decision based on sensors!)")
                else:
                    print(f"  [PATCH SHAPE] 🟡 NORMAL (RL decision based on sensors!)")
            
            env.render()
            
            # ========================================
            # REWARD COMPUTATION (for RL training)
            # ========================================
            
            # Compute reward for this step
            team_speed = np.mean([np.linalg.norm(v) for v in agent_velocities])
            progress = -np.min([np.linalg.norm(np.array(p) - team_goal) 
                               for p in agent_positions])
            
            collision_penalty = 0.0
            for i in range(num_agents):
                if obs["collisions"][i] > 0.5:
                    collision_penalty -= 100.0
            
            # LEFT BEHIND PENALTY: Agents should stay in patch together!
            left_behind_penalty = 0.0
            # Use RL-learned patch radius (not hardcoded!)
            max_patch_radius = patch.formation_distance * patch.patch_radius_scale
            
            # Compute team centroid
            team_centroid = np.mean(agent_positions, axis=0)
            
            # Penalty if any agent is too far from centroid (left behind)
            for i, pos in enumerate(agent_positions):
                dist_from_centroid = np.linalg.norm(np.array(pos) - team_centroid)
                if dist_from_centroid > max_patch_radius:
                    # Agent is outside patch! Heavy penalty
                    left_behind_penalty -= 50.0 * (dist_from_centroid - max_patch_radius)
            
            # Also penalize high spread (agents too far apart)
            inter_agent_distances = []
            for i in range(len(agent_positions)):
                for j in range(i+1, len(agent_positions)):
                    dist = np.linalg.norm(np.array(agent_positions[i]) - np.array(agent_positions[j]))
                    inter_agent_distances.append(dist)
            
            if len(inter_agent_distances) > 0:
                max_inter_dist = max(inter_agent_distances)
                if max_inter_dist > max_patch_radius * 2:
                    # Agents too spread out!
                    left_behind_penalty -= 20.0 * (max_inter_dist - max_patch_radius * 2)
            
            step_reward_rl = (
                5.0 * team_speed +          # Reward high speed (flow!)
                10.0 * progress +           # Reward progress to goal
                collision_penalty +         # Penalize collisions
                left_behind_penalty         # Penalize agents leaving patch
            )
            
            episode_reward += step_reward_rl
            
            # RL policy update (accumulate for episode)
            rl_policy.update(step_reward_rl)
            
            step_count += 1
            
            # Check if goal reached
            min_dist_to_goal = min([np.linalg.norm(np.array(p) - team_goal) 
                                   for p in agent_positions])
            if min_dist_to_goal < 2.0:  # Within 2m of goal
                print(f"\n🎯 Goal reached at step {step_count}!")
                success_count += 1
                done = True
            
            # Logging
            if step_count % 100 == 0:
                elapsed = time.time() - episode_start_time
                avg_speed = np.mean([np.linalg.norm(v) for v in agent_velocities])
                
                # Compute team cohesion
                team_centroid = np.mean(agent_positions, axis=0)
                max_dist_from_centroid = max([np.linalg.norm(np.array(p) - team_centroid) 
                                             for p in agent_positions])
                
                print(f"  Step {step_count}: time={elapsed:.1f}s, "
                      f"reward={episode_reward:.1f}, avg_speed={avg_speed:.2f}m/s")
                print(f"    Patch: radius={patch_radius:.1f}m, "
                      f"cohesion={max_dist_from_centroid:.1f}m, "
                      f"left_behind_penalty={left_behind_penalty:.1f}")
                for i, pos in enumerate(agent_positions):
                    dist = np.linalg.norm(np.array(pos) - team_goal)
                    print(f"    Agent {i}: pos=({pos[0]:.1f},{pos[1]:.1f}), "
                          f"goal_dist={dist:.1f}m")
        
        # Episode complete
        episode_time = time.time() - episode_start_time
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1} Summary:")
        print(f"  Reward: {episode_reward:.1f}")
        print(f"  Steps: {step_count}")
        print(f"  Time: {episode_time:.2f}s")
        print(f"  Success rate: {success_count}/{episode + 1} = {100*success_count/(episode+1):.1f}%")
        print(f"{'='*60}")
    
    # Training complete
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total episodes: {num_episodes}")
    print(f"Success rate: {success_count}/{num_episodes} = {100*success_count/num_episodes:.1f}%")
    print(f"Average reward: {np.mean(episode_rewards):.1f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"{'='*60}")
    
    # Close visualization
    if _patch_fig is not None:
        plt.ioff()
        plt.close(_patch_fig)
    
    env.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYBRID SE-MPC + RL SYSTEM")
    print("Patch (RL) generates flow field → SE-MPC (fixed) executes locally")
    print("="*60 + "\n")
    
    # Run with multiple episodes for training
    run_hybrid_se_mpc_rl(num_episodes=5, max_steps_per_episode=2000)

