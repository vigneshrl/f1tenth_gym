#!/usr/bin/env python3
"""
DUAL POLICY ELLIPSOID SYSTEM (CORRECTED)
========================================
Two-policy hierarchical architecture for multi-agent coordination:

1. PATCH POLICY (High-Level RL):
   - Controls ellipsoid SHAPE ONLY: (axis_a, axis_b, heading)
   - CENTER IS ALWAYS TEAM CENTROID (NOT learned!)
   - Learns when to compress/expand based on environment (corridor width)
   - 3-dimensional action space
   
2. AGENT POLICY (Low-Level RL):
   - Shared policy for all 4 agents (same network, different observations)
   - Stay inside ellipsoid + avoid inter-collision + adapt to deformation
   - Car-like dynamics (acceleration, steering)
   - Agents start at KNOWN positions relative to ellipsoid center
   - 2-dimensional action per agent (8 total for 4 agents)

KEY INSIGHT:
- Ellipsoid center = team centroid (follows agents automatically)
- RL only learns SHAPE and ORIENTATION (a, b, θ)
- This ensures ellipsoid is ALWAYS around the agents!

Why not PSD matrix? 
- Directly optimizing shape matrix requires PSD enforcement (hard in RL)
- (a, b, θ) tuple always produces valid ellipsoid shape!

Action Space: 3 (patch) + 8 (agents) = 11 dimensions
Uses: Stable-Baselines3 (PPO)
"""

import time
import os
import json
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from collections import deque
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D
import torch
import torch.nn as nn

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    SB3_AVAILABLE = True
except ImportError:
    print("WARNING: stable-baselines3 not installed. Install with: pip install stable-baselines3")
    SB3_AVAILABLE = False


# ============================================================================
# 1. ELLIPSOID MATH UTILITIES
# ============================================================================

class Ellipsoid:
    """
    Ellipsoid defined by (center_x, center_y, axis_a, axis_b, heading_angle).
    
    Equation in rotated frame:
        ((x-cx)*cos(θ) + (y-cy)*sin(θ))² / a² +
        (-(x-cx)*sin(θ) + (y-cy)*cos(θ))² / b² <= 1
    """
    
    def __init__(self, cx, cy, axis_a, axis_b, theta):
        """
        Args:
            cx, cy: Center coordinates
            axis_a: Semi-major axis (length in direction of theta)
            axis_b: Semi-minor axis (length perpendicular to theta)
            theta: Heading angle (radians)
        """
        self.cx = cx
        self.cy = cy
        self.a = max(axis_a, 0.5)  # Minimum 0.5m
        self.b = max(axis_b, 0.5)
        self.theta = theta
        
        # Precompute rotation
        self.cos_t = np.cos(theta)
        self.sin_t = np.sin(theta)
    
    def update(self, cx, cy, axis_a, axis_b, theta):
        """Update ellipsoid parameters."""
        self.cx = cx
        self.cy = cy
        self.a = max(axis_a, 0.5)
        self.b = max(axis_b, 0.5)
        self.theta = theta
        self.cos_t = np.cos(theta)
        self.sin_t = np.sin(theta)
    
    def is_inside(self, x, y, margin=0.0):
        """
        Check if point (x,y) is inside ellipsoid.
        
        Args:
            x, y: Point coordinates
            margin: Safety margin (negative = smaller ellipsoid)
        
        Returns:
            bool: True if inside
        """
        dx = x - self.cx
        dy = y - self.cy
        
        # Rotate to ellipsoid frame
        x_rot = dx * self.cos_t + dy * self.sin_t
        y_rot = -dx * self.sin_t + dy * self.cos_t
        
        # Ellipsoid equation
        a_eff = max(self.a - margin, 0.1)
        b_eff = max(self.b - margin, 0.1)
        
        value = (x_rot / a_eff)**2 + (y_rot / b_eff)**2
        return value <= 1.0
    
    def signed_distance(self, x, y):
        """
        Approximate signed distance to ellipsoid boundary.
        Negative = inside, Positive = outside
        
        Args:
            x, y: Point coordinates
        
        Returns:
            float: Approximate signed distance
        """
        dx = x - self.cx
        dy = y - self.cy
        
        # Rotate to ellipsoid frame
        x_rot = dx * self.cos_t + dy * self.sin_t
        y_rot = -dx * self.sin_t + dy * self.cos_t
        
        # Normalized distance (1.0 = on boundary)
        normalized_dist = np.sqrt((x_rot / self.a)**2 + (y_rot / self.b)**2)
        
        # Approximate signed distance
        # Using average radius for scaling
        avg_radius = (self.a + self.b) / 2
        signed_dist = (normalized_dist - 1.0) * avg_radius
        
        return signed_dist
    
    def get_boundary_point(self, angle):
        """
        Get point on ellipsoid boundary at given angle (in ellipsoid frame).
        
        Args:
            angle: Angle in radians (in ellipsoid local frame)
        
        Returns:
            (x, y): World coordinates of boundary point
        """
        # Point in ellipsoid frame
        x_local = self.a * np.cos(angle)
        y_local = self.b * np.sin(angle)
        
        # Rotate back to world frame
        x_world = x_local * self.cos_t - y_local * self.sin_t + self.cx
        y_world = x_local * self.sin_t + y_local * self.cos_t + self.cy
        
        return x_world, y_world
    
    def get_direction_to_center(self, x, y):
        """Get unit vector from point (x,y) toward ellipsoid center."""
        dx = self.cx - x
        dy = self.cy - y
        dist = np.sqrt(dx**2 + dy**2) + 1e-8
        return dx / dist, dy / dist
    
    def get_params(self):
        """Return parameters as array."""
        return np.array([self.cx, self.cy, self.a, self.b, self.theta])
    
    def area(self):
        """Return ellipsoid area (for reward scaling)."""
        return np.pi * self.a * self.b


# ============================================================================
# 2. CUSTOM GYMNASIUM ENVIRONMENTS
# ============================================================================

class EllipsoidPatchEnv(gym.Env):
    """
    Environment for training the HIGH-LEVEL Patch Policy.
    
    Observation: Team state (positions, velocities, goal, obstacles)
    Action: Ellipsoid parameters (cx, cy, a, b, theta)
    Reward: Team progress + cohesion + efficiency
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, base_env, num_agents=4, team_goal=None, render_mode=None):
        super().__init__()
        
        self.base_env = base_env
        self.num_agents = num_agents
        self.team_goal = np.array(team_goal if team_goal else [50.0, 0.0])
        self.render_mode = render_mode
        
        # Ellipsoid bounds
        self.cx_range = (-50.0, 100.0)  # Reasonable track range
        self.cy_range = (-50.0, 50.0)
        self.a_range = (0.5, 5.0)       # Axis lengths
        self.b_range = (0.5, 5.0)
        self.theta_range = (0.0, 2 * np.pi)
        
        # Action space: (cx, cy, a, b, theta) normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )
        
        # Observation space
        # Per agent: (x, y, vx, vy, theta) = 5
        # Global: (goal_x, goal_y, avg_corridor_width, obstacle_density) = 4
        # Ellipsoid state: (cx, cy, a, b, theta) = 5
        obs_dim = num_agents * 5 + 4 + 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Current ellipsoid
        self.ellipsoid = Ellipsoid(0, 0, 2.0, 2.0, 0.0)
        
        # State
        self.agent_positions = []
        self.agent_velocities = []
        self.current_obs = None
        
    def _denormalize_action(self, action):
        """Convert [-1,1] action to actual ellipsoid parameters."""
        cx = (action[0] + 1) / 2 * (self.cx_range[1] - self.cx_range[0]) + self.cx_range[0]
        cy = (action[1] + 1) / 2 * (self.cy_range[1] - self.cy_range[0]) + self.cy_range[0]
        a = (action[2] + 1) / 2 * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
        b = (action[3] + 1) / 2 * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
        theta = (action[4] + 1) / 2 * (self.theta_range[1] - self.theta_range[0]) + self.theta_range[0]
        return cx, cy, a, b, theta
    
    def _get_observation(self, base_obs):
        """Construct observation from base environment observation."""
        obs_list = []
        
        # Per-agent state
        for i in range(self.num_agents):
            x = base_obs["poses_x"][i]
            y = base_obs["poses_y"][i]
            vx = base_obs["linear_vels_x"][i]
            vy = base_obs["linear_vels_y"][i]
            theta = base_obs["poses_theta"][i]
            obs_list.extend([x, y, vx, vy, theta])
        
        # Global state
        obs_list.extend([self.team_goal[0], self.team_goal[1]])
        
        # Corridor width estimate (from LIDAR)
        corridor_widths = []
        for i in range(self.num_agents):
            scan = base_obs["scans"][i]
            mid = len(scan) // 2
            quarter = len(scan) // 4
            left_min = np.min(scan[:quarter]) if len(scan[:quarter]) > 0 else 30.0
            right_min = np.min(scan[-quarter:]) if len(scan[-quarter:]) > 0 else 30.0
            corridor_widths.append(left_min + right_min)
        avg_corridor = np.mean(corridor_widths)
        obs_list.append(avg_corridor)
        
        # Obstacle density
        all_scans = [base_obs["scans"][i] for i in range(self.num_agents)]
        avg_scan = np.mean(all_scans, axis=0)
        obstacle_density = np.sum(avg_scan < 3.0) / len(avg_scan)
        obs_list.append(obstacle_density)
        
        # Current ellipsoid state
        obs_list.extend(self.ellipsoid.get_params())
        
        return np.array(obs_list, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        base_obs, info = self.base_env.reset()
        
        # Initialize ellipsoid at team centroid
        positions = [[base_obs["poses_x"][i], base_obs["poses_y"][i]] 
                     for i in range(self.num_agents)]
        centroid = np.mean(positions, axis=0)
        self.ellipsoid.update(centroid[0], centroid[1], 2.5, 2.5, 0.0)
        
        self.agent_positions = positions
        self.current_obs = base_obs
        
        obs = self._get_observation(base_obs)
        return obs, info
    
    def step(self, action):
        """
        Step the patch policy - update ellipsoid, then let agents respond.
        """
        # Denormalize action to ellipsoid parameters
        cx, cy, a, b, theta = self._denormalize_action(action)
        self.ellipsoid.update(cx, cy, a, b, theta)
        
        # The actual agent stepping is done by AgentEnv
        # Here we just return reward for the patch choice
        
        # Compute reward based on current state
        reward = self._compute_patch_reward()
        
        obs = self._get_observation(self.current_obs)
        
        # Check termination
        terminated = False
        truncated = False
        
        return obs, reward, terminated, truncated, {}
    
    def _compute_patch_reward(self):
        """Compute reward for patch policy."""
        reward = 0.0
        
        # 1. Progress reward (ellipsoid closer to goal)
        dist_to_goal = np.sqrt((self.ellipsoid.cx - self.team_goal[0])**2 + 
                               (self.ellipsoid.cy - self.team_goal[1])**2)
        reward -= 0.1 * dist_to_goal  # Closer = better
        
        # 2. Cohesion reward (all agents inside)
        agents_inside = 0
        for pos in self.agent_positions:
            if self.ellipsoid.is_inside(pos[0], pos[1]):
                agents_inside += 1
            else:
                # Penalty for agent outside
                signed_dist = self.ellipsoid.signed_distance(pos[0], pos[1])
                reward -= 10.0 * max(0, signed_dist)
        
        # Bonus for all inside
        if agents_inside == self.num_agents:
            reward += 5.0
        
        # 3. Efficiency reward (smaller ellipsoid is more efficient)
        area = self.ellipsoid.area()
        reward -= 0.01 * area  # Small penalty for large area
        
        return reward
    
    def update_agent_states(self, base_obs):
        """Called by training loop to update agent states."""
        self.current_obs = base_obs
        self.agent_positions = [[base_obs["poses_x"][i], base_obs["poses_y"][i]] 
                                for i in range(self.num_agents)]
        self.agent_velocities = [[base_obs["linear_vels_x"][i], base_obs["linear_vels_y"][i]] 
                                 for i in range(self.num_agents)]


class AgentEnv(gym.Env):
    """
    Environment for training the LOW-LEVEL Agent Policy.
    
    Same policy used by ALL agents (parameter sharing).
    Each agent gets its own observation, outputs its own action.
    
    Observation: Local state (own pose, ellipsoid relative, neighbors, LIDAR)
    Action: (acceleration, steering_angle)
    Reward: Stay in ellipsoid + avoid collisions + progress
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, base_env, ellipsoid, num_agents=4, agent_idx=0, 
                 team_goal=None, render_mode=None):
        super().__init__()
        
        self.base_env = base_env
        self.ellipsoid = ellipsoid
        self.num_agents = num_agents
        self.agent_idx = agent_idx
        self.team_goal = np.array(team_goal if team_goal else [50.0, 0.0])
        self.render_mode = render_mode
        
        # Car-like dynamics bounds
        self.accel_max = 8.0
        self.steering_max = 0.4
        self.v_max = 15.0
        self.v_min = 0.5
        
        # Robot parameters
        self.robot_radius = 0.15
        self.safety_distance = 0.5  # Min distance between agents
        
        # Action space: (acceleration, steering) normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space
        # Own state: (x, y, theta, v) = 4
        # Ellipsoid relative: (cx_rel, cy_rel, a, b, ellip_theta, signed_dist) = 6
        # Goal relative: (gx_rel, gy_rel, dist_to_goal) = 3
        # Neighbors relative: (num_agents-1) * (dx, dy, dist) = (num_agents-1) * 3
        # LIDAR compressed: 16 beams
        neighbor_dim = (num_agents - 1) * 3
        lidar_dim = 16
        obs_dim = 4 + 6 + 3 + neighbor_dim + lidar_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # State
        self.all_agent_states = None
        self.current_obs = None
        self.prev_velocity = 0.0
        
    def _compress_lidar(self, scan, num_output=16):
        """Compress 1080-beam LIDAR to fewer beams."""
        scan = np.array(scan)
        
        # Clamp and normalize
        scan = np.clip(scan, 0.1, 30.0) / 30.0  # Normalize to [0, 1]
        
        # Downsample by averaging
        chunk_size = len(scan) // num_output
        compressed = []
        for i in range(num_output):
            start = i * chunk_size
            end = start + chunk_size
            compressed.append(np.min(scan[start:end]))  # Use min (closest obstacle)
        
        return np.array(compressed, dtype=np.float32)
    
    def _get_observation(self, base_obs, agent_idx):
        """Construct local observation for this agent."""
        obs_list = []
        
        # Own state
        x = base_obs["poses_x"][agent_idx]
        y = base_obs["poses_y"][agent_idx]
        theta = base_obs["poses_theta"][agent_idx]
        vx = base_obs["linear_vels_x"][agent_idx]
        vy = base_obs["linear_vels_y"][agent_idx]
        v = np.sqrt(vx**2 + vy**2)
        obs_list.extend([x, y, theta, v])
        
        # Ellipsoid relative (in agent's frame)
        cx_rel = self.ellipsoid.cx - x
        cy_rel = self.ellipsoid.cy - y
        signed_dist = self.ellipsoid.signed_distance(x, y)
        obs_list.extend([cx_rel, cy_rel, self.ellipsoid.a, self.ellipsoid.b, 
                        self.ellipsoid.theta, signed_dist])
        
        # Goal relative
        gx_rel = self.team_goal[0] - x
        gy_rel = self.team_goal[1] - y
        dist_to_goal = np.sqrt(gx_rel**2 + gy_rel**2)
        obs_list.extend([gx_rel, gy_rel, dist_to_goal])
        
        # Neighbors relative
        for j in range(self.num_agents):
            if j != agent_idx:
                nx = base_obs["poses_x"][j]
                ny = base_obs["poses_y"][j]
                dx = nx - x
                dy = ny - y
                dist = np.sqrt(dx**2 + dy**2)
                obs_list.extend([dx, dy, dist])
        
        # LIDAR compressed
        scan = base_obs["scans"][agent_idx]
        compressed_lidar = self._compress_lidar(scan, num_output=16)
        obs_list.extend(compressed_lidar)
        
        return np.array(obs_list, dtype=np.float32)
    
    def _denormalize_action(self, action):
        """Convert [-1,1] action to actual control."""
        accel = action[0] * self.accel_max
        steering = action[1] * self.steering_max
        return accel, steering
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Don't reset base_env here - managed by training loop
        if self.current_obs is None:
            base_obs, info = self.base_env.reset()
            self.current_obs = base_obs
        else:
            info = {}
        
        obs = self._get_observation(self.current_obs, self.agent_idx)
        return obs, info
    
    def step(self, action):
        """
        Step for single agent.
        Note: Actual env stepping is coordinated by training loop.
        """
        accel, steering = self._denormalize_action(action)
        
        # Compute reward
        reward = self._compute_reward(accel, steering)
        
        # Get observation
        obs = self._get_observation(self.current_obs, self.agent_idx)
        
        # Check termination
        terminated = False
        truncated = False
        
        # Check collision with wall
        if self.current_obs["collisions"][self.agent_idx] > 0.5:
            reward -= 100.0
            terminated = True
        
        return obs, reward, terminated, truncated, {"accel": accel, "steering": steering}
    
    def _compute_reward(self, accel, steering):
        """Compute reward for this agent."""
        reward = 0.0
        
        x = self.current_obs["poses_x"][self.agent_idx]
        y = self.current_obs["poses_y"][self.agent_idx]
        vx = self.current_obs["linear_vels_x"][self.agent_idx]
        vy = self.current_obs["linear_vels_y"][self.agent_idx]
        v = np.sqrt(vx**2 + vy**2)
        
        # 1. STAY IN ELLIPSOID (most important!)
        signed_dist = self.ellipsoid.signed_distance(x, y)
        if signed_dist > 0:  # Outside
            reward -= 50.0 * signed_dist  # Heavy penalty
        else:  # Inside
            reward += 2.0  # Small bonus for staying inside
        
        # 2. AVOID INTER-COLLISION (spring-damper model)
        for j in range(self.num_agents):
            if j != self.agent_idx:
                nx = self.current_obs["poses_x"][j]
                ny = self.current_obs["poses_y"][j]
                dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                
                # Collision penalty (very close)
                if dist < self.safety_distance:
                    reward -= 100.0 * (self.safety_distance - dist)
                
                # Spring-damper: prefer equilibrium distance
                equilibrium_dist = 1.0
                spring_error = abs(dist - equilibrium_dist)
                reward -= 2.0 * spring_error
        
        # 3. PROGRESS TOWARD GOAL
        dist_to_goal = np.sqrt((x - self.team_goal[0])**2 + (y - self.team_goal[1])**2)
        reward -= 0.1 * dist_to_goal
        
        # 4. SPEED REWARD (move fast!)
        reward += 1.0 * v
        
        # 5. SMOOTHNESS (penalize jerky control)
        reward -= 0.01 * abs(accel)
        reward -= 0.1 * abs(steering)
        
        return reward
    
    def update_observation(self, base_obs):
        """Update current observation (called by training loop)."""
        self.current_obs = base_obs


# ============================================================================
# 3. MULTI-AGENT WRAPPER (Coordinates all agents)
# ============================================================================

class MultiAgentEllipsoidEnv(gym.Env):
    """
    CORRECTED Dual-Policy Environment:
    
    1. PATCH POLICY: Controls ONLY (a, b, theta) - shape and orientation
       - Center is ALWAYS team centroid (NOT learned!)
       - Learns when to compress/expand based on environment
       
    2. AGENT POLICY: Same policy for all 4 agents
       - Stay inside patch
       - Avoid inter-collision
       - Adapt to patch deformation
       - Agents start at KNOWN positions relative to patch center
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, num_agents=4, team_goal=None, render_mode="human"):
        super().__init__()
        
        self.num_agents = num_agents
        self.team_goal = np.array(team_goal if team_goal else [50.0, 0.0])
        self.render_mode = render_mode
        
        # Create base F1TENTH environment
        self.base_env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": num_agents,
                "timestep": 0.01,
                "integrator": "rk4",
                "control_input": ["speed", "steering_angle"],
                "model": "st",
                "observation_config": {"type": "original"},
                "params": {"mu": 1.0},
                "reset_config": {"type": "rl_random_static"},
            },
            render_mode=render_mode,
        )
        
        # Shared ellipsoid - CENTER IS ALWAYS TEAM CENTROID!
        self.ellipsoid = Ellipsoid(0, 0, 2.5, 2.5, 0.0)
        
        # ===== CORRECTED ACTION SPACE =====
        # Patch: 3 (a, b, theta) - NOT center! Center = team centroid
        # Agents: num_agents * 2 (accel, steering each)
        # Total: 3 + 4*2 = 11
        total_action_dim = 3 + num_agents * 2
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_action_dim,),
            dtype=np.float32
        )
        
        # Combined observation space
        # Patch obs: corridor_width, obstacle_density, wall_clearance, safe_size, current (a, b, theta) = 7
        # Per agent: relative_to_center(2), own_vel(2), heading(1), 
        #            signed_dist_to_boundary(1), formation_offset(2), neighbors_rel(6), lidar(16) = 30
        self.patch_obs_dim = 7  # Added wall_clearance and safe_size
        self.agent_obs_dim = 30  # 2+2+1+1+2+6+16 = 30
        total_obs_dim = self.patch_obs_dim + self.agent_obs_dim * num_agents  # 7 + 30*4 = 127
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
        
        # State tracking
        self.current_base_obs = None
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Bounds
        self.accel_max = 8.0
        self.steering_max = 0.4
        self.v_max = 15.0
        self.v_min = 0.5
        
        # Ellipsoid bounds (ONLY for a, b, theta - NOT center!)
        # MINIMUM SIZE: Must fit 4 agents in formation!
        # Formation is diamond with max offset ~1.0m, so minimum radius ~1.5m
        self.min_ellipsoid_size = 1.5   # HARD FLOOR - 4 agents must fit!
        self.a_range = (self.min_ellipsoid_size, 5.0)  # Semi-major axis
        self.b_range = (self.min_ellipsoid_size, 5.0)  # Semi-minor axis
        self.theta_range = (0.0, np.pi)  # Heading (0 to 180 degrees is enough)
        
        # Previous velocities for integration
        self.prev_velocities = [0.0] * num_agents
        
        # Initial formation offsets (KNOWN positions relative to center)
        # Diamond formation: front, back, left, right
        self.formation_offsets = [
            [1.0, 0.0],    # Agent 0: Front
            [-1.0, 0.0],   # Agent 1: Back
            [0.0, 0.8],    # Agent 2: Left
            [0.0, -0.8],   # Agent 3: Right
        ]
        
        # Visualization
        self._fig = None
        self._ax = None
        
    def _denormalize_patch_action(self, action):
        """
        Convert normalized patch action to ellipsoid params.
        ONLY (a, b, theta) - NOT center!
        
        IMPORTANT: Enforces MINIMUM size so 4 agents can always fit!
        """
        a = (action[0] + 1) / 2 * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
        b = (action[1] + 1) / 2 * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
        theta = (action[2] + 1) / 2 * (self.theta_range[1] - self.theta_range[0]) + self.theta_range[0]
        
        # ENFORCE MINIMUM SIZE - 4 agents must always fit!
        a = max(a, self.min_ellipsoid_size)
        b = max(b, self.min_ellipsoid_size)
        
        return a, b, theta
    
    def _denormalize_agent_action(self, action):
        """Convert normalized agent action to control."""
        accel = action[0] * self.accel_max
        steering = action[1] * self.steering_max
        return accel, steering
    
    def _get_team_centroid(self, base_obs):
        """Compute team centroid - this is ALWAYS the ellipsoid center."""
        positions = [[base_obs["poses_x"][i], base_obs["poses_y"][i]] 
                     for i in range(self.num_agents)]
        return np.mean(positions, axis=0)
    
    def _compress_lidar(self, scan, num_output=16):
        """Compress LIDAR scan."""
        scan = np.clip(scan, 0.1, 30.0) / 30.0
        chunk_size = len(scan) // num_output
        compressed = []
        for i in range(num_output):
            start = i * chunk_size
            end = start + chunk_size
            compressed.append(np.min(scan[start:end]))
        return np.array(compressed, dtype=np.float32)
    
    def _estimate_corridor_width(self, base_obs):
        """Estimate corridor width from all agents' LIDAR."""
        corridor_widths = []
        for i in range(self.num_agents):
            scan = base_obs["scans"][i]
            quarter = len(scan) // 4
            left_min = np.min(scan[:quarter]) if len(scan[:quarter]) > 0 else 30.0
            right_min = np.min(scan[-quarter:]) if len(scan[-quarter:]) > 0 else 30.0
            corridor_widths.append(left_min + right_min)
        return np.mean(corridor_widths)
    
    def _estimate_obstacle_density(self, base_obs):
        """Estimate obstacle density from LIDAR."""
        all_scans = [base_obs["scans"][i] for i in range(self.num_agents)]
        avg_scan = np.mean(all_scans, axis=0)
        return np.sum(avg_scan < 3.0) / len(avg_scan)
    
    def _check_ellipsoid_wall_collision(self, base_obs):
        """
        Check if ellipsoid boundary hits track walls using LIDAR.
        
        Returns:
            collision_severity: 0 if no collision, >0 if ellipsoid extends into wall
            min_clearance: Minimum distance from ellipsoid boundary to wall
        """
        # Get ellipsoid boundary points and check against LIDAR
        centroid = np.array([self.ellipsoid.cx, self.ellipsoid.cy])
        
        # Sample points around ellipsoid boundary
        num_samples = 16
        collision_severity = 0.0
        min_clearance = float('inf')
        
        for i in range(self.num_agents):
            x = base_obs["poses_x"][i]
            y = base_obs["poses_y"][i]
            theta_agent = base_obs["poses_theta"][i]
            scan = base_obs["scans"][i]
            
            # For each LIDAR beam direction, check if ellipsoid extends past obstacle
            num_beams = len(scan)
            fov = 4.7  # radians
            
            for beam_idx in range(0, num_beams, num_beams // num_samples):
                # Beam angle in world frame
                beam_angle = -fov/2 + (beam_idx / num_beams) * fov + theta_agent
                obstacle_dist = scan[beam_idx]
                
                if obstacle_dist < 30.0:  # Real obstacle detected
                    # Point where obstacle is
                    obs_x = x + obstacle_dist * np.cos(beam_angle)
                    obs_y = y + obstacle_dist * np.sin(beam_angle)
                    
                    # Check if this obstacle point is inside our ellipsoid
                    if self.ellipsoid.is_inside(obs_x, obs_y):
                        # Wall is INSIDE ellipsoid - collision!
                        penetration = -self.ellipsoid.signed_distance(obs_x, obs_y)
                        collision_severity += penetration
                    
                    # Also check clearance from ellipsoid boundary to wall
                    signed_dist = self.ellipsoid.signed_distance(obs_x, obs_y)
                    if signed_dist > 0:  # Wall is outside ellipsoid
                        min_clearance = min(min_clearance, signed_dist)
        
        if min_clearance == float('inf'):
            min_clearance = 10.0  # Default large clearance
            
        return collision_severity, min_clearance
    
    def _get_safe_ellipsoid_size(self, base_obs):
        """
        Get maximum safe ellipsoid size that won't hit walls.
        Uses LIDAR to find minimum distance to walls in all directions.
        """
        min_distances = []
        
        for i in range(self.num_agents):
            scan = base_obs["scans"][i]
            # Get minimum distance in each quadrant
            quarter = len(scan) // 4
            
            # Front, left, back, right minimums
            front_min = np.min(scan[quarter:3*quarter]) if len(scan) > 0 else 30.0
            left_min = np.min(scan[:quarter]) if len(scan) > 0 else 30.0
            right_min = np.min(scan[3*quarter:]) if len(scan) > 0 else 30.0
            
            min_distances.extend([front_min, left_min, right_min])
        
        # Safe size is minimum distance minus safety margin
        safety_margin = 0.5
        safe_size = max(0.5, np.min(min_distances) - safety_margin)
        
        return safe_size
    
    def _get_observation(self, base_obs):
        """
        Build combined observation.
        
        Structure:
        - Patch obs (5): corridor_width, obstacle_density, a, b, theta
        - Per agent (28 each): rel_to_center(2), vel(2), heading(1), 
                               signed_dist(1), formation_offset(2), neighbors(6), lidar(16)
        """
        obs_list = []
        
        # Get team centroid (= ellipsoid center)
        centroid = self._get_team_centroid(base_obs)
        
        # === PATCH OBSERVATION (7 dims) ===
        corridor_width = self._estimate_corridor_width(base_obs)
        obstacle_density = self._estimate_obstacle_density(base_obs)
        _, wall_clearance = self._check_ellipsoid_wall_collision(base_obs)
        safe_size = self._get_safe_ellipsoid_size(base_obs)
        
        obs_list.extend([
            corridor_width / 10.0,   # Normalized
            obstacle_density,         # Already [0, 1]
            wall_clearance / 5.0,     # Normalized - how far ellipsoid is from walls
            safe_size / 5.0,          # Normalized - max safe size based on LIDAR
            self.ellipsoid.a / 5.0,   # Normalized
            self.ellipsoid.b / 5.0,
            self.ellipsoid.theta / np.pi  # Normalized to [0, 1]
        ])
        
        # === AGENT OBSERVATIONS (28 dims each) ===
        for agent_idx in range(self.num_agents):
            x = base_obs["poses_x"][agent_idx]
            y = base_obs["poses_y"][agent_idx]
            theta = base_obs["poses_theta"][agent_idx]
            vx = base_obs["linear_vels_x"][agent_idx]
            vy = base_obs["linear_vels_y"][agent_idx]
            v = np.sqrt(vx**2 + vy**2)
            
            # Position relative to centroid (ellipsoid center)
            rel_x = x - centroid[0]
            rel_y = y - centroid[1]
            obs_list.extend([rel_x, rel_y])
            
            # Velocity
            obs_list.extend([vx / 10.0, vy / 10.0])  # Normalized
            
            # Heading
            obs_list.append(theta / np.pi)  # Normalized
            
            # Signed distance to ellipsoid boundary (negative = inside)
            signed_dist = self.ellipsoid.signed_distance(x, y)
            obs_list.append(signed_dist)
            
            # Known formation offset (where agent SHOULD be)
            offset = self.formation_offsets[agent_idx]
            obs_list.extend(offset)
            
            # Neighbors relative (2 dims per neighbor = 6 total for 3 neighbors)
            for j in range(self.num_agents):
                if j != agent_idx:
                    nx = base_obs["poses_x"][j]
                    ny = base_obs["poses_y"][j]
                    dx = (nx - x) / 5.0  # Normalized
                    dy = (ny - y) / 5.0
                    obs_list.extend([dx, dy])
            
            # LIDAR compressed (16 dims)
            compressed = self._compress_lidar(base_obs["scans"][agent_idx])
            obs_list.extend(compressed)
        
        return np.array(obs_list, dtype=np.float32)
    
    def _compute_reward(self, base_obs, agent_actions):
        """
        Compute combined reward for both policies.
        
        HIERARCHY:
        - Environment → Patch adapts to environment
        - Patch → Agents adapt to patch (agents follow the patch!)
        
        PATCH POLICY REWARDS (adapts to ENVIRONMENT, not agents):
        - Don't hit walls (LIDAR)
        - Fit the corridor
        - Progress toward goal
        - Maintain minimum size for 4 agents
        
        AGENT POLICY REWARDS (adapts to PATCH):
        - Stay inside the patch (agents must keep up!)
        - Avoid inter-collision
        - Maintain formation
        """
        total_reward = 0.0
        
        positions = [[base_obs["poses_x"][i], base_obs["poses_y"][i]] 
                     for i in range(self.num_agents)]
        velocities = [[base_obs["linear_vels_x"][i], base_obs["linear_vels_y"][i]] 
                      for i in range(self.num_agents)]
        centroid = self._get_team_centroid(base_obs)
        
        # =======================================================
        # PATCH POLICY REWARDS (adapts to ENVIRONMENT)
        # =======================================================
        
        # 1. DON'T HIT WALLS! (uses LIDAR - most important for patch!)
        wall_collision, wall_clearance = self._check_ellipsoid_wall_collision(base_obs)
        if wall_collision > 0:
            # Heavy penalty for ellipsoid hitting walls
            total_reward -= 100.0 * wall_collision
        
        # Bonus for good wall clearance
        if wall_clearance > 0.3:
            total_reward += 2.0
        
        # 2. Size should fit corridor (adapt to environment!)
        safe_size = self._get_safe_ellipsoid_size(base_obs)
        
        # Penalty if ellipsoid is larger than safe size (hitting walls)
        if self.ellipsoid.a > safe_size:
            total_reward -= 20.0 * (self.ellipsoid.a - safe_size)
        if self.ellipsoid.b > safe_size:
            total_reward -= 20.0 * (self.ellipsoid.b - safe_size)
        
        # 3. Progress toward goal (team moves forward)
        dist_to_goal = np.sqrt((centroid[0] - self.team_goal[0])**2 + 
                               (centroid[1] - self.team_goal[1])**2)
        total_reward += 5.0 * (50.0 - dist_to_goal) / 50.0
        
        # 4. Efficiency bonus - smaller patch (but still fits agents) is more efficient
        # Only reward being small if we're NOT hitting walls
        if wall_collision == 0:
            # Reward for being compact (but above minimum)
            compactness = (5.0 - self.ellipsoid.a) + (5.0 - self.ellipsoid.b)
            total_reward += 0.2 * compactness
        
        # =======================================================
        # AGENT POLICY REWARDS (adapts to PATCH)
        # Agents must KEEP UP with the patch - patch doesn't wait!
        # =======================================================
        
        agents_inside = 0
        for i in range(self.num_agents):
            x, y = positions[i]
            vx, vy = velocities[i]
            v = np.sqrt(vx**2 + vy**2)
            
            # 1. STAY INSIDE PATCH (agents must keep up!)
            if self.ellipsoid.is_inside(x, y):
                agents_inside += 1
                total_reward += 3.0  # Good - agent is keeping up
            else:
                # Agent fell behind / went outside - AGENT's fault, not patch's!
                signed_dist = self.ellipsoid.signed_distance(x, y)
                total_reward -= 30.0 * signed_dist  # Agent penalty
            
            # 2. Speed - agents should move!
            total_reward += 0.3 * v
            
            # 3. Wall collision (agent hit wall)
            if base_obs["collisions"][i] > 0.5:
                total_reward -= 100.0
            
            # 4. Inter-agent collision avoidance
            for j in range(i + 1, self.num_agents):
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                               (positions[i][1] - positions[j][1])**2)
                
                # Hard collision penalty
                if dist < 0.4:
                    total_reward -= 80.0 * (0.4 - dist)
                
                # Soft spring - prefer equilibrium
                equilibrium = 0.8
                spring_error = abs(dist - equilibrium)
                total_reward -= 1.0 * spring_error
            
            # 5. Formation maintenance (stay near assigned offset relative to centroid)
            rel_x = x - centroid[0]
            rel_y = y - centroid[1]
            target_x, target_y = self.formation_offsets[i]
            formation_error = np.sqrt((rel_x - target_x)**2 + (rel_y - target_y)**2)
            total_reward -= 0.5 * formation_error
        
        # Bonus if ALL agents are inside (team cohesion)
        if agents_inside == self.num_agents:
            total_reward += 10.0
        
        # === AGENT POLICY REWARDS ===
        
        for i in range(self.num_agents):
            x, y = positions[i]
            vx, vy = velocities[i]
            v = np.sqrt(vx**2 + vy**2)
            
            # 1. STAY INSIDE (most important!)
            signed_dist = self.ellipsoid.signed_distance(x, y)
            if signed_dist > 0:  # Outside
                total_reward -= 50.0 * signed_dist  # Heavy penalty
            else:  # Inside
                total_reward += 1.0
            
            # 2. Speed toward goal
            total_reward += 0.3 * v
            
            # 3. Wall collision
            if base_obs["collisions"][i] > 0.5:
                total_reward -= 100.0
            
            # 4. Inter-agent collision avoidance (spring-damper)
            for j in range(i + 1, self.num_agents):
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                               (positions[i][1] - positions[j][1])**2)
                
                # Hard collision penalty
                if dist < 0.4:
                    total_reward -= 100.0 * (0.4 - dist)
                
                # Soft spring: prefer equilibrium distance
                equilibrium = 1.0
                spring_error = abs(dist - equilibrium)
                total_reward -= 1.0 * spring_error
            
            # 5. Formation maintenance (stay near assigned offset)
            rel_x = x - centroid[0]
            rel_y = y - centroid[1]
            target_x, target_y = self.formation_offsets[i]
            formation_error = np.sqrt((rel_x - target_x)**2 + (rel_y - target_y)**2)
            total_reward -= 0.5 * formation_error
        
        return total_reward
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        base_obs, info = self.base_env.reset()
        self.current_base_obs = base_obs
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Initialize ellipsoid at team centroid with default shape
        centroid = self._get_team_centroid(base_obs)
        corridor_width = self._estimate_corridor_width(base_obs)
        initial_size = min(corridor_width * 0.4, 2.5)
        self.ellipsoid.update(centroid[0], centroid[1], initial_size, initial_size, 0.0)
        
        # Reset velocities
        self.prev_velocities = [np.sqrt(base_obs["linear_vels_x"][i]**2 + 
                                        base_obs["linear_vels_y"][i]**2)
                                for i in range(self.num_agents)]
        
        obs = self._get_observation(base_obs)
        return obs, info
    
    def step(self, action):
        """
        Execute combined action:
        - First 3 dims: patch action (a, b, theta) - NOT center!
        - Next num_agents*2 dims: agent actions
        
        Ellipsoid center is ALWAYS team centroid (computed, not learned).
        """
        # Parse action
        # Patch: 3 dims (a, b, theta)
        # Agents: 4*2 = 8 dims
        patch_action = action[:3]
        agent_actions_raw = action[3:]
        
        # Get team centroid - this IS the ellipsoid center
        centroid = self._get_team_centroid(self.current_base_obs)
        
        # Update ellipsoid shape (center = centroid, shape from RL)
        a, b, theta = self._denormalize_patch_action(patch_action)
        self.ellipsoid.update(centroid[0], centroid[1], a, b, theta)
        
        # Convert agent actions to environment actions
        env_actions = np.zeros((self.num_agents, 2))
        for i in range(self.num_agents):
            accel, steering = self._denormalize_agent_action(
                agent_actions_raw[i*2:(i+1)*2]
            )
            
            # Integrate velocity
            v_new = self.prev_velocities[i] + accel * 0.01  # dt = 0.01
            v_new = np.clip(v_new, self.v_min, self.v_max)
            self.prev_velocities[i] = v_new
            
            env_actions[i] = [steering, v_new]  # [steering, speed]
        
        # Step base environment
        base_obs, _, base_done, base_truncated, base_info = self.base_env.step(env_actions)
        self.current_base_obs = base_obs
        
        # Update ellipsoid center to new centroid (follows agents!)
        new_centroid = self._get_team_centroid(base_obs)
        self.ellipsoid.cx = new_centroid[0]
        self.ellipsoid.cy = new_centroid[1]
        
        # Compute reward
        reward = self._compute_reward(base_obs, env_actions)
        self.episode_reward += reward
        
        # Build observation
        obs = self._get_observation(base_obs)
        
        # Check termination
        self.step_count += 1
        terminated = base_done or any(base_obs["collisions"][i] > 0.5 for i in range(self.num_agents))
        truncated = self.step_count >= 2000
        
        # Goal reached?
        positions = [[base_obs["poses_x"][i], base_obs["poses_y"][i]] 
                     for i in range(self.num_agents)]
        min_dist = min(np.sqrt((p[0] - self.team_goal[0])**2 + 
                               (p[1] - self.team_goal[1])**2) for p in positions)
        if min_dist < 2.0:
            reward += 500.0  # Big bonus
            terminated = True
        
        # Visualization
        if self.step_count % 10 == 0:
            self._visualize(positions)
        
        # Render
        if self.render_mode == "human":
            self.base_env.render()
        
        return obs, reward, terminated, truncated, {"episode_reward": self.episode_reward}
    
    def _visualize(self, positions):
        """Real-time matplotlib visualization of ellipsoid AROUND agents."""
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(10, 8))
        
        self._ax.clear()
        self._ax.set_aspect('equal')
        self._ax.grid(True, alpha=0.3)
        
        # Count agents inside
        agents_inside = sum(1 for pos in positions if self.ellipsoid.is_inside(pos[0], pos[1]))
        
        self._ax.set_title(f'🎯 Ellipsoid AROUND Agents - Step {self.step_count} | '
                          f'Inside: {agents_inside}/{self.num_agents} | '
                          f'Reward: {self.episode_reward:.1f}', fontsize=14)
        
        # Draw ellipsoid (centered on team centroid!)
        ellipse = Ellipse(
            xy=(self.ellipsoid.cx, self.ellipsoid.cy),
            width=self.ellipsoid.a * 2,
            height=self.ellipsoid.b * 2,
            angle=np.degrees(self.ellipsoid.theta),
            facecolor='cyan',
            edgecolor='darkblue',
            alpha=0.3,
            linewidth=3
        )
        self._ax.add_patch(ellipse)
        
        # Draw ellipsoid center marker
        self._ax.plot(self.ellipsoid.cx, self.ellipsoid.cy, 'b+', markersize=15,
                     markeredgewidth=3, label='Patch Center (=Centroid)')
        
        # Draw agents
        colors = ['red', 'orange', 'purple', 'green']
        for i, pos in enumerate(positions):
            inside = self.ellipsoid.is_inside(pos[0], pos[1])
            color = colors[i % len(colors)]
            marker = 'o' if inside else 'X'
            size = 15 if inside else 20
            self._ax.plot(pos[0], pos[1], marker, color=color, markersize=size,
                         markeredgecolor='black', markeredgewidth=2)
            status = "✓" if inside else "✗"
            self._ax.text(pos[0] + 0.3, pos[1] + 0.3, f'R{i}{status}', 
                         fontsize=10, fontweight='bold',
                         color='green' if inside else 'red')
        
        # Draw formation target positions
        centroid = np.array([self.ellipsoid.cx, self.ellipsoid.cy])
        for i, offset in enumerate(self.formation_offsets):
            target = centroid + np.array(offset)
            self._ax.plot(target[0], target[1], 's', color=colors[i], 
                         markersize=8, alpha=0.3, markeredgecolor='black')
        
        # Draw goal
        self._ax.plot(self.team_goal[0], self.team_goal[1], 'g*', markersize=20,
                     markeredgecolor='black', markeredgewidth=2, label='Goal')
        
        # Draw inter-agent connections (spring-damper visualization)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                               (positions[i][1] - positions[j][1])**2)
                # Color based on distance (green=good, red=too close)
                if dist < 0.5:
                    color = 'red'
                elif dist < 1.5:
                    color = 'green'
                else:
                    color = 'orange'
                self._ax.plot([positions[i][0], positions[j][0]],
                             [positions[i][1], positions[j][1]],
                             '-', color=color, linewidth=2, alpha=0.7)
        
        # Set limits (centered on ellipsoid)
        margin = max(self.ellipsoid.a, self.ellipsoid.b) + 5
        self._ax.set_xlim(self.ellipsoid.cx - margin, self.ellipsoid.cx + margin)
        self._ax.set_ylim(self.ellipsoid.cy - margin, self.ellipsoid.cy + margin)
        
        # Info text
        info = f'Ellipsoid: ({self.ellipsoid.cx:.1f}, {self.ellipsoid.cy:.1f})\n'
        info += f'Axes: a={self.ellipsoid.a:.1f}, b={self.ellipsoid.b:.1f}\n'
        info += f'Heading: {np.degrees(self.ellipsoid.theta):.1f}°'
        self._ax.text(0.02, 0.98, info, transform=self._ax.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.pause(0.001)
    
    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
        self.base_env.close()


# ============================================================================
# 4. TRAINING WITH CHECKPOINTS
# ============================================================================

def train_dual_policy(
    total_timesteps=100000,
    num_agents=4,
    team_goal=None,
    save_path="dual_policy_model",
    checkpoint_freq=5000,
    checkpoint_dir="checkpoints",
    resume_from=None
):
    """
    Train the dual-policy system using PPO with automatic checkpointing.
    
    Both policies are trained simultaneously in a single combined policy.
    
    Args:
        total_timesteps: Total training steps
        num_agents: Number of agents
        team_goal: Goal position [x, y]
        save_path: Final model save path
        checkpoint_freq: Save checkpoint every N timesteps
        checkpoint_dir: Directory for checkpoints
        resume_from: Path to checkpoint to resume from (optional)
    """
    
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 is required!")
        print("Install with: pip install stable-baselines3")
        return None
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Generate run ID for this training session
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_checkpoint_dir = os.path.join(checkpoint_dir, f"run_{run_id}")
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    
    print("=" * 70)
    print("DUAL POLICY ELLIPSOID TRAINING WITH CHECKPOINTS")
    print("=" * 70)
    print(f"Agents: {num_agents}")
    print(f"Goal: {team_goal}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Checkpoint frequency: Every {checkpoint_freq} steps")
    print(f"Checkpoint directory: {run_checkpoint_dir}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  [Patch Policy] → Ellipsoid (cx, cy, a, b, θ)")
    print("  [Agent Policy × N] → Controls (accel, steering)")
    print("  Both trained simultaneously via combined PPO")
    print("=" * 70)
    
    # Create environment
    team_goal = team_goal if team_goal else [50.0, 0.0]
    
    def make_env():
        return MultiAgentEllipsoidEnv(
            num_agents=num_agents,
            team_goal=team_goal,
            render_mode="human"
        )
    
    env = DummyVecEnv([make_env])
    
    # Custom callback for logging AND checkpointing
    class CheckpointCallback(BaseCallback):
        """
        Callback that saves checkpoints at regular intervals.
        Also handles crash recovery by saving on exceptions.
        """
        def __init__(self, save_freq, save_path, checkpoint_dir, verbose=1):
            super().__init__(verbose)
            self.save_freq = save_freq
            self.save_path = save_path
            self.checkpoint_dir = checkpoint_dir
            self.episode_rewards = []
            self.episode_count = 0
            self.best_mean_reward = -np.inf
            self.last_checkpoint_step = 0
            
        def _on_step(self):
            # Track episodes
            if self.locals.get("dones", [False])[0]:
                self.episode_count += 1
                reward = self.locals.get("infos", [{}])[0].get("episode_reward", 0)
                self.episode_rewards.append(reward)
                
                if self.episode_count % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    print(f"\n[Episode {self.episode_count}] "
                          f"Avg Reward (last 10): {avg_reward:.1f}")
                    
                    # Save best model
                    if avg_reward > self.best_mean_reward:
                        self.best_mean_reward = avg_reward
                        best_path = os.path.join(self.checkpoint_dir, "best_model")
                        self.model.save(best_path)
                        print(f"   🏆 New best model saved! (reward: {avg_reward:.1f})")
            
            # Save checkpoint at intervals
            if self.num_timesteps - self.last_checkpoint_step >= self.save_freq:
                self._save_checkpoint()
                self.last_checkpoint_step = self.num_timesteps
            
            return True
        
        def _save_checkpoint(self):
            """Save a checkpoint with metadata."""
            checkpoint_name = f"checkpoint_{self.num_timesteps}"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            
            # Save model
            self.model.save(checkpoint_path)
            
            # Save metadata
            metadata = {
                "timesteps": self.num_timesteps,
                "episodes": self.episode_count,
                "best_mean_reward": float(self.best_mean_reward),
                "last_10_rewards": [float(r) for r in self.episode_rewards[-10:]] if self.episode_rewards else [],
                "timestamp": datetime.now().isoformat()
            }
            metadata_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n💾 Checkpoint saved: {checkpoint_path}")
            print(f"   Timesteps: {self.num_timesteps}, Episodes: {self.episode_count}")
        
        def _on_training_end(self):
            """Save final checkpoint when training ends."""
            self._save_checkpoint()
            print("\n✅ Final checkpoint saved on training end")
    
    # Create callback
    callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_path,
        checkpoint_dir=run_checkpoint_dir,
        verbose=1
    )
    
    # Create or load PPO model
    if resume_from and os.path.exists(resume_from + ".zip"):
        print(f"\n📂 Loading model from checkpoint: {resume_from}")
        model = PPO.load(resume_from, env=env)
        
        # Load metadata if available
        metadata_path = resume_from + "_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"   Resuming from timestep: {metadata.get('timesteps', 'unknown')}")
            print(f"   Previous best reward: {metadata.get('best_mean_reward', 'unknown')}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./dual_policy_tensorboard/"
        )
    
    print("\n🚀 Starting training...")
    print("   Watch the matplotlib window for real-time visualization!")
    print("   TensorBoard logs: ./dual_policy_tensorboard/")
    print(f"   Checkpoints: {run_checkpoint_dir}/")
    print()
    print("   💾 Checkpoints saved every", checkpoint_freq, "steps")
    print("   🏆 Best model auto-saved when reward improves")
    print("   ⚠️  If training crashes, resume with --resume flag")
    print()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save final model
        model.save(save_path)
        print(f"\n✅ Final model saved to: {save_path}")
        
        # Also save to checkpoint dir
        final_path = os.path.join(run_checkpoint_dir, "final_model")
        model.save(final_path)
        print(f"✅ Final model also saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        interrupted_path = os.path.join(run_checkpoint_dir, f"interrupted_{callback.num_timesteps}")
        model.save(interrupted_path)
        print(f"💾 Interrupted model saved to: {interrupted_path}")
        print(f"   Resume with: --resume {interrupted_path}")
        
    except Exception as e:
        print(f"\n❌ Training crashed with error: {e}")
        crash_path = os.path.join(run_checkpoint_dir, f"crash_{callback.num_timesteps}")
        try:
            model.save(crash_path)
            print(f"💾 Crash recovery model saved to: {crash_path}")
            print(f"   Resume with: --resume {crash_path}")
        except:
            print("   Could not save crash recovery model")
        raise  # Re-raise the exception
    
    finally:
        env.close()
    
    return model


def evaluate_dual_policy(model_path="dual_policy_model", num_episodes=5):
    """Evaluate trained dual-policy model."""
    
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 is required!")
        return
    
    print("=" * 70)
    print("EVALUATING DUAL POLICY MODEL")
    print("=" * 70)
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = MultiAgentEllipsoidEnv(
        num_agents=4,
        team_goal=[50.0, 0.0],
        render_mode="human"
    )
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        print(f"\nEpisode {ep + 1}/{num_episodes}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
            
            if step % 100 == 0:
                print(f"  Step {step}: Reward so far: {total_reward:.1f}")
        
        episode_rewards.append(total_reward)
        print(f"  Episode {ep + 1} finished: Total reward = {total_reward:.1f}")
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.1f}")
    print(f"Std deviation: {np.std(episode_rewards):.1f}")
    print(f"Min reward: {min(episode_rewards):.1f}")
    print(f"Max reward: {max(episode_rewards):.1f}")
    print("=" * 70)
    
    env.close()


def run_demo(num_steps=2000):
    """
    Run a demo with heuristic policy to test the environment.
    
    Action space:
    - Patch: 3 dims (a, b, theta) - center is always team centroid!
    - Agents: 4*2 = 8 dims (accel, steering per agent)
    - Total: 11 dims
    """
    print("=" * 70)
    print("DUAL POLICY ELLIPSOID DEMO")
    print("=" * 70)
    print()
    print("CORRECTED ARCHITECTURE:")
    print("  - Patch Policy: (a, b, θ) only - 3 outputs")
    print("  - Ellipsoid center = team centroid (ALWAYS!)")
    print("  - Agent Policy: (accel, steering) × 4 - 8 outputs")
    print("  - Total action dim: 11")
    print("=" * 70)
    
    env = MultiAgentEllipsoidEnv(
        num_agents=4,
        team_goal=[50.0, 0.0],
        render_mode="human"
    )
    
    obs, info = env.reset()
    
    print("\n🎬 Running demo...")
    print("   Watch the ellipsoid stay AROUND agents!")
    print("   Ellipsoid center = team centroid (follows agents)")
    print("   Press Ctrl+C to stop\n")
    
    total_reward = 0
    try:
        for step in range(num_steps):
            # Heuristic action: 3 (patch) + 8 (agents) = 11 dims
            action = np.zeros(3 + 4 * 2)
            
            # === PATCH ACTION (3 dims: a, b, theta) ===
            # Heuristic: moderate size, no rotation
            # Note: center is computed automatically from team centroid!
            corridor_width = obs[0] * 10.0  # Unnormalize corridor width
            
            # Adapt size to corridor (compress in narrow, expand in wide)
            if corridor_width < 4.0:
                size = 0.2  # Smaller (maps to ~1.4m)
            else:
                size = 0.5  # Moderate (maps to ~2.75m)
            
            action[0] = size  # a (semi-major axis)
            action[1] = size  # b (semi-minor axis)
            action[2] = 0.0   # theta (no rotation)
            
            # === AGENT ACTIONS (8 dims: 2 per agent) ===
            for i in range(4):
                # Simple heuristic: accelerate forward, small random steering
                action[3 + i*2] = 0.3      # Accelerate (moderate)
                action[3 + i*2 + 1] = np.random.uniform(-0.05, 0.05)  # Tiny steering
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 100 == 0:
                print(f"Step {step}: Reward = {total_reward:.1f}, "
                      f"Ellipsoid: a={env.ellipsoid.a:.1f}, b={env.ellipsoid.b:.1f}")
            
            if terminated or truncated:
                print(f"\n🏁 Episode ended at step {step}")
                print(f"   Total reward: {total_reward:.1f}")
                obs, info = env.reset()
                total_reward = 0
                
    except KeyboardInterrupt:
        print("\n⚠️ Demo stopped by user")
    
    finally:
        env.close()


# ============================================================================
# 5. MAIN
# ============================================================================

def list_checkpoints(checkpoint_dir="checkpoints"):
    """List all available checkpoints."""
    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoint directory found: {checkpoint_dir}")
        return
    
    print("\n" + "=" * 70)
    print("📂 AVAILABLE CHECKPOINTS")
    print("=" * 70)
    
    runs = sorted([d for d in os.listdir(checkpoint_dir) 
                   if os.path.isdir(os.path.join(checkpoint_dir, d))])
    
    if not runs:
        print("No checkpoints found.")
        return
    
    for run in runs:
        run_path = os.path.join(checkpoint_dir, run)
        checkpoints = sorted([f for f in os.listdir(run_path) if f.endswith('.zip')])
        
        print(f"\n📁 {run}/")
        for cp in checkpoints:
            cp_path = os.path.join(run_path, cp.replace('.zip', ''))
            metadata_path = cp_path + "_metadata.json"
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                print(f"   └─ {cp.replace('.zip', '')}")
                print(f"      Steps: {meta.get('timesteps', '?')}, "
                      f"Episodes: {meta.get('episodes', '?')}, "
                      f"Best Reward: {meta.get('best_mean_reward', '?'):.1f}")
            else:
                print(f"   └─ {cp.replace('.zip', '')}")
    
    print("\n" + "=" * 70)
    print("To resume training:")
    print("  python3 dual_policy_ellipsoid.py --mode train --resume <checkpoint_path>")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dual Policy Ellipsoid System with Checkpointing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo (no training)
  python3 dual_policy_ellipsoid.py --mode demo
  
  # Train with default settings
  python3 dual_policy_ellipsoid.py --mode train
  
  # Train with custom timesteps and checkpoint frequency
  python3 dual_policy_ellipsoid.py --mode train --timesteps 50000 --checkpoint-freq 2500
  
  # Resume training from checkpoint
  python3 dual_policy_ellipsoid.py --mode train --resume checkpoints/run_xxx/checkpoint_10000
  
  # List all checkpoints
  python3 dual_policy_ellipsoid.py --mode list
  
  # Evaluate trained model
  python3 dual_policy_ellipsoid.py --mode eval --model dual_policy_model
        """
    )
    parser.add_argument("--mode", type=str, default="demo",
                       choices=["demo", "train", "eval", "list"],
                       help="Mode: demo, train, eval, or list (checkpoints)")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Training timesteps (default: 100000)")
    parser.add_argument("--model", type=str, default="dual_policy_model",
                       help="Model save/load path (default: dual_policy_model)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--checkpoint-freq", type=int, default=5000,
                       help="Save checkpoint every N timesteps (default: 5000)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory for checkpoints (default: checkpoints)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🎯 DUAL POLICY ELLIPSOID SYSTEM")
    print("   Hierarchical RL for Multi-Agent Coordination")
    print("   WITH AUTOMATIC CHECKPOINTING")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    if args.mode == "train":
        print(f"Checkpoint frequency: Every {args.checkpoint_freq} steps")
        print(f"Checkpoint directory: {args.checkpoint_dir}/")
        if args.resume:
            print(f"Resuming from: {args.resume}")
    print("=" * 70 + "\n")
    
    if args.mode == "demo":
        run_demo(num_steps=2000)
        
    elif args.mode == "train":
        if not SB3_AVAILABLE:
            print("ERROR: stable-baselines3 required for training!")
            print("Install: pip install stable-baselines3")
        else:
            train_dual_policy(
                total_timesteps=args.timesteps,
                num_agents=4,
                team_goal=[50.0, 0.0],
                save_path=args.model,
                checkpoint_freq=args.checkpoint_freq,
                checkpoint_dir=args.checkpoint_dir,
                resume_from=args.resume
            )
            
    elif args.mode == "eval":
        if not SB3_AVAILABLE:
            print("ERROR: stable-baselines3 required for evaluation!")
            print("Install: pip install stable-baselines3")
        else:
            evaluate_dual_policy(
                model_path=args.model,
                num_episodes=args.episodes
            )
    
    elif args.mode == "list":
        list_checkpoints(checkpoint_dir=args.checkpoint_dir)

