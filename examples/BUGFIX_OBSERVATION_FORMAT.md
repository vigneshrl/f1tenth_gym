# Bug Fix: Observation Format Mismatch

## 🐛 Bug Report

**Error:**
```
TypeError: 'int' object is not subscriptable
at line: x = obs[agent_id]["poses_x"][i]
```

**Cause:**
The code was written for per-agent observation dictionaries, but the environment was configured with `"observation_config": {"type": "original"}` which returns a different format.

---

## 📊 Observation Format Comparison

### Type: "original" (What was configured)
```python
obs = {
    'ego_idx': int,
    'scans': np.array(shape=(num_agents, 1080)),      # All agents
    'poses_x': np.array(shape=(num_agents,)),         # All agents
    'poses_y': np.array(shape=(num_agents,)),         # All agents
    'poses_theta': np.array(shape=(num_agents,)),
    'linear_vels_x': np.array(shape=(num_agents,)),
    'linear_vels_y': np.array(shape=(num_agents,)),
    'ang_vels_z': np.array(shape=(num_agents,)),
    'collisions': np.array(shape=(num_agents,)),
    ...
}

# Access: obs["poses_x"][i] for agent i
```

### Type: "features" or "kinematic_state" (What code expected)
```python
obs = {
    'agent_0': {
        'scan': np.array(1080,),
        'pose_x': float,
        'pose_y': float,
        'pose_theta': float,
        'linear_vel_x': float,
        'linear_vel_y': float,
        ...
    },
    'agent_1': {
        'scan': np.array(1080,),
        'pose_x': float,
        ...
    }
}

# Access: obs[agent_id]["pose_x"]
```

---

## ✅ Fix Applied

### Before (Incorrect):
```python
agent_ids = list(obs.keys())  # Returns ['ego_idx', 'scans', 'poses_x', ...]
for i, agent_id in enumerate(agent_ids):
    x = obs[agent_id]["poses_x"][i]  # ERROR! agent_id is not a dict key
```

### After (Correct):
```python
for i in range(num_agents):
    x = obs["poses_x"][i]  # Directly access array by index
    y = obs["poses_y"][i]
    scan = obs["scans"][i]
```

---

## 🔧 Changes Made

### Line ~612-631: Team State Gathering
```python
# OLD:
agent_ids = list(obs.keys())
for i, agent_id in enumerate(agent_ids):
    x = obs[agent_id]["poses_x"][i]

# NEW:
for i in range(num_agents):
    x = obs["poses_x"][i]
```

### Line ~653-667: Agent Loop
```python
# OLD:
for i, agent_id in enumerate(agent_ids):
    scan = obs[agent_id]["scans"][i]

# NEW:
for i in range(num_agents):
    scan = obs["scans"][i]
```

### Line ~710-713: Collision Check
```python
# OLD:
for i, agent_id in enumerate(agent_ids):
    if obs[agent_id]["collisions"][i] > 0.5:

# NEW:
for i in range(num_agents):
    if obs["collisions"][i] > 0.5:
```

---

## 🎯 Testing

After fix, the code should:
- ✅ Start without errors
- ✅ Access LIDAR scans correctly
- ✅ Read agent positions/velocities
- ✅ Process observations for SE-MPC

**Test command:**
```bash
python3 examples/se_mpc_rl_hybrid.py
```

**Expected output:**
```
============================================================
HYBRID SE-MPC + RL SYSTEM
...
Starting hybrid control loop...
Num agents: 2
Team goal: [50.0, 0.0]
Step 100: time=X.Xs, reward=Y.Y, avg_speed=Z.Zm/s
  Agent 0: pos=(...), goal_dist=...m, width=...m
  Agent 1: pos=(...), goal_dist=...m, width=...m
...
```

---

## 📝 Alternative: Use "features" Observation Type

If you prefer per-agent dictionaries, change the config:

```python
config={
    "observation_config": {"type": "features"},  # Per-agent dicts
    ...
}
```

Then revert to the original code structure. However, **"original" type is fine** now that the fix is applied!

---

## ✅ Status: FIXED

The code now correctly handles the "original" observation format from f1tenth_gym.

