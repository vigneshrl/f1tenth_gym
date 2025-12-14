# Quick Start: Hybrid SE-MPC + RL System

## 🎯 What You Have

Three new files implementing the **"Robots Flow Like Fluid"** architecture:

1. **`se_mpc_rl_hybrid.py`** - Main hybrid system (Patch + SE-MPC)
2. **`train_patch_rl.py`** - RL training wrapper
3. **`HYBRID_ARCHITECTURE.md`** - Detailed documentation

## 🚀 Run It Now (No Training)

### Option 1: Test Hybrid System with Heuristic Patch

```bash
cd /home/mrvik/f1tenth_gym/examples
python3 se_mpc_rl_hybrid.py
```

**What happens:**
- 2 robots spawn in F1TENTH Spielberg track
- Patch generates flow field (heuristic, not learned yet)
- Each robot runs SE-MPC using:
  - LIDAR for local obstacles
  - Flow guidance from patch
  - Formation constraints
- Watch them navigate!

**Expected behavior:**
- Robots move towards goal (x=50)
- Maintain rough formation
- Avoid obstacles from LIDAR
- Print logs every 100 steps

---

## 📚 Understand the Architecture

Read `HYBRID_ARCHITECTURE.md` for:
- Complete system diagram
- Data flow explanation
- What RL learns
- Key differences from old code

**TL;DR:**
- **Patch (RL)**: Centralized coordinator → outputs flow field + topology
- **SE-MPC (Fixed)**: Decentralized executor → uses LIDAR + follows flow

---

## 🤖 Train the RL Policy

### Step 1: Install Dependencies

```bash
pip install stable-baselines3 torch
```

### Step 2: Prepare Training Script

Open `train_patch_rl.py` and uncomment the PPO training code (lines ~80-100).

### Step 3: Train

```bash
python3 train_patch_rl.py --train --timesteps 100000
```

**Training parameters:**
- Algorithm: PPO (Proximal Policy Optimization)
- State: Agent positions, velocities, corridor width
- Action: Flow parameters, formation distance, topology
- Reward: Speed + progress - collisions

**Training time:** ~1-2 hours for 100k steps (depends on hardware)

### Step 4: Monitor Training

```bash
tensorboard --logdir=./patch_rl_logs/
```

Open browser: http://localhost:6006

Watch:
- Episode reward (should increase)
- Episode length (should vary)
- Policy loss, value loss

---

## 🔬 Test Trained Policy

```bash
python3 train_patch_rl.py --test --model patch_policy
```

**Compare:**
- Heuristic policy: Fixed parameters, no adaptation
- Trained policy: Adapts to corridor width, obstacles, formation

---

## ⚙️ Customize

### Change Number of Robots

In `se_mpc_rl_hybrid.py`:
```python
num_agents = 4  # Change from 2 to 4
```

### Change Track

```python
config={
    "map": "Monza",  # Or: Austin, Spa, Silverstone, etc.
    ...
}
```

### Tune Reward Weights

In `train_patch_rl.py`, `_compute_reward()`:
```python
reward = (
    5.0 * avg_speed +      # Increase for more speed focus
    10.0 * progress +      # Increase for goal focus
    2.0 * formation +      # Increase for tighter formation
    collision_penalty
)
```

### Adjust MPC Parameters

In `se_mpc_rl_hybrid.py`, `setup_se_mpc_with_flow()`:
```python
T_p = 2.0   # Prediction horizon (increase for smoother)
N = 10      # MPC steps (increase for better optimization)
W_flow = 50.0  # Flow tracking weight (increase for better following)
```

---

## 🐛 Troubleshooting

### Issue: IPOPT fails to solve

**Cause:** Too many obstacles, infeasible problem

**Fix:**
- Reduce `sample_beams` in `process_lidar_scan()` (fewer constraints)
- Increase `acceptable_tol` in solver options
- Check if obstacles are too close to start position

### Issue: Robots don't move

**Cause:** MPC returning None, fallback control not working

**Fix:**
- Check initial velocity is non-zero
- Verify flow field is not zero
- Print `v_desired` to debug

### Issue: Training reward not increasing

**Cause:** Bad reward design, or policy not learning

**Fix:**
- Check reward components are balanced (print them)
- Increase training timesteps
- Try different RL algorithm (SAC instead of PPO)
- Reduce action space complexity (fewer parameters)

### Issue: "ModuleNotFoundError: stable_baselines3"

**Fix:**
```bash
pip install stable-baselines3
```

---

## 📊 What to Expect

### **Without Training (Heuristic):**
- Robots move forward
- Some obstacle avoidance (from SE-MPC)
- Rigid formation
- May not adapt well to environment changes

### **With Training (100k steps):**
- Better speed control
- Adaptive formation (wider in open, tighter in narrow)
- Smoother flow field
- Higher success rate

### **With More Training (1M steps):**
- Near-optimal flow patterns
- Dynamic split/merge
- Anticipatory slow-down before obstacles
- Emergent coordination behaviors

---

## 📈 Key Metrics to Track

1. **Episode Reward:** Should increase over training
2. **Average Speed:** Should approach max while staying safe
3. **Collision Rate:** Should decrease to near-zero
4. **Success Rate:** % of episodes reaching goal
5. **Formation Error:** Distance from desired formation

---

## 🎓 Next Steps

1. **Run heuristic version** → Understand baseline
2. **Read architecture doc** → Understand system
3. **Train for 10k steps** → See if reward increases
4. **Train for 100k steps** → Get decent policy
5. **Visualize results** → Compare heuristic vs learned
6. **Experiment** → Try different rewards, parameters

---

## 💡 Pro Tips

- Start with 2 agents (simpler), then scale to 4+
- Use short training runs (10k steps) to debug rewards
- Log everything (positions, speeds, formation errors)
- Visualize flow field (add rendering of desired velocities)
- Compare multiple RL algorithms (PPO, SAC, TD3)

---

## 🌟 Success Criteria

**Your policy is working well when:**
- ✅ Robots reach goal >80% of episodes
- ✅ Zero collisions with walls
- ✅ Average speed >3 m/s
- ✅ Formation maintained (error <0.5m)
- ✅ Adapts to different corridor widths

---

## 📞 Need Help?

**Check:**
1. `HYBRID_ARCHITECTURE.md` - Detailed system docs
2. Code comments - Inline explanations
3. Print debug info - Add print statements
4. Visualize - Render environment, plot trajectories

**Common Questions:**
- "Why no global obstacles?" → SE-MPC uses LIDAR only (decentralized)
- "What does patch do?" → Generates flow field (where to go)
- "What does SE-MPC do?" → Executes locally (how to get there safely)
- "What does RL learn?" → Optimal patch parameters (when to split/merge, how fast)

---

**Ready to train!** 🚀🤖🌊

