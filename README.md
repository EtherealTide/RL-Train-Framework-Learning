# RL Train Framework Learning

A systematic, 4-week journey exploring modern Reinforcement Learning training frameworks and distributed RL infrastructure: from single-device algorithm fundamentals to distributed runtimes (Ray), embodied robotics frameworks (UniLab), and LLM post-training infrastructures (TRL / verl / OpenRLHF).

---

## 🎯 Core Engineering Objectives

The goal is not merely calling pre-packaged APIs, but mastering the architectural and infrastructure trade-offs in modern RL systems:
- **Decoupled System Architecture**: Decoupling rollout generation, trajectory buffering, and learner gradient updates.
- **Distributed Runtimes**: Resource allocation, actor-learner topologies, placement groups, and object store semantics via Ray.
- **Bottlenecks & Optimization**: Identifying rollout-bound vs. learner-bound regimes, communication/weight synchronization latency, and CUDA Graph stability across dynamic batching and distributed ranks.

---

## 🗺️ Roadmap & Framework Stack

| Level | Target Framework / Module | Focus & Core Takeaways |
| :--- | :--- | :--- |
| **01. Reference Algorithmic Core** | CleanRL / Native PyTorch | GAE, PPO/SAC training loops, continuous/discrete distributions |
| **02. Classical Frameworks** | Stable-Baselines3 | Vectorized environments (`VecEnv`, `SubprocVecEnv`), Rollout/Replay buffers |
| **03. Distributed Runtime** | Ray Core | Tasks, Actors, Object Store (`ray.put`/`get`), Placement Groups |
| **04. Distributed RL System** | Ray RLlib | `EnvRunner`, `LearnerGroup`, multi-GPU/multi-node scaling |
| **05. Robotics / Embodied RL** | UniLab / unilab-rl | FastSAC, shared-memory transitions, heterogeneous physics-policy pipeline |
| **06. Post-training & LLM RL** | TRL / verl / OpenRLHF | Multi-worker dataflow graphs, vLLM co-location, large-scale RL infra |

---

## 📅 Progress Tracker

### Week 1: Algorithmic Core & Execution Flow
- **Day 01: Gymnasium & Minimal RL Pipeline**
  - Implemented 1D navigation environment (`LineWorldEnv`) adhering to Gymnasium API standards.
  - Clarified semantics of `terminated` (environment goal/hazard) vs. `truncated` (horizon timeout / bootstrap required).
  - Built trajectory collection loop and reversed discounted return ($G_t$) calculation.
- **Day 02: Native REINFORCE (Policy Gradient)**
  - Implemented a PyTorch policy network for discrete action spaces.
  - Sampled actions using torch.distributions.Categorical.
  - Preserved trajectory log-probabilities for autograd.
  - Implemented Monte Carlo REINFORCE policy loss.
  - Compared policy probabilities before and after training.
  - Evaluated deterministic success rate after policy optimization.

---

## 🛠️ Repository Structure

```text
.
├── .gitignore
├── README.md
└── Gymnasium-MiniRL/
    ├── minimal_env.py      # Custom Gymnasium environment implementation
    └── reinforce.py         # Minimal REINFORCE algorithm implementation