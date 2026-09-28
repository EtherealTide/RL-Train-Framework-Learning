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
- **Day 03: Actor-Critic & Generalized Advantage Estimation**
  - **Basic task — LineWorld Actor-Critic:**
    - Replaced the standalone policy network with a shared Actor-Critic network.
    - Implemented the correct TD residual:
      `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`.
    - Implemented reverse-scan GAE and bootstrapped value targets.
    - Correctly separated Gymnasium `terminated` and `truncated` semantics.
    - Added advantage normalization, entropy regularization, gradient clipping,
      and training diagnostics.
  - **Improved experiment — CartPole-200:**
    - Moved from the one-dimensional LineWorld task to a four-dimensional
      CartPole control task.
    - Used a shared feature layer followed by private actor and critic towers.
    - Separated rollout collection from gradient computation.
    - Combined four episodes in each update and normalized advantages over the
      complete batch.
    - Replaced critic MSE with Huber loss and reduced the critic loss weight.
    - Used orthogonal initialization and linear learning-rate decay.
    - Reached a deterministic evaluation score of `200/200` for every
      evaluation from episode 800 through episode 2000 in the final seed-42 run.
    - See the full experiment record in
      [Day 3 Improvement Report](Gymnasium-MiniRL/DAY3_IMPROVEMENT_REPORT.md).

---

## 🛠️ Repository Structure

```text
.
├── .gitignore
├── README.md
├── requirements.txt        # Python dependencies for the project
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI workflow for linting and testing
└── Gymnasium-MiniRL/
    ├── minimal_env.py                  # LineWorld and CartPole environment setup
    ├── reinforce.py                    # Batched GAE Actor-Critic training loop
    ├── actor_critic_gae.py             # Reverse-scan GAE implementation
    └── DAY3_IMPROVEMENT_REPORT.md      # Detailed Day 3 experiment record
```
