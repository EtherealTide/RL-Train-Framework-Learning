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
- **Day 04: Proximal Policy Optimization (PPO-Clip)**
  - Implemented PPO-Clip with a shared Actor-Critic network.
  - Collected fixed-length rollouts of exactly 1,024 environment steps,
    preserving environment state and unfinished episode returns between rollouts.
  - Recorded behavior-policy action log-probabilities and value estimates
    during sampling under `torch.no_grad()`.
  - Computed GAE before shuffling, with separate masks for value
    bootstrapping and advantage continuation across episode boundaries.
  - Bootstrapped time-limit truncations and unfinished rollout boundaries;
    used zero bootstrap for true terminations.
  - Reused the next step's stored value within uninterrupted trajectories,
    avoiding redundant value-network evaluations.
  - Trained for four epochs per rollout, reshuffling samples each epoch
    into mini-batches of 256 transitions: 16 optimizer steps per rollout.
  - Used mini-batch advantage normalization, entropy regularization,
    Huber value loss, gradient clipping, and linear learning-rate decay.
  - Logged training returns, success rates, losses, policy entropy,
    clipping fraction, and pre-normalization advantage standard deviation.
  - Evaluated greedy actions over 50 episodes in a separate CartPole-200
    environment.

  **Experiment observations**
  - Switching from full-batch to mini-batch updates increased optimizer
    steps per rollout from 4 to 16.
  - Retaining the initial learning rate of `1e-3` produced repeated
    performance regressions, including zero evaluation success in the
    final logged evaluations.
  - Reducing the initial learning rate to `3e-4`, while retaining
    `gamma=0.99` and linear decay, improved late-training stability.
  - In the final seed-42 run, all 20 logged evaluations from iteration
    110 through 300 achieved `100%` success and a mean return of `200/200`.
  - The run used exactly 307,200 environment steps. Intermediate
    regressions still occurred, including at iterations 90–100.
  - Logged mean pre-normalization advantage standard deviations remained
    above 1; this run provided no evidence of near-zero advantage variance
    being amplified by normalization.
  - Results apply to this training seed and the 200-step environment limit;
    robustness across training seeds has not yet been established.

  **Architecture takeaway**
  - Old log-probabilities must correspond to the policy that generated
    each action. Recording them during collection avoids reconstructing
    behavior-policy probabilities with potentially newer learner weights.
  - The implementation remains synchronous and single-environment.
    Distributed execution will additionally require policy-version
    tracking and synchronization.

- **Day 05: Soft Actor-Critic (SAC)**
  - Implemented continuous-action SAC from scratch in PyTorch on `Pendulum-v1`.
  - Built a circular replay buffer and sampled random mini-batches for
    off-policy learning.
  - Implemented a squashed Gaussian policy with reparameterized sampling,
    action rescaling, and log-probability Jacobian correction.
  - Implemented independent twin Q networks, target Q networks,
    entropy-regularized Bellman targets, and Polyak soft updates.
  - Optimized the actor through differentiable sampled actions and learned
    the entropy temperature automatically.
  - Preserved bootstrapping for time-limit truncations and disabled it
    only for true terminations.
  - Trained for 100,000 environment steps with seed 42, batch size 256,
    `gamma=0.99`, `tau=0.005`, and learning rates of `3e-4`.

  **Evaluation protocol**
  - Evaluated deterministic actions every 5,000 training steps over
    50 episodes, using the same episode seeds (10042–10091) at each checkpoint.
  - Computed deterministic actions directly from the policy mean,
    avoiding random sampling during evaluation.
  - Defined a custom success criterion: all final 50 post-action states
    must satisfy an absolute angle error below 10 degrees and an absolute
    angular velocity below 0.5 rad/s.
  - Logged mean episode return and its standard deviation, success rate,
    whole-episode upright ratio, and mean reward over the final 50 steps.
  - The success criterion is an evaluation metric, not an environment
    termination condition.

  **Experiment observations**

  | Training steps | Evaluation return (mean ± std) | Success rate | Upright ratio | Final-50-step mean reward |
  | ---: | ---: | ---: | ---: | ---: |
  | 5,000 | -125.35 ± 88.27 | 100% | 84.8% | -0.0014 |
  | 30,000 | -118.62 ± 69.10 | 0% | 28.5% | -0.0520 |
  | 35,000 | -110.97 ± 67.76 | 100% | 83.9% | approximately 0 |
  | 100,000 | -112.38 ± 67.31 | 100% | 79.2% | -0.0029 |

  - All 14 logged evaluations from 35,000 through 100,000 steps achieved
    100% success under the custom stability criterion.
  - A temporary stability regression occurred at 30,000 steps despite
    a relatively small change in total return, showing why return alone
    does not fully describe balancing performance.
  - At the final checkpoint, every evaluation episode met the stability
    criterion, and the final-window reward was close to zero. Negative
    total returns therefore remained compatible with successful swing-up
    and balancing.
  - Return standard deviations describe variation across evaluation
    episodes, not across independent training runs.
  - Results apply to one training seed and a fixed evaluation set;
    robustness across training seeds and unseen initial states remains
    untested.

  **Architecture takeaway**
  - SAC separates environment interaction from learning through a replay
    buffer and reuses previously collected transitions.
  - Critic targets, actor updates, entropy-temperature updates, and target
    network updates have distinct gradient paths.
  - The implementation remains synchronous and single-environment,
    providing a baseline for later actor-learner separation.
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
    ├── ppo.py                          # Fixed-length PPO rollouts and shuffled mini-batch updates
    ├── sac.py                          # Off-policy SAC and Pendulum stability evaluation
    └── DAY3_IMPROVEMENT_REPORT.md      # Detailed Day 3 experiment record
```
