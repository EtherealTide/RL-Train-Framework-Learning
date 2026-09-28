# Day 3 改进实验报告：GAE Actor-Critic on CartPole

## 1. 实验目标

Day 3 的基础任务是在 `LineWorldEnv` 上实现共享网络的 Actor-Critic 和
GAE。基础版本修正 GAE 公式后已经可以正常训练。

改进实验希望回答三个问题：

1. GAE Actor-Critic 能否解决比 LineWorld 更复杂的环境？
2. actor 和 critic 共享网络时，怎样减少 critic 对策略学习的干扰？
3. 怎样让已经学会的策略在后期保持稳定？

最终选择的环境是 CartPole。训练使用 200 步上限，因此本文称它为
CartPole-200。

## 2. 基础版本的问题

### 2.1 最初的 GAE 公式错误

最初代码中的 TD residual 是：

```python
delta_t = reward_t + gamma * next_value
```

正确公式应该是：

```python
delta_t = reward_t + gamma * next_value - current_value
```

也就是原公式漏掉了 `-V(s_t)`。

### 2.2 错误公式造成的现象

value 越大，下一次计算出的 advantage 就越大。增大的 advantage 又会生成更大的
value target，形成不断放大的循环。

原始日志中的代表性数据如下：

| Episode | 回报 | Critic loss | Advantage mean | Value mean | 梯度范数 |
|---:|---:|---:|---:|---:|---:|
| 90 | 18 | 10,051.64 | 88.94 | 28.26 | 506.96 |
| 130 | -50 | 120,578.69 | -333.98 | -40.79 | 1,910.42 |
| 200 | -50 | 283,824.19 | -512.40 | -63.39 | 2,943.40 |
| 240 | -48 | 405,926.59 | -612.75 | -76.06 | 3,519.45 |
| 300 | 16 | 61,382.48 | 211.81 | 96.25 | 1,216.75 |

环境中的真实 episode 回报只有几十，但 critic loss 已达到几十万，value 和
advantage 也持续增大。这说明 critic 学习的目标本身有问题。

修正公式后，LineWorld 可以正常训练。因此，基础 Day 3 的核心结论是：问题不在
GAE 方法本身，而在 TD residual 的实现。

## 3. 改进实验环境

### 3.1 为什么选择 CartPole

LineWorld 只有一个位置状态，最优策略基本上是一直向右。它适合检查算法是否写对，
但不适合检查训练框架是否稳定。

CartPole 的 observation 包含四个连续变量：

```text
cart position
cart velocity
pole angle
pole angular velocity
```

动作仍然是两个离散动作：向左推或向右推。因此原来的 `Categorical` actor 可以
继续使用，同时环境比 LineWorld 更复杂。

### 3.2 终止信号的区别

两个环境对结束信号的解释不同：

| 环境 | `terminated=True` | `truncated=True` |
|---|---|---|
| LineWorld | 到达目标，成功 | 超过步数限制，失败 |
| CartPole | 杆倒下或小车越界，失败 | 坚持到时间上限，成功 |

因此 CartPole 的成功判断是：

```python
success = truncated and not terminated
```

如果是 `terminated`，bootstrap value 为 0。如果只是时间限制导致的
`truncated`，则使用最后状态的 `V(s)` 进行 bootstrap。

未训练网络的测试结果为：

```text
success_rate = 0.0
avg_reward   = 14.6
avg_steps    = 14.6
```

这可以作为随机初始策略的参考值。

## 4. 改进一：调整共享网络结构

### 4.1 修改意图

最初的 actor 和 critic 直接使用同一份隐藏特征。actor 学习动作，critic 学习数值
回归，这两个任务的梯度可能互相干扰。

改进后的结构为：

```text
observation
     |
shared backbone
     |
     +-- actor tower  -- actor head  -- action logits
     |
     +-- critic tower -- critic head -- state value
```

第一层仍然共享，满足共享 backbone 的实验要求。actor 和 critic 后面各自拥有一个
私有隐藏层，可以学习不同的高级特征。

### 4.2 Shape 测试

使用 `hidden_dim=64` 做结构测试时，输入 8 条四维 observation，输出结果为：

```text
logits: torch.Size([8, 2])
values: torch.Size([8])
parameters: 8835
```

shape 符合预期。正式训练为了保持模型较小，使用的是 `hidden_dim=32`。

## 5. 改进二：正交初始化

### 5.1 修改意图

如果初始 actor 明显偏向某一个动作，采样数据也会产生偏差。隐藏层使用正交初始化，
actor 输出层使用较小的 `0.01` gain，critic 输出层使用 `1.0` gain。

所有 bias 初始化为 0。

### 5.2 初始化测试

零 observation 的输出为：

```text
logits:        [[0.0, 0.0]]
probabilities: [[0.5, 0.5]]
entropy:       0.6931
value:         0.0
```

两个动作的初始概率相同，entropy 等于二分类分布的最大值 `ln(2)`。初始化符合预期。

## 6. 改进三：分离 rollout 和梯度计算

### 6.1 修改意图

原代码在采样时保存带计算图的 `log_prob`、`value` 和 `entropy`。整条 episode 的
计算图会一直保留到更新阶段，不方便合并多条 trajectory。

改进后：

- rollout 只保存 observation、action、reward 和结束标志；
- rollout 使用 `torch.no_grad()`；
- 更新时把 observation batch 再送入网络；
- 在更新阶段统一计算 log probability、value 和 entropy。

### 6.2 单 episode 测试

```text
episode length = 55
total loss     = 39.1025
actor loss     = 0.00005
critic loss    = 78.2188
entropy        = 0.6931
advantage std  = 2.3441
gradient norm  = 18.3201
```

数据流可以正常工作，但 critic loss 和梯度仍然很大。

## 7. 改进四：多 episode batch

### 7.1 修改意图

单条 CartPole episode 较短，而且相邻状态非常相似。如果只在一条 episode 内标准化
advantage，样本之间缺少足够差异。

改进方法是先收集多条 episode：

1. 每条 episode 独立计算 GAE，避免递推跨越 episode 边界；
2. 将所有 episode 的 advantage 拼接起来；
3. 在完整 batch 上统一标准化；
4. 使用完整 batch 做一次梯度更新。

### 7.2 四条 episode 合并测试

```text
episode lengths:  [14, 19, 16, 9]
total transitions: 58
critic loss:       33.2098
gradient norm:     10.4236
```

与单 episode 测试相比：

```text
critic loss:   78.22 -> 33.21
gradient norm: 18.32 -> 10.42
```

批量数据降低了单条 trajectory 带来的波动，但 critic 梯度仍然偏大。

## 8. 改进五：Huber critic loss

### 8.1 MSE 阶段的数据

最初使用 MSE critic loss、每批 8 条 episode：

| Episode | Train return | Critic loss | Gradient norm | Eval return | Eval success |
|---:|---:|---:|---:|---:|---:|
| 104 | 22.69 | 56.09 | 13.41 | - | - |
| 200 | 22.02 | 48.28 | 13.13 | 105.88 | 0.02 |
| 400 | 25.18 | 60.79 | 14.95 | 110.03 | 0.01 |
| 600 | 32.03 | 63.80 | 16.02 | 110.93 | 0.04 |

MSE 会平方放大 value error。虽然设置了梯度裁剪，但裁剪只能限制大小，不能改变更新
方向主要由 critic 决定的问题。

### 8.2 修改方法

最终代码使用：

```python
critic_loss = F.smooth_l1_loss(
    values,
    returns.detach(),
    beta=0.5,
)
```

同时把：

```text
value_coef: 0.5 -> 0.1
```

Huber loss 在误差较大时按线性方式增长，因此异常 value target 不会产生平方级梯度。

### 8.3 修改后的数据

| Episode | Train return | Critic loss | Gradient norm | Eval return | Eval success |
|---:|---:|---:|---:|---:|---:|
| 104 | 22.69 | 6.75 | 0.29 | - | - |
| 200 | 22.02 | 6.20 | 0.23 | 106.90 | 0.02 |
| 400 | 25.80 | 7.44 | 0.20 | 116.06 | 0.05 |
| 600 | 28.25 | 7.03 | 0.22 | 114.73 | 0.09 |
| 800 | 32.88 | 6.72 | 0.23 | 102.03 | 0.05 |

结果说明：

- critic loss 从几十下降到约 7；
- 裁剪前梯度从约 13～16 下降到约 0.2～0.4；
- value 更新不再控制整个共享网络；
- 但训练速度仍然较慢，需要调整 actor 的有效更新速度。

## 9. 改进六：提高 actor 的有效学习速度

### 9.1 修改意图

每批 8 条 episode 时，800 episode 只有 100 次参数更新。与此同时，entropy 系数
`0.01` 会让策略长时间保持接近随机。

修改如下：

```text
episodes_per_update: 8     -> 4
learning_rate:       3e-4  -> 1e-3
entropy_coef:        0.01  -> 0.001
```

每批仍然包含多条 episode，但相同 episode 数下的更新次数增加了一倍。

### 9.2 实验数据

| Episode | Train return | Train success | Entropy | Eval return | Eval success |
|---:|---:|---:|---:|---:|---:|
| 200 | 34.78 | 0.00 | 0.6356 | 136.73 | 0.25 |
| 400 | 115.62 | 0.11 | 0.5944 | 200.00 | 1.00 |
| 600 | 173.26 | 0.42 | 0.5790 | 200.00 | 1.00 |
| 800 | 194.63 | 0.81 | 0.5625 | 200.00 | 1.00 |

与之前相比，评估回报从约 100～116 提升到满分 200。entropy 缓慢下降，但没有
接近 0，说明策略在学习明确动作的同时仍保留一定探索。

## 10. 发现新问题：后期策略退化

使用固定 `1e-3` 学习率继续训练时，策略出现明显波动：

| Episode | Train return | Eval return | Eval success |
|---:|---:|---:|---:|
| 800 | 194.63 | 200.00 | 1.00 |
| 1000 | 127.70 | 143.74 | 0.00 |
| 1200 | 133.79 | 144.46 | 0.00 |
| 1400 | 184.43 | 199.18 | 0.88 |
| 1600 | 189.06 | 192.13 | 0.45 |
| 1800 | 170.74 | 183.19 | 0.27 |
| 2000 | 200.00 | 200.00 | 1.00 |

最终结果虽然回到满分，但训练过程并不稳定。已经学会的策略仍然用相同学习率更新，
少量 advantage 噪声可能产生过大的参数变化。

## 11. 改进七：线性学习率衰减

### 11.1 修改意图

前期需要较大的学习率快速学习，后期需要较小的学习率保护已经学好的策略。因此使用：

```python
progress = episodes_seen / num_episodes
current_lr = initial_lr * (1.0 - progress)
```

初始学习率是 `1e-3`，训练结束时接近 0。

### 11.2 最终实验数据

| Episode | Learning rate | Train return | Train success | Critic loss | Entropy | Eval return | Eval success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 800 | 0.000602 | 179.93 | 0.62 | 5.12 | 0.5592 | 200.00 | 1.00 |
| 1000 | 0.000502 | 191.43 | 0.79 | 4.25 | 0.5567 | 200.00 | 1.00 |
| 1200 | 0.000402 | 189.54 | 0.70 | 3.58 | 0.5570 | 200.00 | 1.00 |
| 1400 | 0.000302 | 189.43 | 0.75 | 3.11 | 0.5654 | 200.00 | 1.00 |
| 1600 | 0.000202 | 190.22 | 0.76 | 2.84 | 0.5476 | 200.00 | 1.00 |
| 1800 | 0.000102 | 189.04 | 0.70 | 2.67 | 0.5492 | 200.00 | 1.00 |
| 2000 | 0.000002 | 190.12 | 0.71 | 2.66 | 0.5473 | 200.00 | 1.00 |

加入学习率衰减后，从 episode 800 到 episode 2000 的每次确定性评估都达到
`200/200`，没有再次出现成功率突然降到 0 的情况。

## 12. 最终配置

| 参数 | 最终值 |
|---|---:|
| Environment | CartPole-v1，200 步上限 |
| Training episodes | 2000 |
| Hidden dimension | 32 |
| Episodes per update | 4 |
| Gamma | 0.95 |
| GAE lambda | 0.95 |
| Initial learning rate | 0.001 |
| Learning-rate schedule | Linear decay |
| Value coefficient | 0.1 |
| Entropy coefficient | 0.001 |
| Critic loss | Smooth L1，`beta=0.5` |
| Gradient clipping | 0.5 |
| Training seed | 42 |
| Evaluation episodes | 100 |
| Evaluation policy | Deterministic argmax |

## 13. 最终结论

最终版本可以稳定解决 CartPole-200：

```text
episodes 800～2000:
    evaluation average reward = 200.00
    evaluation success rate   = 1.00
```

训练策略的回报约为 190，而确定性评估为 200。这是因为训练阶段使用
`Categorical.sample()` 保留探索，评估阶段使用 `argmax`。

本实验得到的主要结论是：

1. GAE 的公式必须包含 `-V(s_t)`，否则 value target 会不断放大。
2. 共享 backbone 后加入 actor/critic 私有层，可以减少任务之间的直接冲突。
3. rollout 与梯度计算分离后，更容易实现批量训练。
4. 多 episode advantage normalization 比单 episode 更稳定。
5. Huber loss 和较低的 value coefficient 可以防止 critic 主导共享网络。
6. 合理的 batch 大小、学习率和 entropy 系数决定了策略学习速度。
7. 线性学习率衰减可以明显减少后期策略退化。

当前没有加入 PPO。实验已经证明普通 batched GAE Actor-Critic 足以解决
CartPole-200，因此没有必要为了复杂度而继续增加算法组件。

## 14. 实验限制和后续工作

本报告的最终完整数据来自 `seed=42`。它证明该配置在这个 seed 上可以稳定训练，但
还不能证明对所有随机种子都同样稳定。

后续可以继续进行：

1. 使用 seed 0、1、2 重复实验，报告均值和标准差；
2. 将按 episode 收集改为固定 transition 数量的 rollout buffer；
3. 在更长的 CartPole-500 上测试稳定性；
4. 如果长时间训练仍发生策略退化，再引入 PPO clipped objective。

## 15. 运行方法

```powershell
conda activate RL
cd Gymnasium-MiniRL
python reinforce.py
```

程序会每 100 个 episode 输出训练统计，每 200 个 episode使用 100 条 episode 做
确定性评估。
