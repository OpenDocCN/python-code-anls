# `.\lucidrains\anymal-belief-state-encoder-decoder-pytorch\anymal_belief_state_encoder_decoder_pytorch\trainer.py`

```
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from collections import deque
from einops import rearrange

from anymal_belief_state_encoder_decoder_pytorch import Anymal

# 定义一个继承自Dataset的类，用于存储经验数据
class ExperienceDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))

# 创建一个DataLoader对象，用于批量加载数据
def create_dataloader(data, batch_size):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size = batch_size, drop_last = True)

# 定义一个继承自nn.Module的类，用于训练学生模型
class StudentTrainer(nn.Module):
    def __init__(
        self,
        *,
        anymal,
        env,
        epochs = 2,
        lr = 5e-4,
        max_timesteps = 10000,
        update_timesteps = 5000,
        minibatch_size = 16,
        truncate_tpbtt = 10
    ):
        super().__init__()
        self.env = env
        self.anymal = anymal
        self.optimizer = Adam(anymal.student.parameters(), lr = lr)
        self.epochs = epochs

        self.max_timesteps = max_timesteps
        self.update_timesteps = update_timesteps
        self.minibatch_size = minibatch_size
        self.truncate_tpbtt = truncate_tpbtt

        self.running_proprio, self.running_extero = anymal.get_observation_running_stats()

    # 从记忆中学习
    def learn_from_memories(
        self,
        memories,
        next_states,
        noise_strength = 0.
    ):
        device = next(self.parameters()).device

        # 从记忆中检索和准备数据进行训练

        states = []
        teacher_states = []
        hiddens = []
        dones = []

        for (state, teacher_state, hidden, done) in memories:
            states.append(state)
            teacher_states.append(teacher_state)
            hiddens.append(hidden)
            dones.append(torch.Tensor([done]))

        states = tuple(zip(*states))
        teacher_states = tuple(zip(*teacher_states))

        # 将值转换为torch张量

        to_torch_tensor = lambda t: torch.stack(t).to(device).detach()

        states = map(to_torch_tensor, states)
        teacher_states = map(to_torch_tensor, teacher_states)
        hiddens = to_torch_tensor(hiddens)
        dones = to_torch_tensor(dones)

        # 为策略阶段训练准备数据加载器

        dl = create_dataloader([*states, *teacher_states, hiddens, dones], self.minibatch_size)

        current_hiddens = self.anymal.student.get_gru_hiddens()
        current_hiddens = rearrange(current_hiddens, 'l d -> 1 l d')

        for _ in range(self.epochs):
            for ind, (proprio, extero, privileged, teacher_proprio, teacher_extero, episode_hiddens, done) in enumerate(dl):

                straight_through_hiddens = current_hiddens - current_hiddens.detach() + episode_hiddens

                loss, current_hiddens = self.anymal(
                    proprio,
                    extero,
                    privileged,
                    teacher_states = (teacher_proprio, teacher_extero),
                    hiddens = straight_through_hiddens,
                    noise_strength = noise_strength
                )

                loss.backward(retain_graph = True)

                tbptt_limit = not ((ind + 1) % self.truncate_tpbtt)
                if tbptt_limit: # 控制梯度回传的时间跨度
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    current_hiddens = current_hiddens.detach()

                # 根据是否是新的一集，分离隐藏状态
                # 待办事项：重新构建数据加载器以每批行加载一个集

                maybe_detached_hiddens = []
                for current_hidden, done in zip(current_hiddens.unbind(dim = 0), dones.unbind(dim = 0)):
                    maybe_detached_hiddens.append(current_hidden.detach() if done else current_hidden)

                current_hiddens = torch.stack(maybe_detached_hiddens)

    # 前向传播函数
    def forward(
        self,
        noise_strength = 0.
    ):
        device = next(self.parameters()).device

        time = 0
        done = False
        states = self.env.reset()
        memories = deque([])

        hidden = self.anymal.student.get_gru_hiddens()
        hidden = rearrange(hidden, 'l d -> 1 l d')

        self.running_proprio.clear()
        self.running_extero.clear()

        for timestep in range(self.max_timesteps):
            time += 1

            states = list(map(lambda t: t.to(device), states))
            anymal_states = list(map(lambda t: rearrange(t, '... -> 1 ...'), states))

            # 教师需要有归一化的观测值

            (proprio, extero, privileged) = states

            self.running_proprio.push(proprio)
            self.running_extero.push(extero)

            teacher_states = (
                self.running_proprio.norm(proprio),
                self.running_extero.norm(extero)
            )

            teacher_anymal_states = list(map(lambda t: rearrange(t, '... -> 1 ...'), teacher_states))

            # 将状态添加到记忆中

            memories.append((
                states,
                teacher_states,
                rearrange(hidden, '1 ... -> ...'),
                done
            ))

            dist, hidden = self.anymal.forward_student(
                *anymal_states[:-1],
                hiddens = hidden,
                return_action_categorical_dist = True
            )

            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action = action.item()

            next_states, _, done, _ = self.env.step(action)

            states = next_states

            if time % self.update_timesteps == 0:
                self.learn_from_memories(memories, next_states, noise_strength = noise_strength)
                memories.clear()

            if done:
                break
```