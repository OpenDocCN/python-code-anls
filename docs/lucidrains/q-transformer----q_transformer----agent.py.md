# `.\lucidrains\q-transformer\q_transformer\agent.py`

```
# 导入必要的库
import sys
from pathlib import Path

# 导入 numpy 的相关模块
from numpy.lib.format import open_memmap

# 导入 torch 相关模块
import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset

# 导入 einops 库
from einops import rearrange

# 导入自定义的 QRoboticTransformer 类
from q_transformer.q_robotic_transformer import QRoboticTransformer

# 导入 torchtyping 库
from torchtyping import TensorType

# 导入 beartype 库
from beartype import beartype
from beartype.typing import Iterator, Tuple, Union

# 导入 tqdm 库
from tqdm import tqdm

# 确保在 64 位系统上进行训练
assert sys.maxsize > (2 ** 32), 'you need to be on 64 bit system to store > 2GB experience for your q-transformer agent'

# 定义常量
TEXT_EMBEDS_FILENAME = 'text_embeds.memmap.npy'
STATES_FILENAME = 'states.memmap.npy'
ACTIONS_FILENAME = 'actions.memmap.npy'
REWARDS_FILENAME = 'rewards.memmap.npy'
DONES_FILENAME = 'dones.memmap.npy'

DEFAULT_REPLAY_MEMORIES_FOLDER = './replay_memories_data'

# 定义辅助函数
def exists(v):
    return v is not None

def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

# 定义回放记忆数据集类
class ReplayMemoryDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder: str = DEFAULT_REPLAY_MEMORIES_FOLDER,
        num_timesteps: int = 1
    ):
        # 确保时间步数大于等于 1
        assert num_timesteps >= 1
        self.is_single_timestep = num_timesteps == 1
        self.num_timesteps = num_timesteps

        # 检查文件夹是否存在
        folder = Path(folder)
        assert folder.exists() and folder.is_dir()

        # 打开并读取相关文件
        text_embeds_path = folder / TEXT_EMBEDS_FILENAME
        states_path = folder / STATES_FILENAME
        actions_path = folder / ACTIONS_FILENAME
        rewards_path = folder / REWARDS_FILENAME
        dones_path = folder / DONES_FILENAME

        self.text_embeds = open_memmap(str(text_embeds_path), dtype='float32', mode='r')
        self.states = open_memmap(str(states_path), dtype='float32', mode='r')
        self.actions = open_memmap(str(actions_path), dtype='int', mode='r')
        self.rewards = open_memmap(str(rewards_path), dtype='float32', mode='r')
        self.dones = open_memmap(str(dones_path), dtype='bool', mode='r')

        self.num_timesteps = num_timesteps

        # 根据结束标志计算每个 episode 的长度
        self.episode_length = (self.dones.cumsum(axis=-1) == 0).sum(axis=-1) + 1

        # 过滤出长度足够的 episode
        trainable_episode_indices = self.episode_length >= num_timesteps

        self.text_embeds = self.text_embeds[trainable_episode_indices]
        self.states = self.states[trainable_episode_indices]
        self.actions = self.actions[trainable_episode_indices]
        self.rewards = self.rewards[trainable_episode_indices]
        self.dones = self.dones[trainable_episode_indices]

        self.episode_length = self.episode_length[trainable_episode_indices]

        # 确保存在可训练的 episode
        assert self.dones.size > 0, 'no trainable episodes'

        self.num_episodes, self.max_episode_len = self.dones.shape

        timestep_arange = torch.arange(self.max_episode_len)

        timestep_indices = torch.stack(torch.meshgrid(
            torch.arange(self.num_episodes),
            timestep_arange
        ), dim=-1)

        trainable_mask = timestep_arange < rearrange(torch.from_numpy(self.episode_length) - num_timesteps, 'e -> e 1')
        self.indices = timestep_indices[trainable_mask]

    # 返回数据集的长度
    def __len__(self):
        return self.indices.shape[0]
    # 重载索引操作符，根据索引获取数据
    def __getitem__(self, idx):
        # 从索引中获取当前 episode 和 timestep 的索引
        episode_index, timestep_index = self.indices[idx]

        # 创建一个切片对象，用于获取当前 timestep 到 num_timesteps 之间的数据
        timestep_slice = slice(timestep_index, (timestep_index + self.num_timesteps))

        # 复制当前 episode 的文本嵌入数据
        text_embeds = self.text_embeds[episode_index, timestep_slice].copy()
        # 复制当前 episode 的状态数据
        states = self.states[episode_index, timestep_slice].copy()
        # 复制当前 episode 的动作数据
        actions = self.actions[episode_index, timestep_slice].copy()
        # 复制当前 episode 的奖励数据
        rewards = self.rewards[episode_index, timestep_slice].copy()
        # 复制当前 episode 的完成标志数据
        dones = self.dones[episode_index, timestep_slice].copy()

        # 获取下一个状态数据，如果当前 timestep 已经是最后一个，则获取最后一个状态数据
        next_state = self.states[episode_index, min(timestep_index, self.max_episode_len - 1)].copy()

        # 返回文本嵌入数据、状态数据、动作数据、下一个状态数据、奖励数据、完成标志数据
        return text_embeds, states, actions, next_state, rewards, dones
# 定义一个基础环境类，用于扩展
class BaseEnvironment(Module):
    # 初始化方法，接受状态形状和文本嵌入形状作为参数
    @beartype
    def __init__(
        self,
        *,
        state_shape: Tuple[int, ...],
        text_embed_shape: Union[int, Tuple[int, ...]]
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置状态形状和文本嵌入形状属性
        self.state_shape = state_shape
        self.text_embed_shape = cast_tuple(text_embed_shape)
        # 注册一个缓冲区
        self.register_buffer('dummy', torch.zeros(0), persistent=False)

    # 返回缓冲区所在设备
    @property
    def device(self):
        return self.dummy.device

    # 初始化方法，返回指令和初始状态
    def init(self) -> Tuple[str, Tensor]:
        raise NotImplementedError

    # 前向传播方法，接受动作作为参数，返回奖励、下一个状态和是否结束的元组
    def forward(
        self,
        actions: Tensor
    ) -> Tuple[
        TensorType[(), float],     # reward
        Tensor,                    # next state
        TensorType[(), bool]       # done
    ]:
        raise NotImplementedError

# 代理类
class Agent(Module):
    # 初始化方法，接受 QRoboticTransformer 对象、环境对象和一些参数
    @beartype
    def __init__(
        self,
        q_transformer: QRoboticTransformer,
        *,
        environment: BaseEnvironment,
        memories_dataset_folder: str = DEFAULT_REPLAY_MEMORIES_FOLDER,
        num_episodes: int = 1000,
        max_num_steps_per_episode: int = 10000,
        epsilon_start: float = 0.25,
        epsilon_end: float = 0.001,
        num_steps_to_target_epsilon: int = 1000
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置 QRoboticTransformer 对象
        self.q_transformer = q_transformer
        # 设置是否在文本上进行条件
        condition_on_text = q_transformer.condition_on_text
        self.condition_on_text = condition_on_text
        # 设置环境对象
        self.environment = environment

        # 断言环境对象具有状态形状和文本嵌入形状属性
        assert hasattr(environment, 'state_shape') and hasattr(environment, 'text_embed_shape')

        # 断言参数的取值范围
        assert 0. <= epsilon_start <= 1.
        assert 0. <= epsilon_end <= 1.
        assert epsilon_start >= epsilon_end

        # 设置一些参数
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.num_steps_to_target_epsilon = num_steps_to_target_epsilon
        self.epsilon_slope = (epsilon_end - epsilon_start) / num_steps_to_target_epsilon

        self.num_episodes = num_episodes
        self.max_num_steps_per_episode = max_num_steps_per_episode

        # 创建存储回忆的文件夹
        mem_path = Path(memories_dataset_folder)
        self.memories_dataset_folder = mem_path

        mem_path.mkdir(exist_ok=True, parents=True)
        assert mem_path.is_dir()

        # 设置存储状态、动作、奖励和结束标志的文件路径
        states_path = mem_path / STATES_FILENAME
        actions_path = mem_path / ACTIONS_FILENAME
        rewards_path = mem_path / REWARDS_FILENAME
        dones_path = mem_path / DONES_FILENAME

        # 设置先验形状和动作数量
        prec_shape = (num_episodes, max_num_steps_per_episode)
        num_actions = q_transformer.num_actions
        state_shape = environment.state_shape

        # 如果在文本上进行条件
        if condition_on_text:
            text_embeds_path = mem_path / TEXT_EMBEDS_FILENAME
            text_embed_shape = environment.text_embed_shape
            self.text_embed_shape = text_embed_shape
            # 创建文本嵌入的内存映射
            self.text_embeds = open_memmap(str(text_embeds_path), dtype='float32', mode='w+', shape=(*prec_shape, *text_embed_shape))

        # 创建状态、动作、奖励和结束标志的内存映射
        self.states = open_memmap(str(states_path), dtype='float32', mode='w+', shape=(*prec_shape, *state_shape))
        self.actions = open_memmap(str(actions_path), dtype='int', mode='w+', shape=(*prec_shape, num_actions))
        self.rewards = open_memmap(str(rewards_path), dtype='float32', mode='w+', shape=prec_shape)
        self.dones = open_memmap(str(dones_path), dtype='bool', mode='w+', shape=prec_shape)

    # 根据步数获取 epsilon 值
    def get_epsilon(self, step):
        return max(self.epsilon_end, self.epsilon_slope * float(step) + self.epsilon_start)

    # 无需梯度的装饰器
    @beartype
    @torch.no_grad()
    # 定义一个方法，用于执行前向传播
    def forward(self):
        # 将 Q-Transformer 设置为评估模式
        self.q_transformer.eval()

        # 循环执行多个 episode
        for episode in range(self.num_episodes):
            # 打印当前 episode 的信息
            print(f'episode {episode}')

            # 初始化环境，获取指令和当前状态
            instruction, curr_state = self.environment.init()

            # 在每个 episode 中执行多个步骤
            for step in tqdm(range(self.max_num_steps_per_episode)):
                # 判断是否是最后一个步骤
                last_step = step == (self.max_num_steps_per_episode - 1)

                # 根据当前步骤获取 epsilon 值
                epsilon = self.get_epsilon(step)

                # 初始化文本嵌入为 None
                text_embed = None

                # 如果需要根据文本条件执行动作
                if self.condition_on_text:
                    # 获取指令的文本嵌入
                    text_embed = self.q_transformer.embed_texts([instruction])

                # 获取动作
                actions = self.q_transformer.get_actions(
                    rearrange(curr_state, '... -> 1 ...'),
                    text_embeds = text_embed,
                    prob_random_action = epsilon
                )

                # 执行动作，获取奖励、下一个状态和是否结束的标志
                reward, next_state, done = self.environment(actions)

                # 判断是否结束或是最后一个步骤
                done = done | last_step

                # 使用 memmap 存储记忆，以便后续回顾和学习

                # 如果需要根据文本条件执行动作
                if self.condition_on_text:
                    # 断言文本嵌入的形状符合预期
                    assert text_embed.shape[1:] == self.text_embed_shape
                    # 将文本嵌入存储到指定位置
                    self.text_embeds[episode, step] = text_embed

                # 存储当前状态、动作、奖励和结束标志
                self.states[episode, step]      = curr_state
                self.actions[episode, step]     = actions
                self.rewards[episode, step]     = reward
                self.dones[episode, step]       = done

                # 如果已经结束，跳出当前 episode 的循环
                if done:
                    break

                # 更新当前状态为下一个状态
                curr_state = next_state

            # 如果需要根据文本条件执行动作
            if self.condition_on_text:
                # 刷���文本嵌入的存储
                self.text_embeds.flush()

            # 刷新当前状态、动作、奖励和结束标志的存储
            self.states.flush()
            self.actions.flush()
            self.rewards.flush()
            self.dones.flush()

        # 关闭 memmap

        # 如果需要根据文本条件执行动作
        if self.condition_on_text:
            # 删除文本嵌入
            del self.text_embeds

        # 删除当前状态、动作、奖励和结束标志
        del self.states
        del self.actions
        del self.rewards
        del self.dones

        # 打印完成信息，存储的记忆位置
        print(f'completed, memories stored to {self.memories_dataset_folder.resolve()}')
```