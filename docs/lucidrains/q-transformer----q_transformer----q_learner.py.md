# `.\lucidrains\q-transformer\q_transformer\q_learner.py`

```
# 导入所需的模块
from pathlib import Path
from functools import partial
from contextlib import nullcontext
from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, DataLoader

# 导入自定义的类型注解模块
from torchtyping import TensorType

# 导入 einops 相关模块
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# 导入 beartype 相关模块
from beartype import beartype
from beartype.typing import Optional, Union, List, Tuple

# 导入自定义的 QRoboticTransformer 类
from q_transformer.q_robotic_transformer import QRoboticTransformer

# 导入自定义的优化器获取函数
from q_transformer.optimizer import get_adam_optimizer

# 导入 accelerate 相关模块
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# 导入 EMA 模块
from ema_pytorch import EMA

# 定义常量

# 定义 QIntermediates 命名元组，包含 Q 学习中的中间变量
QIntermediates = namedtuple('QIntermediates', [
    'q_pred_all_actions',
    'q_pred',
    'q_next',
    'q_target'
])

# 定义 Losses 命名元组，包含损失函数中的损失项
Losses = namedtuple('Losses', [
    'td_loss',
    'conservative_reg_loss'
])

# 定义辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 判断两个数是否整除
def is_divisible(num, den):
    return (num % den) == 0

# 将单个张量按指定模式打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 将单个张量按指定模式解包
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 生成数据集的无限循环迭代器
def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# 张量操作辅助函数

# 从张量中选择指定索引的元素
def batch_select_indices(t, indices):
    indices = rearrange(indices, '... -> ... 1')
    selected = t.gather(-1, indices)
    return rearrange(selected, '... 1 -> ...')

# Q 学习在机器人变压器上的实现

# 定义 QLearner 类，继承自 Module
class QLearner(Module):

    # 初始化函数
    @beartype
    def __init__(
        self,
        model: Union[QRoboticTransformer, Module],
        *,
        dataset: Dataset,
        batch_size: int,
        num_train_steps: int,
        learning_rate: float,
        min_reward: float = 0.,
        grad_accum_every: int = 1,
        monte_carlo_return: Optional[float] = None,
        weight_decay: float = 0.,
        accelerator: Optional[Accelerator] = None,
        accelerator_kwargs: dict = dict(),
        dataloader_kwargs: dict = dict(
            shuffle = True
        ),
        q_target_ema_kwargs: dict = dict(
            beta = 0.99,
            update_after_step = 10,
            update_every = 5
        ),
        max_grad_norm = 0.5,
        n_step_q_learning = False,
        discount_factor_gamma = 0.98,
        conservative_reg_loss_weight = 1., # they claim 1. is best in paper
        optimizer_kwargs: dict = dict(),
        checkpoint_folder = './checkpoints',
        checkpoint_every = 1000,
    # 初始化函数，继承父类的初始化方法
    def __init__(
        self,
        model,
        discount_factor_gamma,
        n_step_q_learning,
        conservative_reg_loss_weight,
        q_target_ema_kwargs,
        max_grad_norm,
        learning_rate,
        weight_decay,
        optimizer_kwargs,
        accelerator,
        accelerator_kwargs,
        min_reward,
        monte_carlo_return,
        dataset,
        batch_size,
        dataloader_kwargs,
        checkpoint_every,
        checkpoint_folder,
        num_train_steps,
        grad_accum_every
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 判断是否有多个动作
        self.is_multiple_actions = model.num_actions > 1

        # Q-learning 相关超参数
        self.discount_factor_gamma = discount_factor_gamma
        self.n_step_q_learning = n_step_q_learning

        # 是否有保守正则化损失
        self.has_conservative_reg_loss = conservative_reg_loss_weight > 0.
        self.conservative_reg_loss_weight = conservative_reg_loss_weight

        # 注册缓冲区
        self.register_buffer('discount_matrix', None, persistent = False)

        # 在线 Q 模型
        self.model = model

        # EMA（目标）Q 模型
        self.ema_model = EMA(
            model,
            include_online_model = False,
            **q_target_ema_kwargs
        )

        # 最大梯度范数
        self.max_grad_norm = max_grad_norm

        # 获取 Adam 优化器
        self.optimizer = get_adam_optimizer(
            model.parameters(),
            lr = learning_rate,
            wd = weight_decay,
            **optimizer_kwargs
        )

        # 如果加速器不存在，则创建一个
        if not exists(accelerator):
            accelerator = Accelerator(
                kwargs_handlers = [
                    DistributedDataParallelKwargs(find_unused_parameters = True)
                ],
                **accelerator_kwargs
            )

        self.accelerator = accelerator

        # 最小奖励和蒙特卡洛回报
        self.min_reward = min_reward
        self.monte_carlo_return = monte_carlo_return

        # 创建数据加载器
        self.dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            **dataloader_kwargs
        )

        # 准备模型、EMA 模型、优化器和数据加载器
        (
            self.model,
            self.ema_model,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.ema_model,
            self.optimizer,
            self.dataloader
        )

        # 检查点相关
        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)

        # 创建检查点文件夹
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
        assert self.checkpoint_folder.is_dir()

        # 创建一个零张量作为虚拟损失
        self.register_buffer('zero', torch.tensor(0.))

        # 训练步骤相关
        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every

        # 注册步骤计数器
        self.register_buffer('step', torch.tensor(0))

    # 保存模型
    def save(
        self,
        checkpoint_num = None,
        overwrite = True
    ):
        name = 'checkpoint'
        if exists(checkpoint_num):
            name += f'-{checkpoint_num}'

        path = self.checkpoint_folder / (name + '.pt')

        assert overwrite or not path.exists()

        # 打包模型、EMA 模型、优化器和步骤计数器
        pkg = dict(
            model = self.unwrap(self.model).state_dict(),
            ema_model = self.unwrap(self.ema_model).state_dict(),
            optimizer = self.optimizer.state_dict(),
            step = self.step.item()
        )

        # 保存模型
        torch.save(pkg, str(path))

    # 加载模型
    def load(self, path):
        path = Path(path)
        assert exists(path)

        pkg = torch.load(str(path))

        # 加载模型、EMA 模型和优化器
        self.unwrap(self.model).load_state_dict(pkg['model'])
        self.unwrap(self.ema_model).load_state_dict(pkg['ema_model'])

        self.optimizer.load_state_dict(pkg['optimizer'])
        self.step.copy_(pkg['step'])

    # 获取设备
    @property
    def device(self):
        return self.accelerator.device

    # 判断是否为主进程
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 解包模型
    def unwrap(self, module):
        return self.accelerator.unwrap_model(module)

    # 打印信息
    def print(self, msg):
        return self.accelerator.print(msg)

    # 等待所有进程完成
    def wait(self):
        return self.accelerator.wait_for_everyone()
    def get_discount_matrix(self, timestep):
        # 检查是否已存在折扣矩阵并且其时间步长大于等于当前时间步长
        if exists(self.discount_matrix) and self.discount_matrix.shape[-1] >= timestep:
            # 如果满足条件，则返回已存在的折扣矩阵的子矩阵
            return self.discount_matrix[:timestep, :timestep]

        # 创建一个时间步长范围的张量
        timestep_arange = torch.arange(timestep, device=self.accelerator.device)
        # 计算时间步长之间的幂次
        powers = (timestep_arange[None, :] - timestep_arange[:, None])
        # 根据幂次计算折扣矩阵
        discount_matrix = torch.triu(self.discount_factor_gamma ** powers)

        # 将折扣矩阵注册为缓冲区
        self.register_buffer('discount_matrix', discount_matrix, persistent=False)
        # 返回折扣矩阵
        return self.discount_matrix

    def q_learn(
        self,
        text_embeds: TensorType['b', 'd', float],
        states: TensorType['b', 'c', 'f', 'h', 'w', float],
        actions: TensorType['b', int],
        next_states: TensorType['b', 'c', 'f', 'h', 'w', float],
        reward: TensorType['b', float],
        done: TensorType['b', bool],
        *,
        monte_carlo_return=None
    ) -> Tuple[TensorType[()], QIntermediates]:
        # 'next'代表下一个时间步（无论是状态、q值、动作等）

        γ = self.discount_factor_gamma
        # 计算非终止状态的掩码
        not_terminal = (~done).float()

        # 使用在线Q机器人变换器进行预测
        q_pred_all_actions = self.model(states, text_embeds=text_embeds)
        # 选择出采取的动作对应的Q值
        q_pred = batch_select_indices(q_pred_all_actions, actions)

        # 使用指数平滑的模型副本作为未来的Q目标。比在每个批次之后将q_target设置为q_eval更稳定
        # 最大Q值被视为最优动作，隐含地是具有最高Q分数的动作
        q_next = self.ema_model(next_states, text_embeds=text_embeds).amax(dim=-1)
        q_next.clamp_(min=default(monte_carlo_return, -1e4))

        # 贝尔曼方程。最重要的代码行，希望正确执行
        q_target = reward + not_terminal * (γ * q_next)

        # 强制在线模型能够预测这个目标
        loss = F.mse_loss(q_pred, q_target)

        # 这就是全部。对于Q学习的核心，大约5行代码
        # 返回损失和一些中间结果以便记录
        return loss, QIntermediates(q_pred_all_actions, q_pred, q_next, q_target)

    def n_step_q_learn(
        self,
        text_embeds: TensorType['b', 'd', float],
        states: TensorType['b', 't', 'c', 'f', 'h', 'w', float],
        actions: TensorType['b', 't', int],
        next_states: TensorType['b', 'c', 'f', 'h', 'w', float],
        rewards: TensorType['b', 't', float],
        dones: TensorType['b', 't', bool],
        *,
        monte_carlo_return=None
    ) -> Tuple[TensorType[()], QIntermediates]:
        """
        einops

        b - batch
        c - channels
        f - frames
        h - height
        w - width
        t - timesteps
        a - action bins
        q - q values
        d - text cond dimension
        """

        num_timesteps, device = states.shape[1], states.device

        # fold time steps into batch

        states, time_ps = pack_one(states, '* c f h w')
        text_embeds, _ = pack_one(text_embeds, '* d')

        # repeat text embeds per timestep

        repeated_text_embeds = repeat(text_embeds, 'b ... -> (b n) ...', n = num_timesteps)

        γ = self.discount_factor_gamma

        # anything after the first done flag will be considered terminal

        dones = dones.cumsum(dim = -1) > 0
        dones = F.pad(dones, (1, 0), value = False)

        not_terminal = (~dones).float()

        # get q predictions

        actions = rearrange(actions, 'b t -> (b t)')

        q_pred_all_actions = self.model(states, text_embeds = repeated_text_embeds)
        q_pred = batch_select_indices(q_pred_all_actions, actions)
        q_pred = unpack_one(q_pred, time_ps, '*')

        q_next = self.ema_model(next_states, text_embeds = text_embeds).amax(dim = -1)
        q_next.clamp_(min = default(monte_carlo_return, -1e4))

        # prepare rewards and discount factors across timesteps

        rewards, _ = pack([rewards, q_next], 'b *')

        γ = self.get_discount_matrix(num_timesteps + 1)[:-1, :]

        # account for discounting using the discount matrix

        q_target = einsum('b t, q t -> b q', not_terminal * rewards, γ)

        # have transformer learn to predict above Q target

        loss = F.mse_loss(q_pred, q_target)

        # prepare q prediction

        q_pred_all_actions = unpack_one(q_pred_all_actions, time_ps, '* a')

        return loss, QIntermediates(q_pred_all_actions, q_pred, q_next, q_target)

    def autoregressive_q_learn_handle_single_timestep(
        self,
        text_embeds,
        states,
        actions,
        next_states,
        rewards,
        dones,
        *,
        monte_carlo_return = None
    ):
        """
        simply detect and handle single timestep
        and use `autoregressive_q_learn` as more general function
        """
        if states.ndim == 5:
            states = rearrange(states, 'b ... -> b 1 ...')

        if actions.ndim == 2:
            actions = rearrange(actions, 'b ... -> b 1 ...')

        if rewards.ndim == 1:
            rewards = rearrange(rewards, 'b -> b 1')

        if dones.ndim == 1:
            dones = rearrange(dones, 'b -> b 1')

        return self.autoregressive_q_learn(text_embeds, states, actions, next_states, rewards, dones, monte_carlo_return = monte_carlo_return)

    def autoregressive_q_learn(
        self,
        text_embeds:    TensorType['b', 'd', float],
        states:         TensorType['b', 't', 'c', 'f', 'h', 'w', float],
        actions:        TensorType['b', 't', 'n', int],
        next_states:    TensorType['b', 'c', 'f', 'h', 'w', float],
        rewards:        TensorType['b', 't', float],
        dones:          TensorType['b', 't', bool],
        *,
        monte_carlo_return = None
    ) -> Tuple[TensorType[()], QIntermediates]:
        """
        einops

        b - batch
        c - channels
        f - frames
        h - height
        w - width
        t - timesteps
        n - number of actions
        a - action bins
        q - q values
        d - text cond dimension
        """
        # 设置默认的蒙特卡洛回报值
        monte_carlo_return = default(monte_carlo_return, -1e4)
        # 获取状态的时间步数和设备信息
        num_timesteps, device = states.shape[1], states.device

        # 将时间步折叠到批次中

        states, time_ps = pack_one(states, '* c f h w')
        actions, _ = pack_one(actions, '* n')
        text_embeds, _ = pack_one(text_embeds, '* d')

        # 每个时间步重复文本嵌入

        repeated_text_embeds = repeat(text_embeds, 'b ... -> (b n) ...', n = num_timesteps)

        # 第一个完成标志之后的任何内容都将被视为终止

        dones = dones.cumsum(dim = -1) > 0
        dones = F.pad(dones, (1, -1), value = False)

        not_terminal = (~dones).float()

        # 奖励不应在终止步骤及之后给出

        rewards = rewards * not_terminal

        # 因为希腊字母Unicode看起来很好

        γ = self.discount_factor_gamma

        # 获取每个动作的预测 Q 值
        # 解包回 (b, t, n)

        q_pred_all_actions = self.model(states, text_embeds = repeated_text_embeds, actions = actions)
        q_pred = batch_select_indices(q_pred_all_actions, actions)
        q_pred = unpack_one(q_pred, time_ps, '* n')

        # 获取 q_next

        q_next = self.ema_model(next_states, text_embeds = text_embeds)
        q_next = q_next.max(dim = -1).values
        q_next.clamp_(min = monte_carlo_return)

        # 获取目标 Q
        # 解包回 - (b, t, n)

        q_target_all_actions = self.ema_model(states, text_embeds = repeated_text_embeds, actions = actions)
        q_target = q_target_all_actions.max(dim = -1).values

        q_target.clamp_(min = monte_carlo_return)
        q_target = unpack_one(q_target, time_ps, '* n')

        # 论文的主要贡献是以下逻辑
        # 第 4.1 节 - 方程 1

        # 首先处理除最后一个动作之外的所有动作的损失

        q_pred_rest_actions, q_pred_last_action      = q_pred[..., :-1], q_pred[..., -1]
        q_target_first_action, q_target_rest_actions = q_target[..., 0], q_target[..., 1:]

        losses_all_actions_but_last = F.mse_loss(q_pred_rest_actions, q_target_rest_actions, reduction = 'none')

        # 接下来处理最后一个动作，其中包含奖励

        q_target_last_action, _ = pack([q_target_first_action[..., 1:], q_next], 'b *')

        q_target_last_action = rewards + γ * q_target_last_action

        losses_last_action = F.mse_loss(q_pred_last_action, q_target_last_action, reduction = 'none')

        # 展平并平均

        losses, _ = pack([losses_all_actions_but_last, losses_last_action], '*')

        return losses.mean(), QIntermediates(q_pred_all_actions, q_pred, q_next, q_target)

    def learn(
        self,
        *args,
        min_reward: Optional[float] = None,
        monte_carlo_return: Optional[float] = None
    ):
        # 从参数中解包出 actions
        _, _, actions, *_ = args

        # q-learn kwargs
        # 创建包含 monte_carlo_return 参数的字典
        q_learn_kwargs = dict(
            monte_carlo_return = monte_carlo_return
        )

        # main q-learning loss, respectively
        # 1. proposed autoregressive q-learning for multiple actions - (handles single or n-step automatically)
        # 2. single action - single timestep (classic q-learning)
        # 3. single action - n-steps

        # 如果是多个动作
        if self.is_multiple_actions:
            # 使用 autoregressive_q_learn_handle_single_timestep 处理单个时间步的动作
            td_loss, q_intermediates = self.autoregressive_q_learn_handle_single_timestep(*args, **q_learn_kwargs)
            num_timesteps = actions.shape[1]

        # 如果是 n-step Q-learning
        elif self.n_step_q_learning:
            # 使用 n_step_q_learn 处理 n-step Q-learning
            td_loss, q_intermediates = self.n_step_q_learn(*args, **q_learn_kwargs)
            num_timesteps = actions.shape[1]

        else:
            # 使用 q_learn 处理单个时间步的动作
            td_loss, q_intermediates = self.q_learn(*args, **q_learn_kwargs)
            num_timesteps = 1

        # 如果没有保守正则化损失
        if not self.has_conservative_reg_loss:
            # 返回损失和 Losses 对象
            return loss, Losses(td_loss, self.zero)

        # 计算保守正则化
        # 论文中的第 4.2 节，方程式 2

        # 获取批次大小
        batch = actions.shape[0]

        # 获取所有动作的 Q 预测值
        q_preds = q_intermediates.q_pred_all_actions
        q_preds = rearrange(q_preds, '... a -> (...) a')

        # 获取动作的数量
        num_action_bins = q_preds.shape[-1]
        num_non_dataset_actions = num_action_bins - 1

        # 重新排列动作
        actions = rearrange(actions, '... -> (...) 1')

        # 创建数据集动作掩码
        dataset_action_mask = torch.zeros_like(q_preds).scatter_(-1, actions, torch.ones_like(q_preds))

        # 获取未选择的动作的 Q 值
        q_actions_not_taken = q_preds[~dataset_action_mask.bool()]
        q_actions_not_taken = rearrange(q_actions_not_taken, '(b t a) -> b t a', b = batch, a = num_non_dataset_actions)

        # 计算保守正则化损失
        conservative_reg_loss = ((q_actions_not_taken - (min_reward * num_timesteps)) ** 2).sum() / num_non_dataset_actions

        # 总损失
        loss =  0.5 * td_loss + \
                0.5 * conservative_reg_loss * self.conservative_reg_loss_weight

        # 损失细分
        loss_breakdown = Losses(td_loss, conservative_reg_loss)

        return loss, loss_breakdown

    # 前向传播函数
    def forward(
        self,
        *,
        monte_carlo_return: Optional[float] = None,
        min_reward: Optional[float] = None
        ):
            # 如果未提供蒙特卡洛回报和最小奖励，则使用默认值
            monte_carlo_return = default(monte_carlo_return, self.monte_carlo_return)
            min_reward = default(min_reward, self.min_reward)

            # 获取当前步数
            step = self.step.item()

            # 创建一个循环迭代器，用于遍历数据加载器
            replay_buffer_iter = cycle(self.dataloader)

            # 设置模型为训练模式
            self.model.train()
            self.ema_model.train()

            # 在训练步数小于总训练步数时执行循环
            while step < self.num_train_steps:

                # 清空梯度
                self.optimizer.zero_grad()

                # 主要的 Q-learning 算法

                # 对于每个梯度累积步骤
                for grad_accum_step in range(self.grad_accum_every):
                    is_last = grad_accum_step == (self.grad_accum_every - 1)
                    # 如果不是最后一个梯度累积步骤，则使用 partial 函数创建上下文
                    context = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                    # 使用自动混合精度和上下文执行学习过程
                    with self.accelerator.autocast(), context():

                        # 调用 learn 方法进行学习
                        loss, (td_loss, conservative_reg_loss) = self.learn(
                            *next(replay_buffer_iter),
                            min_reward = min_reward,
                            monte_carlo_return = monte_carlo_return
                        )

                        # 反向传播
                        self.accelerator.backward(loss / self.grad_accum_every)

                # 打印 TD 损失
                self.print(f'td loss: {td_loss.item():.3f}')

                # 限制梯度大小（变压器最佳实践）
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # 执行优化器步骤
                self.optimizer.step()

                # 更新目标 EMA
                self.wait()
                self.ema_model.update()

                # 增加步数
                step += 1
                self.step.add_(1)

                # 是否进行检查点
                self.wait()

                if self.is_main and is_divisible(step, self.checkpoint_every):
                    checkpoint_num = step // self.checkpoint_every
                    self.save(checkpoint_num)

                self.wait()

            # 训练完成后打印信息
            self.print('training complete')
```