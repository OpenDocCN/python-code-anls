# `.\lucidrains\self-rewarding-lm-pytorch\self_rewarding_lm_pytorch\dpo.py`

```py
# 导入必要的库
import os
from pathlib import Path
from copy import deepcopy
from functools import lru_cache
from collections import namedtuple
from dataclasses import dataclass

# 导入类型提示相关库
from beartype import beartype
from beartype.typing import Optional, Callable, Union, List
from torchtyping import TensorType

# 导入 PyTorch 相关库
import torch
from torch.nn import Module, Dropout
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

# 导入加速库
from accelerate import Accelerator

# 导入 einops 和 einx 库
from einops import rearrange
from einx import get_at

# 导入 numpy 相关库
from numpy.lib.format import open_memmap

# 导入自定义工具函数
from pytorch_custom_utils import (
    get_adam_optimizer,
    OptimizerWithWarmupSchedule
)

# 导入加速相关工具函数
from pytorch_custom_utils.accelerate_utils import (
    model_forward_contexts
)

# 导入自定义工具函数
from pytorch_custom_utils.utils import (
    masked_mean,
    maybe_and_mask
)

# 导入进度条库
from tqdm import tqdm

# 导入 EMA 库
from ema_pytorch import EMA

# 定义辅助函数

# 判断变量是否存在
def exists(v):
    return v is not None

# 如果变量存在则返回变量值，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 生成循环迭代器
def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# 判断是否处于分布式环境
@lru_cache(maxsize=None)
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

# 从模型和序列中获取对数概率
def log_prob_from_model_and_seq(model, seq):
    logits = model(seq)
    log_probs = logits.log_softmax(dim=-1)
    return get_at('b n [c], b n -> b n', log_probs, seq)

# 根据长度和序列生成掩码
def prompt_mask_from_len(lengths, seq):
    seq_len, device = seq.shape[-1], seq.device
    return torch.arange(seq_len, device=device) < rearrange(lengths, '... -> ... 1')

# 设置模型中的 Dropout 概率
def set_dropout_(model: Module, prob: float):
    for module in model.modules():
        if isinstance(module, Dropout):
            module.p = prob

# 使用线性衰减的 Adam 优化器
def adam_optimizer_with_linear_decay(
    model: Module,
    start_learning_rate: float,
    end_learning_rate: float,
    num_decay_steps: int,
    accelerator: Accelerator,
    weight_decay: float,
    adam_kwargs: dict = dict(),
) -> OptimizerWithWarmupSchedule:

    adam = get_adam_optimizer(
        model.parameters(),
        lr=start_learning_rate,
        wd=weight_decay
    )

    scheduler = None
    if start_learning_rate != end_learning_rate:
        scheduler = LinearLR

    return OptimizerWithWarmupSchedule(
        optimizer=adam,
        accelerator=accelerator,
        scheduler=LinearLR,
        scheduler_kwargs=dict(
            start_factor=1.,
            end_factor=end_learning_rate / start_learning_rate,
            total_iters=num_decay_steps
        )
    )

# 提前停止

# 定义提前停止返回结果的数据类
@dataclass
class EarlyStopperReturn:
    should_stop: bool
    score: float

# 提前停止类
class EarlyStopper(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        evaluator: Module,
        accelerator: Accelerator,
        calculate_should_stop: Callable[..., bool] = lambda scores: len(scores) > 1 and scores[-1] > scores[-2],
        early_stop_checkpoint_folder: str = './early-stop-checkpoint'
    ):
        super().__init__()
        self.model = model
        self.evaluator = evaluator
        self.accelerator = accelerator

        self.scores: List[Union[int, float]] = []
        self.calculate_should_stop = calculate_should_stop

        self.early_stop_checkpoint_folder = Path(early_stop_checkpoint_folder)
        self.early_stop_checkpoint_folder.mkdir(exist_ok=True, parents=True)

        self.register_buffer('break_signal', torch.tensor(0.))

    # 清空提前停止检查点文件夹
    def clear_early_checkpoint_folder(self):
        for file in self.early_stop_checkpoint_folder.glob('*.pt'):
            os.remove(file)

    # 判断是否为主进程
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 等待所有进程完成
    def wait(self):
        return self.accelerator.wait_for_everyone()
    # 保存模型的状态到指定路径
    def save(self, path: str, overwrite: bool = False):
        # 等待所有操作完成
        self.wait()

        # 如果是主进程
        if self.is_main:

            # 设置保存路径为早停检查点文件夹下的指定路径
            path = self.early_stop_checkpoint_folder / path

            # 如果文件已存在且不允许覆盖，则抛出异常
            assert not path.exists() or overwrite, f'file already exists'

            # 构建保存的数据包，包含模型的状态字典
            pkg = dict(
                model = self.model.state_dict()
            )

            # 保存数据包到指定路径
            torch.save(pkg, str(path))

    # 前向传播函数，返回早停器的结果
    @torch.no_grad()
    def forward(self) -> EarlyStopperReturn:
        # 设置模型为评估模式
        self.model.eval()

        score = None

        # 如果是主进程
        if self.is_main:

            # 计算模型的评估分数
            score = self.evaluator(self.model)

            # 如果评估分数是张量
            if torch.is_tensor(score):
                # 确保张量元素个数为1
                assert score.numel() == 1
                # 将张量展平为标量
                score = score.flatten().item()

            # 确保评估分数为整数或浮点数
            assert isinstance(score, (int, float))

            # 将评估分数添加到分数列表中
            self.scores.append(score)

            # 计算是否应该停止训练
            should_stop = self.calculate_should_stop(self.scores)

            # 如果应该停止，则设置中断信号为1
            if should_stop:
                self.break_signal.copy_(1.)

        # 处理分布式环境下的早停中断信号
        if is_distributed():
            dist.all_reduce(self.break_signal)
            should_stop = self.break_signal.item() == 1.

        # 处理在评估分数下降之前恢复到检查点的逻辑
        if should_stop:
            # 获取上一个评估分数对应的检查点文件名
            prev_checkpoint_filename = f'model.ckpt.{len(self.scores) - 1}.pt'
            prev_checkpoint_path = self.early_stop_checkpoint_folder / prev_checkpoint_filename
            # 加载上一个检查点的模型状态
            pkg = torch.load(str(prev_checkpoint_path))

            self.model.load_state_dict(pkg['model'])
        else:
            # 生成当前评估分数对应的检查点文件名，并保存当前模型状态
            checkpoint_filename = f'model.ckpt.{len(self.scores)}.pt'
            self.save(checkpoint_filename)

        # 返回早停器的结果，包括评估分数和是否应该停止训练的标志
        return EarlyStopperReturn(score, self.break_signal.item() == 1)
# 从两个 memmap numpy 文件中读取数据集

# 数据集包含首选和非首选序列的形状 - (<样本数>, <偏好 (2) - 首选后跟非首选>, <序列长度>)
# 提示长度 (<样本数>,)

class DPODataset(Dataset):
    def __init__(
        self,
        data_folder: str = './',
        preference_seq_memmap_file: str = 'preference_seq.memmap.npy',
        prompt_len_memmap_file: str = 'prompt_len.memmap.npy',
    ):
        self.data_folder = Path(data_folder)
        assert self.data_folder.exists() and self.data_folder.is_dir()

        preference_seq_memmap_path = self.data_folder / preference_seq_memmap_file
        prompt_len_memmap_path = self.data_folder / prompt_len_memmap_file

        assert preference_seq_memmap_path.exists()
        assert prompt_len_memmap_path.exists()

        self.paired_sequences = open_memmap(str(preference_seq_memmap_path), dtype = 'int', mode = 'r')
        self.prompt_len = open_memmap(str(prompt_len_memmap_path), dtype = 'int', mode = 'r')

        self.seq_len = self.paired_sequences.shape[1]
        assert self.paired_sequences.shape[0] == self.prompt_len.shape[0]

    def __len__(self):
        return self.paired_sequences.shape[0]

    def __getitem__(self, idx):
        sequences = self.paired_sequences[idx].copy()
        prompt_lens = self.prompt_len[idx].copy()

        preferred_seq, unpreferred_seq = sequences

        return preferred_seq, unpreferred_seq, prompt_lens

# 主类

class DPO(Module):
    def __init__(
        self,
        model: Module,
        *,
        beta = 0.1,
        ref_model_ema_decay = 1.,
        pad_id: Optional[int] = None,
        ema_kwargs: dict = dict()
    ):
        super().__init__()
        self.policy_model = model

        self.ref_model = EMA(
            model,
            beta = ref_model_ema_decay,
            **ema_kwargs
        )

        self.beta = beta
        self.pad_id = pad_id

    def update_reference_model_with_policy(self):
        self.ref_model.copy_params_from_model_to_ema()

    def update_ema(self):
        self.ref_model.update()

    def parameters(self):
        return self.policy_model.parameters()

    @property
    def device(self):
        return next(self.parameters()).device

    @autocast(enabled = False)
    def forward(
        self,
        preferred_seq: TensorType['b', 'n', int],
        unpreferred_seq: TensorType['b', 'n', int],
        prompt_len: TensorType['b', int],
        preferred_seq_mask: Optional[TensorType['b', 'n', bool]] = None,
        unpreferred_seq_mask: Optional[TensorType['b', 'n', bool]] = None
    # 设置策略模型为训练模式
    self.policy_model.train()

    """
    b - batch
    n - sequence length
    """

    # 根据提示长度和首选/非首选序列生成掩码
    preferred_prompt_mask = prompt_mask_from_len(prompt_len, preferred_seq)
    unpreferred_prompt_mask = prompt_mask_from_len(prompt_len, unpreferred_seq)

    """
    Following Appendix B in https://arxiv.org/abs/2305.18290
    """

    # 如果存在填充 ID
    if exists(self.pad_id):
        # 确保首选序列掩码和非首选序列掩码不存在
        assert not exists(preferred_seq_mask)
        assert not exists(unpreferred_seq_mask)
        # 创建首选序列掩码
        preferred_seq_mask = preferred_seq != self.pad_id
        preferred_seq.masked_fill_(~preferred_seq_mask, 0)
        # 创建非首选序列掩码
        unpreferred_seq_mask = unpreferred_seq != self.pad_id
        unpreferred_seq.masked_fill_(~unpreferred_seq_mask, 0)            

    # 在不计算梯度的情况下执行以下操作
    with torch.no_grad():
        # 设置参考模型为评估模式
        self.ref_model.eval()
        # 计算首选序列和非首选序列在参考模型下的对数概率
        ref_preferred_logprob = log_prob_from_model_and_seq(self.ref_model, preferred_seq)
        ref_unpreferred_logprob = log_prob_from_model_and_seq(self.ref_model, unpreferred_seq)

    # 计算策略模型下首选序列和非首选序列的对数概率
    policy_preferred_logprob = log_prob_from_model_and_seq(self.policy_model, preferred_seq)
    policy_unpreferred_logprob = log_prob_from_model_and_seq(self.policy_model, unpreferred_seq)

    # 计算掩码平均值

    # 对策略模型和参考模型下的首选序列和非首选序列的对数概率进行掩码平均值计算
    policy_preferred_logprob, ref_preferred_logprob = [masked_mean(seq, maybe_and_mask(preferred_seq_mask, ~preferred_prompt_mask)) for seq in (policy_preferred_logprob, ref_preferred_logprob)]
    policy_unpreferred_logprob, ref_unpreferred_logprob = [masked_mean(seq, maybe_and_mask(unpreferred_seq_mask, ~unpreferred_prompt_mask)) for seq in (policy_unpreferred_logprob, ref_unpreferred_logprob)]

    # 计算 DPO 损失

    # 计算策略模型和参考模型下的首选序列和非首选序列的对数概率之差
    policy_logratios = policy_preferred_logprob - policy_unpreferred_logprob
    ref_logratios = ref_preferred_logprob - ref_unpreferred_logprob

    # 计算损失值
    losses = -F.logsigmoid(self.beta * (policy_logratios - ref_logratios))

    # 返回损失值的平均值
    return losses.mean()
# trainer class

class DPOTrainer(Module):
    # 初始化方法
    @beartype
    def __init__(
        self,
        dpo: Union[DPO, Module],
        *,
        dataset_generator: Optional[Callable[[], Dataset]] = None,
        accelerator: Optional[Accelerator] = None,
        batch_size: int = 16,
        grad_accum_steps: int = 2,
        num_decay_steps: int = 1000,
        num_train_steps: Optional[int] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.,
        train_dataset: Optional[Dataset] = None,
        valid_dataset: Optional[Dataset] = None,
        start_learning_rate: float = 1e-6,
        end_learning_rate: float = 1e-7,
        early_stopper: Optional[EarlyStopper] = None,
        dropout: float = 0.1,
        check_early_stop_every: int = 200,
        early_stopper_eval_module: Optional[Module] = None,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        dpo_kwargs: dict = dict(
            beta = 0.1,
            ref_model_ema_decay = 1.
        ),
        early_stopper_kwargs: dict = dict()
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 如果 dpo 不是 DPO 类型，则使用 dpo_kwargs 创建 DPO 对象
        if not isinstance(dpo, DPO):
            dpo = DPO(dpo, **dpo_kwargs)

        # 设置 DPO 对象的 dropout
        set_dropout_(dpo, dropout)

        # 如果 accelerator 不存在，则使用 accelerate_kwargs 创建 Accelerator 对象
        if not exists(accelerator):
            accelerator = Accelerator(**accelerate_kwargs)

        # 设置 accelerator
        self.accelerator = accelerator

        # 准备模型
        self.model = accelerator.prepare(dpo)
        self.dropout = dropout

        # 设置数据集生成器
        self.dataset_generator = dataset_generator

        # 设置批量大小和梯度累积步数
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps

        # 使用 adam_optimizer_with_linear_decay 创建优化器
        self.optimizer = adam_optimizer_with_linear_decay(
            dpo,
            start_learning_rate,
            end_learning_rate,
            num_decay_steps = num_decay_steps,
            accelerator = accelerator,
            weight_decay = weight_decay,
            adam_kwargs = adam_kwargs
        )

        # 如果存在 early_stopper_eval_module，则创建 EarlyStopper 对象
        self.early_stopper = None
        if exists(early_stopper_eval_module):
            self.early_stopper = EarlyStopper(
                dpo.policy_model,
                evaluator = early_stopper_eval_module,
                accelerator = self.accelerator,
                **early_stopper_kwargs
            )

        # 设置检查早停的频率
        self.check_early_stop_every = check_early_stop_every

        # 如果存在 train_dataset，则创建 DataLoader 对象
        self.train_dataloader = None
        if exists(train_dataset):
            self.train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
            self.train_dataloader = accelerator.prepare(self.train_dataloader)

        # 如果存在 valid_dataset，则创建 DataLoader 对象
        self.valid_dataloader = None
        if exists(valid_dataset):
            self.valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size)

        # 初始化步数和训练步数
        self.steps = 0
        self.num_train_steps = num_train_steps

    # 获取未包装的模型
    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    # 判断是否为主进程
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 打印信息
    def print(self, *msg):
        self.accelerator.print(*msg)

    # 等待所有进程完成
    def wait(self):
        return self.accelerator.wait_for_everyone()

    # 记录日志
    def log(self, **data):
        self.accelerator.log(data, step = self.steps)

    # 前向传播方法
    def forward(
        self,
        train_self_reward_dataset: Optional[Dataset] = None
        ):
            # 检查是否存在数据集生成器，如果存在则生成训练用的自我奖励数据集
            if exists(self.dataset_generator):
                train_self_reward_dataset = self.dataset_generator()

            # 更新参考模型的策略
            self.model.update_reference_model_with_policy()

            # 如果存在早停器，清除早停检查点文件夹
            if exists(self.early_stopper):
                self.early_stopper.clear_early_checkpoint_folder()

            # 获取训练数据加载器
            train_dataloader = self.train_dataloader

            # 如果训练数据加载器不存在，则创建一个并准备好
            if not exists(train_dataloader):
                assert exists(train_self_reward_dataset)
                train_dataloader = DataLoader(train_self_reward_dataset, batch_size = self.batch_size, drop_last = True, shuffle = True)
                train_dataloader = self.accelerator.prepare(train_dataloader)

            # 创建数据加载器的迭代器
            iter_dl = cycle(train_dataloader)

            # 创建进度条
            pbar = tqdm(desc = 'dpo fine-tuning', total = self.num_train_steps)

            # 设置模型的 dropout
            set_dropout_(self.model, self.dropout)

            # 进入训练循环
            while True:
                self.model.train()

                # 遍历模型前向上下文
                for forward_context in model_forward_contexts(self.accelerator, self.model, self.grad_accum_steps):
                    with forward_context():
                        batch = next(iter_dl)

                        # 计算 DPO 损失
                        dpo_loss = self.model(*batch)
                        self.accelerator.backward(dpo_loss / self.grad_accum_steps)

                # 打印 DPO 损失值
                self.print(f'dpo loss: {dpo_loss.item():.3f}')
                self.log(loss = dpo_loss.item())

                # 执行优化器的步骤
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 等待
                self.wait()

                # 更新指数移动平均模型
                self.unwrapped_model.update_ema()

                # 更新步数并更新进度条
                self.steps += 1
                pbar.update(1)

                # 如果达到训练步数上限，则结束训练
                if exists(self.num_train_steps) and self.steps >= self.num_train_steps:
                    break

                # 检查是否需要早停
                self.wait()

                if not (self.steps % self.check_early_stop_every) and exists(self.early_stopper):

                    # 执行早停逻辑
                    early_stop_return = self.early_stopper()

                    if self.is_main:
                        self.print(f'valid dpo loss: {early_stop_return.score:.3f}')
                        self.log(dpo_valid_score = early_stop_return.score)

                    if early_stop_return.should_stop:
                        self.print('early stopping')
                        break

            # 关闭进度条
            pbar.close()
            self.print('dpo training finished')
```