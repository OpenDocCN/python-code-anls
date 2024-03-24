# `.\lucidrains\self-rewarding-lm-pytorch\self_rewarding_lm_pytorch\spin.py`

```
from pathlib import Path
# 导入 Path 模块，用于处理文件路径

from beartype import beartype
from beartype.typing import Optional, Callable, Union
# 导入 beartype 模块，用于类型注解

from torchtyping import TensorType
# 导入 TensorType 类型注解

import torch
# 导入 torch 模块

from torch.nn import Module, Dropout
# 从 torch.nn 模块中导入 Module 和 Dropout 类

import torch.nn.functional as F
# 导入 torch.nn.functional 模块，用于神经网络函数

from torch.cuda.amp import autocast
# 导入 autocast 函数，用于混合精度训练

from torch.utils.data import Dataset, DataLoader
# 导入 Dataset 和 DataLoader 类，用于处理数据集和数据加载

from torch.nn.utils.rnn import pad_sequence
# 导入 pad_sequence 函数，用于填充序列

from accelerate import Accelerator
# 导入 Accelerator 类，用于加速训练

from einops import rearrange
# 导入 rearrange 函数，用于重排张量维度

from einx import get_at
# 导入 get_at 函数，用于获取张量的特定位置

from pytorch_custom_utils.utils import (
    masked_mean,
    maybe_and_mask
)
# 从 pytorch_custom_utils.utils 模块中导入 masked_mean 和 maybe_and_mask 函数

from pytorch_custom_utils.accelerate_utils import (
    model_forward_contexts
)
# 从 pytorch_custom_utils.accelerate_utils 模块中导入 model_forward_contexts 函数

from self_rewarding_lm_pytorch.dpo import (
    adam_optimizer_with_linear_decay
)
# 从 self_rewarding_lm_pytorch.dpo 模块中导入 adam_optimizer_with_linear_decay 函数

from self_rewarding_lm_pytorch.sampling_utils import (
    sample,
    top_p,
    top_k
)
# 从 self_rewarding_lm_pytorch.sampling_utils 模块中导入 sample、top_p 和 top_k 函数

from tqdm import tqdm
# 导入 tqdm 模块，用于显示进度条

from ema_pytorch import EMA
# 导入 EMA 类，用于指数移动平均

# helper functions

def exists(v):
    return v is not None
# 定义 exists 函数，判断变量是否为 None

def cycle(dl):
    while True:
        for batch in dl:
            yield batch
# 定义 cycle 函数，用于循环迭代数据加载器中的批次数据

def log_prob_from_model_and_seq(model, seq):
    logits = model(seq)
    log_probs = logits.log_softmax(dim = -1)
    return get_at('b n [c], b n -> b n', log_probs, seq)
# 定义 log_prob_from_model_and_seq 函数，计算模型生成序列的对数概率

def prompt_mask_from_len(lengths, seq):
    seq_len, device = seq.shape[-1], seq.device
    return torch.arange(seq_len, device = device) < rearrange(lengths, '... -> ... 1')
# 定义 prompt_mask_from_len 函数，根据序列长度生成掩码

def set_dropout_(model: Module, prob: float):
    for module in model.modules():
        if isinstance(module, Dropout):
            module.p = prob
# 定义 set_dropout_ 函数，设置模型中的 Dropout 层的概率

# main class

class SPIN(Module):
    def __init__(
        self,
        model: Module,
        *,
        λ = 0.1,
        pad_id: Optional[int] = None,
        ref_model_ema_decay = 1.,
        ema_kwargs: dict = dict()
    ):
        super().__init__()
        self.policy_model = model

        self.ref_model = EMA(
            model,
            beta = ref_model_ema_decay,
            **ema_kwargs
        )
        # 初始化 SPIN 类，包括策略模型、参考模型和参数

        self.λ = λ
        self.pad_id = pad_id
        # 设置 λ 和 pad_id 属性

    def update_reference_model_with_policy(self):
        self.ref_model.copy_params_from_model_to_ema()
    # 更新参考模型参数为策略模型参数

    def update_ema(self):
        self.ref_model.update()
    # 更新指数��动平均

    def parameters(self):
        return self.policy_model.parameters()
    # 返回策略模型的参数

    @property
    def device(self):
        return next(self.parameters()).device
    # 返回模型所在设备

    @autocast(enabled = False)
    def forward(
        self,
        generated_seq: TensorType['b', 'n', int],
        real_seq: TensorType['b', 'n', int],
        prompt_len: TensorType['b', int],
        generated_seq_mask: Optional[TensorType['b', 'n', bool]] = None,
        real_seq_mask: Optional[TensorType['b', 'n', bool]] = None
    # 设置策略模型为训练模式
    self.policy_model.train()

    """
    b - batch
    n - sequence length
    """

    # 根据提示长度和实际序列生成实际提示掩码和生成提示掩码
    real_prompt_mask = prompt_mask_from_len(prompt_len, real_seq)
    generated_prompt_mask = prompt_mask_from_len(prompt_len, generated_seq)

    """
    Equation 4.7 in https://arxiv.org/abs/2401.01335v1
    """

    # 如果存在填充 ID
    if exists(self.pad_id):
        # 确保生成序列掩码和实际序列掩码不存在
        assert not exists(generated_seq_mask)
        assert not exists(real_seq_mask)
        # 生成生成序列掩码并填充
        generated_seq_mask = generated_seq != self.pad_id
        generated_seq.masked_fill_(~generated_seq_mask, 0)

        # 生成实际序列掩码并填充
        real_seq_mask = real_seq != self.pad_id
        real_seq.masked_fill_(~real_seq_mask, 0)

    # 禁用梯度计算
    with torch.no_grad():
        # 设置参考模型为评估模式
        self.ref_model.eval()
        # 计算生成序列和实际序列的参考模型对数概率
        ref_generated_logprob = log_prob_from_model_and_seq(self.ref_model, generated_seq)
        ref_real_logprob = log_prob_from_model_and_seq(self.ref_model, real_seq)

    # 计算策略模型对生成序列和实际序列的对数概率
    policy_generated_logprob = log_prob_from_model_and_seq(self.policy_model, generated_seq)
    policy_real_logprob = log_prob_from_model_and_seq(self.policy_model, real_seq)

    # 对变长序列进行掩码平均值计算

    # 对生成序列和实际序列的策略模型对数概率和参考模型对数概率进行掩码平均值计算
    policy_generated_logprob, ref_generated_logprob = [masked_mean(seq, maybe_and_mask(generated_seq_mask, ~generated_prompt_mask)) for seq in (policy_generated_logprob, ref_generated_logprob)]
    policy_real_logprob, ref_real_logprob = [masked_mean(seq, maybe_and_mask(real_seq_mask, ~real_prompt_mask)) for seq in (policy_real_logprob, ref_real_logprob)]

    # 计算 SPIN 损失

    # 计算损失值
    losses = -F.logsigmoid(self.λ * ((policy_real_logprob - ref_real_logprob) - (policy_generated_logprob - ref_generated_logprob)))

    # 返回损失值的平均值
    return losses.mean()
class SPINTrainer(Module):
    # 定义 SPINTrainer 类，继承自 Module 类
    def __init__(
        self,
        model: Union[Module, SPIN],
        *,
        train_sft_dataset: Dataset,
        max_seq_len: int,
        valid_sft_dataset: Optional[Dataset] = None,
        valid_every = 100,
        accelerator: Optional[Accelerator] = None,
        accelerate_kwargs: dict = dict(),
        batch_size = 16,
        grad_accum_steps = 2,
        epochs = 2,
        start_learning_rate = 1e-6,
        end_learning_rate = 1e-7,
        learning_rate_num_decay_steps = 1000,
        dropout = 0.,
        weight_decay = 0.,
        adam_kwargs: dict = dict(),
        temperature = 0.7,
        filter_fn = top_p,
        filter_kwargs = dict(thres = 0.9),
        pad_id: int = -1,
        ref_model_ema_decay = 1.,
        checkpoint_every = None,
        checkpoint_folder = './spin-checkpoints',
        spin_kwargs: dict = dict(
            λ = 0.1,
        )
    ):
        # 初始化函数，接受多个参数
        super().__init__()

        self.accelerator = accelerator
        # 设置 accelerator 属性为传入的 accelerator 参数
        if not exists(self.accelerator):
            self.accelerator = Accelerator(**accelerate_kwargs)
            # 如果 accelerator 不存在，则根据 accelerate_kwargs 创建一个 Accelerator 对象

        if not isinstance(model, SPIN):
            model = SPIN(
                model,
                pad_id = pad_id,
                ref_model_ema_decay = ref_model_ema_decay,
                **spin_kwargs
            )
            # 如果 model 不是 SPIN 类型，则根据传入参数创建一个 SPIN 对象

        self.model = model
        self.dropout = dropout
        self.train_dataloader = DataLoader(train_sft_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
        # 设置模型、dropout 和训练数据加载器属性

        self.grad_accum_steps = grad_accum_steps
        self.num_train_steps = len(self.train_dataloader) // self.grad_accum_steps * epochs
        # 设置梯度累积步数和训练步数

        self.optimizer = adam_optimizer_with_linear_decay(
            model,
            start_learning_rate,
            end_learning_rate,
            num_decay_steps = learning_rate_num_decay_steps,
            accelerator = self.accelerator,
            weight_decay = weight_decay,
            adam_kwargs = adam_kwargs
        )
        # 使用 adam_optimizer_with_linear_decay 函数创建优化器

        (
            self.model,
            self.train_dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.train_dataloader
        )
        # 准备模型和训练数据加载器

        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        # 设置最大序列长度和 pad_id

        # sampling

        self.temperature = temperature
        self.filter_fn = filter_fn
        self.filter_kwargs = filter_kwargs
        # 设置采样相关参数

        # validation

        self.valid_dataloader = None
        self.valid_every = valid_every
        # 初始化验证数据加载器和验证频率

        if exists(valid_sft_dataset):
            self.valid_dataloader = DataLoader(valid_sft_dataset, batch_size = batch_size)
            # 如果存在验证数据集，则创建验证数据加载器

        # checkpointing

        self.should_checkpoint = exists(checkpoint_every)
        self.checkpoint_every = checkpoint_every
        # 设置是否需要检查点和检查点频率

        if self.should_checkpoint:
            self.checkpoint_folder = Path(checkpoint_folder)
            self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
            # 如果需要检查点，则创建检查点文件夹

        self.steps = 0
        # 初始化步数为 0

    @property
    def is_main(self):
        return self.accelerator.is_main_process
        # 返回是否为主进程的属性

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)
        # 返回解封装后的模型属性

    def print(self, *msg):
        self.accelerator.print(*msg)
        # 打印函数

    def log(self, **data):
        self.accelerator.log(data, step = self.steps)
        # 记录日志函数

    def wait(self):
        return self.accelerator.wait_for_everyone()
        # 等待所有进程函数

    def save(self, path: str, overwrite: bool = False):
        self.wait()

        if self.is_main:

            path = self.checkpoint_folder / path

            assert not path.exists() or overwrite, f'file already exists'

            pkg = dict(
                model = self.unwrapped_model.state_dict()
            )

            torch.save(pkg, str(path))
            # 保存模型函数

    def calc_spin_loss(
        self,
        real_seq: TensorType['b', 'n', int],
        prompt_len: TensorType['b', int]
        # 计算 SPIN 损失函数
    ):
        # 根据实际序列长度和掩码生成提示掩码
        prompt_mask = prompt_mask_from_len(prompt_len, real_seq)
        # 根据提示掩码拆分实际序列，得到提示列表
        prompts = real_seq[prompt_mask].split(prompt_len.tolist())

        # 使用策略模型生成序列
        generated_seqs = sample(
            self.unwrapped_model.policy_model,
            prompts = prompts,
            seq_len = self.max_seq_len,
            temperature = self.temperature,
            filter_fn = self.filter_fn,
            filter_kwargs = self.filter_kwargs,
            output_keep_prompt = True
        )

        # 计算 SPIN 损失
        spin_loss = self.model(
            real_seq = real_seq,
            generated_seq = generated_seqs,
            prompt_len = prompt_len
        )

        return spin_loss

    def forward(self, overwrite_checkpoints: bool = True):
        """
        Algorithm 1 - https://arxiv.org/abs/2401.01335v1
        """

        # 更新参考模型
        self.model.update_reference_model_with_policy()

        self.steps = 0

        # 设置模型的 dropout
        set_dropout_(self.model, self.dropout)

        # 创建训练数据加载器的迭代器
        train_dataloader_iter = cycle(self.train_dataloader)

        # 循环进行自我训练
        for _ in tqdm(range(self.num_train_steps), desc = 'spin fine-tuning'):

            self.model.train()
            # 遍历模型前向计算上下文
            for forward_context in model_forward_contexts(self.accelerator, self.model, self.grad_accum_steps):
                with forward_context():
                    # 从训练数据加载器中获取实际序列和提示长度
                    real_seq, prompt_len = next(train_dataloader_iter)

                    # 计算 SPIN 损失
                    train_loss = self.calc_spin_loss(real_seq, prompt_len)

                    # 反向传播
                    self.accelerator.backward(train_loss / self.grad_accum_steps)

            # 打印训练损失
            self.print(f'train spin loss: {train_loss.item():.3f}')
            self.log(loss = train_loss.item())

            # 更新优化器
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.steps += 1

            # 等待
            self.wait()

            # 更新指数移动平均模型
            self.unwrapped_model.update_ema()

            # 如果存在验证数据加载器且满足验证频率条件
            if exists(self.valid_dataloader) and not (self.valid_every % self.steps):
                self.wait()

                if self.is_main:
                    total_loss = 0.
                    total_batches = 0.

                    with torch.no_grad():
                        self.model.eval()

                        # 遍历验证数据加载器
                        for valid_seq, prompt_len in tqdm(self.valid_dataloader, desc = 'valid spin'):
                            batch = valid_seq.shape[0]
                            # 计算验证 SPIN 损失
                            valid_spin_loss = self.calc_spin_loss(valid_seq, prompt_len)

                            total_batches += batch
                            total_loss += valid_spin_loss * batch

                        valid_loss = total_loss / total_batches

                        # 打印验证损失
                        self.print(f'valid spin loss: {valid_loss.item():.3f}')
                        self.log(valid_spin_loss = valid_loss.item())

            # 如果需要保存检查点且满足检查点频率条件
            if self.should_checkpoint and not (self.checkpoint_every % self.steps):
                checkpoint_num = self.steps // self.checkpoint_every
                self.save(f'spin.ckpt.{checkpoint_num}.pt', overwrite = overwrite_checkpoints)

        self.print(f'self-play training complete')
```