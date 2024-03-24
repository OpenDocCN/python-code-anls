# `.\lucidrains\CALM-pytorch\CALM_pytorch\CALM.py`

```
# 从 math 模块中导入 ceil 函数
from math import ceil
# 从 pathlib 模块中导入 Path 类
from pathlib import Path
# 从 functools 模块中导入 partial 函数
from functools import partial
# 从 contextlib 模块中导入 nullcontext 和 contextmanager 函数
from contextlib import nullcontext, contextmanager

# 从 dataclasses 模块中导入 dataclass 装饰器
from dataclasses import dataclass

# 导入 torch 库
import torch
# 从 torch.nn.functional 模块中导入 F 别名
import torch.nn.functional as F
# 从 torch.nn 模块中导入 Module 和 ModuleList 类
from torch.nn import Module, ModuleList
# 从 torch.utils.data 模块中导入 Dataset 和 DataLoader 类
from torch.utils.data import Dataset, DataLoader
# 从 torch.optim.lr_scheduler 模块中导入 _LRScheduler 类
from torch.optim.lr_scheduler import _LRScheduler
# 从 torch 模块中导入 nn、einsum 和 Tensor 类
from torch import nn, einsum, Tensor

# 导入 beartype 库
from beartype import beartype
from beartype.door import is_bearable
# 从 beartype.typing 模块中导入 List、Optional、Callable、Type、Tuple、Union、Literal 类型
from beartype.typing import List, Optional, Callable, Type, Tuple, Union, Literal

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 从 x_transformers.x_transformers 模块中导入 RMSNorm、Attention 和 TransformerWrapper 类
from x_transformers.x_transformers import (
    RMSNorm,
    Attention,
    TransformerWrapper,
)

# 导入 accelerate 库
from accelerate import Accelerator

# 从 pytorch_custom_utils 模块中导入 OptimizerWithWarmupSchedule、get_adam_optimizer 和 auto_unwrap_model 函数
from pytorch_custom_utils import (
    OptimizerWithWarmupSchedule,
    get_adam_optimizer,
    auto_unwrap_model
)

# 从 pytorch_custom_utils.accelerate_utils 模块中导入 model_forward_contexts 函数
from pytorch_custom_utils.accelerate_utils import (
    model_forward_contexts
)

# 从 CALM_pytorch.sampling_utils 模块中导入 sample、top_p 和 top_k 函数

# types

# 定义 Sequence 类型为 Tuple 或 List
Sequence = Union[Tuple, List]

# 定义 HiddenPosition 类型为 'input' 或 'output'
HiddenPosition = Union[Literal['input'], Literal['output']]

# 定义 SequenceOf 函数，接受类型参数 t，返回 Tuple[t, ...] 或 List[t]
def SequenceOf(t):
    return Union[Tuple[t, ...], List[t]]

# 定义 SingularOrMany 函数，接受类型参数 t，返回 t 或 SequenceOf(t)
def SingularOrMany(t):
    return Union[t, SequenceOf(t)]

# helpers

# 定义 exists 函数，判断变量是否存在
def exists(v):
  return v is not None

# 定义 default 函数，返回第一个参���或默认值
def default(v, d):
    return v if exists(v) else d

# 定义 xnor 函数，实现逻辑异或操作
def xnor(x, y):
    return not (x ^ y)

# 定义 cast_tuple 函数，将参数转换为元组
def cast_tuple(t, length = 1):
    return t if is_bearable(t, Sequence) else ((t,) * length)

# 定义 get_block_output_from_hook_outputs 函数，从钩子输出中获取模块输出
def get_block_output_from_hook_outputs(
    hidden_position: HiddenPosition,
    _, inp, out
):
    maybe_tensor = out if hidden_position == 'output' else inp

    if isinstance(maybe_tensor, tuple):
        maybe_tensor = maybe_tensor[0]

    assert torch.is_tensor(maybe_tensor)
    return maybe_tensor

# freezing llms

# 定义 set_module_requires_grad_ 函数，设置模块参数是否需要梯度
@beartype
def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

# 定义 freeze_all_layers_ 函数，冻结所有层的参数
def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

# function for returning an ordered list of modules, where the output of the module is the output of that transformer block layer
# ex. for x-transformers TransformerWrapper

# 定义 x_transformer_blocks 函数，返回 TransformerWrapper 中每个 transformer block 的模块列表
@beartype
def x_transformer_blocks(transformer: TransformerWrapper) -> List[Module]:
    blocks = []
    for layer in transformer.attn_layers.layers:
        blocks.append(layer[-1])
    return blocks[1::2]

# helper classes

# 定义 Recorder 类
class Recorder:
    # Recorder 类的构造函数
    @beartype
    def __init__(
        self,
        outputs: Optional[List] = None,
        forward_hook_get_hidden: HiddenPosition = 'output',
        modules: Optional[List] = None,
    ):
        self.output = default(outputs, [])
        self.modules = modules
        self.get_output_fn = partial(get_block_output_from_hook_outputs, forward_hook_get_hidden)

    # Recorder 类的调用函数
    def __call__(self, *args):

        if exists(self.modules):
            self.modules.append(args[0])

        hidden = self.get_output_fn(*args)
        self.output.append(hidden.detach())

# 定义 ExtractHiddensWrapper 类
class ExtractHiddensWrapper(Module):
    # ExtractHiddensWrapper 类的构造函数
    @beartype
    def __init__(
        self,
        model: Module,
        blocks: List[Module],
        hidden_positions: SingularOrMany(HiddenPosition) = 'output'
    ):
        super().__init__()
        hidden_positions = cast_tuple(hidden_positions, len(blocks))
        assert len(hidden_positions) == len(blocks)

        self.model = model

        self.outputs = []
        self.modules = []
        self.recorders = []

        for block, hidden_position in zip(blocks, hidden_positions):
            recorder = Recorder(self.outputs, hidden_position, self.modules)
            self.recorders.append(recorder)
            block.register_forward_hook(recorder)
    # 定义一个方法用于前向传播，接受任意参数和关键字参数，可以选择是否返回被挂钩的模块
    def forward(self, *args, return_hooked_modules = False, **kwargs):
        # 调用模型的前向传播方法，传入参数和关键字参数
        self.model(*args, **kwargs)

        # 复制输出和模块字典
        outputs = self.outputs.copy()
        modules = self.modules.copy()

        # 清空输出和模块字典
        self.outputs.clear()
        self.modules.clear()

        # 如果不需要返回被挂钩的模块，则返回输出字典
        if not return_hooked_modules:
            return outputs

        # 如果需要返回被挂钩的模块，则同时返回输出字典和模块字典
        return outputs, modules
# 定义交叉注意力块类
class CrossAttentionBlock(Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        dim,
        dim_context,
        linear_project_context = True,  # 在论文中，他们对增强隐藏状态进行了投影。不确定是否需要，但最好先准确
        pre_rmsnorm = False,
        forward_hook_get_hidden: Union[
            Literal['output'],
            Literal['input']
        ] = 'output',
        **kwargs
    ):
        super().__init__()
        # 如果需要预先进行 RMS 归一化，则创建 RMSNorm 对象
        self.pre_rmsnorm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        self.context_proj = None

        self.dim = dim
        self.dim_context = dim_context

        # 如果需要线性投影上下文，则创建线性层对象
        if linear_project_context:
            self.context_proj = nn.Linear(dim_context, dim)
            dim_context = dim

        # 创建注意力对象
        self.attn = Attention(
            dim = dim,
            dim_context = dim_context,
            zero_init_output = True,
            gate_value_heads = True,
            **kwargs
        )

        self.context = None
        self.context_mask = None
        self.forward_hook_get_hidden = forward_hook_get_hidden

    # 设置掩码
    def set_mask(self, mask: Tensor):
        self.context_mask = mask

    # 取消掩码
    def unset_mask(self):
        self.context_mask = None

    # 前向传播函数
    def forward(self, *hook_args):
        x = get_block_output_from_hook_outputs(self.forward_hook_get_hidden, *hook_args)

        context = self.context
        assert exists(context)

        maybe_enable_grad = torch.enable_grad if self.training else nullcontext

        with maybe_enable_grad():
            res = x
            x = self.pre_rmsnorm(x)

            if exists(self.context_proj):
                context = self.context_proj(context)

            out = self.attn(x, context, context_mask = self.context_mask) + res

        return out

# 主类
@dataclass
class AugmentParams:
    model: Module
    hidden_position: SingularOrMany(HiddenPosition) = 'output'
    transformer_blocks: Optional[List[Module]] = None
    extract_blocks_fn: Optional[Callable[[Module], List[Module]]] = None
    model_return_hiddens: bool = False
    input_shape: Optional[Tuple[int, ...]] = None
    connections: Optional[Tuple[Tuple[int, int], ...]] = None
    connect_every_num_layers: int = 4 # 在论文中，他们做了 4 层
    mask_kwarg: Optional[str] = None

# CALM 类
class CALM(Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        anchor_llm: Module,
        augment_llms: SingularOrMany(AugmentParams),
        *,
        attn_kwargs: dict = dict(
            linear_project_context = True,
            pre_rmsnorm = True,
            flash = True
        ),
        anchor_extract_blocks_fn: Callable[[Module], List[Module]] = None,
        anchor_transformer_blocks: Optional[List[Module]] = None,
        anchor_hidden_position: SingularOrMany(HiddenPosition) = 'output',
        pad_id: int = -1
    def state_dict(self):
        return self.cross_attns.state_dict()

    def load_state_dict(self, pkg, strict = False):
        self.cross_attns.load_state_dict(pkg, strict = strict)

    def parameters(self):
        return self.cross_attns.parameters()

    def release_cross_attn_contexts(self):
        for one_augment_cross_attns in self.cross_attns:
            for cross_attn in one_augment_cross_attns:
                cross_attn.context = None

    def forward_augments(
        self,
        prompt: Tensor,
        prompt_mask: Optional[SingularOrMany(SequenceOf(Tensor))] = None
    ):
        # 如果只提供一个提示并且有多个增强LLM，则将该提示输入到所有增强LLM中

        num_augment_llms = len(self.augment_llms)

        prompts = cast_tuple(prompt, num_augment_llms)

        assert len(prompts) == num_augment_llms

        # 提示掩码

        if not exists(prompt_mask):
            prompt_mask = tuple((p != self.pad_id if not torch.is_floating_point(p) else None) for p in prompts)

        prompt_mask = cast_tuple(prompt_mask, num_augment_llms)

        prompt_masks = prompt_mask # 在这一点上，应该是复数

        assert len(prompt_masks) == num_augment_llms

        # 调用增强LLM，使用前向钩子收集隐藏状态

        augments_hiddens = []

        with torch.no_grad():

            self.augment_llms.eval()

            for augment_llm, params, prompt, prompt_mask in zip(self.augment_llms, self.augment_llms_params, prompts, prompt_masks):
                augment_llm_kwarg = dict()

                if exists(params.mask_kwarg):
                    augment_llm_kwarg = {params.mask_kwarg: prompt_mask}

                one_augment_hiddens = augment_llm(prompt, **augment_llm_kwarg)

                augments_hiddens.append(one_augment_hiddens)

        # 为锚点前向设置每个交叉注意力块的上下文

        for one_augment_hiddens, one_augment_cross_attns, one_augment_connections in zip(augments_hiddens, self.cross_attns, self.connections):

            for (augment_layer_index, _), cross_attn in zip(one_augment_connections, one_augment_cross_attns):
            
                cross_attn.context = one_augment_hiddens[augment_layer_index - 1]

        return prompts, prompt_masks

    @contextmanager
    def set_cross_attn_masks(self, masks):
        # 为交叉注意力设置上下文掩码

        for one_cross_attn, mask in zip(self.cross_attns, masks):
            for cross_attn in one_cross_attn:
                cross_attn.set_mask(mask)

        yield

        # 取消设置上下文掩码

        for one_cross_attn in self.cross_attns:
            for cross_attn in one_cross_attn:
                cross_attn.unset_mask()


    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        seq_len: int,
        prompt_mask: Optional[SingularOrMany(SequenceOf(Tensor))] = None,
        filter_fn: Callable = top_p,
        filter_kwargs: dict = dict(
            thres = 0.9
        )
    ):
        batch, device = prompt.shape[0], next(self.cross_attns.parameters()).device

        self.eval()

        # 在所有增强模型上运行前向并收集隐藏状态

        prompts, prompt_masks = self.forward_augments(prompt = prompt, prompt_mask = prompt_mask)

        with self.set_cross_attn_masks(prompt_masks):

            # 采样

            generated =  sample(
                self.anchor_llm,
                prompt,
                seq_len = seq_len,
                filter_fn = filter_fn,
                filter_kwargs = filter_kwargs
            )

            self.release_cross_attn_contexts()

        return generated

    @beartype
    def forward(
        self,
        seq: Tensor,
        *,
        prompt: SingularOrMany(Tensor),
        prompt_mask: Optional[SingularOrMany(Tensor)] = None,
        mask: Optional[Tensor] = None,
        return_loss = True,
        anchor_llm_in_train_mode = True  # 对此不确定
        ):
        # 如果需要返回损失值，则将交叉注意力模型设置为训练模式
        if return_loss:
            self.cross_attns.train()

            # 如果锚定语言模型需要在训练模式下，则设置为训练模式，否则设置为评估模式
            if anchor_llm_in_train_mode:
                self.anchor_llm.train()
            else:
                self.anchor_llm.eval()

            # 将序列截断，去掉最后一个字符，用于输入和标签
            seq, labels = seq[:, :-1], seq[:, 1:]

        # 在所有数据增强模型上运行前向传播，并收集隐藏状态

        prompts, prompt_masks = self.forward_augments(prompt=prompt, prompt_mask=prompt_mask)

        # 设置交叉注意力模型的掩码
        with self.set_cross_attn_masks(prompt_masks):
            # 调用锚定语言模型，该模型应该处理与增强语言模型隐藏状态的交叉注意力

            logits = self.anchor_llm(seq)

            # 释放交叉注意力上下文
            self.release_cross_attn_contexts()

            # 断言锚定语言模型返回的 logits 维度应为 (batch, seq, num tokens)
            assert logits.ndim == 3, 'anchor llm should return logits in the shape (batch, seq, num tokens)'

        # 返回用于解码的 logits

        if not return_loss:
            return logits

        # 考虑提示掩码

        if exists(mask):
            # 如果存在掩码，则使用掩码填充标签
            labels = labels.masked_fill(~mask[:, 1:], self.pad_id)

        # 用于微调

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index=self.pad_id
        )

        return loss
# 定义一个循环生成器，用于循环遍历数据加载器中的批次数据
def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# 使用装饰器自动解包模型
@auto_unwrap_model()
class FineTuner:

    # 初始化方法，接收多个参数
    @beartype
    def __init__(
        self,
        calm: CALM,
        *,
        num_train_steps: int,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        dataset: Dataset,
        data_kwarg_names: Tuple[str, ...] = ('seq', 'mask', 'prompt'),
        accelerate_kwargs: dict = dict(),
        checkpoint_every: int = 1000,
        checkpoint_path: str = './checkpoints',
        scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        warmup_steps: int = 1000,
        max_grad_norm = 0.5,
        grad_accum_steps = 1
    ):
        # 初始化加速器
        self.accelerator = Accelerator(**accelerate_kwargs)

        # 创建数据加载器
        self.dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)
        self.data_kwarg_names = data_kwarg_names

        # 设置模型
        self.model = calm

        # 创建 Adam 优化器
        adam = get_adam_optimizer(
            calm.parameters(),
            lr = learning_rate,
            wd = weight_decay
        )

        # 初始化优化器和学习率调度器
        self.optimizer = OptimizerWithWarmupSchedule(
            accelerator = self.accelerator,
            optimizer = adam,
            scheduler = scheduler,
            scheduler_kwargs = scheduler_kwargs,
            warmup_steps = warmup_steps,
            max_grad_norm = max_grad_norm
        )

        self.step = 0
        self.num_train_steps = num_train_steps
        self.grad_accum_steps = grad_accum_steps

        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.mkdir(exist_ok = True, parents = True)

    # 判断当前进程是否为主进程
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 打印信息
    def print(self, msg):
        self.accelerator.print(msg)

    # 保存模型和优化器状态
    def save(self, filename: str, overwrite: bool = True):
        path = self.checkpoint_path / filename
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            step = self.step
        )

        torch.save(pkg, str(path))

    # 加载模型和优化器状态
    def load(self, filename: str):
        path = self.checkpoint_path / filename
        assert path.exists()

        pkg = torch.load(str(path))

        self.model.load_state_dict(pkg['model'])
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.step = pkg['step']

    # 定义 FineTuner 类的调用方法
    def __call__(self, forward_kwargs: dict = dict()):
        dl_iter = cycle(self.dl)
        self.model.train()

        for step in range(self.step, self.num_train_steps):

            for context in model_forward_contexts(
                model = self.model,
                accelerator = self.accelerator,
                grad_accum_steps = self.grad_accum_steps
            ):
                with context():
                    data = next(dl_iter)

                    if not isinstance(data, dict):
                        data = dict(zip(self.data_kwarg_names, data))

                    loss = self.model(**data, **forward_kwargs)

                    self.accelerator.backward(loss / self.grad_accum_steps)

            self.print(f'{step + 1}: {loss.item():.3f}')

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.step += 1

            self.accelerator.wait_for_everyone()

            if self.is_main and not (self.step % self.checkpoint_every):
                num = self.step // self.checkpoint_every
                self.save(f'checkpoint.{num}.pt')

            self.accelerator.wait_for_everyone()

        self.print('training complete')
        self.save('checkpoint.-1.pt')
```