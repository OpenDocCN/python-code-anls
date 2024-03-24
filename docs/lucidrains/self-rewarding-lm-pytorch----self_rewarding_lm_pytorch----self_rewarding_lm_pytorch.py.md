# `.\lucidrains\self-rewarding-lm-pytorch\self_rewarding_lm_pytorch\self_rewarding_lm_pytorch.py`

```py
# 导入所需的库
import re
import sys
from functools import partial
from random import randrange
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field
from functools import wraps
from textwrap import dedent

# 导入类型提示相关的库
from beartype import beartype
from beartype.typing import Optional, Dict, List, Tuple, Union, Callable
from torchtyping import TensorType

# 导入 PyTorch 相关的库
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Dropout
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 导入 NumPy 相关的库
import numpy as np
from numpy.lib.format import open_memmap

# 导入自定义的模块
from self_rewarding_lm_pytorch.dpo import (
    DPO,
    DPODataset,
    DPOTrainer,
    EarlyStopper,
    set_dropout_,
    adam_optimizer_with_linear_decay
)

from self_rewarding_lm_pytorch.spin import (
    SPIN,
    SPINTrainer
)

from einops import rearrange, repeat

from accelerate import Accelerator

from pytorch_custom_utils.utils import pad_or_slice_to

from pytorch_custom_utils.accelerate_utils import (
    model_forward_contexts
)

from self_rewarding_lm_pytorch.sampling_utils import (
    sample,
    top_p,
    top_k
)

from self_rewarding_lm_pytorch.mocks import always

from tqdm import tqdm

# 如果系统是 32 位，则给出警告
if sys.maxsize <= (2 ** 32):
    print('you need to be on 64 bit system to use memmapped files of > 2GB')

# 基本模板引擎
import jinja2
jinja2_env = jinja2.Environment()

# 从 Jinja 模板中查找变量
def find_variables_from_jinja_template(template: str):
    from jinja2 import meta
    ast = jinja2_env.parse(template)
    return meta.find_undeclared_variables(ast)

# 辅助函数
# 判断变量是否存在
def exists(v):
    return v is not None

# 如果变量存在则返回变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 返回数组的第一个元素
def first(arr):
    return arr[0]

# 无限循环生成器
def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# 返回输入本身
def identity(t, *args, **kwargs):
    return t

# 根据长度生成掩码
def prompt_mask_from_len(length, seq):
    seq_len, device = seq.shape[-1], seq.device
    return torch.arange(seq_len, device=device) < rearrange(length, '... -> ... 1')

# 将输入转换为元组
def cast_tuple(t, length=1, validate=False):
    out = t if isinstance(t, tuple) else ((t,) * length)
    assert not validate or len(out) == length
    return out

# 转换输入数据类型的装饰器
def cast_input(cast_fn):
    def decorator(fn):
        @wraps(fn)
        def inner(t, *args, **kwargs):
            t = cast_fn(t)
            return fn(t, *args, **kwargs)
        return inner

    return decorator

# 转换输出数据类型的装饰器
def cast_output(cast_fn):
    def decorator(fn):
        @wraps(fn)
        def output(*args, **kwargs):
            out = fn(*args, **kwargs)
            out = cast_fn(out)
            return out
        return output

    return decorator

# 常量
# llm-as-judge prompt
# https://openreview.net/forum?id=uccHPGDlao

# 默认的评分模板
DEFAULT_LLM_AS_JUDGE_PROMPT = """
Review the user’s question and the corresponding response using the additive 5-point
scoring system described below. Points are accumulated based on the satisfaction of each
criterion:
- Add 1 point if the response is relevant and provides some information related to
the user’s inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s question,
but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user’s question in a
useful way, regardless of whether it seems to have been written by an AI Assistant or if it
has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective,
addressing the user’s question directly and comprehensively, and is well-organized and
helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s question
"""
# 定义默认的奖励正则表达式模板
DEFAULT_REWARD_REGEX_TEMPLATE = """
Score: {{ reward }}
"""

# 创建解析奖励函数，根据奖励正则表达式模板
def create_parse_reward_fn(reward_regex_template):
    # 确保奖励模板包含"score"变量
    assert find_variables_from_jinja_template(reward_regex_template) == {'reward'}, 'reward template must include "score" variable'
    # 渲染奖励正则表达式模板
    reward_regex_str = jinja2_env.from_string(reward_regex_template).render(reward = "([0-9\.]+)")

    # 解析奖励函数
    def parse_reward_fn(llm_response: str) -> float:
        # 使用正则表达式匹配奖励
        result = re.search(rf"{reward_regex_str}", llm_response)

        # 如果没有匹配结果或者没有分组
        if not exists(result) or result.groups == 0:
            return None

        # 如果匹配结果不是数字
        if not result.groups(1).isnumeric():
            return None

        # 返回解析后的奖励值
        return float(result.groups(1))

    return parse_reward_fn

# 奖励配置
@dataclass
class RewardConfig:
    prompt_template: str
    reward_regex_template: Optional[str] = None
    parse_reward: Optional[Callable[[str], Optional[float]]] = None
    template_fn: Optional[Callable[..., str]] = None
    auto_dedent: bool = True

    # 初始化函数
    def init(self):

        # 可能需要去除缩进
        if self.auto_dedent:
            self.prompt_template = dedent(self.prompt_template)

            # 如果奖励正则表达式模板存在，也需要去除缩进
            if exists(self.reward_regex_template):
                self.reward_regex_template = dedent(self.reward_regex_template)

        # 初始化用于渲染提示和响应模板的函数
        prompt_template = self.prompt_template
        assert find_variables_from_jinja_template(prompt_template) == {'prompt', 'response'}, 'template must include prompt and response templating variables'
        self.template_fn = jinja2_env.from_string(prompt_template).render

        # 如果没有传入解析奖励函数，则根据奖励正则表达式模板创建解析函数
        if not exists(self.parse_reward):
            assert exists(self.reward_regex_template), 'reward_regex_template must be given if parse_reward is not passed in'
            self.parse_reward = create_parse_reward_fn(self.reward_regex_template)

        return self

# 默认奖励提示配置
SELF_REWARD_PROMPT_CONFIG = dict(
    default = RewardConfig(
        prompt_template = DEFAULT_LLM_AS_JUDGE_PROMPT,
        reward_regex_template = DEFAULT_REWARD_REGEX_TEMPLATE
    )
)

# 默认的有效奖励对选择函数
default_is_valid_reward_pair = lambda preferred_reward, unpreferred_reward: (preferred_reward != unpreferred_reward).all()

# 默认的选择配对奖励函数
@beartype
def default_pick_paired_rewards_fn(rewards: Tensor):
    is_nan_mask = torch.isnan(rewards)
    rewards_max, rewards_min = rewards.clone(), rewards.clone()
    rewards_max[is_nan_mask] = -1e6
    rewards_min[is_nan_mask] = 1e6
    return torch.stack((rewards_max.argmax(dim = -1), rewards_min.argmin(dim = -1)))

# SFT训练器类
class SFTTrainer(Module):
    @beartype
    # 初始化模型训练器，设置各种参数
    def __init__(
        self,
        model: Module,
        *,
        accelerator: Accelerator,
        train_dataset: Union[List[Dataset], Dataset],
        valid_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        grad_accum_steps: int = 2,
        num_epochs: int = 3,
        start_learning_rate: float = 5.5e-6,
        end_learning_rate: float = 1.1e-6,
        learning_rate_num_decay_steps: Optional[int] = None,
        dropout: float = 0.,
        weight_decay: float = 0.,
        ignore_index: int = -1,
        adam_kwargs: dict = dict(),
        valid_every: int = 1
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置加速器和模型
        self.accelerator = accelerator
        self.model = model
        self.dropout = dropout

        self.num_epochs = num_epochs
        self.ignore_index = ignore_index

        # 如果训练数据集是列表，则将其合并为一个数据集
        if isinstance(train_dataset, list):
            train_dataset = ConcatDataset(train_dataset)

        # 创建训练数据加载器
        self.train_dataloader = DataLoader(train_dataset, batch_size = batch_size, drop_last = True, shuffle = True)

        # 计算总的训练步数
        self.num_train_steps = len(self.train_dataloader) // grad_accum_steps * num_epochs
        self.grad_accum_steps = grad_accum_steps

        # 准备模型和训练数据加载器
        (
            self.model,
            self.train_dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.train_dataloader
        )

        # 如果学习率衰减步数不存在，则设置为训练数据集长度的一半
        if not exists(learning_rate_num_decay_steps):
            learning_rate_num_decay_steps = len(train_dataset) // 2

        # 创建优化器
        self.optimizer = adam_optimizer_with_linear_decay(
            model,
            start_learning_rate,
            end_learning_rate,
            num_decay_steps = learning_rate_num_decay_steps,
            accelerator = accelerator,
            weight_decay = weight_decay,
            adam_kwargs = adam_kwargs
        )

        self.valid_every = valid_every

        self.valid_dataloader = None
        # 如果验证数据集存在，则创建验证数据加载器
        if exists(valid_dataset):
            self.valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size)

        self.steps = 0

    # 记录日志
    def log(self, **data):
        self.accelerator.log(data, step = self.steps)

    # 等待所有进程完成
    def wait(self):
        return self.accelerator.wait_for_everyone()

    # 计算交叉熵损失
    def get_cross_entropy_loss(
        self,
        seq: TensorType['batch', 'seq', int],
        prompt_len_or_mask: Union[
            TensorType['batch', int],
            TensorType['batch', 'seq', bool]
        ]
    ):
        # 根据输入的 prompt_len_or_mask 类型，生成 prompt_mask
        if prompt_len_or_mask.dtype == torch.long:
            prompt_mask = prompt_mask_from_len(prompt_len_or_mask, seq)
        else:
            prompt_mask = prompt_len_or_mask

        # 将输入序列和标签序列分开
        seq, labels = seq[:, :-1], seq[:, 1:]

        # 根据 prompt_mask 填充标签
        labels.masked_fill_(prompt_mask[:, 1:], self.ignore_index)

        # 获取模型的预测结果
        logits = self.model(seq)

        # 计算交叉熵损失
        return F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels,
            ignore_index = self.ignore_index
        )
    # 定义 forward 方法，用于模型的前向传播
    def forward(self):
        
        # 从训练数据加载器中创建一个循环迭代器
        train_dl_iter = cycle(self.train_dataloader)
        
        # 设置模型中的 dropout 层
        set_dropout_(self.model, self.dropout)
        
        # 循环执行训练步骤
        for _ in tqdm(range(self.num_train_steps), desc='sft fine-tuning'):
            self.model.train()
            
            # 遍历模型前向传播上下文
            for forward_context in model_forward_contexts(self.accelerator, self.model, self.grad_accum_steps):
                with forward_context():
                    # 从训练数据加载器中获取下一个序列和提示长度或掩码
                    seq, prompt_len_or_mask = next(train_dl_iter)
                    
                    # 计算交叉熵损失
                    loss = self.get_cross_entropy_loss(seq, prompt_len_or_mask)
                    
                    # 反向传播计算梯度
                    self.accelerator.backward(loss / self.grad_accum_steps)
            
            # 更新优化器参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 记录损失值
            self.log(loss=loss.item())
            
            # 更新步数
            self.steps += 1
            
            # 如果存在验证数据加载器且满足验证频率条件
            if exists(self.valid_dataloader) and not (step % self.valid_every):
                self.wait()
                
                # 如果是主进程
                if self.accelerator.is_main_process:
                    total_valid_loss = 0.
                    total_batches = 0.
                    
                    # 将模型设置为评估模式
                    self.model.eval()
                    
                    # 在无梯度计算的情况下进行验证
                    with torch.no_grad():
                        for seq, prompt_len_or_mask in self.valid_dataloader:
                            batch = seq.shape[0]
                            
                            # 计算验证集的交叉熵损失
                            loss = self.get_cross_entropy_loss(seq, prompt_len_or_mask)
                            
                            total_valid_loss += loss.item() * batch
                            total_batches += batch
                    
                    # 计算验证集的平均损失
                    valid_loss = total_valid_loss / total_batches
                    
                    # 记录验证集损失值
                    self.log(valid_loss=valid_loss)
# 定义一个 DPODatasetGenerator 类，用于生成奖励数据集

class DPODatasetGenerator(Module):
    # 初始化方法，接受以下参数
    @beartype
    def __init__(
        self,
        model: Module,  # 模型对象
        prompt_dataset: Dataset,  # 提示数据集
        num_preference_pairs: int,  # 偏好对数量
        accelerator: Accelerator,  # 加速器对象
        tokenizer_encode: Callable[[str], TensorType['seq', int]],  # 编码器函数
        tokenizer_decode: Callable[[TensorType['seq', int]], str],  # 解码器函数
        self_reward_model: Optional[Module] = None,  # 自我奖励模型，默认为 None
        batch_size: int = 16,  # 批处理大小，默认为 16
        num_candidate_responses: int = 4,  # 候选响应数量，默认为 4
        gen_temperature: float = 0.7,  # 生成温度，默认为 0.7
        gen_filter_fn = top_p,  # 生成过滤函数，默认为 top_p
        gen_filter_kwargs: dict = dict(thres = 0.9),  # 生成过滤函数的参数，默认为 {'thres': 0.9}
        eval_temperature: float = 0.7,  # 评估温度，默认为 0.7
        eval_filter_fn = top_p,  # 评估过滤函数，默认为 top_p
        eval_filter_kwargs: dict = dict(thres = 0.9),  # 评估过滤函数的参数，默认为 {'thres': 0.9}
        num_evals_to_average: int = 3,  # 平均评估次数，默认为 3
        *,
        reward_config: RewardConfig,  # 奖励配置对象
        reward_model: Optional[Module] = None,  # 奖励模型，默认为 None
        data_folder: str = './',  # 数据文件夹，默认为当前目录
        preference_seq_memmap_file: str = 'preference_seq.memmap.npy',  # 偏好序列内存映射文件名，默认为 'preference_seq.memmap.npy'
        prompt_len_memmap_file: str = 'prompt_len.memmap.npy',  # 提示长度内存映射文件名，默认为 'prompt_len.memmap.npy'
        self_reward_memmap_file: str = 'self_reward.memmap.npy',  # 自我奖励内存映射文件名，默认为 'self_reward.memmap.npy'
        preference_max_seq_len: int = 1024,  # 偏好最大序列长度，默认为 1024
        generate_reward_max_seq_len: int = 256,  # 生成奖励最大序列长度，默认为 256
        is_valid_reward: Callable[[float], bool] = lambda *args: True,  # 是否有效奖励的函数，默认为始终返回 True
        is_valid_reward_pair: Optional[Callable[[float, float], bool]] = None,  # 是否有效奖励对的函数，默认为 None
        pick_paired_rewards: Callable[[Tensor], Tensor] = default_pick_paired_rewards_fn,  # 选择配对奖励的函数，默认为 default_pick_paired_rewards_fn
        pad_id: int = -1  # 填充 ID，默认为 -1
    # 初始化函数，继承父类的初始化方法
    def __init__(
        self,
        model,
        num_candidate_responses,
        self_reward_model,
        reward_config,
        batch_size,
        prompt_dataset,
        gen_filter_fn,
        gen_filter_kwargs,
        gen_temperature,
        eval_filter_fn,
        eval_filter_kwargs,
        eval_temperature,
        tokenizer_encode,
        tokenizer_decode,
        num_evals_to_average,
        is_valid_reward,
        is_valid_reward_pair,
        pick_paired_rewards,
        reward_model,
        generate_reward_max_seq_len,
        num_preference_pairs,
        preference_max_seq_len,
        pad_id,
        data_folder,
        preference_seq_memmap_file,
        prompt_len_memmap_file,
        self_reward_memmap_file,
        accelerator
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化属性
        self.model = model
        self.num_candidate_responses = num_candidate_responses

        self.self_reward_model = default(self_reward_model, model)
        self.reward_config = reward_config.init()

        self.batch_size = batch_size
        self.prompt_dataset = prompt_dataset
        self.prompt_dataloader = DataLoader(prompt_dataset, batch_size = batch_size, shuffle = True)

        self.gen_filter_fn = gen_filter_fn
        self.gen_filter_kwargs = gen_filter_kwargs
        self.gen_temperature = gen_temperature

        self.eval_filter_fn = eval_filter_fn
        self.eval_filter_kwargs = eval_filter_kwargs
        self.eval_temperature = eval_temperature

        self.tokenizer_encode = cast_output(lambda t: t.long())(tokenizer_encode)
        self.tokenizer_decode = cast_input(lambda t: t.long() if torch.is_tensor(t) else [*map(int, t)])(tokenizer_decode)

        self.num_evals_to_average = num_evals_to_average

        # 逻辑用于采样奖励对并在将其添加到生成的偏好数据集之前进行验证

        self.is_valid_reward = is_valid_reward
        self.is_valid_reward_pair = default(is_valid_reward_pair, lambda *args: True)
        self.pick_paired_rewards = pick_paired_rewards

        # 准备外部奖励模型，如果传入的话

        self.has_external_reward_model = exists(reward_model)
        self.reward_model = reward_model

        # 形状和填充

        self.generate_reward_max_seq_len = generate_reward_max_seq_len

        self.num_preference_pairs = num_preference_pairs

        self.preference_max_seq_len = preference_max_seq_len

        self.pad_id = pad_id

        memmap_shape = (num_preference_pairs, 2, preference_max_seq_len)

        # 保存以便在最后返回 DPO 数据集的实例

        self.dpo_dataset_kwargs = dict(
            data_folder = data_folder,
            preference_seq_memmap_file = preference_seq_memmap_file,
            prompt_len_memmap_file = prompt_len_memmap_file
        )

        # npy 文件的 memmap

        self.data_folder_path = Path(data_folder)
        self.data_folder_path.mkdir(exist_ok = True, parents = True)

        self.preference_seq_memmap_path = self.data_folder_path / preference_seq_memmap_file
        self.prompt_len_memmap_path = self.data_folder_path / prompt_len_memmap_file
        self.self_reward_mmemap_path = self.data_folder_path / self_reward_memmap_file

        self.preference_seq_memmap = open_memmap(str(self.preference_seq_memmap_path), dtype = 'int', mode = 'w+', shape = memmap_shape)
        self.prompt_len_memmap = open_memmap(str(self.prompt_len_memmap_path), dtype = 'int', mode = 'w+', shape = (num_preference_pairs,))
        self.self_reward_memmap_file = open_memmap(str(self.self_reward_mmemap_path), dtype = 'float32', mode = 'w+', shape = (num_preference_pairs, 2))

        self.accelerator = accelerator

    # 返回加速器设备
    @property
    def device(self):
        return self.accelerator.device

    # 生成奖励
    def generate_reward(
        self,
        prompt: str,
        response: str
        ) -> Optional[float]:

        """
        main contribution of the paper is the logic in this function
        in paper, they sample it 3 times and then average
        """

        # 获取模型参数的设备信息
        device = next(self.model.parameters()).device

        # 获取奖励配置中的模板函数和解析奖励函数
        template_fn = self.reward_config.template_fn
        parse_reward = self.reward_config.parse_reward

        # 根据模板函数生成奖励提示字符串，并使用分词器编码为张量
        reward_prompt_str = template_fn(prompt=prompt, response=response)
        reward_prompt = self.tokenizer_encode(reward_prompt_str).to(device)

        # 复制奖励提示张量，重复次数为 self.num_evals_to_average
        reward_prompt = repeat(reward_prompt, 'n -> b n', b=self.num_evals_to_average)

        reward_prompt = reward_prompt.to(device)
        self_reward_model = self_reward_model.to(device)

        # 使用自我奖励模型生成奖励响应
        reward_responses = sample(
            self_reward_model,
            prompts=reward_prompt,
            seq_len=self.generate_reward_max_seq_len,
            temperature=self.eval_temperature,
            filter_fn=self.eval_filter_fn,
            filter_kwargs=self.eval_filter_kwargs
        )

        # 将奖励响应转换为字符串列表
        reward_responses_as_str: List[str] = [self.tokenizer_decode(resp[resp != self.pad_id].cpu()) for resp in reward_responses]
        
        # 解析奖励字符串列表，得到奖励值列表
        rewards: List[Optional[float]] = [parse_reward(resp_str) for resp_str in reward_responses_as_str]

        # 过滤掉不存在的奖励值
        rewards = [*filter(exists, rewards)] # for now, just filter out any failed responses

        # 如果奖励值列表为空，则返回 None
        if len(rewards) == 0:
            return None

        # 计算奖励值列表的平均值
        avg_reward = Tensor(rewards).mean().item()
        return avg_reward

    @torch.no_grad()
# 定义了一个类 FinetuneConfig，用于存储微调配置信息
class FinetuneConfig:
    pass

# 导入 partial 函数，用于创建带有默认参数的函数
default_dict = partial(field, default_factory = dict)

# 使用 dataclass 装饰器定义了一个类 SFTConfig，继承自 FinetuneConfig，用于存储自训练配置信息
@dataclass
class SFTConfig(FinetuneConfig):
    train_dataset: Union[Dataset, List[Dataset]]  # 训练数据集，可以是单个数据集或数据集列表
    valid_dataset: Optional[Dataset] = None  # 验证数据集，默认为 None
    dropout: float = 0.1  # dropout 概率，默认为 0.1
    trainer_kwargs: dict = default_dict()  # 训练器参数，默认为一个空字典

# 使用 dataclass 装饰器定义了一个类 SelfRewardDPOConfig，继承自 FinetuneConfig，用于存储自奖励 DPO 配置信息
@dataclass
class SelfRewardDPOConfig(FinetuneConfig):
    prompt_dataset: Dataset  # 提示数据集
    num_generated_preference_pairs: int  # 生成的偏好对数量
    dpo_beta: float = 0.1  # DPO beta 参数，默认为 0.1
    max_seq_len: int = 1024  # 最大序列长度，默认为 1024
    rewarding_model: Optional[Module] = None  # 奖励模型，默认为 None
    self_reward_config_keyname: str = 'default'  # 自奖励配置键名，默认为 'default'
    is_valid_reward: Callable[[float], bool] = lambda reward: reward >= 0  # 验证奖励是否有效的函数，默认为 lambda 函数
    is_valid_reward_pair: Callable[[Tensor, Tensor], bool] = default_is_valid_reward_pair  # 验证奖励对是否有效的函数
    pick_paired_rewards_fn: Callable[[Tensor], Tensor] = default_pick_paired_rewards_fn  # 选择配对奖励的函数
    dropout: float = 0.1  # dropout 概率，默认为 0.1
    early_stopper_eval_module: Optional[Module] = None  # 早停评估模块，默认为 None
    num_train_steps: Optional[Module] = None  # 训练步数，默认为 None
    num_candidate_responses: int = 4  # 候选响应数量，默认为 4
    num_sampled_reward_responses: int = 3  # 采样奖励响应数量，默认为 3
    gen_temperature: float = 0.7  # 生成温度，默认为 0.7
    gen_filter_fn: Callable = top_p  # 生成过滤函数，默认为 top_p
    gen_filter_kwargs: dict = default_dict()  # 生成过滤函数参数，默认为一个空字典
    eval_temperature: float = 0.7  # 评估温度，默认为 0.7
    eval_filter_fn: Callable = top_p  # 评估过滤函数，默认为 top_p
    eval_filter_kwargs: dict = default_dict()  # 评估过滤函数参数，默认为一个空字典
    trainer_kwargs: dict = field(default_factory = dict)  # 训练器参数，默认为一个空字典
    reward_generator_kwargs: dict = default_dict()  # 奖��生成器参数，默认为一个空字典

# 使用 dataclass 装饰器定义了一个类 ExternalRewardDPOConfig，继承自 FinetuneConfig，用于存储外部奖励 DPO 配置信息
@dataclass
class ExternalRewardDPOConfig(FinetuneConfig):
    reward_model: Module  # 奖励模型
    dpo_beta: float = 0.1  # DPO beta 参数，默认为 0.1
    max_seq_len: int = 1024  # 最大序列长度，默认为 1024
    gen_temperature: float = 0.7  # 生成温度，默认为 0.7
    gen_filter_fn: Callable = top_p  # 生成过滤函数，默认为 top_p
    gen_filter_kwargs: dict = default_dict()  # 生成过滤函数参数，默认为一个空字典
    dropout: float = 0.1  # dropout 概率，默认为 0.1
    trainer_kwargs: dict = default_dict()  # 训练器参数，默认为一个空字典
    reward_generator_kwargs: dict = default_dict()  # 奖励生成器参数，默认为一个空字典

# 使用 dataclass 装饰器定义了一个类 SelfPlayConfig，继承自 FinetuneConfig，用于存储自对弈配置信息
@dataclass
class SelfPlayConfig(FinetuneConfig):
    train_dataset: Dataset  # 训练数据集
    valid_dataset: Optional[Dataset] = None  # 验证数据集，默认为 None
    max_seq_len: int = 1024  # 最大序列长度，默认为 1024
    spin_λ: float = 0.1  # spin_λ 参数，默认为 0.1
    dropout: float = 0.1  # dropout 概率，默认为 0.1
    temperature: float = 0.7  # 温度，默认为 0.7
    filter_fn: Callable = top_p  # 过滤函数，默认为 top_p
    filter_kwargs: dict = default_dict()  # 过滤函数参数，默认为一个空字典
    trainer_kwargs: dict = default_dict()  # 训练器参数，默认为一个空字典
    spin_kwargs: dict =  default_dict()  # spin 参数，默认为一个空字典

# 定义了一个函数 create_default_paper_config，用于生成默认的论文配置信息
@beartype
def create_default_paper_config(
    *,
    train_sft_dataset: Union[Dataset, List[Dataset],  # 训练 SFT 数据集，可以是单个数据集或数据集列表
    self_reward_prompt_dataset: Union[Dataset, Tuple[Dataset, Dataset]],  # 自奖励提示数据集，可以是单个数据集或数据集元组
    valid_sft_dataset: Optional[Dataset] = None,  # 验证 SFT 数据集，默认为 None
    num_generated_preference_pairs = (3964, 6942),  # 生成的偏好对数量，默认为 (3964, 6942)
    early_stopper_eval_module: Optional[Module] = None,  # 早停评估模块，默认为 None
    dpo_num_train_steps: Optional[int] = None,  # DPO 训练步数，默认为 None
    sft_config: dict = dict(),  # SFT 配置信息，默认为一个空字典
    self_reward_config: dict = dict()  # 自奖励配置信息，默认为一个空字典
) -> List[FinetuneConfig]:  # 返回值为 FinetuneConfig 类型的列表

    prompt_dataset_iter1, prompt_dataset_iter2 = cast_tuple(self_reward_prompt_dataset, 2, validate = True)  # 将自奖励提示数据集转换为元组
    num_generated_iter1, num_generated_iter2 = num_generated_preference_pairs  # 解包生成的偏好对数量

    return [
        SFTConfig(
            train_dataset = train_sft_dataset,  # 训练 SFT 数据集
            valid_dataset = valid_sft_dataset,  # 验证 SFT 数据集
            **sft_config  # 其他 SFT 配置信息
        ),
        SelfRewardDPOConfig(
            num_generated_preference_pairs = num_generated_iter1,  # 生成的偏好对数量
            prompt_dataset = prompt_dataset_iter1,  # 提示数据集
            num_train_steps = dpo_num_train_steps,  # DPO 训练步数
            early_stopper_eval_module = early_stopper_eval_module,  # 早停评估模块
            **self_reward_config  # 其他自奖励配置信息
        ),
        SelfRewardDPOConfig(
            num_generated_preference_pairs = num_generated_iter2,  # 生成的偏好对数量
            prompt_dataset = prompt_dataset_iter2,  # 提示数据集
            num_train_steps = dpo_num_train_steps,  # DPO 训练步数
            early_stopper_eval_module = early_stopper_eval_module,  # 早停评估模块
            **self_reward_config  # 其他自奖励配置信息
        )
    ]

# 定义了一个类 SelfRewardingTrainer，继承自 Module，用于自奖励训练
class SelfRewardingTrainer(Module):
    @beartype  # 类型注解装饰器
    # 初始化方法，接受模型、微调配置、编码和解码函数等参数
    def __init__(
        self,
        model: Module,
        *,
        finetune_configs: Union[Dict, List[FinetuneConfig]],
        tokenizer_encode: Callable[[str], TensorType['seq', int]],
        tokenizer_decode: Callable[[TensorType['seq', int]], str],
        self_reward_prompt_config: Union[RewardConfig, Dict[str, RewardConfig]] = SELF_REWARD_PROMPT_CONFIG,
        pad_id: int = -1,
        checkpoints_folder: str = './checkpoints',
        accelerate_kwargs: dict = dict()
    # 获取未加速的模型
    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    # 打印方法，用于输出信息
    def print(self, *msg):
        self.accelerator.print(*msg)

    # 等待方法，等待所有进程完成
    def wait(self):
        return self.accelerator.wait_for_everyone()

    # 保存方法，保存模型参数到指定路径
    def save(self, path: str, overwrite: bool = False):
        self.wait()

        # 如果是主进程
        if self.accelerator.is_main_process:

            # 拼接保存路径
            path = self.checkpoints_folder / path

            # 如果文件已存在且不允许覆盖，则报错
            assert not path.exists() or overwrite, f'file already exists'

            # 封装模型参数并保存
            pkg = dict(
                model = self.unwrapped_model.state_dict()
            )

            torch.save(pkg, str(path))

    # 前向传播方法，用于微调训练
    def forward(
        self,
        overwrite_checkpoints: bool = False
    ):

        # 遍历训练器类型和训练器
        for ind, (trainer_type, trainer) in enumerate(self.trainers):
            finetuning_stage = ind + 1
            trainer()

            # 保存微调阶段的模型参数
            self.save(f'{finetuning_stage}.{trainer_type}.ckpt.pt', overwrite = overwrite_checkpoints)

        # 输出训练完成信息
        self.print(f'self-reward training done')
```