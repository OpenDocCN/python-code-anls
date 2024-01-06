# `qlora\qlora.py`

```
# 导入 defaultdict 模块，用于创建默认字典
from collections import defaultdict
# 导入 copy 模块，用于复制对象
import copy
# 导入 json 模块，用于处理 JSON 数据
import json
# 导入 os 模块，用于操作系统相关功能
import os
# 导入 exists、join、isdir 函数，用于判断文件或目录是否存在，拼接路径，判断是否为目录
from os.path import exists, join, isdir
# 导入 dataclass、field 模块，用于创建数据类和字段
from dataclasses import dataclass, field
# 导入 sys 模块，用于访问与 Python 解释器和其环境有关的变量和函数
import sys
# 导入 Optional、Dict、Sequence 模块，用于类型提示
from typing import Optional, Dict, Sequence
# 导入 numpy 模块，用于科学计算
import numpy as np
# 导入 tqdm 模块，用于显示进度条
from tqdm import tqdm
# 导入 logging 模块，用于记录日志
import logging
# 导入 bitsandbytes 模块，自定义模块
import bitsandbytes as bnb
# 导入 pandas 模块，用于数据分析
import pandas as pd
# 导入 importlib 模块，用于动态加载模块
import importlib
# 导入 packaging、version 模块，用于处理版本号
from packaging import version
from packaging.version import parse
# 导入 torch 库
import torch
# 导入 transformers 库
import transformers
# 从 torch.nn.utils.rnn 中导入 pad_sequence 函数
from torch.nn.utils.rnn import pad_sequence
# 导入 argparse 库
import argparse
# 从 transformers 中导入 AutoTokenizer, AutoModelForCausalLM, set_seed, Seq2SeqTrainer, BitsAndBytesConfig, LlamaTokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer
)
# 从 datasets 中导入 load_dataset, Dataset
from datasets import load_dataset, Dataset
# 导入 evaluate 模块
import evaluate
# 从 peft 中导入 prepare_model_for_kbit_training, LoraConfig, get_peft_model
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
# 导入所需的模块和库
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
# 定义函数用于检查是否安装了 Intel Extension for PyTorch (IPEX)
def is_ipex_available():
    # 定义函数用于从版本号中获取主版本号和次版本号
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)
    # 获取当前安装的 torch 版本
    _torch_version = importlib.metadata.version("torch")
    # 如果未安装 Intel Extension for PyTorch (IPEX)，返回 False
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    # 获取安装的 Intel Extension for PyTorch (IPEX) 的版本号
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    # 如果未找到 Intel Extension for PyTorch (IPEX) 的版本号，返回 False
    except importlib.metadata.PackageNotFoundError:
        return False
    # 获取 torch 和 IPEX 的主版本号和次版本号
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
# 检查 PyTorch 主版本号和次版本号是否与 Intel Extension for PyTorch 的要求相符，如果不符则发出警告并返回 False
if torch_major_and_minor != ipex_major_and_minor:
    warnings.warn(
        f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
        f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
    )
    return False
# 如果符合要求则返回 True
return True

# 如果 CUDA 可用，则设置 PyTorch 后端的相关参数
if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

# 创建日志记录器
logger = logging.getLogger(__name__)

# 定义常量
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

# 定义数据类，用于存储模型参数
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
# 定义一个数据类 DataArguments，用于存储数据相关的参数
@dataclass
class DataArguments:
    # 定义验证数据集的大小，默认为 1024
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    # 定义最大训练样本数，默认为 None
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
        }
    )
```
In this code, a data class `DataArguments` is defined to store data-related parameters. It includes `eval_dataset_size` to define the size of the validation dataset and `max_train_samples` to define the maximum number of training samples. Each field is annotated with a default value and metadata to provide help information.
    # 最大评估样本数，用于调试目的或加快训练速度，如果设置了，则截断评估样本数
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    # 源序列的最大长度。序列将进行右填充（可能截断）。
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # 目标序列的最大长度。序列将进行右填充（可能截断）。
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # 数据集名称，默认为'alpaca'
    dataset: str = field(
        default='alpaca',
# 定义一个数据类，用于存储训练参数
@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    # 可选的缓存目录，用于存储训练数据
    cache_dir: Optional[str] = field(
        default=None
    )
    # 是否在训练时使用源文本
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    # MMLU 分割类型
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
# 定义一个可选的字符串类型的变量mmlu_dataset，设置默认值为'mmlu-fs'，并添加元数据说明
# 元数据说明：MMLU数据集的选项，可以选择`mmlu-zs`表示零样本学习，或者`mmlu-fs`表示少样本学习
mmlu_dataset: Optional[str] = field(
    default='mmlu-fs',
    metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
)

# 定义一个可选的布尔类型的变量do_mmlu_eval，设置默认值为False，并添加元数据说明
# 元数据说明：是否运行MMLU评估
do_mmlu_eval: Optional[bool] = field(
    default=False,
    metadata={"help": "Whether to run the MMLU evaluation."}
)

# 定义一个可选的整数类型的变量max_mmlu_samples，设置默认值为None，并添加元数据说明
# 元数据说明：如果设置了值，只对MMMLU数据集的`max_mmlu_samples`进行评估
max_mmlu_samples: Optional[int] = field(
    default=None,
    metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
)

# 定义一个整数类型的变量mmlu_source_max_len，设置默认值为2048，并添加元数据说明
# 元数据说明：MMLU的最大源序列长度
mmlu_source_max_len: int = field(
    default=2048,
    metadata={"help": "Maximum source sequence length for mmlu."}
)

# 定义一个布尔类型的变量full_finetune，设置默认值为False，并添加元数据说明
# 元数据说明：在不使用适配器的情况下对整个模型进行微调
full_finetune: bool = field(
    default=False,
    metadata={"help": "Finetune the entire model without adapters."}
)
    # 定义一个布尔类型的字段，表示是否使用8位的adam
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    # 定义一个布尔类型的字段，表示是否通过双重量化压缩量化统计数据
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    # 定义一个字符串类型的字段，表示要使用的量化数据类型，应该是`fp4`或`nf4`之一
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    # 定义一个整数类型的字段，表示要使用的位数
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    # 定义一个整数类型的字段，表示Lora R维度
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
# 定义一个名为 lora_alpha 的浮点型变量，设置默认值为 16，同时添加描述信息
lora_alpha: float = field(
    default=16,
    metadata={"help": " Lora alpha."}
)

# 定义一个名为 lora_dropout 的浮点型变量，设置默认值为 0.0，同时添加描述信息
lora_dropout: float = field(
    default=0.0,
    metadata={"help":"Lora dropout."}
)

# 定义一个名为 max_memory_MB 的整型变量，设置默认值为 80000，同时添加描述信息
max_memory_MB: int = field(
    default=80000,
    metadata={"help": "Free memory per gpu."}
)

# 定义一个名为 report_to 的字符串变量，设置默认值为 'none'，同时添加描述信息
report_to: str = field(
    default='none',
    metadata={"help": "To use wandb or something else for reporting."}
)

# 定义一个名为 output_dir 的字符串变量，设置默认值为 './output'，同时添加描述信息
output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})

# 定义一个名为 optim 的字符串变量，设置默认值为 'paged_adamw_32bit'，同时添加描述信息
optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})

# 定义一个名为 per_device_train_batch_size 的整型变量，设置默认值为 1，同时添加描述信息
per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})

# 定义一个名为 gradient_accumulation_steps 的整型变量，设置默认值为 16，同时添加描述信息
gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    # 定义最大优化器更新步数，默认为10000步
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    # 定义AdamW的L2权重衰减率，默认为0.0，如果需要正则化，则使用lora dropout代替
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    # 定义学习率，默认为0.0002
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    # 定义是否移除未使用的列，默认为False，需要使代码库正常工作
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    # 定义梯度裁剪的最大范数，默认为0.3，经过调整后对所有测试模型都表现良好
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    # 定义是否使用梯度检查点，默认为True，建议使用
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    # 定义是否进行训练，默认为True
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    # 定义学习率调度类型，默认为'constant'，常数稍好于余弦，对分析有优势
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    # 定义进行热身的步数比例，默认为0.03
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    # 定义记录损失的更新步骤频率，默认为10步
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    # 定义是否将序列按长度分组成批处理，默认为True，可以节省内存并显著加快训练速度
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    # 定义保存检查点的策略，默认为'steps'
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    # 定义保存模型的频率，默认为250步
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    # 定义在最老的检查点被覆盖之前要保存多少个检查点，默认为40个

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    # 更多超参数请查看上述链接
    # 定义最大生成新标记数，在评估或预测循环中使用predict_with_generate时
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    # 定义生成新标记的最小数量
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # 生成策略
    # 是否采样
    do_sample: Optional[bool] = field(default=False)
    # Beam搜索的数量
    num_beams: Optional[int] = field(default=1)
    # Beam搜索的组数
    num_beam_groups: Optional[int] = field(default=1)
    # 惩罚alpha值
    penalty_alpha: Optional[float] = field(default=None)
    # 是否使用缓存
    use_cache: Optional[bool] = field(default=True)

    # 对数概率操作的超参数
    # 温度
    temperature: Optional[float] = field(default=1.0)
    # top-k采样的k值
    top_k: Optional[int] = field(default=50)
# 定义了一些可选的参数，设置了默认值
top_p: Optional[float] = field(default=1.0)
typical_p: Optional[float] = field(default=1.0)
diversity_penalty: Optional[float] = field(default=0.0)
repetition_penalty: Optional[float] = field(default=1.0)
length_penalty: Optional[float] = field(default=1.0)
no_repeat_ngram_size: Optional[int] = field(default=0)

# 根据参数和模型找到所有线性模块的名称
def find_all_linear_names(args, model):
    # 根据参数的位数选择不同的线性模块类
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    # 创建一个空集合用于存储线性模块的名称
    lora_module_names = set()
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果模块属于选定的线性模块类，则将其名称加入集合
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # 如果集合中包含'lm_head'，则移除它（用于16位）
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    # 将集合转换为列表并返回
    return list(lora_module_names)
# 定义一个保存 PEFT 模型的回调函数，继承自 transformers.TrainerCallback 类
class SavePeftModelCallback(transformers.TrainerCallback):
    # 保存模型的方法，接受参数 args, state, kwargs
    def save_model(self, args, state, kwargs):
        # 打印保存 PEFT 检查点的信息
        print('Saving PEFT checkpoint...')
        # 如果存在最佳模型检查点，则将检查点文件夹设置为最佳模型检查点下的 adapter_model 文件夹
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        # 否则，将检查点文件夹设置为输出目录下的 PREFIX_CHECKPOINT_DIR-global_step 文件夹
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        # 设置 PEFT 模型路径为检查点文件夹下的 adapter_model 文件夹
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        # 调用模型对象的 save_pretrained 方法保存 PEFT 模型
        kwargs["model"].save_pretrained(peft_model_path)

        # 设置 PyTorch 模型路径为检查点文件夹下的 pytorch_model.bin 文件
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        # 如果 pytorch_model_path 路径存在，则删除该文件
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    # 在保存模型时调用 save_model 方法
    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        # 返回控制信号
        return control
# 在训练结束时执行的函数，接受参数args, state, control, kwargs
def on_train_end(self, args, state, control, **kwargs):
    # 定义一个函数touch，用于更新文件的访问和修改时间
    def touch(fname, times=None):
        # 以追加模式打开文件，如果文件不存在则创建
        with open(fname, 'a'):
            # 更新文件的访问和修改时间
            os.utime(fname, times)

    # 调用touch函数，更新输出目录下的'completed'文件的访问和修改时间
    touch(join(args.output_dir, 'completed'))
    # 调用save_model函数，保存模型
    self.save_model(args, state, kwargs)

# 获取加速模型的函数，接受参数args, checkpoint_dir
def get_accelerate_model(args, checkpoint_dir):

    # 如果有GPU可用，则获取GPU数量
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    # 如果有IPex可用并且XPU可用，则获取XPU数量
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    # 设置最大内存为args.max_memory_MB指定的大小
    max_memory = f'{args.max_memory_MB}MB'
    # 为每个GPU设置最大内存
    max_memory = {i: max_memory for i in range(n_gpus)}
    # 设置设备映射为"auto"
    device_map = "auto"

    # 如果在分布式环境中，需要设置设备映射和每个设备的最大内存
    # （缺少代码，需要补充）
# 如果环境变量中存在'LOCAL_RANK'，则将其转换为整数并赋值给local_rank
if os.environ.get('LOCAL_RANK') is not None:
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    # 创建一个空字符串对应local_rank的设备映射
    device_map = {'': local_rank}
    # 创建一个空字符串对应local_rank的最大内存映射
    max_memory = {'': max_memory[local_rank]}

# 如果args.full_finetune为真，则断言args.bits为16或32
if args.full_finetune: assert args.bits in [16, 32]

# 打印加载基础模型的信息
print(f'loading base model {args.model_name_or_path}...')
# 根据参数设置计算数据类型
compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
# 从预训练模型中加载自动回归语言模型
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
    load_in_4bit=args.bits == 4,
    load_in_8bit=args.bits == 8,
    device_map=device_map,
    max_memory=max_memory,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
# 设置模型量化的参数，包括量化类型、计算精度、是否使用双量化等
model.set_quantization_params(
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=args.double_quant,
    bnb_4bit_quant_type=args.quant_type,
),
# 设置模型的数据类型，根据参数选择使用float32、bfloat16或者float16
torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
# 设置是否信任远程代码
trust_remote_code=args.trust_remote_code,
# 设置是否使用授权令牌
use_auth_token=args.use_auth_token
)

# 如果计算精度为float16且量化位数为4
if compute_dtype == torch.float16 and args.bits == 4:
    # 如果GPU支持bfloat16
    if torch.cuda.is_bf16_supported():
        # 打印提示信息
        print('='*80)
        print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
        print('='*80)
        
# 如果计算精度为float16且Intel XPU可用
if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
    # 将计算精度切换为bfloat16
    compute_dtype = torch.bfloat16
    # 打印提示信息
    print('Intel XPU does not support float16 yet, so switching to bfloat16')
    # 设置模型的属性，使其支持模型并行
    setattr(model, 'model_parallel', True)
    # 设置模型的属性，使其支持并行化
    setattr(model, 'is_parallelizable', True)

    # 根据参数设置模型的 torch 数据类型
    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # 初始化 tokenizer，根据参数设置相关属性
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # 由于快速分词器存在问题，禁用快速分词器
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # 针对 HF 名称更改，需要指定 tokenizer 类型
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )
    # 如果 tokenizer 的填充标记为空，则调用 smart_tokenizer_and_embedding_resize 函数进行处理
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
    # 检查模型名称中是否包含'llama'，或者tokenizer是否为LlamaTokenizer的实例
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer可能没有设置正确的特殊标记。
        # 检查并添加缺失的特殊标记，以防止它们被解析为不同的标记。
        # 注意，这些特殊标记存在于词汇表中。
        # 还要注意，`model.config.pad_token_id`为0，对应于`<unk>`标记。
        print('Adding special tokens.')
        # 添加特殊标记到tokenizer中
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })
    
    # 如果不是完全微调模型
    if not args.full_finetune:
        # 准备模型以进行KBIT训练，使用梯度检查点
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # 如果不是完全微调模型
    if not args.full_finetune:
        # 如果存在检查点目录
        if checkpoint_dir is not None:
# 打印信息，表示正在从检查点加载适配器
print("Loading adapters from checkpoint.")
# 从预训练模型中加载适配器模型，如果is_trainable为True，则适配器模型可训练
model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
# 如果没有从检查点加载适配器，则添加LoRA模块
else:
    print(f'adding LoRA modules...')
    # 查找所有线性模块的名称
    modules = find_all_linear_names(args, model)
    # 配置LoRA模块的参数
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # 获取PEFT模型
    model = get_peft_model(model, config)

# 遍历模型的所有模块
for name, module in model.named_modules():
    # 如果模块是LoraLayer类型
    if isinstance(module, LoraLayer):
        # 如果使用bf16，则将模块转换为torch.bfloat16类型
        if args.bf16:
            module = module.to(torch.bfloat16)
    # 如果模块名称中包含'norm'
    if 'norm' in name:
# 将模块转换为 float32 类型
module = module.to(torch.float32)
# 检查模块名称中是否包含 'lm_head' 或 'embed_tokens'，并且模块具有 'weight' 属性
if 'lm_head' in name or 'embed_tokens' in name:
    if hasattr(module, 'weight'):
        # 如果参数 args.bf16 为真并且模块的权重数据类型为 float32，则将模块转换为 bfloat16 类型
        if args.bf16 and module.weight.dtype == torch.float32:
            module = module.to(torch.bfloat16)
# 返回模型和分词器
return model, tokenizer

# 打印模型中可训练参数的数量
def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    # 遍历模型中的命名参数
    for _, param in model.named_parameters():
        # 统计所有参数的数量
        all_param += param.numel()
        # 如果参数需要梯度，则统计可训练参数的数量
        if param.requires_grad:
            trainable_params += param.numel()
    # 如果参数 args.bits 的值为 4，则可训练参数数量除以 2
    if args.bits == 4: trainable_params /= 2
    # 打印可训练参数的数量
    print(
        f"trainable params: {trainable_params} || "
# 创建一个包含所有参数和可训练参数比例的字符串
f"all params: {all_param} || "
f"trainable: {100 * trainable_params / all_param}"
)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """调整分词器和嵌入大小。

    注意：这是未经优化的版本，可能会导致您的嵌入大小不能被64整除。
    """
    # 添加特殊标记到分词器中，并返回新添加的标记数量
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # 调整模型的分词嵌入大小
    model.resize_token_embeddings(len(tokenizer))
    
    # 如果有新的标记被添加
    if num_new_tokens > 0:
        # 获取输入嵌入数据
        input_embeddings_data = model.get_input_embeddings().weight.data
        # 获取输出嵌入数据
        output_embeddings_data = model.get_output_embeddings().weight.data
        # 计算输入嵌入的平均值，去掉最后 num_new_tokens 个元素
        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        # 计算输出嵌入的平均值，去掉最后 num_new_tokens 个元素
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        # 将输入嵌入的最后 num_new_tokens 个元素替换为平均值
        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        # 将输出嵌入的最后 num_new_tokens 个元素替换为平均值
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 提取元素
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # 标记化
        tokenized_sources_with_prompt = self.tokenizer(
# 使用tokenizer对源文本进行tokenization，设置最大长度，截断，不添加特殊token
tokenized_sources = self.tokenizer(
    sources,
    max_length=self.source_max_len,
    truncation=True,
    add_special_tokens=False,
)
# 使用tokenizer对目标文本进行tokenization，设置最大长度，截断，不添加特殊token
tokenized_targets = self.tokenizer(
    targets,
    max_length=self.target_max_len,
    truncation=True,
    add_special_tokens=False,
)
# 为因果语言模型构建输入和标签
input_ids = []  # 存储输入的tokenized_source和tokenized_target的id
labels = []  # 存储标签
for tokenized_source, tokenized_target in zip(
    tokenized_sources_with_prompt['input_ids'],  # 源文本的tokenized id
    tokenized_targets['input_ids']  # 目标文本的tokenized id
):
    if not self.predict_with_generate:  # 如果不是使用生成模式进行预测
        input_ids.append(torch.tensor(tokenized_source + tokenized_target))  # 将源文本和目标文本的tokenized id拼接并存储在input_ids中
# 如果不在源数据上训练，则将标签添加到标签列表中，使用IGNORE_INDEX填充源数据的长度，然后复制目标数据
if not self.train_on_source:
    labels.append(
        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
    )
# 如果在源数据上训练，则将标签添加到标签列表中，使用深拷贝的源数据和目标数据
else:
    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target))
# 如果存在非自然指令，则将源数据添加到输入数据列表中
if extract_reformulations:
    input_ids.append(torch.tensor(tokenized_source))
# 对输入数据进行填充
input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
# 对标签进行填充，如果不是使用生成模式则为None
labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
# 创建数据字典，包括输入数据和注意力掩码
data_dict = {
    'input_ids': input_ids,
    'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
}
# 如果存在标签，则将标签添加到数据字典中
if labels is not None:
    data_dict['labels'] = labels
# 返回数据字典
return data_dict

# 提取非自然指令数据的函数
def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    # 初始化一个包含输入和输出的字典
    out = {
        'input': [],
        'output': [],
    }
    # 遍历示例中的实例，将指令和输入添加到输出字典中
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    # 如果需要提取重述，将重述的指令和输入添加到输出字典中
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    # 返回输出字典
    return out

# 定义一个包含提示输入的字典
ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
# 定义了一个包含不同格式的提示信息的字典
ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

# 根据示例中的输入是否为空，选择相应的提示格式
def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    # 返回包含格式化后提示信息的字典
    return {'input': prompt_format.format(**example)}

# 根据数据集名称加载本地数据集
def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        # 从 JSON 文件加载数据集
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        # 从 CSV 文件加载数据集
# 从 CSV 文件中读取数据集，转换为 Dataset 对象
full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))

# 如果数据集文件名以 .tsv 结尾，使用制表符作为分隔符读取数据集
elif dataset_name.endswith('.tsv'):
    full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))

# 如果数据集格式不受支持，抛出 ValueError 异常
else:
    raise ValueError(f"Unsupported dataset format: {dataset_name}")

# 将完整数据集划分为训练集和测试集
split_dataset = full_dataset.train_test_split(test_size=0.1)
return split_dataset

# 创建数据模块，用于监督式微调
def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
    """
# 定义了一个函数load_data，用于加载指定数据集的数据
def load_data(dataset_name):
    # 如果数据集名称为'alpaca'，则加载"tatsu-lab/alpaca"数据集
    if dataset_name == 'alpaca':
        return load_dataset("tatsu-lab/alpaca")
    # 如果数据集名称为'alpaca-clean'，则加载"yahma/alpaca-cleaned"数据集
    elif dataset_name == 'alpaca-clean':
        return load_dataset("yahma/alpaca-cleaned")
    # 如果数据集名称为'chip2'，则加载"laion/OIG"数据集的'unified_chip2.jsonl'文件
    elif dataset_name == 'chip2':
        return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
# 如果数据集名称为'self-instruct'，则加载名为'self_instruct'的数据集
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
# 如果数据集名称为'hh-rlhf'，则加载名为'Anthropic/hh-rlhf'的数据集
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
# 如果数据集名称为'longform'，则加载名为'akoksal/LongForm'的数据集
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
# 如果数据集名称为'oasst1'，则加载名为'timdettmers/openassistant-guanaco'的数据集
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
# 如果数据集名称为'vicuna'，则抛出未实现的错误
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
# 如果数据集名称不在上述情况中
        else:
            # 如果数据集名称对应的文件存在
            if os.path.exists(dataset_name):
                try:
                    # 如果未指定数据集格式，则默认为"input-output"
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    # 加载本地数据集
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    # 如果加载数据集出错，则抛出数值错误
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                # 如果数据集名称对应的文件不存在，则抛出未实现的错误
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")
# 格式化数据集，根据指定的数据集格式进行处理
def format_dataset(dataset, dataset_format):
    # 如果数据集格式为'alpaca'、'alpaca-clean'，或者数据集格式为None且参数中的数据集为'alpaca'或'alpaca-clean'
    if (
        dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
        (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
    ):
        # 对数据集进行映射，调用extract_alpaca_dataset函数，移除'insturction'列
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
    # 如果数据集格式为'chip2'，或者数据集格式为None且参数中的数据集为'chip2'
    elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
        # 对数据集进行映射，使用lambda函数对数据进行处理，将'text'按照'<bot>: '分割，分别作为'input'和'output'
        dataset = dataset.map(lambda x: {
            'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
            'output': x['text'].split('\n<bot>: ')[1],
        })
    # 如果数据集格式为'self-instruct'，或者数据集格式为None且参数中的数据集为'self-instruct'
    elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
        # 对数据集进行重命名，将'prompt'列重命名为'input'，将'completion'列重命名为'output'
        for old, new in [["prompt", "input"], ["completion", "output"]]:
            dataset = dataset.rename_column(old, new)
    # 如果数据集格式为'hh-rlhf'，或者数据集格式为None且参数中的数据集为'hh-rlhf'
    elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
        # 对数据集进行映射，使用lambda函数对数据进行处理，将'input'设为空字符串，'output'设为'chosen'列的值
        dataset = dataset.map(lambda x: {
            'input': '',
            'output': x['chosen']
        })
# 如果数据集格式为'oasst1'或者参数中未指定数据集格式但数据集为'oasst1'，则将数据集映射为只包含'output'字段的字典
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        # 如果数据集格式为'input-output'，则保持不变
        elif dataset_format == 'input-output':
            # 保持数据集不变
            pass
        # 移除未使用的列
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        # 返回处理后的数据集
        return dataset

    # 加载数据集
    dataset = load_data(args.dataset)
    # 格式化数据集
    dataset = format_dataset(dataset, args.dataset_format)

    # 分割训练集/评估集，减小数据集大小
    if args.do_eval or args.do_predict:
# 如果数据集中包含'eval'，则将其赋值给eval_dataset
if 'eval' in dataset:
    eval_dataset = dataset['eval']
# 如果数据集中不包含'eval'，则根据参数`eval_dataset_size`将训练数据集划分为训练集和验证集
else:
    print('Splitting train dataset in train and validation according to `eval_dataset_size`')
    dataset = dataset["train"].train_test_split(
        test_size=args.eval_dataset_size, shuffle=True, seed=42
    )
    eval_dataset = dataset['test']
# 如果设置了最大验证样本数，并且验证集样本数超过最大值，则选择前args.max_eval_samples个样本
if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
    eval_dataset = eval_dataset.select(range(args.max_eval_samples))
# 如果设置了按长度分组，则对验证集进行处理
if args.group_by_length:
    eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

# 如果需要训练
if args.do_train:
    # 将训练数据集赋值给train_dataset
    train_dataset = dataset['train']
    # 如果设置了最大训练样本数，并且训练集样本数超过最大值，则选择前args.max_train_samples个样本
    if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    # 如果设置了按长度分组，则对训练集进行处理
    if args.group_by_length:
        train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

# 创建用于CausalLM的数据收集器
data_collator = DataCollatorForCausalLM(
# 返回一个包含训练、评估和预测数据集的字典，以及数据收集器
def get_datasets(tokenizer, args, train_dataset, eval_dataset):
    # 根据参数设置创建数据收集器
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    # 返回包含训练、评估和预测数据集的字典，以及数据收集器
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )

# 获取最近的检查点
def get_last_checkpoint(checkpoint_dir):
    # 如果检查点目录存在
    if isdir(checkpoint_dir):
        # 检查是否存在已完成的标记文件
        is_completed = exists(join(checkpoint_dir, 'completed'))
        # 如果已完成，则返回空值和True
        if is_completed: return None, True # already finished
        # 初始化最大步数为0
        max_step = 0
        # 遍历检查点目录下的文件
        for filename in os.listdir(checkpoint_dir):
            # 如果是目录并且以'checkpoint'开头
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
# 定义一个函数用于训练模型
def train():
    # 使用 HfArgumentParser 解析命令行参数，包括模型参数、数据参数、训练参数和生成参数
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    # 将生成参数转换为 transformers.GenerationConfig 对象
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    # 将所有参数整合到一个 argparse.Namespace 对象中
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    # 打印参数
    print(args)
    
    # 获取最后一个检查点的目录和训练是否已完成的标志
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
```
    # 如果训练已经完成，则打印提示信息
    if completed_training:
        print('Detected that training was already completed!')

    # 获取加速模型和分词器
    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    # 设置模型配置不使用缓存
    model.config.use_cache = False
    print('loaded model')

    # 设置随机种子
    set_seed(args.seed)

    # 根据分词器和参数创建数据模块
    data_module = make_data_module(tokenizer=tokenizer, args=args)
    
    # 创建序列到序列的训练器
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    # 如果不是完全微调，则执行回调
    if not args.full_finetune:
# 添加 SavePeftModelCallback 回调函数到训练器中
trainer.add_callback(SavePeftModelCallback)
# 如果需要进行 MMLU 评估
if args.do_mmlu_eval:
    # 如果使用的是 mmlu-zs 数据集
    if args.mmlu_dataset == 'mmlu-zs':
        # 加载 mmlu-zs 数据集的评估和测试数据
        mmlu_dataset = load_dataset("json", data_files={
            'eval': 'data/mmlu/zero_shot_mmlu_val.json',
            'test': 'data/mmlu/zero_shot_mmlu_test.json',
        })
        # 移除数据集中的 'subject' 列
        mmlu_dataset = mmlu_dataset.remove_columns('subject')
    # 如果使用的是 mmlu 或者 mmlu-fs 数据集
    elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
        # 加载 mmlu 或者 mmlu-fs 数据集的评估和测试数据
        mmlu_dataset = load_dataset("json", data_files={
            'eval': 'data/mmlu/five_shot_mmlu_val.json',
            'test': 'data/mmlu/five_shot_mmlu_test.json',
        })
        # 可能需要移除数据集中的 'subject' 列
        # mmlu_dataset = mmlu_dataset.remove_columns('subject')
    # 选择指定的数据集划分
    mmlu_dataset = mmlu_dataset[args.mmlu_split]
    # 如果设置了最大的 MMLU 样本数，则进行样本选择
    if args.max_mmlu_samples is not None:
        mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
    # 定义 ABCD 索引
    abcd_idx = [
        tokenizer("A", add_special_tokens=False).input_ids[0],
# 调用tokenizer函数，传入字符串"B"，禁用特殊标记，获取其input_ids列表中的第一个元素
tokenizer("B", add_special_tokens=False).input_ids[0],
# 调用tokenizer函数，传入字符串"C"，禁用特殊标记，获取其input_ids列表中的第一个元素
tokenizer("C", add_special_tokens=False).input_ids[0],
# 调用tokenizer函数，传入字符串"D"，禁用特殊标记，获取其input_ids列表中的第一个元素
tokenizer("D", add_special_tokens=False).input_ids[0],

# 加载名为"accuracy"的评估器
accuracy = evaluate.load("accuracy")

# 定义MMLUEvalCallback类，继承自transformers.TrainerCallback
class MMLUEvalCallback(transformers.TrainerCallback):
    # 在评估时触发的回调函数
    def on_evaluate(self, args, state, control, model, **kwargs):
        # 获取评估数据加载器
        data_loader = trainer.get_eval_dataloader(mmlu_dataset)
        # 获取训练器的数据整合器中的source_max_len属性
        source_max_len = trainer.data_collator.source_max_len
        # 将训练器的数据整合器中的source_max_len属性设置为args.mmlu_source_max_len
        trainer.data_collator.source_max_len = args.mmlu_source_max_len
        # 设置训练器的模型为评估模式
        trainer.model.eval()
        preds, refs = [], []
        loss_mmlu = 0
        # 遍历数据加载器中的每个批次
        for batch in tqdm(data_loader, total=len(data_loader)):
            # 调用训练器的prediction_step函数，传入模型和批次数据，获取损失、预测结果和标签
            (loss, logits, labels) = trainer.prediction_step(trainer.model, batch, prediction_loss_only=False)
            # 遍历logits列表
            for i, logit in enumerate(logits):
                # 找到标签中非-100的索引
                label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                # 获取logit中的特定位置的值
                logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                # 将logit_abcd的最大值的索引添加到preds列表中
                preds.append(torch.argmax(logit_abcd).item())
# 从标签中去除忽略索引，并将其转换为一维数组，然后取出每个元素的第一个值
labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
# 将标签转换为列表，并根据标签在 abcd_idx 中的索引添加到 refs 列表中
refs += [abcd_idx.index(label) for label in labels.tolist()]
# 将损失值加到 mmlu_loss 中
loss_mmlu += loss.item()
# 提取每个主题的结果
results = {'mmlu_loss':loss_mmlu/len(data_loader)}
# 获取数据集中的主题
subject = mmlu_dataset['subject']
# 创建一个包含每个主题的参考和预测的字典
subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
# 将每个主题的参考、预测和主题一起打包成元组，并添加到 subjects 字典中
for s,p,r in zip(subject, preds, refs):
    subjects[s]['preds'].append(p)
    subjects[s]['refs'].append(r)
# 创建一个空列表来存储每个主题的分数
subject_scores = []
# 计算每个主题的准确率，并将结果添加到 results 字典中
for subject in subjects:
    subject_score = accuracy.compute(
        references=subjects[subject]['refs'],
        predictions=subjects[subject]['preds']
    )['accuracy']
    results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
    subject_scores.append(subject_score)
# 计算所有主题的平均准确率，并将结果添加到 results 字典中
results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
# 将结果记录到训练器中
trainer.log(results)
    # 设置数据收集器的源最大长度
    trainer.data_collator.source_max_len = source_max_len

    # 添加 MMLU 评估回调函数
    trainer.add_callback(MMLUEvalCallback)

    # 在训练之前验证数据类型和参数数量
    print_trainable_parameters(args, model)
    dtypes = {}
    # 遍历模型的所有参数，获取其数据类型和数量
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    # 计算总参数数量
    for k, v in dtypes.items(): total+= v
    # 打印每种数据类型的参数数量占比
    for k, v in dtypes.items():
        print(k, v, v/total)

    # 初始化所有指标的字典，设置运行名称
    all_metrics = {"run_name": args.run_name}
    # 如果需要训练
    if args.do_train:
        # 打印训练信息
        logger.info("*** Train ***")
# 注意：HF 不支持适配器检查点的 `resume_from_checkpoint`。
# 当前适配器检查点可以按预期重新加载，但优化器/调度器状态不会。
# 训练模型并获取训练结果
train_result = trainer.train()
# 获取训练指标并记录
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
# 保存训练指标
trainer.save_metrics("train", metrics)
# 保存训练状态
trainer.save_state()
# 更新所有指标
all_metrics.update(metrics)

# 如果需要进行评估
if args.do_eval:
    logger.info("*** Evaluate ***")
    # 评估模型并获取评估指标
    metrics = trainer.evaluate(metric_key_prefix="eval")
    # 记录评估指标
    trainer.log_metrics("eval", metrics)
    # 保存评估指标
    trainer.save_metrics("eval", metrics)
    # 更新所有指标
    all_metrics.update(metrics)

# 如果需要进行预测
if args.do_predict:
    logger.info("*** Predict ***")
    # 对测试数据集进行预测并获取预测输出
    prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
    # 获取预测指标
    prediction_metrics = prediction_output.metrics
# 获取预测结果
predictions = prediction_output.predictions
# 将预测结果中值为-100的元素替换为tokenizer的填充标记的ID
predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
# 将预测结果解码为文本
predictions = tokenizer.batch_decode(
    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
# 打开文件，将预测结果写入jsonl文件
with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
    # 遍历数据集中的每个样本
    for i, example in enumerate(data_module['predict_dataset']):
        # 将预测结果添加到样本中
        example['prediction_with_input'] = predictions[i].strip()
        example['prediction'] = predictions[i].replace(example['input'], '').strip()
        # 将样本以json格式写入文件
        fout.write(json.dumps(example) + '\n')
# 打印预测指标
print(prediction_metrics)
# 记录预测指标
trainer.log_metrics("predict", prediction_metrics)
# 保存预测指标
trainer.save_metrics("predict", prediction_metrics)
# 更新所有指标
all_metrics.update(prediction_metrics)

# 如果需要训练、评估或预测
if (args.do_train or args.do_eval or args.do_predict):
    # 打开文件，将所有指标写入metrics.json文件
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
        fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
# 调用名为train的函数，开始执行训练操作。
```