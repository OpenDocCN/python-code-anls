# `qlora\qlora.py`

```py
# 导入所需的库
from collections import defaultdict  # 导入 defaultdict 类
import copy  # 导入 copy 模块
import json  # 导入 json 模块
import os  # 导入 os 模块
from os.path import exists, join, isdir  # 从 os.path 模块中导入 exists, join, isdir 函数
from dataclasses import dataclass, field  # 从 dataclasses 模块中导入 dataclass, field 装饰器
import sys  # 导入 sys 模块
from typing import Optional, Dict, Sequence  # 从 typing 模块中导入 Optional, Dict, Sequence 类型
import numpy as np  # 导入 numpy 库并重命名为 np
from tqdm import tqdm  # 导入 tqdm 模块
import logging  # 导入 logging 模块
import bitsandbytes as bnb  # 导入 bitsandbytes 模块并重命名为 bnb
import pandas as pd  # 导入 pandas 库并重命名为 pd
import importlib  # 导入 importlib 模块
from packaging import version  # 从 packaging 模块中导入 version 类
from packaging.version import parse  # 从 packaging.version 模块中导入 parse 函数

import torch  # 导入 torch 库
import transformers  # 导入 transformers 库
from torch.nn.utils.rnn import pad_sequence  # 从 torch.nn.utils.rnn 模块中导入 pad_sequence 函数
import argparse  # 导入 argparse 模块
from transformers import (  # 从 transformers 模块中导入以下类和函数
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer
)
from datasets import load_dataset, Dataset  # 从 datasets 模块中导入 load_dataset, Dataset 类
import evaluate  # 导入 evaluate 模块

from peft import (  # 从 peft 模块中导入以下函数和类
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer  # 从 peft.tuners.lora 模块中导入 LoraLayer 类
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR  # 从 transformers.trainer_utils 模块中导入 PREFIX_CHECKPOINT_DIR 常量

# 定义函数用于检查是否安装了 Intel PyTorch 扩展
def is_ipex_available():
    # 定义函数用于从版本号中获取主版本号和次版本号
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    # 获取当前 torch 版本号
    _torch_version = importlib.metadata.version("torch")
    # 如果未安装 Intel PyTorch 扩展，则返回 False
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        # 获取 Intel PyTorch 扩展的版本号
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    # 获取 torch 和 Intel PyTorch 扩展的主版本号和次版本号
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    # 如果当前使用的 PyTorch 主版本号和次版本号与 Intel Extension for PyTorch 的不一致
    if torch_major_and_minor != ipex_major_and_minor:
        # 发出警告，提示用户当前的 Intel Extension for PyTorch 需要与特定版本的 PyTorch 兼容
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        # 返回 False，表示不兼容
        return False
    # 如果版本一致，返回 True，表示兼容
    return True
# 检查是否有可用的 CUDA 设备，如果有则设置允许使用 TF32 精度
if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

# 创建一个名为 logger 的日志记录器
logger = logging.getLogger(__name__)

# 定义一个常量，表示忽略的索引值
IGNORE_INDEX = -100
# 定义一个默认的填充标记
DEFAULT_PAD_TOKEN = "[PAD]"

# 定义一个名为 ModelArguments 的数据类，用于存储模型相关的参数
@dataclass
class ModelArguments:
    # 模型名称或路径，默认为 "EleutherAI/pythia-12b"
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    # 是否信任远程代码，默认为 False，帮助信息为 "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    # 是否使用 Huggingface 的认证令牌，默认为 False，帮助信息为 "Enables using Huggingface auth token from Git Credentials."
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

# 定义一个名为 DataArguments 的数据类，用于存储数据相关的参数
@dataclass
class DataArguments:
    # 验证数据集的大小，默认为 1024
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    # 最大训练样本数，默认为 None，用于调试目的或加快训练速度
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    # 最大评估样本数，默认为 None，用于调试目的或加快训练速度
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    # 最大源序列长度，默认为 1024，帮助信息为 "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # 最大目标序列长度，默认为 256，帮助信息为 "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # 数据集名称，默认为 'alpaca'，帮助信息为 "Which dataset to finetune on. See datamodule for options."
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    # 数据集格式，默认为 None，帮助信息为 "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )
# 定义一个名为 TrainingArguments 的类，继承自 transformers 库中的 Seq2SeqTrainingArguments 类
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    # 定义一个可选的缓存目录，默认为 None
    cache_dir: Optional[str] = field(
        default=None
    )
    # 定义一个可选的布尔类型变量，用于指示是否在训练时同时使用输入文本
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    # 定义一个可选的字符串类型变量，用于指定要运行的 MMLU 分割
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    # 定义一个可选的字符串类型变量，用于指定要使用的 MMLU 数据集
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    # 定义一个可选的布尔类型变量，用于指示是否运行 MMLU 评估
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    # 定义一个可选的整数类型变量，如果设置了值，则只对 MMMLU 数据集的 `max_mmlu_samples` 进行评估
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    # 定义一个整数类型变量，用于指定 MMLU 的最大源序列长度
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    # 定义一个布尔类型变量，用于指示是否在没有适配器的情况下对整个模型进行微调
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    # 定义一个布尔类型变量，用于指示是否使用 8 位的 adam 优化器
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    # 定义一个布尔类型变量，用于指示是否通过双重量化来压缩量化统计信息
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    # 定义一个字符串类型变量，用于指定要使用的量化数据类型
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    # 定义一个整数类型变量，用于指定要使用的位数
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    # 定义一个整数类型变量，用于指定 Lora 的 R 维度
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    # 定义一个浮点数类型变量，用于指定 Lora 的 alpha 值
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    # 定义一个浮点数类型变量，用于指定 Lora 的 dropout 值
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    # 设置最大内存限制，单位为 MB
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    # 设置报告输出方式，可以选择 wandb 或其他方式
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    # 设置输出目录，默认为当前目录下的 output 文件夹
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    # 设置优化器类型，默认为 paged_adamw_32bit
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    # 设置每个 GPU 的训练批次大小，默认为 1
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    # 设置梯度累积步数，默认为 16
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    # 设置最大更新步数，默认为 10000
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    # 设置权重衰减率，默认为 0.0
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    # 设置学习率，默认为 0.0002
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    # 设置是否移除未使用的列，默认为 False
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    # 设置梯度裁剪最大范数，默认为 0.3
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    # 设置是否使用梯度检查点，默认为 True
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    # 设置是否进行训练，默认为 True
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    # 设置学习率调度类型，默认为 constant
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    # 设置预热比例，默认为 0.03
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    # 定义日志步数，即每隔多少步更新后记录损失
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    # 是否按照长度将序列分组成相同长度的批次，可以节省内存并显著加快训练速度
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    # 保存策略，即何时保存检查点
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    # 保存模型的频率，即多久保存一次模型
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    # 总共保存的检查点数量限制，超过限制时会覆盖最旧的检查点
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
# 定义一个数据类 GenerationArguments，用于存储生成文本时的参数
@dataclass
class GenerationArguments:
    # 最大生成的新标记数
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    # 最小生成的新标记数
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # 生成策略
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # 对数概率调整的超参数
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

# 查找所有线性层的名称
def find_all_linear_names(args, model):
    # 根据参数的位数选择不同的线性层类
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果模块是指定的线性层类，则将模块的名称加入集合中
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # 如果 'lm_head' 在 lora_module_names 中，则移除它（用于16位）
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    # 返回线性层的名称列表
    return list(lora_module_names)

# 定义一个保存 Peft 模型的回调函数
class SavePeftModelCallback(transformers.TrainerCallback):
    # 保存模型的方法，接受参数、状态和关键字参数
    def save_model(self, args, state, kwargs):
        # 打印保存 PEFT 检查点的信息
        print('Saving PEFT checkpoint...')
        # 如果状态中有最佳模型检查点，则将检查点文件夹设置为最佳模型检查点下的 adapter_model 文件夹
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        # 否则将检查点文件夹设置为输出目录下的 PREFIX_CHECKPOINT_DIR-global_step 文件夹
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        # 设置 PEFT 模型路径为检查点文件夹下的 adapter_model 文件夹
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        # 保存模型到 PEFT 模型路径
        kwargs["model"].save_pretrained(peft_model_path)

        # 设置 PyTorch 模型路径为检查点文件夹下的 pytorch_model.bin 文件
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        # 如果 PyTorch 模型路径已存在，则删除该文件
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    # 在保存时调用 save_model 方法
    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        # 返回控制参数
        return control

    # 在训练结束时调用的方法
    def on_train_end(self, args, state, control, **kwargs):
        # 定义一个 touch 方法，用于更新文件的访问和修改时间
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        # 更新输出目录下的 completed 文件的访问和修改时间
        touch(join(args.output_dir, 'completed'))
        # 保存模型，传入参数、状态和关键字参数
        self.save_model(args, state, kwargs)
def get_accelerate_model(args, checkpoint_dir):

    # 检查是否有可用的 CUDA 设备，获取 GPU 数量
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    # 如果同时支持 IPex 和 XPU，获取 XPU 设备数量
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    # 设置每个 GPU 的最大内存
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    # 设置设备映射为自动
    device_map = "auto"

    # 如果在分布式环境中，需要设置设备映射和每个设备的最大内存
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    # 如果进行完整微调，确保位数为16或32
    if args.full_finetune: assert args.bits in [16, 32]

    # 打印加载基础模型的信息
    print(f'loading base model {args.model_name_or_path}...')
    # 根据参数设置计算数据类型
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    # 从预训练模型中加载自动模型
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
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token
    )
    # 如果计算数据类型为torch.float16且位数为4，且CUDA支持bfloat16
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            # 打印提示信息，GPU支持bfloat16，可以使用--bf16参数加速训练
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)
    # 如果计算数据类型为 torch.float16 并且（Intel XPU 可用并且 torch.xpu 可用）
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        # 将计算数据类型设置为 torch.bfloat16
        compute_dtype = torch.bfloat16
        # 打印提示信息
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    # 设置 model 的 model_parallel 属性为 True
    setattr(model, 'model_parallel', True)
    # 设置 model 的 is_parallelizable 属性为 True
    setattr(model, 'is_parallelizable', True)

    # 设置 model.config.torch_dtype 为 torch.float32（如果 args.fp16 为 True），否则为 torch.bfloat16（如果 args.bf16 为 True），否则为 torch.float32
    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # 初始化 tokenizer，根据给定的模型名称或路径
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )
    # 如果 tokenizer 的 _pad_token 为 None
    if tokenizer._pad_token is None:
        # 调整 tokenizer 和 embedding，添加默认的 pad_token
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    # 如果模型名称或路径中包含 'llama' 或 tokenizer 是 LlamaTokenizer 的实例
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer 可能没有设置正确的特殊 token
        # 检查并添加缺失的特殊 token，以防止它们被解析为不同的 token
        # 注意这些特殊 token 存在于词汇表中
        # 注意 model.config.pad_token_id 为 0，对应于 `<unk>` token
        # 打印提示信息
        print('Adding special tokens.')
        # 添加特殊 token
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })
    # 如果不是完全微调模式，则准备模型以进行 KBIT 训练，可以选择是否使用梯度检查点
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # 如果不是完全微调模式
    if not args.full_finetune:
        # 如果存在检查点目录，则从检查点加载适配器
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        # 否则，添加 LoRA 模块
        else:
            print(f'adding LoRA modules...')
            # 查找所有线性层的名称
            modules = find_all_linear_names(args, model)
            # 配置 LoRA 模块
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            # 获取 PEFT 模型
            model = get_peft_model(model, config)

    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果是 LoraLayer 类型的模块
        if isinstance(module, LoraLayer):
            # 如果使用 bf16，则将模块转换为 torch.bfloat16 类型
            if args.bf16:
                module = module.to(torch.bfloat16)
        # 如果模块名称中包含 'norm'
        if 'norm' in name:
            # 将模块转换为 torch.float32 类型
            module = module.to(torch.float32)
        # 如果模块名称中包含 'lm_head' 或 'embed_tokens'
        if 'lm_head' in name or 'embed_tokens' in name:
            # 如果模块具有 'weight' 属性，并且使用 bf16 且权重类型为 torch.float32
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    # 将模块的权重转换为 torch.bfloat16 类型
                    module = module.to(torch.bfloat16)
    # 返回模型和分词器
    return model, tokenizer
# 打印模型中可训练参数的数量
def print_trainable_parameters(args, model):
    # 初始化可训练参数和所有参数的数量
    trainable_params = 0
    all_param = 0
    # 遍历模型中的参数
    for _, param in model.named_parameters():
        # 统计所有参数的数量
        all_param += param.numel()
        # 如果参数需要梯度更新，则统计可训练参数的数量
        if param.requires_grad:
            trainable_params += param.numel()
    # 如果参数的位数为4，则可训练参数数量除以2
    if args.bits == 4: trainable_params /= 2
    # 打印可训练参数数量、所有参数数量和可训练参数占比
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

# 调整分词器和嵌入大小
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # 添加特殊标记到分词器中
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # 调整模型的分词嵌入大小
    model.resize_token_embeddings(len(tokenizer))
    
    # 如果添加了新的特殊标记
    if num_new_tokens > 0:
        # 获取输入和输出的嵌入数据
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        # 计算输入和输出嵌入的平均值
        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        # 更新输入和输出嵌入数据
        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

# 用于CausalLM数据收集的数据收集器
@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

# 提取非自然指令数据
def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    # 初始化输入和输出数据
    out = {
        'input': [],
        'output': [],
    }
    # 遍历示例数据中的实例列表
    for example_instances in examples['instances']:
        # 遍历每个实例
        for instance in example_instances:
            # 将实例的指令和输入添加到输出字典中
            out['input'].append(instance['instruction_with_input'])
            # 将实例的输出添加到输出字典中
            out['output'].append(instance['output'])
    # 如果需要提取重组，继续处理
    if extract_reformulations:
        # 遍历示例数据中的重组列表
        for example_reformulations in examples['reformulations']:
            # 如果重组列表不为空
            if example_reformulations is not None:
                # 遍历每个重组实例
                for instance in example_reformulations:
                    # 将重组实例的指令和输入添加到输出字典中
                    out['input'].append(instance['instruction_with_input'])
                    # 将重组实例的输出添加到输出字典中
                    out['output'].append(instance['output'])
    # 返回输出字典
    return out
# 定义一个包含两个字符串的字典，用于存储提示信息的格式
ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

# 从示例中提取 Alpaca 数据集
def extract_alpaca_dataset(example):
    # 如果示例中包含输入，则使用带输入的提示格式
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    # 否则使用不带输入的提示格式
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    # 返回包含提示格式的示例字典
    return {'input': prompt_format.format(**example)}

# 本地数据集处理函数
def local_dataset(dataset_name):
    # 根据数据集文件名的后缀选择不同的数据集加载方式
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        # 如果数据集格式不受支持，则抛出异常
        raise ValueError(f"Unsupported dataset format: {dataset_name}")
    # 将完整数据集划分为训练集和测试集
    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

# 创建数据模块函数，用于监督式微调
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
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples
    """
    # 定义一个函数，用于加载指定数据集的数据
    def load_data(dataset_name):
        # 如果数据集名称为'alpaca'，则加载"tatsu-lab/alpaca"数据集
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        # 如果数据集名称为'alpaca-clean'，则加载"yahma/alpaca-cleaned"数据集
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        # 如果数据集名称为'chip2'，则加载"laion/OIG"数据集中的'unified_chip2.jsonl'文件
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        # 如果数据集名称为'self-instruct'，则加载"yizhongw/self_instruct"数据集中的'self_instruct'文件
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        # 如果数据集名称为'hh-rlhf'，则加载"Anthropic/hh-rlhf"数据集
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        # 如果数据集名称为'longform'，则加载"akoksal/LongForm"数据集
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        # 如果数据集名称为'oasst1'，则加载"timdettmers/openassistant-guanaco"数据集
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        # 如果数据集名称为'vicuna'，则抛出未实现的错误
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        # 如果数据集名称不在上述列表中
        else:
            # 如果本地存在该数据集
            if os.path.exists(dataset_name):
                try:
                    # 设置数据集格式为输入-输出，加载本地数据集
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                # 如果加载数据集出错
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            # 如果本地不存在该数据集
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")
    # 格式化数据集，根据指定的数据集格式进行处理
    def format_dataset(dataset, dataset_format):
        # 如果数据集格式是 'alpaca' 或 'alpaca-clean'，或者数据集格式为空且参数中的数据集是 'alpaca' 或 'alpaca-clean'
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            # 对数据集进行映射，提取 'instruction' 列并移除
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        # 如果数据集格式是 'chip2' 或者数据集格式为空且参数中的数据集是 'chip2'
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            # 对数据集进行映射，将 'text' 列按照指定规则拆分为 'input' 和 'output'
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        # 如果数据集格式是 'self-instruct' 或者数据集格式为空且参数中的数据集是 'self-instruct'
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            # 对数据集进行列重命名，将 'prompt' 列重命名为 'input'，将 'completion' 列重命名为 'output'
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        # 如果数据集格式是 'hh-rlhf' 或者数据集格式为空且参数中的数据集是 'hh-rlhf'
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            # 对数据集进行映射，将 'input' 置为空字符串，将 'output' 设置为 'chosen' 列的值
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        # 如果数据集格式是 'oasst1' 或者数据集格式为空且参数中的数据集是 'oasst1'
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            # 对数据集进行映射，将 'input' 置为空字符串，将 'output' 设置为 'text' 列的值
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        # 如果数据集格式是 'input-output'
        elif dataset_format == 'input-output':
            # 保持不变
            pass
        # 移除未使用的列
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        # 返回格式化后的数据集
        return dataset

     # 加载数据集
    dataset = load_data(args.dataset)
    # 格式化数据集
    dataset = format_dataset(dataset, args.dataset_format)

    # 分割训练集和评估集，减小数据集大小
    # 如果需要进行评估或预测
    if args.do_eval or args.do_predict:
        # 如果数据集中包含评估数据集
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        # 如果数据集中没有评估数据集，则将训练数据集按照指定的比例分割成训练集和验证集
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        # 如果设置了最大评估样本数，并且评估数据集的长度超过了最大评估样本数，则进行样本选择
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        # 如果设置了按长度分组，则对评估数据集进行映射操作，计算输入和输出的长度并添加到数据中
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    # 如果需要进行训练
    if args.do_train:
        # 获取训练数据集
        train_dataset = dataset['train']
        # 如果设置了最大训练样本数，并且训练数据集的长度超过了最大训练样本数，则进行样本选择
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        # 如果设置了按长度分组，则对训练数据集进行映射操作，计算输入和输出的长度并添加到数据中
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    # 创建数据收集器，用于训练和评估
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    # 返回包含训练数据集、评估数据集、预测数据集和数据收集器的字典
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )
# 获取最后一个检查点的路径和训练完成状态
def get_last_checkpoint(checkpoint_dir):
    # 检查检查点目录是否存在
    if isdir(checkpoint_dir):
        # 检查是否存在 'completed' 文件
        is_completed = exists(join(checkpoint_dir, 'completed'))
        # 如果已完成训练，则返回 None 和 True
        if is_completed: return None, True # already finished
        # 初始化最大步数为 0
        max_step = 0
        # 遍历检查点目录下的文件
        for filename in os.listdir(checkpoint_dir):
            # 如果是目录且以 'checkpoint' 开头
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                # 获取检查点的步数并更新最大步数
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        # 如果最大步数为 0，则返回 None 和 is_completed
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        # 更新检查点目录为最大步数对应的检查点目录
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        # 打印找到的先前检查点的路径
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        # 返回检查点目录和 is_completed
        return checkpoint_dir, is_completed # checkpoint found!
    # 如果检查点目录不存在，则返回 None 和 False
    return None, False # first training

# 训练模型
def train():
    # 使用 HfArgumentParser 解析命令行参数
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    # 解析参数并返回数据类
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    # 设置 generation_config
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    # 将参数转换为命名空间
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    # 打印参数
    print(args)
    
    # 获取最后一个检查点的路径和训练完成状态
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    # 如果训练已完成，则打印提示信息
    if completed_training:
        print('Detected that training was already completed!')

    # 获取加速模型和分词器
    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    # 设置模型配置的 use_cache 为 False
    model.config.use_cache = False
    # 打印加载的模型信息
    print('loaded model')
    # 设置随机种子
    set_seed(args.seed)

    # 创建数据模块
    data_module = make_data_module(tokenizer=tokenizer, args=args)
    
    # 创建 Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    # 回调函数
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    # 在训练之前验证数据类型和参数数量。
    print_trainable_parameters(args, model)
    # 创建一个空字典用于存储参数的数据类型和数量
    dtypes = {}
    # 遍历模型的所有参数
    for _, p in model.named_parameters():
        # 获取参数的数据类型
        dtype = p.dtype
        # 如果数据类型不在字典中，则添加并初始化数量为0
        if dtype not in dtypes: dtypes[dtype] = 0
        # 增加该数据类型的参数数量
        dtypes[dtype] += p.numel()
    # 初始化参数总数量
    total = 0
    # 计算所有数据类型的参数总数量
    for k, v in dtypes.items(): total+= v
    # 遍历所有数据类型，打印数据类型、参数数量和占总数量的比例
    for k, v in dtypes.items():
        print(k, v, v/total)

    # 创建一个包含运行名称的字典
    all_metrics = {"run_name": args.run_name}
    # 训练
    if args.do_train:
        logger.info("*** Train ***")
        # 注意：HF 不支持适配器检查点的 `resume_from_checkpoint`。
        # 目前适配器检查点可以按预期重新加载，但优化器/调度器状态不会。
        # 开始训练并获取训练结果的指标
        train_result = trainer.train()
        metrics = train_result.metrics
        # 记录训练指标
        trainer.log_metrics("train", metrics)
        # 保存训练指标
        trainer.save_metrics("train", metrics)
        # 保存训练状态
        trainer.save_state()
        # 更新所有指标字典
        all_metrics.update(metrics)
    # 评估
    if args.do_eval:
        logger.info("*** Evaluate ***")
        # 获取评估指标
        metrics = trainer.evaluate(metric_key_prefix="eval")
        # 记录评估指标
        trainer.log_metrics("eval", metrics)
        # 保存评估指标
        trainer.save_metrics("eval", metrics)
        # 更新所有指标字典
        all_metrics.update(metrics)
    # 预测
    # 如果需要进行预测
    if args.do_predict:
        # 打印日志信息
        logger.info("*** Predict ***")
        # 使用训练器进行预测，得到预测输出
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        # 获取预测指标
        prediction_metrics = prediction_output.metrics
        # 获取预测结果
        predictions = prediction_output.predictions
        # 将预测结果中为-100的值替换为tokenizer的pad_token_id
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        # 将预测结果解码为文本
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # 将预测结果写入到predictions.jsonl文件中
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        # 打印预测指标
        print(prediction_metrics)
        # 记录预测指标
        trainer.log_metrics("predict", prediction_metrics)
        # 保存预测指标
        trainer.save_metrics("predict", prediction_metrics)
        # 更新所有指标
        all_metrics.update(prediction_metrics)

    # 如果需要进行训练、评估或预测
    if (args.do_train or args.do_eval or args.do_predict):
        # 将所有指标写入到metrics.json文件中
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))
# 如果当前模块被直接执行，而不是被导入到其他模块中
if __name__ == "__main__":
    # 调用 train 函数，开始执行程序的训练过程
    train()
```