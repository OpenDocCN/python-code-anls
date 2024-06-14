# `.\simcse\trainers.py`

```py
# 导入需要的库和模块
import collections  # 提供了额外的数据容器和工具
import inspect  # 提供了用于检查源代码的工具
import math  # 提供了数学函数
import sys  # 提供了与解释器交互的变量和函数
import os  # 提供了与操作系统交互的功能
import re  # 提供了正则表达式操作
import json  # 提供了 JSON 编码和解码器
import shutil  # 提供了高级文件操作功能
import time  # 提供了时间操作功能
import warnings  # 提供了警告处理功能
from pathlib import Path  # 提供了操作文件和目录路径的类
import importlib.util  # 提供了导入模块的工具
from packaging import version  # 提供了版本语义规范工具
from transformers import Trainer  # 导入 Transformers 库中的 Trainer 类
from transformers.modeling_utils import PreTrainedModel  # 导入预训练模型类
from transformers.training_args import ParallelMode, TrainingArguments  # 导入并行模式和训练参数类
from transformers.utils import logging  # 导入日志工具
from transformers.trainer_utils import (  # 导入训练器工具函数和类
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
    speed_metrics,
)
from transformers.file_utils import (  # 导入文件工具函数和常量
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_torch_tpu_available,
)
from transformers.trainer_callback import (  # 导入训练器回调相关类
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (  # 导入 PyTorch 训练工具函数
    reissue_pt_warnings,
)
from transformers.utils import logging  # 再次导入日志工具（可能是为了保证一致性）
from transformers.data.data_collator import (  # 导入数据集处理工具
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator,
)
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型注解相关的类和函数
from torch.utils.data.dataloader import DataLoader  # 导入 PyTorch 数据加载器
from torch.utils.data.dataset import Dataset  # 导入 PyTorch 数据集
from torch.utils.data.distributed import DistributedSampler  # 导入分布式数据采样器
from torch.utils.data.sampler import RandomSampler, SequentialSampler  # 导入随机和顺序数据采样器

if is_torch_tpu_available():  # 如果可用 TPU
    import torch_xla.core.xla_model as xm  # 导入 TPU 模型核心库
    import torch_xla.debug.metrics as met  # 导入 TPU 调试指标库
    import torch_xla.distributed.parallel_loader as pl  # 导入 TPU 并行加载器

if is_apex_available():  # 如果可用 Apex
    from apex import amp  # 导入 Apex 混合精度训练工具

if version.parse(torch.__version__) >= version.parse("1.6"):  # 如果 PyTorch 版本 >= 1.6
    _is_native_amp_available = True  # 设置本地混合精度训练可用标志为 True
    from torch.cuda.amp import autocast  # 导入自动混合精度支持

if is_datasets_available():  # 如果可用 datasets 库
    import datasets  # 导入 datasets 库

from transformers.trainer import _model_unwrap  # 导入模型解封装函数
from transformers.optimization import Adafactor, AdamW, get_scheduler  # 导入优化器和调度器
import copy  # 提供了浅复制和深复制操作
# 设置 SentEval 的路径
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# 导入 SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval  # 导入 SentEval 库
import numpy as np  # 导入 NumPy 库
from datetime import datetime  # 导入日期时间类
from filelock import FileLock  # 导入文件锁定工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

class CLTrainer(Trainer):  # 定义一个继承自 Trainer 类的 CLTrainer 类

    def evaluate(  # 定义评估函数
        self,
        eval_dataset: Optional[Dataset] = None,  # 评估数据集（可选）
        ignore_keys: Optional[List[str]] = None,  # 忽略的键（可选）
        metric_key_prefix: str = "eval",  # 指标键前缀
        eval_senteval_transfer: bool = False,  # 是否评估 SentEval 转移任务（默认为 False）
    ) -> Dict[str, float]:
        # 定义函数的输入参数和返回类型，返回一个字典，键为字符串，值为浮点数

        # SentEval 准备和批处理函数
        def prepare(params, samples):
            return

        # SentEval 批处理函数
        def batcher(params, batch):
            # 将批次中的句子列表转换为空格分隔的句子字符串
            sentences = [' '.join(s) for s in batch]
            # 使用自定义的 tokenizer 对批次进行编码，并返回 PyTorch 张量
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            # 将批次中的每个项移动到指定设备上
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            # 在无梯度计算环境中，使用模型生成输出，包括隐藏状态，并返回池化后的输出
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                pooler_output = outputs.pooler_output
            return pooler_output.cpu()

        # 设置 SentEval 的参数（快速模式）
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                            'tenacity': 3, 'epoch_size': 2}

        # 初始化 SentEval 引擎
        se = senteval.engine.SE(params, batcher, prepare)
        # 定义需要评估的任务列表
        tasks = ['STSBenchmark', 'SICKRelatedness']
        # 如果需要评估 SentEval 转移任务或者在参数中指定了评估转移任务，则扩展任务列表
        if eval_senteval_transfer or self.args.eval_transfer:
            tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
        # 将模型设置为评估模式
        self.model.eval()
        # 执行 SentEval 的评估，并获取结果
        results = se.eval(tasks)
        
        # 提取 STS-Benchmark 和 SICK-Relatedness 的 Spearman 相关系数
        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

        # 计算平均的 STS 评估值
        metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2} 
        # 如果需要评估 SentEval 转移任务或者在参数中指定了评估转移任务，则进一步处理
        if eval_senteval_transfer or self.args.eval_transfer:
            avg_transfer = 0
            # 遍历每个转移任务，计算平均准确率和记录每个任务的准确率
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                avg_transfer += results[task]['devacc']
                metrics['eval_{}'.format(task)] = results[task]['devacc']
            avg_transfer /= 7  # 计算平均转移任务准确率
            metrics['eval_avg_transfer'] = avg_transfer

        # 记录评估结果到日志
        self.log(metrics)
        # 返回评估的所有指标
        return metrics
```