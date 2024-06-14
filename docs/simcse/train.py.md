# `.\train.py`

```py
# 导入必要的模块和库
import logging  # 导入日志模块
import math  # 导入数学模块
import os  # 导入操作系统模块
import sys  # 导入系统模块
from dataclasses import dataclass, field  # 导入数据类相关模块
from typing import Optional, Union, List, Dict, Tuple  # 导入类型提示相关模块
import torch  # 导入PyTorch深度学习框架
import collections  # 导入集合模块
import random  # 导入随机数模块

from datasets import load_dataset  # 从datasets库中导入数据集加载函数

import transformers  # 导入transformers库
from transformers import (
    CONFIG_MAPPING,  # 导入配置映射
    MODEL_FOR_MASKED_LM_MAPPING,  # 导入掩码语言模型映射
    AutoConfig,  # 导入自动配置类
    AutoModelForMaskedLM,  # 导入自动掩码语言模型类
    AutoModelForSequenceClassification,  # 导入自动序列分类模型类
    AutoTokenizer,  # 导入自动分词器类
    DataCollatorForLanguageModeling,  # 导入语言建模数据收集器类
    DataCollatorWithPadding,  # 导入带填充的数据收集器类
    HfArgumentParser,  # 导入Hugging Face参数解析器
    Trainer,  # 导入训练器类
    TrainingArguments,  # 导入训练参数类
    default_data_collator,  # 导入默认数据收集器函数
    set_seed,  # 导入设置随机种子函数
    EvalPrediction,  # 导入评估预测类
    BertModel,  # 导入BERT模型类
    BertForPreTraining,  # 导入BERT预训练模型类
    RobertaModel  # 导入RoBERTa模型类
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase  # 导入分词器基础类相关模块
from transformers.trainer_utils import is_main_process  # 导入是否主进程函数
from transformers.data.data_collator import DataCollatorForLanguageModeling  # 导入语言建模数据收集器类
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available  # 导入文件工具相关函数
from simcse.models import RobertaForCL, BertForCL  # 从simcse模块中导入RoBERTa和BERT模型
from simcse.trainers import CLTrainer  # 从simcse模块中导入对比学习训练器类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())  # 获取掩码语言模型映射的所有配置类列表
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)  # 获取所有配置类对应的模型类型元组

@dataclass
class ModelArguments:
    """
    与我们将要微调或从头训练的模型/配置/分词器相关的参数。
    """

    # Huggingface原始参数
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    # 定义温度参数，用于 softmax 操作

    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    )
    # 定义池化器类型，决定使用哪种池化方式来处理输入的特征

    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    # 定义硬负样本权重的对数值（仅在使用硬负样本时有效）

    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    # 是否启用 MLM 辅助目标任务

    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    # 定义 MLM 辅助目标任务的权重（仅在启用 MLM 时有效）

    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )
    # 是否仅在训练阶段使用 MLP（多层感知机）
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    def __post_init__(self):
        # Check if either dataset_name or train_file is provided; both cannot be None.
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                # Ensure that the provided train_file has a valid extension (csv, json, txt).
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    @cached_property
    @torch_required
    # 设置设备环境，返回 torch.device 对象
    def _setup_devices(self) -> "torch.device":
        # 记录日志信息，指示 PyTorch 正在设置设备
        logger.info("PyTorch: setting up devices")
        # 如果禁用 CUDA，则使用 CPU 设备
        if self.no_cuda:
            device = torch.device("cpu")
            # 设置 GPU 数量为 0
            self._n_gpu = 0
        # 如果有 TPU 设备可用
        elif is_torch_tpu_available():
            # 导入 torch_xla 库，获取 TPU 设备
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            # 设置 GPU 数量为 0
            self._n_gpu = 0
        # 如果未指定本地 GPU 的索引
        elif self.local_rank == -1:
            # 如果存在多个 GPU，则使用 nn.DataParallel 进行并行计算。
            # 如果只想使用特定的一些 GPU，可以设置环境变量 CUDA_VISIBLE_DEVICES=0
            # 明确地将 CUDA 设备设置为第一个（索引为 0）CUDA 设备，否则 `set_device` 会触发缺少设备索引的错误。
            # 索引 0 考虑了环境中可用的 GPU，所以 `CUDA_VISIBLE_DEVICES=1,2` 与 `cuda:0` 将使用环境中的第一个 GPU，即 GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # 有时在此之前尚未运行 postinit 中的行，所以只需检查我们不是在默认值。
            self._n_gpu = torch.cuda.device_count()
        # 如果指定了本地 GPU 的索引
        else:
            # 在这里，我们将使用 torch.distributed 进行分布式计算。
            # 初始化分布式后端，它将负责同步节点/GPU
            #
            # deepspeed 在内部执行自己的 DDP，并要求使用以下命令启动程序：
            # deepspeed  ./program.py
            # 而不是：
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                # 检查 deepspeed 是否可用
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    # 如果 deepspeed 不可用，则引发 ImportError
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                # 初始化 deepspeed 的分布式环境
                deepspeed.init_distributed()
            else:
                # 初始化分布式进程组
                torch.distributed.init_process_group(backend="nccl")
            # 指定设备为具有本地索引的 CUDA 设备
            device = torch.device("cuda", self.local_rank)
            # 设置 GPU 数量为 1
            self._n_gpu = 1

        # 如果设备类型为 CUDA
        if device.type == "cuda":
            # 设置当前 CUDA 设备
            torch.cuda.set_device(device)

        # 返回设备对象
        return device
def main():
    # 创建参数解析器，用于解析命令行参数
    # 可以在 src/transformers/training_args.py 中查看所有可能的参数
    # 或者通过传递 --help 标志给此脚本查看帮助信息。
    # 现在我们保持不同的参数集，以更清晰地分离关注点。

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    
    # 如果脚本只有一个参数且是一个 JSON 文件路径，
    # 则解析该文件以获取参数。
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # 否则，解析命令行参数到数据类中。
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 如果输出目录已经存在且不为空，并且指定了不覆盖的选项，则抛出异常。
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # 配置日志记录格式和级别
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # 在每个进程上记录小的摘要信息：
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    
    # 如果是主进程，则设置 Transformers 日志的详细信息级别。
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        
    # 记录训练/评估的参数信息。
    logger.info("Training/evaluation parameters %s", training_args)

    # 在初始化模型之前设置种子。
    set_seed(training_args.seed)

    # 获取数据集：可以提供自己的 CSV/JSON/TXT 训练和评估文件，或者提供公共数据集的名称
    # 数据集将从 https://huggingface.co/datasets/ 自动下载。
    #
    # 对于 CSV/JSON 文件，此脚本将使用名为 'text' 的列或第一列作为数据列。您可以轻松调整此行为。
    #
    # 在分布式训练中，load_dataset 函数保证只有一个本地进程可以同时下载数据集。
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    
    # 获取训练文件的扩展名，并根据扩展名设置相应的数据集标识符。
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    # 如果文件扩展名是 "csv"，则使用特定的参数加载数据集
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        # 否则，使用默认参数加载数据集
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")

    # 查看更多关于加载任何类型标准或自定义数据集（从文件、Python 字典、Pandas DataFrame 等）的信息，请访问：
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # 加载预训练模型和分词器
    #
    # 分布式训练：
    # .from_pretrained 方法确保只有一个本地进程能够同时下载模型和词汇表。
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        # 如果指定了配置名称，则从预训练模型库中加载配置
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        # 否则，根据模型名称或路径加载配置
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        # 如果两者都没有指定，则根据模型类型从预定义的配置映射中实例化配置对象
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        # 如果指定了分词器名称，则从预训练模型库中加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        # 否则，根据模型名称或路径加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        # 如果没有指定分词器名称，抛出数值错误，因为脚本不支持从头开始实例化分词器
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # 检查模型参数中是否指定了模型名称或路径
    if model_args.model_name_or_path:
        # 如果模型名称或路径中包含 'roberta' 字符串
        if 'roberta' in model_args.model_name_or_path:
            # 使用预训练的 RoBERTa 模型创建模型实例
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
        # 如果模型名称或路径中包含 'bert' 字符串
        elif 'bert' in model_args.model_name_or_path:
            # 使用预训练的 BERT 模型创建模型实例
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
            # 如果设置了 MLM (Masked Language Modeling) 的操作标志
            if model_args.do_mlm:
                # 使用预训练的 BERT 模型创建 MLM 任务的模型实例
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                # 将预训练模型的预测头加载到当前模型中
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else:
            # 如果既不是 RoBERTa 也不是 BERT 模型，则抛出未实现的异常
            raise NotImplementedError
    else:
        # 如果没有指定模型名称或路径，则抛出未实现的异常
        raise NotImplementedError

    # 调整模型的 token embeddings 大小以匹配 tokenizer 的词汇表大小
    model.resize_token_embeddings(len(tokenizer))

    # 准备特征数据
    column_names = datasets["train"].column_names
    sent2_cname = None
    # 根据列名数量决定数据集的类型和配置
    if len(column_names) == 2:
        # 如果有两列，说明是成对数据集
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names) == 3:
        # 如果有三列，说明是带有硬负例的成对数据集
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # 如果只有一列，说明是无监督数据集
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        # 如果列数不符合以上任何一种情况，则抛出未实现的异常
        raise NotImplementedError
    # 定义函数 prepare_features，用于准备模型训练所需的特征
    def prepare_features(examples):
        # 计算样本集合中 sent0_cname 对应列的总数量
        total = len(examples[sent0_cname])

        # 避免 examples[sent0_cname] 和 examples[sent1_cname] 中的 None 值
        for idx in range(total):
            # 如果 examples[sent0_cname][idx] 是 None，则替换为一个空字符串
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            # 如果 examples[sent1_cname][idx] 是 None，则替换为一个空字符串
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
        
        # 将 sent0_cname 和 sent1_cname 列的内容合并成一个列表 sentences
        sentences = examples[sent0_cname] + examples[sent1_cname]

        # 如果存在 sent2_cname 列
        if sent2_cname is not None:
            # 遍历 sent2_cname 列，处理其中的 None 值
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
            # 将 sent2_cname 列的内容加入 sentences 列表中
            sentences += examples[sent2_cname]

        # 使用 tokenizer 对 sentences 进行编码，生成句子特征
        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 初始化特征字典 features
        features = {}
        # 如果存在 sent2_cname 列
        if sent2_cname is not None:
            # 对每个特征键进行处理
            for key in sent_features:
                # 将 sent_features 中的特征组合成 triplets 形式，并加入 features 字典中
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            # 对每个特征键进行处理
            for key in sent_features:
                # 将 sent_features 中的特征组合成 pairs 形式，并加入 features 字典中
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
            
        # 返回最终生成的特征字典 features
        return features

    # 如果设置了训练参数 do_train
    if training_args.do_train:
        # 对训练集 datasets["train"] 应用 prepare_features 函数，进行批处理处理
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # 数据收集器
    @dataclass
    # 如果 data_args.pad_to_max_length 为真，则使用默认的数据收集器 default_data_collator
    # 否则使用 OurDataCollatorWithPadding(tokenizer) 自定义的数据收集器
    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)

    # 初始化 CLTrainer 对象 trainer，用于模型的训练
    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # 将模型参数 model_args 赋给 trainer 的 model_args 属性
    trainer.model_args = model_args

    # 开始训练
    # 如果设置了训练参数 do_train，则执行训练过程
    if training_args.do_train:
        # 确定模型路径，如果指定的模型路径存在且是目录，则使用该路径；否则设为 None
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        # 使用 Trainer 对象进行模型训练，返回训练结果
        train_result = trainer.train(model_path=model_path)
        # 保存训练好的模型和其对应的分词器，便于后续上传使用
        trainer.save_model()

        # 定义保存训练结果的输出文件路径
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        # 如果当前进程是全局的第一个进程
        if trainer.is_world_process_zero():
            # 打开输出文件，准备写入训练结果
            with open(output_train_file, "w") as writer:
                # 记录训练结果日志信息
                logger.info("***** Train results *****")
                # 遍历排序后的训练结果指标项，记录到日志并写入文件
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # 需要保存 Trainer 的状态信息，因为 Trainer.save_model 只保存了模型和分词器
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # 如果设置了评估参数 do_eval，则执行评估过程
    # Evaluation
    results = {}
    if training_args.do_eval:
        # 记录评估过程日志信息
        logger.info("*** Evaluate ***")
        # 执行评估，并返回评估结果
        results = trainer.evaluate(eval_senteval_transfer=True)

        # 定义保存评估结果的输出文件路径
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        # 如果当前进程是全局的第一个进程
        if trainer.is_world_process_zero():
            # 打开输出文件，准备写入评估结果
            with open(output_eval_file, "w") as writer:
                # 记录评估结果日志信息
                logger.info("***** Eval results *****")
                # 遍历排序后的评估结果项，记录到日志并写入文件
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    # 返回评估结果字典
    return results
# 定义一个函数 _mp_fn，用于 xla_spawn（TPUs）中的多进程处理，接受一个参数 index
def _mp_fn(index):
    # 调用主函数 main() 来执行实际的处理逻辑
    main()

# 如果当前脚本作为主程序执行，则执行主函数 main()
if __name__ == "__main__":
    main()
```