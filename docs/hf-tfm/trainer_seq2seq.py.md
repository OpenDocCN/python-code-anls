# `.\trainer_seq2seq.py`

```
# 引入警告模块，用于处理可能的警告信息
import warnings
# 从标准库中复制深度拷贝函数
from copy import deepcopy
# 引入处理路径操作的 Path 类
from pathlib import Path
# 引入类型检查相关的工具
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

# 引入 PyTorch 库
import torch
# 从 PyTorch 中引入神经网络模块
from torch import nn
# 引入数据集类
from torch.utils.data import Dataset

# 从当前包中的指定模块导入 GenerationConfig 类
from .generation.configuration_utils import GenerationConfig
# 从当前包中的指定模块导入 is_deepspeed_zero3_enabled 函数
from .integrations.deepspeed import is_deepspeed_zero3_enabled
# 从当前包中导入 Trainer 类
from .trainer import Trainer
# 从当前包中导入 logging 工具
from .utils import logging

# 如果是类型检查环境，导入以下几个类型
if TYPE_CHECKING:
    # 从当前包中导入 DataCollator 类
    from .data.data_collator import DataCollator
    # 从当前包中导入 PreTrainedModel 类
    from .modeling_utils import PreTrainedModel
    # 从当前包中导入 PreTrainedTokenizerBase 类
    from .tokenization_utils_base import PreTrainedTokenizerBase
    # 从当前包中导入 TrainerCallback 类
    from .trainer_callback import TrainerCallback
    # 从当前包中导入 EvalPrediction 和 PredictionOutput 类
    from .trainer_utils import EvalPrediction, PredictionOutput
    # 从当前包中导入 TrainingArguments 类
    from .training_args import TrainingArguments

# 获取 logger 对象，用于记录日志信息
logger = logging.get_logger(__name__)

# 定义 Seq2SeqTrainer 类，继承自 Trainer 类
class Seq2SeqTrainer(Trainer):
    # 初始化函数，接受多个参数用于模型训练
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,  # 模型参数，可以是预训练模型或者 PyTorch nn.Module
        args: "TrainingArguments" = None,  # 训练参数，类型为 TrainingArguments
        data_collator: Optional["DataCollator"] = None,  # 数据收集器，可选的 DataCollator 类型
        train_dataset: Optional[Dataset] = None,  # 训练数据集，可选的 Dataset 类型
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,  # 评估数据集，可选的 Dataset 或字典类型
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,  # 分词器，可选的 PreTrainedTokenizerBase 类型
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,  # 模型初始化函数，可选的无参函数
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,  # 计算评估指标的函数，可选的输入为 EvalPrediction 输出为字典类型
        callbacks: Optional[List["TrainerCallback"]] = None,  # 回调函数列表，可选的 TrainerCallback 类型列表
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),  # 优化器和学习率调度器元组，默认为 (None, None)
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,  # 预处理 logits 的函数，可选的输入为两个张量，输出为张量
        ):
            # 调用父类的初始化方法，传入以下参数来初始化模型训练器
            super().__init__(
                model=model,  # 模型对象
                args=args,  # 训练过程中的参数配置
                data_collator=data_collator,  # 数据收集器，用于处理批量数据
                train_dataset=train_dataset,  # 训练数据集
                eval_dataset=eval_dataset,  # 评估数据集
                tokenizer=tokenizer,  # 分词器对象
                model_init=model_init,  # 模型初始化函数
                compute_metrics=compute_metrics,  # 计算评估指标的函数
                callbacks=callbacks,  # 回调函数列表
                optimizers=optimizers,  # 优化器对象
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # 用于评估指标的逻辑预处理函数
            )

            # 如果在参数中指定了生成配置 GenerationConfig，则覆盖模型的生成配置
            # 优先级：args.generation_config > model.generation_config > 默认的 GenerationConfig
            if self.args.generation_config is not None:
                # 加载指定路径下的生成配置文件
                gen_config = self.load_generation_config(self.args.generation_config)
                # 将加载的生成配置设置为模型的生成配置
                self.model.generation_config = gen_config

        @staticmethod
    # 加载生成配置信息，可以接受字符串或GenerationConfig类型的参数，并返回一个GenerationConfig对象
    def load_generation_config(gen_config_arg: Union[str, GenerationConfig]) -> GenerationConfig:
        """
        Loads a `~generation.GenerationConfig` from the `Seq2SeqTrainingArguments.generation_config` arguments.

        Args:
            gen_config_arg (`str` or [`~generation.GenerationConfig`]):
                `Seq2SeqTrainingArguments.generation_config` argument.

        Returns:
            A `~generation.GenerationConfig`.
        """

        # 如果gen_config_arg是GenerationConfig类型，则进行深拷贝并返回
        if isinstance(gen_config_arg, GenerationConfig):
            gen_config = deepcopy(gen_config_arg)
        else:
            # 如果gen_config_arg是str或Path类型
            pretrained_model_name = Path(gen_config_arg) if isinstance(gen_config_arg, str) else gen_config_arg
            config_file_name = None

            # 确定pretrained_model_name指向的是文件路径、目录路径还是模型ID或URL
            # 这一步是为了确定config_file_name的值
            if pretrained_model_name.is_file():
                config_file_name = pretrained_model_name.name
                pretrained_model_name = pretrained_model_name.parent
            # 如果是目录路径
            elif pretrained_model_name.is_dir():
                pass
            # 如果是模型ID或URL
            else:
                pretrained_model_name = gen_config_arg

            # 使用pretrained_model_name和config_file_name创建GenerationConfig对象
            gen_config = GenerationConfig.from_pretrained(pretrained_model_name, config_file_name)

        # 严格验证以便在早期发现问题。在训练结束时，GenerationConfig.save_pretrained()运行，如果验证时有警告则会抛出异常。
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                gen_config.validate()
            if len(caught_warnings) > 0:
                raise ValueError(str([w.message for w in caught_warnings]))
        except ValueError as exc:
            # 如果验证失败，则抛出异常，指示生成的配置实例无效
            raise ValueError(
                "The loaded generation config instance is invalid -- `GenerationConfig.validate()` throws warnings "
                "and/or exceptions. Fix these issues to train your model.\n\nThrown during validation:\n" + str(exc)
            )
        # 返回生成的GenerationConfig对象
        return gen_config
        ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            # Set max_length from training args if not already set in gen_kwargs
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            # Set num_beams from training args if not already set in gen_kwargs
            gen_kwargs["num_beams"] = self.args.generation_num_beams

        # Assign the gather function to use for predictions
        self.gather_function = self.accelerator.gather
        # Store the generated kwargs internally for later use
        self._gen_kwargs = gen_kwargs

        # Call the evaluate method from the superclass with provided arguments
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
    ) -> "PredictionOutput":
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        # Copy gen_kwargs to avoid modifying the original
        gen_kwargs = gen_kwargs.copy()

        # Legacy argument setting: If max_length or max_new_tokens is not explicitly set, use values from training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        
        # Legacy argument setting: If num_beams is not explicitly set, use value from training args
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        
        # Set gather function to the accelerator's gather method
        self.gather_function = self.accelerator.gather
        
        # Store the modified gen_kwargs internally
        self._gen_kwargs = gen_kwargs

        # Call the predict method of the superclass to perform predictions
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
    # 定义一个方法 `prediction_step`，用于执行预测步骤
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ):
        # 如果存在自定义的分词器并且有定义 PAD 标记的 ID
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # 设置 PAD 标记 ID，如果未定义则使用 EOS 标记 ID
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            # 如果未定义分词器或者 PAD 标记 ID，检查模型配置中是否有定义 PAD 标记 ID
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                # 如果都未定义，则抛出数值错误异常
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        # 创建一个形状为 (tensor.shape[0], max_length) 的张量，填充为 PAD 标记 ID
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        # 将输入张量的内容复制到创建的填充张量中，保留原始内容的形状
        padded_tensor[:, : tensor.shape[-1]] = tensor
        # 返回填充后的张量
        return padded_tensor
```