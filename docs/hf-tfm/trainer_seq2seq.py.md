# `.\transformers\trainer_seq2seq.py`

```py
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入所需的模块和类
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.utils.data import Dataset
from .generation.configuration_utils import GenerationConfig
from .integrations.deepspeed import is_deepspeed_zero3_enabled
from .trainer import Trainer
from .utils import logging

# 如果是类型检查，则导入额外的类
if TYPE_CHECKING:
    from .data.data_collator import DataCollator
    from .modeling_utils import PreTrainedModel
    from .tokenization_utils_base import PreTrainedTokenizerBase
    from .trainer_callback import TrainerCallback
    from .trainer_utils import EvalPrediction, PredictionOutput
    from .training_args import TrainingArguments

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 Seq2SeqTrainer 类，继承自 Trainer 类
class Seq2SeqTrainer(Trainer):
    # 初始化方法
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        # 调用父类的构造函数，初始化 Seq2SeqTrainer 实例
        super().__init__(
            # 将传入的模型、参数、数据收集器、训练数据集、评估数据集、分词器、模型初始化器、计算指标函数、回调函数、优化器、指标预处理函数传递给父类构造函数
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # 如果 args 中指定了 generation_config，则覆盖 self.model.generation_config
        # 优先级：args.generation_config > model.generation_config > 默认 GenerationConfig
        if self.args.generation_config is not None:
            # 载入指定的 GenerationConfig
            gen_config = self.load_generation_config(self.args.generation_config)
            # 将载入的 GenerationConfig 赋值给 self.model.generation_config
            self.model.generation_config = gen_config

    @staticmethod
    def load_generation_config(gen_config_arg: Union[str, GenerationConfig]) -> GenerationConfig:
        """
        从 Seq2SeqTrainingArguments.generation_config 参数中加载 GenerationConfig。

        Args:
            gen_config_arg (str 或 GenerationConfig):
                Seq2SeqTrainingArguments.generation_config 参数。

        Returns:
            一个 GenerationConfig 实例。
        """

        # 如果传入的是 GenerationConfig 实例，则直接返回深拷贝
        if isinstance(gen_config_arg, GenerationConfig):
            return deepcopy(gen_config_arg)

        # 如果传入的是 str 或 Path 类型
        pretrained_model_name = Path(gen_config_arg) if isinstance(gen_config_arg, str) else gen_config_arg
        config_file_name = None

        # 判断传入的路径是文件还是目录，或者是模型ID或URL，并确定配置文件名
        if pretrained_model_name.is_file():
            config_file_name = pretrained_model_name.name
            pretrained_model_name = pretrained_model_name.parent
        # 如果是目录路径，则不进行处理
        elif pretrained_model_name.is_dir():
            pass
        # 如果是模型ID或URL，则直接将其作为预训练模型名称
        else:
            pretrained_model_name = gen_config_arg

        # 从预训练模型名称和配置文件名加载 GenerationConfig
        gen_config = GenerationConfig.from_pretrained(pretrained_model_name, config_file_name)
        return gen_config

    def evaluate(
        self,
        # 评估数据集（可选）
        eval_dataset: Optional[Dataset] = None,
        # 忽略的键列表（可选）
        ignore_keys: Optional[List[str]] = None,
        # 指标键前缀，默认为 "eval"
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        运行评估并返回指标。

        调用脚本将负责提供计算指标的方法，因为它们是与任务相关的（将其传递给init中的`compute_metrics`参数）。

        您还可以子类化并覆盖此方法以注入自定义行为。

        Args:
            eval_dataset (`Dataset`, *optional*):
                如果希望覆盖`self.eval_dataset`，请传递一个数据集。如果它是一个[`~datasets.Dataset`]，则会自动删除`model.forward()`方法不接受的列。它必须实现`__len__`方法。
            ignore_keys (`List[str]`, *optional*):
                模型输出中应在聚合预测时忽略的键列表（如果它是一个字典）。
            metric_key_prefix (`str`, *optional*, 默认为`"eval"`):
                用作指标键前缀的可选前缀。例如，如果前缀是`"eval"`（默认），则指标"bleu"将被命名为"eval_bleu"。
            max_length (`int`, *optional*):
                在使用generate方法进行预测时要使用的最大目标长度。
            num_beams (`int`, *optional*):
                在使用generate方法进行预测时要使用的beam搜索的数量。1表示不进行beam搜索。
            gen_kwargs:
                附加的`generate`特定kwargs。

        Returns:
            包含评估损失和从预测中计算的潜在指标的字典。字典还包含来自训练状态的时代数。
        """

        gen_kwargs = gen_kwargs.copy()

        # 如果 a) 未显式传递选项；并且 b) 在训练参数中设置了参数，则使用旧版本的参数设置
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        # 一般情况下我们不想丢弃样本
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
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

        # Copying the generate kwargs to ensure immutability
        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None  # If max_length is not explicitly passed
            and gen_kwargs.get("max_new_tokens") is None  # If max_new_tokens is not explicitly passed
            and self.args.generation_max_length is not None  # If generation_max_length is set in training args
        ):
            # Set max_length to the value from training args
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            # If num_beams is not explicitly passed and generation_num_beams is set in training args
            # Set num_beams to the value from training args
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        # Set gather_function to the accelerator's gather method
        self.gather_function = self.accelerator.gather
        # Set _gen_kwargs to the generated keyword arguments
        self._gen_kwargs = gen_kwargs

        # Call the predict method of the superclass (Trainer class) with the provided arguments
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
    # 定义预测步骤函数，用于生成模型的预测结果
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    # 将张量填充到最大长度的函数
    def _pad_tensors_to_max_len(self, tensor, max_length):
        # 检查分词器是否存在且有定义 PAD 标记的 ID
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # 如果 PAD 标记未定义，则至少需要定义 EOS 标记
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            # 如果模型配置中定义了 PAD 标记的 ID
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                # 抛出异常，要求在模型配置中设置 PAD 标记的 ID 以便填充张量
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        # 使用 PAD 标记的 ID 创建与输入张量相同形状的填充张量
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        # 将输入张量的值填充到填充张量中
        padded_tensor[:, : tensor.shape[-1]] = tensor
        # 返回填充后的张量
        return padded_tensor
```