# `.\transformers\keras_callbacks.py`

```
import logging
import os
from pathlib import Path
from time import sleep
from typing import Callable, List, Optional, Union

import numpy as np
import tensorflow as tf
from huggingface_hub import Repository, create_repo
from packaging.version import parse
from tensorflow.keras.callbacks import Callback

from . import IntervalStrategy, PreTrainedTokenizerBase
from .modelcard import TrainingSummary

# 获取 logger 对象
logger = logging.getLogger(__name__)

class KerasMetricCallback(Callback):
    """
    Callback to compute metrics at the end of every epoch. Unlike normal Keras metrics, these do not need to be
    compilable by TF. It is particularly useful for common NLP metrics like BLEU and ROUGE that require string
    operations or generation loops that cannot be compiled. Predictions (or generations) will be computed on the
    `eval_dataset` before being passed to the `metric_fn` in `np.ndarray` format. The `metric_fn` should compute
    metrics and return a dict mapping metric names to metric values.

    We provide an example of a suitable metric_fn that computes ROUGE scores for a summarization model below. Note that
    this example skips some post-processing for readability and simplicity, and should probably not be used as-is!

    ```py
    from datasets import load_metric

    rouge_metric = load_metric("rouge")

    # 定义计算 ROUGE 分数的函数
    def rouge_fn(predictions, labels):
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge_metric.compute(predictions=decoded_predictions, references=decoded_labels)
        return {key: value.mid.fmeasure * 100 for key, value in result.items()}
    ```

    The above function will return a dict containing values which will be logged like any other Keras metric:

    ```
    {'rouge1': 37.4199, 'rouge2': 13.9768, 'rougeL': 34.361, 'rougeLsum': 35.0781
    ```
    Args:
        metric_fn (`Callable`):
            用户提供的度量函数。将以两个参数 - `predictions` 和 `labels` 调用它。这些参数包含模型的输出和数据集中的匹配标签。应返回一个字典，将度量名称映射到数值。
        eval_dataset (`tf.data.Dataset` or `dict` or `tuple` or `np.ndarray` or `tf.Tensor`):
            用于为 `metric_fn` 生成预测的验证数据集。
        output_cols (`List[str], *optional*):
            从模型输出中保留的列的列表作为预测值。默认为全部列。
        label_cols ('`List[str]`, *optional*'):
            从输入数据集中保留的列的列表作为标签。如果未提供此参数，将自动检测。
        batch_size (`int`, *optional*):
            批量大小。仅在数据不是预先分批的 `tf.data.Dataset` 时使用。
        predict_with_generate (`bool`, *optional*, defaults to `False`):
            是否应使用 `model.generate()` 获取模型的输出。
        use_xla_generation (`bool`, *optional*, defaults to `False`):
            如果正在生成，则是否使用 XLA 编译模型的生成部分。这可以大大提高生成的速度（最多提高 100 倍），但会对每个输入形状进行新的 XLA 编译。在使用 XLA 生成时，最好将输入填充到相同的大小，或者在您的 `tokenizer` 或 `DataCollator` 中使用 `pad_to_multiple_of` 参数，这将减少唯一输入形状的数量并节省大量编译时间。如果 `predict_with_generate` 为 `False`，则此选项无效。
        generate_kwargs (`dict`, *optional*):
            在生成时传递给 `model.generate()` 的关键字参数。如果 `predict_with_generate` 为 `False`，则此参数无效。
    """

    def __init__(
        self,
        metric_fn: Callable,
        eval_dataset: Union[tf.data.Dataset, np.ndarray, tf.Tensor, tuple, dict],
        output_cols: Optional[List[str]] = None,
        label_cols: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        predict_with_generate: bool = False,
        use_xla_generation: bool = False,
        generate_kwargs: Optional[dict] = None,
    # 调用父类的构造函数初始化实例
    ):
        super().__init__()
        # 设置度量函数
        self.metric_fn = metric_fn
        # 设置批处理大小
        self.batch_size = batch_size
        # 检查评估数据集是否为 TensorFlow 数据集
        if not isinstance(eval_dataset, tf.data.Dataset):
            # 如果评估数据集不是预先批处理的 TensorFlow 数据集，则要求设置批处理大小
            if batch_size is None:
                raise ValueError(
                    "When passing data to KerasMetricCallback that is not a pre-batched tf.data.Dataset "
                    "the batch_size argument must be set."
                )
            # 将数据包装成 TensorFlow 数据集
            eval_dataset = tf.data.Dataset.from_tensor_slices(eval_dataset).batch(batch_size, drop_remainder=False)
        # 设置评估数据集
        self.eval_dataset = eval_dataset
        # 是否使用生成模式进行预测
        self.predict_with_generate = predict_with_generate
        # 输出列的名称
        self.output_cols = output_cols

        # 下面的代码块尝试解析数据集中哪些元素应该被附加到传递给度量函数的标签列表中
        if isinstance(eval_dataset.element_spec, tuple) and len(eval_dataset.element_spec) == 2:
            input_spec, label_spec = eval_dataset.element_spec
        else:
            input_spec = eval_dataset.element_spec
            label_spec = None
        # 检查是否提供了标签列
        if label_cols is not None:
            for label in label_cols:
                if label not in input_spec:
                    raise ValueError(f"Label {label} is in label_cols but could not be found in the dataset inputs!")
            # 设置标签列和使用 Keras 标签的标志
            self.label_cols = label_cols
            self.use_keras_label = False
        elif label_spec is not None:
            # 如果数据集输入被拆分为输入和标签的 2 元组，则假设第二个元素是标签
            self.label_cols = None
            self.use_keras_label = True
        elif "labels" in input_spec:
            # 如果数据集输入包含标签，则使用默认的 "labels" 键
            self.label_cols = ["labels"]
            self.use_keras_label = False
            logging.warning("No label_cols specified for KerasMetricCallback, assuming you want the 'labels' key.")
        elif "start_positions" in input_spec and "end_positions" in input_spec:
            # 如果数据集输入包含 "start_positions" 和 "end_positions" 键，则使用它们作为标签列
            self.label_cols = ["start_positions", "end_positions"]
            self.use_keras_label = False
            logging.warning(
                "No label_cols specified for KerasMetricCallback, assuming you want the "
                "start_positions and end_positions keys."
            )
        else:
            # 如果无法自动检测标签列，则引发异常
            raise ValueError("Could not autodetect label_cols for KerasMetricCallback, please specify them!")
        # 检查 TensorFlow 版本是否小于 2.7，如果是，则发出警告
        if parse(tf.__version__) < parse("2.7"):
            logging.warning("TF versions less than 2.7 may encounter issues with KerasMetricCallback!")

        # 是否使用 XLA 生成
        self.use_xla_generation = use_xla_generation
        # 生成参数
        self.generate_kwargs = {} if generate_kwargs is None else generate_kwargs

        # 生成函数初始化为 None
        self.generation_function = None

    # 静态方法
    @staticmethod
``` 
    # 将多个批次数据进行拼接，如果所有批次都是一维或者长度相同，则简单拼接
    def _concatenate_batches(batches, padding_index=-100):
        # 如果所有批次都是一维或者长度相同，则简单拼接
        if batches[0].ndim == 1 or all(batch.shape[1] == batches[0].shape[1] for batch in batches):
            return np.concatenate(batches, axis=0)

        # 如果批次长度不同，进行填充操作
        max_len = max([batch.shape[1] for batch in batches])
        num_samples = sum([batch.shape[0] for batch in batches])
        # 创建一个与第一个批次相同形状的填充数组
        output = np.full_like(
            batches[0], fill_value=padding_index, shape=[num_samples, max_len] + list(batches[0].shape[2:])
        )
        # i 用于追踪下一个批次要写入的拼接数组的位置
        i = 0
        for batch in batches:
            output[i : i + len(batch), : batch.shape[1]] = batch
            i += len(batch)
        return output

    # 后处理预测或标签数据
    def _postprocess_predictions_or_labels(self, inputs):
        # 如果输入是字典
        if isinstance(inputs[0], dict):
            outputs = {}
            for key in inputs[0].keys():
                # 对每个键值进行拼接操作
                outputs[key] = self._concatenate_batches([batch[key] for batch in inputs])
            # 如果字典只有一个键，直接返回数组
            if len(outputs) == 1:
                outputs = list(outputs.values())[0]
        # 如果输入是列表或元组
        elif isinstance(inputs[0], list) or isinstance(inputs[0], tuple):
            outputs = []
            for input_list in zip(*inputs):
                outputs.append(self._concatenate_batches(input_list))
            if len(outputs) == 1:
                outputs = outputs[0]  # 如果列表只有一个元素，直接返回数组
        # 如果输入是 numpy 数组
        elif isinstance(inputs[0], np.ndarray):
            outputs = self._concatenate_batches(inputs)
        # 如果输入是 TensorFlow 张量
        elif isinstance(inputs[0], tf.Tensor):
            outputs = self._concatenate_batches([tensor.numpy() for tensor in inputs])
        else:
            raise TypeError(f"Couldn't handle batch of type {type(inputs[0])}!")
        return outputs
class PushToHubCallback(Callback):
    """
    Callback that will save and push the model to the Hub regularly. By default, it pushes once per epoch, but this can
    be changed with the `save_strategy` argument. Pushed models can be accessed like any other model on the hub, such
    as with the `from_pretrained` method.

    ```py
    from transformers.keras_callbacks import PushToHubCallback

    push_to_hub_callback = PushToHubCallback(
        output_dir="./model_save",
        tokenizer=tokenizer,
        hub_model_id="gpt5-7xlarge",
    )

    model.fit(train_dataset, callbacks=[push_to_hub_callback])
    ```

    Args:
        output_dir (`str`):
            The output directory where the model predictions and checkpoints will be written and synced with the
            repository on the Hub.
        save_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"epoch"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                - `"no"`: Save is done at the end of training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`
        save_steps (`int`, *optional*):
            The number of steps between saves when using the "steps" `save_strategy`.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            The tokenizer used by the model. If supplied, will be uploaded to the repo alongside the weights.
        hub_model_id (`str`, *optional*):
            The name of the repository to keep in sync with the local `output_dir`. It can be a simple model ID in
            which case the model will be pushed in your namespace. Otherwise it should be the whole repository name,
            for instance `"user_name/model"`, which allows you to push to an organization you are a member of with
            `"organization_name/model"`.

            Will default to the name of `output_dir`.
        hub_token (`str`, *optional*):
            The token to use to push the model to the Hub. Will default to the token in the cache folder obtained with
            `huggingface-cli login`.
        checkpoint (`bool`, *optional*, defaults to `False`):
            Whether to save full training checkpoints (including epoch and optimizer state) to allow training to be
            resumed. Only usable when `save_strategy` is `"epoch"`.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        save_strategy: Union[str, IntervalStrategy] = "epoch",
        save_steps: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
        checkpoint: bool = False,
        **model_card_args,
    ):
        # 调用父类的构造函数
        super().__init__()
        # 如果有检查点并且保存策略不是“epoch”，则引发值错误异常
        if checkpoint and save_strategy != "epoch":
            raise ValueError("Cannot save checkpoints when save_strategy is not 'epoch'!")
        # 如果保存策略是字符串，则转换为小写的IntervalStrategy对象
        if isinstance(save_strategy, str):
            save_strategy = IntervalStrategy(save_strategy.lower())
        # 保存策略赋值
        self.save_strategy = save_strategy
        # 如果保存策略为IntervalStrategy.STEPS并且保存步数不是正整数，则引发值错误异常
        if self.save_strategy == IntervalStrategy.STEPS and (not isinstance(save_steps, int) or save_steps <= 0):
            raise ValueError("Please supply a positive integer argument for save_steps when save_strategy == 'steps'!")
        # 保存步数赋值
        self.save_steps = save_steps
        # 将输出目录转换为Path对象
        output_dir = Path(output_dir)

        # 创建仓库并获取仓库ID
        if hub_model_id is None:
            hub_model_id = output_dir.absolute().name
        # 创建仓库，并指定是否允许已存在的同名仓库，然后获取仓库ID
        self.hub_model_id = create_repo(repo_id=hub_model_id, exist_ok=True, token=hub_token).repo_id

        # 输出目录和仓库对象赋值
        self.output_dir = output_dir
        self.repo = Repository(str(self.output_dir), clone_from=self.hub_model_id, token=hub_token)

        # 分词器、最后一个作业、检查点、训练历史、模型卡片参数初始化
        self.tokenizer = tokenizer
        self.last_job = None
        self.checkpoint = checkpoint
        self.training_history = None
        self.model_card_args = model_card_args

    def on_train_begin(self, logs=None):
        # 尽管我们可以访问model.history，但不能保证History回调函数会在这个回调函数之前触发，所以在这里也要跟踪训练历史
        self.training_history = []

    def on_train_batch_end(self, batch, logs=None):
        # 如果保存策略是IntervalStrategy.STEPS并且当前批次满足保存步数条件，则执行保存操作
        if self.save_strategy == IntervalStrategy.STEPS and (batch + 1) % self.save_steps == 0:
            # 如果上一个上传作业仍在运行，则返回，不开始另一个上传
            if self.last_job is not None and not self.last_job.is_done:
                return  
            # 保存模型至输出目录
            self.model.save_pretrained(self.output_dir)
            # 如果分词器不为空，则保存分词器至输出目录
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.output_dir)
            # 将结果推送到Hub，并获取推送作业对象
            _, self.last_job = self.repo.push_to_hub(
                commit_message=f"Training in progress steps {batch}", blocking=False
            )
    # 在每个 epoch 结束时调用的函数
    def on_epoch_end(self, epoch, logs=None):
        # 复制 logs，避免意外写入 Keras 后续会读取的内容
        logs = logs.copy()
        # 如果 logs 中没有 "epoch" 键，则添加当前 epoch
        if "epoch" not in logs:
            logs["epoch"] = epoch
        # 将 logs 添加到训练历史中
        self.training_history.append(logs)
        # 如果保存策略是基于 epoch，则执行以下操作
        if self.save_strategy == IntervalStrategy.EPOCH:
            # 如果上一个上传任务仍在运行，则返回，不启动新任务
            if self.last_job is not None and not self.last_job.is_done:
                return
            # 保存模型到指定输出目录
            self.model.save_pretrained(self.output_dir)
            # 如果存在 tokenizer，则保存到指定输出目录
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.output_dir)
            # 如果存在 checkpoint，则保存到指定输出目录
            if self.checkpoint:
                checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
                self.model._save_checkpoint(checkpoint_dir, epoch)
            # 从 Keras 创建 TrainingSummary 对象
            train_summary = TrainingSummary.from_keras(
                model=self.model,
                model_name=self.hub_model_id,
                keras_history=self.training_history,
                **self.model_card_args,
            )
            # 将 TrainingSummary 转换为 model_card
            model_card = train_summary.to_model_card()
            # 将 model_card 写入 README.md 文件
            with (self.output_dir / "README.md").open("w") as f:
                f.write(model_card)
            # 推送到 Hub，并获取最后一个任务
            _, self.last_job = self.repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False
            )

    # 在训练结束时调用的函数
    def on_train_end(self, logs=None):
        # 确保最新版本的模型已上传
        if self.last_job is not None and not self.last_job.is_done:
            logging.info("Pushing the last epoch to the Hub, this may take a while...")
            # 等待最后一个任务完成
            while not self.last_job.is_done:
                sleep(1)
        else:
            # 保存模型到指定输出目录
            self.model.save_pretrained(self.output_dir)
            # 如果存在 tokenizer，则保存到指定输出目录
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.output_dir)
            # 从 Keras 创建 TrainingSummary 对象
            train_summary = TrainingSummary.from_keras(
                model=self.model,
                model_name=self.hub_model_id,
                keras_history=self.training_history,
                **self.model_card_args,
            )
            # 将 TrainingSummary 转换为 model_card
            model_card = train_summary.to_model_card()
            # 将 model_card 写入 README.md 文件
            with (self.output_dir / "README.md").open("w") as f:
                f.write(model_card)
            # 推送到 Hub，提交消息为 "End of training"，阻塞等待完成
            self.repo.push_to_hub(commit_message="End of training", blocking=True)
```