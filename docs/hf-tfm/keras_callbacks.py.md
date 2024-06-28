# `.\keras_callbacks.py`

```py
import logging  # 导入日志模块
import os  # 导入操作系统模块
from pathlib import Path  # 导入路径操作模块
from time import sleep  # 导入睡眠函数
from typing import Callable, List, Optional, Union  # 导入类型提示相关模块

import numpy as np  # 导入NumPy库
import tensorflow as tf  # 导入TensorFlow库
from huggingface_hub import Repository, create_repo  # 导入Hugging Face Hub相关函数
from packaging.version import parse  # 导入版本解析模块

from . import IntervalStrategy, PreTrainedTokenizerBase  # 从当前包导入特定模块
from .modelcard import TrainingSummary  # 从当前包导入模型卡片中的训练摘要
from .modeling_tf_utils import keras  # 从当前包导入TensorFlow工具中的Keras模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

class KerasMetricCallback(keras.callbacks.Callback):
    """
    Callback to compute metrics at the end of every epoch. Unlike normal Keras metrics, these do not need to be
    compilable by TF. It is particularly useful for common NLP metrics like BLEU and ROUGE that require string
    operations or generation loops that cannot be compiled. Predictions (or generations) will be computed on the
    `eval_dataset` before being passed to the `metric_fn` in `np.ndarray` format. The `metric_fn` should compute
    metrics and return a dict mapping metric names to metric values.

    We provide an example of a suitable metric_fn that computes ROUGE scores for a summarization model below. Note that
    this example skips some post-processing for readability and simplicity, and should probably not be used as-is!

    ```
    from datasets import load_metric

    rouge_metric = load_metric("rouge")


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
    """
    pass  # KerasMetricCallback 类暂时不包含具体实现，只是一个占位符
    # 初始化方法，用于创建一个新的评估器对象
    def __init__(
        self,
        metric_fn: Callable,  # 参数：评估指标函数，接受预测值和标签作为输入，返回指标名称到数值的字典
        eval_dataset: Union[tf.data.Dataset, np.ndarray, tf.Tensor, tuple, dict],  # 参数：用于评估的数据集或数据字典/元组/数组
        output_cols: Optional[List[str]] = None,  # 可选参数：模型输出中要保留的列名列表，默认为全部列
        label_cols: Optional[List[str]] = None,  # 可选参数：从输入数据集中要保留的标签列名列表，如果未提供则自动检测
        batch_size: Optional[int] = None,  # 可选参数：批处理大小，仅在数据不是预先批处理的 tf.data.Dataset 时使用
        predict_with_generate: bool = False,  # 可选参数：是否使用 model.generate() 获取模型输出
        use_xla_generation: bool = False,  # 可选参数：如果生成结果，是否使用 XLA 编译模型生成，可以显著提高生成速度
        generate_kwargs: Optional[dict] = None,  # 可选参数：传递给 model.generate() 的关键字参数，仅在 predict_with_generate 为 True 时有效
        ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置度量函数和批次大小
        self.metric_fn = metric_fn
        self.batch_size = batch_size
        # 如果评估数据集不是 tf.data.Dataset 类型，则根据情况处理
        if not isinstance(eval_dataset, tf.data.Dataset):
            if batch_size is None:
                # 如果没有设置批次大小且传入的数据不是预先批处理的 tf.data.Dataset，则抛出异常
                raise ValueError(
                    "When passing data to KerasMetricCallback that is not a pre-batched tf.data.Dataset "
                    "the batch_size argument must be set."
                )
            # 将传入的数据转换为 tf.data.Dataset，并按指定的批次大小进行分批
            eval_dataset = tf.data.Dataset.from_tensor_slices(eval_dataset).batch(batch_size, drop_remainder=False)
        # 存储评估数据集
        self.eval_dataset = eval_dataset
        self.predict_with_generate = predict_with_generate
        self.output_cols = output_cols

        # 下面的代码块尝试解析数据集的元素规范，确定应该将哪些元素附加到传递给 metric_fn 的标签列表中
        if isinstance(eval_dataset.element_spec, tuple) and len(eval_dataset.element_spec) == 2:
            # 如果数据集的元素规范是一个元组且长度为 2，则假设第一个元素是输入，第二个元素是标签
            input_spec, label_spec = eval_dataset.element_spec
        else:
            # 否则，将整个元素规范视为输入规范，标签规范设为 None
            input_spec = eval_dataset.element_spec
            label_spec = None
        # 如果指定了 label_cols
        if label_cols is not None:
            # 检查每个指定的标签是否在输入规范中，如果不在则抛出异常
            for label in label_cols:
                if label not in input_spec:
                    raise ValueError(f"Label {label} is in label_cols but could not be found in the dataset inputs!")
            self.label_cols = label_cols
            self.use_keras_label = False
        elif label_spec is not None:
            # 如果数据集的元素规范是一个元组，且没有指定 label_cols，则假设第二个元素是标签
            self.label_cols = None
            self.use_keras_label = True
        elif "labels" in input_spec:
            # 如果输入规范中有 "labels"，则将其作为标签列
            self.label_cols = ["labels"]
            self.use_keras_label = False
            logging.warning("No label_cols specified for KerasMetricCallback, assuming you want the 'labels' key.")
        elif "start_positions" in input_spec and "end_positions" in input_spec:
            # 如果输入规范中有 "start_positions" 和 "end_positions"，则将它们作为标签列
            self.label_cols = ["start_positions", "end_positions"]
            self.use_keras_label = False
            logging.warning(
                "No label_cols specified for KerasMetricCallback, assuming you want the "
                "start_positions and end_positions keys."
            )
        else:
            # 如果无法自动检测到标签列，则抛出异常
            raise ValueError("Could not autodetect label_cols for KerasMetricCallback, please specify them!")
        # 如果 TensorFlow 版本小于 2.7，给出警告
        if parse(tf.__version__) < parse("2.7"):
            logging.warning("TF versions less than 2.7 may encounter issues with KerasMetricCallback!")

        # 设置是否使用 XLA 生成
        self.use_xla_generation = use_xla_generation
        # 生成文本的额外参数
        self.generate_kwargs = {} if generate_kwargs is None else generate_kwargs

        # 生成函数初始化为 None
        self.generation_function = None

    @staticmethod
    def _concatenate_batches(batches, padding_index=-100):
        # 如果所有批次都是一维的或者长度相同，直接进行简单的拼接
        if batches[0].ndim == 1 or all(batch.shape[1] == batches[0].shape[1] for batch in batches):
            return np.concatenate(batches, axis=0)

        # 如果批次长度不同，进行填充操作
        max_len = max([batch.shape[1] for batch in batches])  # 计算最大长度
        num_samples = sum([batch.shape[0] for batch in batches])  # 计算总样本数
        output = np.full_like(
            batches[0], fill_value=padding_index, shape=[num_samples, max_len] + list(batches[0].shape[2:])
        )
        # i 用于跟踪下一个要写入批次数据的位置
        i = 0
        for batch in batches:
            output[i : i + len(batch), : batch.shape[1]] = batch  # 将每个批次的数据写入到输出中
            i += len(batch)
        return output

    def _postprocess_predictions_or_labels(self, inputs):
        if isinstance(inputs[0], dict):
            outputs = {}
            for key in inputs[0].keys():
                outputs[key] = self._concatenate_batches([batch[key] for batch in inputs])
            # 如果输出是一个只有一个键的字典，直接返回数组
            if len(outputs) == 1:
                outputs = list(outputs.values())[0]
        elif isinstance(inputs[0], list) or isinstance(inputs[0], tuple):
            outputs = []
            for input_list in zip(*inputs):
                outputs.append(self._concatenate_batches(input_list))
            if len(outputs) == 1:
                outputs = outputs[0]  # 如果输出是一个只有一个元素的列表，直接返回数组
        elif isinstance(inputs[0], np.ndarray):
            outputs = self._concatenate_batches(inputs)
        elif isinstance(inputs[0], tf.Tensor):
            outputs = self._concatenate_batches([tensor.numpy() for tensor in inputs])
        else:
            raise TypeError(f"Couldn't handle batch of type {type(inputs[0])}!")  # 处理无法处理的批次类型异常
        return outputs
# 定义一个自定义的回调类，用于定期保存模型并推送到 Hub 上。默认情况下，每个 epoch 结束后进行推送，但可以通过 `save_strategy` 参数进行更改。
class PushToHubCallback(keras.callbacks.Callback):

    """
    Callback that will save and push the model to the Hub regularly. By default, it pushes once per epoch, but this can
    be changed with the `save_strategy` argument. Pushed models can be accessed like any other model on the hub, such
    as with the `from_pretrained` method.

    ```
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
        # 初始化回调函数，接受输出目录、保存策略、保存步数、分词器、Hub 模型 ID、Hub token、是否保存检查点等参数
        super().__init__()
        # 设置输出目录，用于保存模型预测和检查点，并与 Hub 上的仓库同步
        self.output_dir = output_dir
        # 设置保存策略，控制模型保存的频率，默认为每个 epoch 结束时保存
        self.save_strategy = save_strategy
        # 设置保存步数，当保存策略为 "steps" 时，指定每隔多少步保存一次
        self.save_steps = save_steps
        # 设置分词器，如果提供，将与模型权重一起上传到 Hub
        self.tokenizer = tokenizer
        # 设置 Hub 模型 ID，指定要同步的本地输出目录对应的仓库名称
        self.hub_model_id = hub_model_id
        # 设置 Hub token，用于推送模型到 Hub，如果未提供，则使用缓存文件夹中的 token
        self.hub_token = hub_token
        # 设置是否保存完整的训练检查点，包括 epoch 和优化器状态，允许在训练中断后恢复
        self.checkpoint = checkpoint
        # 其他模型卡片参数，以字典形式传递给模型卡片
        self.model_card_args = model_card_args
        ):
            super().__init__()
            # 调用父类的构造方法
            if checkpoint and save_strategy != "epoch":
                raise ValueError("Cannot save checkpoints when save_strategy is not 'epoch'!")
            # 检查是否能够保存检查点，若保存策略不是 'epoch'，则抛出值错误异常
            if isinstance(save_strategy, str):
                save_strategy = IntervalStrategy(save_strategy.lower())
            # 如果保存策略是字符串，则转换为小写后创建 IntervalStrategy 对象
            self.save_strategy = save_strategy
            # 设置保存策略
            if self.save_strategy == IntervalStrategy.STEPS and (not isinstance(save_steps, int) or save_steps <= 0):
                raise ValueError("Please supply a positive integer argument for save_steps when save_strategy == 'steps'!")
            # 如果保存策略为步数，并且保存步数不是正整数或者小于等于零，则抛出值错误异常
            self.save_steps = save_steps
            # 设置保存步数
            output_dir = Path(output_dir)

            # Create repo and retrieve repo_id
            # 创建仓库并获取仓库 ID
            if hub_model_id is None:
                hub_model_id = output_dir.absolute().name
            # 如果未指定 hub_model_id，则将其设为输出目录的绝对路径名
            self.hub_model_id = create_repo(repo_id=hub_model_id, exist_ok=True, token=hub_token).repo_id
            # 创建仓库，获取仓库 ID，并存储到实例变量中

            self.output_dir = output_dir
            # 设置输出目录
            self.repo = Repository(str(self.output_dir), clone_from=self.hub_model_id, token=hub_token)
            # 创建仓库对象，克隆自指定的 hub_model_id，并设置令牌

            self.tokenizer = tokenizer
            # 设置分词器
            self.last_job = None
            # 初始化最后一个作业为 None
            self.checkpoint = checkpoint
            # 设置检查点标志
            self.training_history = None
            # 初始化训练历史为 None
            self.model_card_args = model_card_args
            # 设置模型卡参数

    def on_train_begin(self, logs=None):
        # Although we can access model.history, we have no guarantees that the History callback will fire before this
        # one, so we keep track of it here too
        # 虽然我们可以访问 model.history，但不能保证 History 回调会在当前回调之前触发，因此我们也在这里进行跟踪
        self.training_history = []
        # 初始化训练历史为空列表

    def on_train_batch_end(self, batch, logs=None):
        if self.save_strategy == IntervalStrategy.STEPS and (batch + 1) % self.save_steps == 0:
            # 如果保存策略是基于步数，并且当前批次是保存步数的倍数
            if self.last_job is not None and not self.last_job.is_done:
                return  # The last upload is still running, don't start another
                # 如果上一个上传仍在运行中，则不启动另一个上传
            self.model.save_pretrained(self.output_dir)
            # 保存模型到输出目录
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.output_dir)
                # 如果存在分词器，则保存分词器到输出目录
            _, self.last_job = self.repo.push_to_hub(
                commit_message=f"Training in progress steps {batch}", blocking=False
            )
            # 推送模型和分词器到 Hub 仓库，使用批次号作为提交消息，非阻塞模式
    # 在每个 epoch 结束时调用的方法，用于处理日志和保存模型训练历史
    def on_epoch_end(self, epoch, logs=None):
        logs = logs.copy()  # 复制日志以避免意外影响后续 Keras 的读取操作
        if "epoch" not in logs:
            logs["epoch"] = epoch  # 如果日志中没有 epoch，则添加当前 epoch
        self.training_history.append(logs)  # 将当前 epoch 的日志记录到训练历史中
        if self.save_strategy == IntervalStrategy.EPOCH:
            if self.last_job is not None and not self.last_job.is_done:
                return  # 如果上一个上传任务仍在运行，则不启动新的任务
            self.model.save_pretrained(self.output_dir)  # 保存模型到指定输出目录
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.output_dir)  # 如果存在 tokenizer，则保存到同一输出目录
            if self.checkpoint:
                checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
                self.model._save_checkpoint(checkpoint_dir, epoch)  # 保存检查点信息
            # 从 Keras 历史和模型信息中生成训练摘要
            train_summary = TrainingSummary.from_keras(
                model=self.model,
                model_name=self.hub_model_id,
                keras_history=self.training_history,
                **self.model_card_args,
            )
            model_card = train_summary.to_model_card()  # 转换训练摘要为模型卡片信息
            with (self.output_dir / "README.md").open("w") as f:
                f.write(model_card)  # 将模型卡片信息写入 README.md 文件中
            # 推送到版本控制平台（Hub），并获取推送任务状态
            _, self.last_job = self.repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False
            )

    # 在训练结束时调用的方法，确保最新版本的模型已上传到 Hub
    def on_train_end(self, logs=None):
        if self.last_job is not None and not self.last_job.is_done:
            logging.info("Pushing the last epoch to the Hub, this may take a while...")
            while not self.last_job.is_done:
                sleep(1)  # 等待上一个推送任务完成
        else:
            self.model.save_pretrained(self.output_dir)  # 保存最终版本的模型到输出目录
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.output_dir)  # 如果存在 tokenizer，则保存到同一输出目录
            # 从 Keras 历史和模型信息中生成训练摘要
            train_summary = TrainingSummary.from_keras(
                model=self.model,
                model_name=self.hub_model_id,
                keras_history=self.training_history,
                **self.model_card_args,
            )
            model_card = train_summary.to_model_card()  # 转换训练摘要为模型卡片信息
            with (self.output_dir / "README.md").open("w") as f:
                f.write(model_card)  # 将模型卡片信息写入 README.md 文件中
            self.repo.push_to_hub(commit_message="End of training", blocking=True)  # 推送最终训练结果到版本控制平台（Hub）
```