# `.\transformers\utils\notebook.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入模块
import re
import time
from typing import Optional
# 导入 IPython 显示模块
import IPython.display as disp
# 导入自定义的训练回调模块
from ..trainer_callback import TrainerCallback
# 导入训练工具模块
from ..trainer_utils import IntervalStrategy, has_length

# 格式化时间函数，将秒数格式化为 (h):mm:ss
def format_time(t):
    t = int(t)
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h != 0 else f"{m:02d}:{s:02d}"

# 生成 HTML 进度条
def html_progress_bar(value, total, prefix, label, width=300):
    # docstyle-ignore
    return f"""
    <div>
      {prefix}
      <progress value='{value}' max='{total}' style='width:{width}px; height:20px; vertical-align: middle;'></progress>
      {label}
    </div>
    """

# 将文本放入 HTML 表格中
def text_to_html_table(items):
    html_code = """<table border="1" class="dataframe">\n"""
    html_code += """  <thead>\n <tr style="text-align: left;">\n"""
    for i in items[0]:
        html_code += f"      <th>{i}</th>\n"
    html_code += "    </tr>\n  </thead>\n  <tbody>\n"
    for line in items[1:]:
        html_code += "    <tr>\n"
        for elt in line:
            elt = f"{elt:.6f}" if isinstance(elt, float) else str(elt)
            html_code += f"      <td>{elt}</td>\n"
        html_code += "    </tr>\n"
    html_code += "  </tbody>\n</table><p>"
    return html_code

# 笔记本进度条类
class NotebookProgressBar:
    """
    A progress par for display in a notebook.

    Class attributes (overridden by derived classes)

        - **warmup** (`int`) -- The number of iterations to do at the beginning while ignoring `update_every`.
        - **update_every** (`float`) -- Since calling the time takes some time, we only do it every presumed
          `update_every` seconds. The progress bar uses the average time passed up until now to guess the next value
          for which it will call the update.
    """
    Args:
        total (`int`):
            The total number of iterations to reach.
        prefix (`str`, *optional*):
            A prefix to add before the progress bar.
        leave (`bool`, *optional*, defaults to `True`):
            Whether or not to leave the progress bar once it's completed. You can always call the
            [`~utils.notebook.NotebookProgressBar.close`] method to make the bar disappear.
        parent ([`~notebook.NotebookTrainingTracker`], *optional*):
            A parent object (like [`~utils.notebook.NotebookTrainingTracker`]) that spawns progress bars and handle
            their display. If set, the object passed must have a `display()` method.
        width (`int`, *optional*, defaults to 300):
            The width (in pixels) that the bar will take.

    Example:

    ```python
    import time

    pbar = NotebookProgressBar(100)
    for val in range(100):
        pbar.update(val)
        time.sleep(0.07)
    pbar.update(100)
    ```py"""

    warmup = 5
    update_every = 0.2

    def __init__(
        self,
        total: int,
        prefix: Optional[str] = None,
        leave: bool = True,
        parent: Optional["NotebookTrainingTracker"] = None,
        width: int = 300,
    ):
        # 初始化进度条对象
        self.total = total
        self.prefix = "" if prefix is None else prefix
        self.leave = leave
        self.parent = parent
        self.width = width
        self.last_value = None
        self.comment = None
        self.output = None
    def update(self, value: int, force_update: bool = False, comment: str = None):
        """
        更新进度条到指定的`value`值。

        Args:
            value (`int`):
                要使用的值。必须在0和`total`之间。
            force_update (`bool`, *optional*, 默认为`False`):
                是否强制更新内部状态和显示（默认情况下，进度条将等待`value`达到它预测的对应于自上次更新以来超过`update_every`属性的时间的值，以避免添加样板文件）。
            comment (`str`, *optional*):
                要添加到进度条左侧的注释。
        """
        # 设置当前值为传入的值
        self.value = value
        # 如果有注释，则更新注释
        if comment is not None:
            self.comment = comment
        # 如果是第一次调用update方法
        if self.last_value is None:
            # 初始化计时器和值
            self.start_time = self.last_time = time.time()
            self.start_value = self.last_value = value
            self.elapsed_time = self.predicted_remaining = None
            self.first_calls = self.warmup
            self.wait_for = 1
            # 更新进度条
            self.update_bar(value)
        # 如果值小于等于上次值且不是强制更新，则直接返回
        elif value <= self.last_value and not force_update:
            return
        # 如果是强制更新或者是在预定的更新时间内或者值大于等于上次值加上等待时间
        elif force_update or self.first_calls > 0 or value >= min(self.last_value + self.wait_for, self.total):
            # 如果是第一次调用
            if self.first_calls > 0:
                self.first_calls -= 1
            current_time = time.time()
            self.elapsed_time = current_time - self.start_time
            # 如果值大于起始值，则计算每个项目的平均时间
            if value > self.start_value:
                self.average_time_per_item = self.elapsed_time / (value - self.start_value)
            else:
                self.average_time_per_item = None
            # 如果值大于等于总数
            if value >= self.total:
                value = self.total
                self.predicted_remaining = None
                # 如果不保留进度条，则关闭
                if not self.leave:
                    self.close()
            # 如果平均时间不为空，则计算预计剩余时间
            elif self.average_time_per_item is not None:
                self.predicted_remaining = self.average_time_per_item * (self.total - value)
            # 更新进度条
            self.update_bar(value)
            self.last_value = value
            self.last_time = current_time
            # 如果平均时间为空或者为0，则等待时间为1
            if (self.average_time_per_item is None) or (self.average_time_per_item == 0):
                self.wait_for = 1
            else:
                # 否则，根据平均时间计算等待时间
                self.wait_for = max(int(self.update_every / self.average_time_per_item), 1)
    # 更新进度条的值和注释
    def update_bar(self, value, comment=None):
        # 根据总数和当前值计算空格数，用于对齐显示
        spaced_value = " " * (len(str(self.total)) - len(str(value))) + str(value)
        # 根据不同情况设置进度条的显示标签
        if self.elapsed_time is None:
            self.label = f"[{spaced_value}/{self.total} : < :"
        elif self.predicted_remaining is None:
            self.label = f"[{spaced_value}/{self.total} {format_time(self.elapsed_time)}"
        else:
            self.label = (
                f"[{spaced_value}/{self.total} {format_time(self.elapsed_time)} <"
                f" {format_time(self.predicted_remaining)}"
            )
            # 根据平均每项所需时间设置显示标签
            if self.average_time_per_item == 0:
                self.label += ", +inf it/s"
            else:
                self.label += f", {1/self.average_time_per_item:.2f} it/s"

        # 根据注释是否为空设置显示标签
        self.label += "]" if self.comment is None or len(self.comment) == 0 else f", {self.comment}]"
        # 更新显示
        self.display()

    # 显示进度条
    def display(self):
        # 生成 HTML 格式的进度条代码
        self.html_code = html_progress_bar(self.value, self.total, self.prefix, self.label, self.width)
        if self.parent is not None:
            # 如果是子进度条，则由父进度条处理显示
            self.parent.display()
            return
        if self.output is None:
            # 如果输出为空，则创建显示
            self.output = disp.display(disp.HTML(self.html_code), display_id=True)
        else:
            # 否则更新显示
            self.output.update(disp.HTML(self.html_code))

    # 关闭进度条
    def close(self):
        "Closes the progress bar."
        # 如果是顶层进度条且输出不为空，则清空显示
        if self.parent is None and self.output is not None:
            self.output.update(disp.HTML(""))
class NotebookTrainingTracker(NotebookProgressBar):
    """
    An object tracking the updates of an ongoing training with progress bars and a nice table reporting metrics.

    Args:
        num_steps (`int`): The number of steps during training. column_names (`List[str]`, *optional*):
            The list of column names for the metrics table (will be inferred from the first call to
            [`~utils.notebook.NotebookTrainingTracker.write_line`] if not set).
    """

    def __init__(self, num_steps, column_names=None):
        # 调用父类的构造函数，初始化进度条
        super().__init__(num_steps)
        # 如果未提供列名，则内部表格为空，否则设置为包含列名的列表
        self.inner_table = None if column_names is None else [column_names]
        # 子进度条初始化为空
        self.child_bar = None

    def display(self):
        # 生成 HTML 代码，包括进度条和内部表格
        self.html_code = html_progress_bar(self.value, self.total, self.prefix, self.label, self.width)
        if self.inner_table is not None:
            self.html_code += text_to_html_table(self.inner_table)
        if self.child_bar is not None:
            self.html_code += self.child_bar.html_code
        # 如果输出为空，则显示 HTML 代码，否则更新输出
        if self.output is None:
            self.output = disp.display(disp.HTML(self.html_code), display_id=True)
        else:
            self.output.update(disp.HTML(self.html_code))

    def write_line(self, values):
        """
        Write the values in the inner table.

        Args:
            values (`Dict[str, float]`): The values to display.
        """
        # 如果内部表格为空，则创建新表格
        if self.inner_table is None:
            self.inner_table = [list(values.keys()), list(values.values())]
        else:
            columns = self.inner_table[0]
            # 更新列名
            for key in values.keys():
                if key not in columns:
                    columns.append(key)
            self.inner_table[0] = columns
            if len(self.inner_table) > 1:
                last_values = self.inner_table[-1]
                first_column = self.inner_table[0][0]
                if last_values[0] != values[first_column]:
                    # 写入新行
                    self.inner_table.append([values[c] if c in values else "No Log" for c in columns])
                else:
                    # 更新最后一行
                    new_values = values
                    for c in columns:
                        if c not in new_values.keys():
                            new_values[c] = last_values[columns.index(c)]
                    self.inner_table[-1] = [new_values[c] for c in columns]
            else:
                self.inner_table.append([values[c] for c in columns])
    # 添加一个子进度条显示在指标表格下方。返回子进度条对象以便于更新。
    def add_child(self, total, prefix=None, width=300):
        """
        Add a child progress bar displayed under the table of metrics. The child progress bar is returned (so it can be
        easily updated).

        Args:
            total (`int`): The number of iterations for the child progress bar.
            prefix (`str`, *optional*): A prefix to write on the left of the progress bar.
            width (`int`, *optional*, defaults to 300): The width (in pixels) of the progress bar.
        """
        # 创建一个子进度条对象，设置总迭代次数、前缀和宽度，并将其赋值给self.child_bar
        self.child_bar = NotebookProgressBar(total, prefix=prefix, parent=self, width=width)
        # 返回子进度条对象
        return self.child_bar

    # 移除子进度条
    def remove_child(self):
        """
        Closes the child progress bar.
        """
        # 将self.child_bar置为None，关闭子进度条
        self.child_bar = None
        # 调用display方法显示结果
        self.display()
# 定义一个继承自TrainerCallback的类，用于在Jupyter Notebooks或Google colab中显示训练或评估的进度
class NotebookProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation, optimized for Jupyter Notebooks or
    Google colab.
    """

    # 初始化函数，设置初始变量
    def __init__(self):
        self.training_tracker = None
        self.prediction_bar = None
        self._force_next_update = False

    # 在训练开始时调用，设置训练追踪器
    def on_train_begin(self, args, state, control, **kwargs):
        self.first_column = "Epoch" if args.evaluation_strategy == IntervalStrategy.EPOCH else "Step"
        self.training_loss = 0
        self.last_log = 0
        column_names = [self.first_column] + ["Training Loss"]
        if args.evaluation_strategy != IntervalStrategy.NO:
            column_names.append("Validation Loss")
        self.training_tracker = NotebookTrainingTracker(state.max_steps, column_names)

    # 在每个训练步骤结束时调用，更新训练追踪器
    def on_step_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if int(state.epoch) == state.epoch else f"{state.epoch:.2f}"
        self.training_tracker.update(
            state.global_step + 1,
            comment=f"Epoch {epoch}/{state.num_train_epochs}",
            force_update=self._force_next_update,
        )
        self._force_next_update = False

    # 在预测步骤时调用，更新预测进度条
    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if not has_length(eval_dataloader):
            return
        if self.prediction_bar is None:
            if self.training_tracker is not None:
                self.prediction_bar = self.training_tracker.add_child(len(eval_dataloader))
            else:
                self.prediction_bar = NotebookProgressBar(len(eval_dataloader))
            self.prediction_bar.update(1)
        else:
            self.prediction_bar.update(self.prediction_bar.value + 1)

    # 在预测结束时调用，关闭预测进度条
    def on_predict(self, args, state, control, **kwargs):
        if self.prediction_bar is not None:
            self.prediction_bar.close()
        self.prediction_bar = None

    # 在记录日志时调用，用于没有评估策略的情况下记录训练损失
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 只有在没有评估策略且日志中包含损失时才执行
        if args.evaluation_strategy == IntervalStrategy.NO and "loss" in logs:
            values = {"Training Loss": logs["loss"]}
            # 第一列必须是Step，因为不在epoch评估策略中
            values["Step"] = state.global_step
            self.training_tracker.write_line(values)
    # 在评估过程中，根据训练追踪器记录的日志信息更新评估结果
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 如果训练追踪器不为空
        if self.training_tracker is not None:
            # 初始化值字典，包含"Training Loss"和"Validation Loss"，初始值为"No log"
            values = {"Training Loss": "No log", "Validation Loss": "No log"}
            # 遍历历史日志记录，找到最近的包含"loss"的日志
            for log in reversed(state.log_history):
                if "loss" in log:
                    values["Training Loss"] = log["loss"]
                    break

            # 根据首列名称确定"Epoch"或"Step"的值
            if self.first_column == "Epoch":
                values["Epoch"] = int(state.epoch)
            else:
                values["Step"] = state.global_step
            # 设置指标键前缀为"eval"
            metric_key_prefix = "eval"
            # 遍历指标
            for k in metrics:
                # 如果指标以"_loss"结尾，则更新指标键前缀
                if k.endswith("_loss"):
                    metric_key_prefix = re.sub(r"\_loss$", "", k)
            # 移除特定的指标
            _ = metrics.pop("total_flos", None)
            _ = metrics.pop("epoch", None)
            _ = metrics.pop(f"{metric_key_prefix}_runtime", None)
            _ = metrics.pop(f"{metric_key_prefix}_samples_per_second", None)
            _ = metrics.pop(f"{metric_key_prefix}_steps_per_second", None)
            _ = metrics.pop(f"{metric_key_prefix}_jit_compilation_time", None)
            # 遍历剩余的指标，更新值字典
            for k, v in metrics.items():
                splits = k.split("_")
                name = " ".join([part.capitalize() for part in splits[1:]])
                if name == "Loss":
                    # 单个数据集的情况下，将指标名称更新为"Validation Loss"
                    name = "Validation Loss"
                values[name] = v
            # 写入值字典到训练追踪器
            self.training_tracker.write_line(values)
            # 移除子节点
            self.training_tracker.remove_child()
            self.prediction_bar = None
            # 评估过程耗时较长，强制进行下一次更新
            self._force_next_update = True

    # 在训练结束时更新训练追踪器
    def on_train_end(self, args, state, control, **kwargs):
        # 更新训练追踪器，包括全局步数、注释和���制更新标志
        self.training_tracker.update(
            state.global_step,
            comment=f"Epoch {int(state.epoch)}/{state.num_train_epochs}",
            force_update=True,
        )
        # 将训练追踪器置空
        self.training_tracker = None
```