# `.\utils\notebook.py`

```
# 导入正则表达式模块
import re
# 导入时间模块
import time
# 导入类型提示模块中的 Optional 类型
from typing import Optional
# 导入 IPython 显示模块中的 disp 别名
import IPython.display as disp
# 导入自定义的训练回调模块
from ..trainer_callback import TrainerCallback
# 导入训练工具模块中的 IntervalStrategy 和 has_length 函数
from ..trainer_utils import IntervalStrategy, has_length


def format_time(t):
    "Format `t` (in seconds) to (h):mm:ss"
    # 将秒数 `t` 格式化为时:分:秒格式的字符串
    t = int(t)
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h != 0 else f"{m:02d}:{s:02d}"


def html_progress_bar(value, total, prefix, label, width=300):
    # docstyle-ignore
    # 生成 HTML 格式的进度条
    return f"""
    <div>
      {prefix}
      <progress value='{value}' max='{total}' style='width:{width}px; height:20px; vertical-align: middle;'></progress>
      {label}
    </div>
    """


def text_to_html_table(items):
    "Put the texts in `items` in an HTML table."
    # 将文本数据 `items` 转换为 HTML 表格代码
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


class NotebookProgressBar:
    """
    A progress bar for display in a notebook.

    Class attributes (overridden by derived classes)

        - **warmup** (`int`) -- The number of iterations to do at the beginning while ignoring `update_every`.
        - **update_every** (`float`) -- Since calling the time takes some time, we only do it every presumed
          `update_every` seconds. The progress bar uses the average time passed up until now to guess the next value
          for which it will call the update.
    """
    # 定义类 `NotebookProgressBar`，用于在笔记本中显示进度条
    class NotebookProgressBar:
        
        # 类属性：热身次数
        warmup = 5
        # 类属性：每次更新的时间间隔（秒）
        update_every = 0.2

        # 初始化方法，设置进度条的各种参数
        def __init__(
            self,
            total: int,                              # 参数：总的迭代次数
            prefix: Optional[str] = None,            # 参数：前缀字符串，默认为 None
            leave: bool = True,                      # 参数：是否在完成后保留进度条，默认为 True
            parent: Optional["NotebookTrainingTracker"] = None,  # 参数：父对象，用于显示进度条，默认为 None
            width: int = 300,                        # 参数：进度条的宽度（像素），默认为 300
        ):
            self.total = total                       # 实例变量：总的迭代次数
            self.prefix = "" if prefix is None else prefix  # 实例变量：前缀字符串，若为 None 则为空字符串
            self.leave = leave                       # 实例变量：是否保留进度条
            self.parent = parent                     # 实例变量：父对象，用于显示进度条
            self.width = width                       # 实例变量：进度条的宽度
            self.last_value = None                   # 实例变量：上一次更新的数值
            self.comment = None                      # 实例变量：注释信息，默认为 None
            self.output = None                       # 实例变量：输出信息，默认为 None
    def update(self, value: int, force_update: bool = False, comment: str = None):
        """
        更新进度条到指定的 `value` 值的主要方法。

        Args:
            value (`int`):
                要使用的值。必须在 0 和 `total` 之间。
            force_update (`bool`, *optional*, 默认为 `False`):
                是否强制更新内部状态和显示（默认情况下，进度条将等待 `value` 达到它预测的对应于自上次更新以来超过 `update_every` 属性的时间的值，以避免添加样板文件）。
            comment (`str`, *optional*):
                要添加到进度条左侧的注释。
        """
        self.value = value  # 设置实例的值为给定的 `value`

        if comment is not None:
            self.comment = comment  # 如果提供了注释，设置实例的注释属性为给定的注释内容

        if self.last_value is None:
            # 如果上次的值为 None，表示第一次调用更新方法
            self.start_time = self.last_time = time.time()
            self.start_value = self.last_value = value
            self.elapsed_time = self.predicted_remaining = None
            self.first_calls = self.warmup
            self.wait_for = 1
            self.update_bar(value)  # 更新进度条显示

        elif value <= self.last_value and not force_update:
            # 如果给定的值小于等于上次的值并且不强制更新，则直接返回，不执行更新操作
            return

        elif force_update or self.first_calls > 0 or value >= min(self.last_value + self.wait_for, self.total):
            # 如果强制更新或者还处于初始调用阶段或者值超过了预期阈值，则执行更新操作
            if self.first_calls > 0:
                self.first_calls -= 1  # 减少初始调用次数计数器

            current_time = time.time()
            self.elapsed_time = current_time - self.start_time

            # 如果值大于起始值，则计算每个项目的平均时间
            if value > self.start_value:
                self.average_time_per_item = self.elapsed_time / (value - self.start_value)
            else:
                self.average_time_per_item = None

            if value >= self.total:
                value = self.total
                self.predicted_remaining = None
                if not self.leave:
                    self.close()  # 如果值达到或超过总数，则关闭进度条

            elif self.average_time_per_item is not None:
                # 如果存在平均每个项目的时间，则预测剩余时间
                self.predicted_remaining = self.average_time_per_item * (self.total - value)

            self.update_bar(value)  # 更新进度条显示
            self.last_value = value
            self.last_time = current_time

            if (self.average_time_per_item is None) or (self.average_time_per_item == 0):
                self.wait_for = 1
            else:
                # 根据平均每个项目的时间和更新频率计算等待时间
                self.wait_for = max(int(self.update_every / self.average_time_per_item), 1)
    # 更新进度条的显示，根据给定的值和可选的注释更新进度条标签
    def update_bar(self, value, comment=None):
        # 根据总数值和当前值计算填充空格，以对齐显示的数值
        spaced_value = " " * (len(str(self.total)) - len(str(value))) + str(value)
        # 如果尚未计算过已用时间，则设置标签显示格式
        if self.elapsed_time is None:
            self.label = f"[{spaced_value}/{self.total} : < :"
        # 如果尚未计算过预计剩余时间，则设置标签显示格式
        elif self.predicted_remaining is None:
            self.label = f"[{spaced_value}/{self.total} {format_time(self.elapsed_time)}"
        # 如果已计算过预计剩余时间，则设置带预计时间信息的标签显示格式
        else:
            self.label = (
                f"[{spaced_value}/{self.total} {format_time(self.elapsed_time)} <"
                f" {format_time(self.predicted_remaining)}"
            )
            # 如果每项平均处理时间为零，则显示无穷大速率
            if self.average_time_per_item == 0:
                self.label += ", +inf it/s"
            else:
                self.label += f", {1/self.average_time_per_item:.2f} it/s"

        # 如果有注释，将注释添加到标签末尾
        self.label += "]" if self.comment is None or len(self.comment) == 0 else f", {self.comment}]"
        # 更新进度条的显示
        self.display()

    # 更新 HTML 代码以反映当前进度条的状态
    def display(self):
        self.html_code = html_progress_bar(self.value, self.total, self.prefix, self.label, self.width)
        # 如果存在父进度条，由父进度条负责显示
        if self.parent is not None:
            self.parent.display()
            return
        # 如果输出对象为空，则创建新的输出显示
        if self.output is None:
            self.output = disp.display(disp.HTML(self.html_code), display_id=True)
        # 否则更新现有输出对象的 HTML 内容
        else:
            self.output.update(disp.HTML(self.html_code))

    # 关闭进度条的显示，仅当为根进度条且输出对象存在时才会执行
    def close(self):
        "Closes the progress bar."
        if self.parent is None and self.output is not None:
            # 清空输出对象的 HTML 内容，关闭进度条显示
            self.output.update(disp.HTML(""))
# 定义一个名为 NotebookTrainingTracker 的类，继承自 NotebookProgressBar 类
# 用于跟踪正在进行的训练的更新状态，包括进度条和报告指标的漂亮表格显示
class NotebookTrainingTracker(NotebookProgressBar):
    """
    An object tracking the updates of an ongoing training with progress bars and a nice table reporting metrics.

    Args:
        num_steps (`int`): The number of steps during training.
        column_names (`List[str]`, *optional*): The list of column names for the metrics table
            (will be inferred from the first call to `~utils.notebook.NotebookTrainingTracker.write_line` if not set).
    """

    # 初始化方法，接受 num_steps 参数和可选的 column_names 参数
    def __init__(self, num_steps, column_names=None):
        # 调用父类 NotebookProgressBar 的初始化方法
        super().__init__(num_steps)
        # 如果 column_names 为 None，则将 inner_table 设置为 None；否则将 column_names 封装成一个包含列表的列表
        self.inner_table = None if column_names is None else [column_names]
        # 初始化 child_bar 属性为 None
        self.child_bar = None

    # 显示方法，生成 HTML 代码用于展示进度条和内部表格
    def display(self):
        # 生成进度条的 HTML 代码
        self.html_code = html_progress_bar(self.value, self.total, self.prefix, self.label, self.width)
        # 如果内部表格不为 None，则将其转换成 HTML 表格代码添加到 html_code 中
        if self.inner_table is not None:
            self.html_code += text_to_html_table(self.inner_table)
        # 如果 child_bar 不为 None，则将其 HTML 代码添加到 html_code 中
        if self.child_bar is not None:
            self.html_code += self.child_bar.html_code
        # 如果 output 属性为 None，则使用 disp.display 方法显示 html_code
        if self.output is None:
            self.output = disp.display(disp.HTML(self.html_code), display_id=True)
        # 否则，更新 output 中的 HTML 内容为 html_code
        else:
            self.output.update(disp.HTML(self.html_code))

    # 写入新行到内部表格中的方法
    def write_line(self, values):
        """
        Write the values in the inner table.

        Args:
            values (`Dict[str, float]`): The values to display.
        """
        # 如果内部表格为 None，则将 values 的键和值分别作为列名和第一行数据
        if self.inner_table is None:
            self.inner_table = [list(values.keys()), list(values.values())]
        else:
            # 否则，获取当前的列名列表
            columns = self.inner_table[0]
            # 遍历 values 的键，将不在列名列表中的键添加到列名列表中
            for key in values.keys():
                if key not in columns:
                    columns.append(key)
            self.inner_table[0] = columns
            # 如果内部表格行数大于 1，则更新最后一行数据或添加新行数据
            if len(self.inner_table) > 1:
                last_values = self.inner_table[-1]
                first_column = self.inner_table[0][0]
                if last_values[0] != values[first_column]:
                    # 写入新行
                    self.inner_table.append([values[c] if c in values else "No Log" for c in columns])
                else:
                    # 更新最后一行数据
                    new_values = values
                    for c in columns:
                        if c not in new_values.keys():
                            new_values[c] = last_values[columns.index(c)]
                    self.inner_table[-1] = [new_values[c] for c in columns]
            else:
                # 如果内部表格只有一行，则直接添加新行数据
                self.inner_table.append([values[c] for c in columns])
    # 添加一个子进度条显示在指标表格下方。返回子进度条对象，以便进行更新操作。
    def add_child(self, total, prefix=None, width=300):
        """
        Add a child progress bar displayed under the table of metrics. The child progress bar is returned (so it can be
        easily updated).

        Args:
            total (`int`): The number of iterations for the child progress bar.
            prefix (`str`, *optional*): A prefix to write on the left of the progress bar.
            width (`int`, *optional*, defaults to 300): The width (in pixels) of the progress bar.
        """
        # 创建一个 NotebookProgressBar 对象作为子进度条，设置父对象为当前对象，并指定进度条的宽度和前缀
        self.child_bar = NotebookProgressBar(total, prefix=prefix, parent=self, width=width)
        # 返回新创建的子进度条对象，以便可以后续更新进度
        return self.child_bar

    # 移除子进度条
    def remove_child(self):
        """
        Closes the child progress bar.
        """
        # 将子进度条对象设置为 None，从而关闭并释放相关资源
        self.child_bar = None
        # 调用 display() 方法，可能用于刷新界面以反映子进度条的移除
        self.display()
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation, optimized for Jupyter Notebooks or
    Google colab.
    """

    # 初始化回调对象
    def __init__(self):
        # 记录训练进度的追踪器对象
        self.training_tracker = None
        # 预测进度条对象
        self.prediction_bar = None
        # 强制下一次更新标志
        self._force_next_update = False

    # 在训练开始时调用
    def on_train_begin(self, args, state, control, **kwargs):
        # 确定第一列的名称是 Epoch 还是 Step，根据评估策略
        self.first_column = "Epoch" if args.evaluation_strategy == IntervalStrategy.EPOCH else "Step"
        # 初始化训练损失
        self.training_loss = 0
        # 上次记录日志的步骤
        self.last_log = 0
        # 创建列名列表
        column_names = [self.first_column] + ["Training Loss"]
        # 如果评估策略不是 NO，则添加验证损失列
        if args.evaluation_strategy != IntervalStrategy.NO:
            column_names.append("Validation Loss")
        # 初始化训练追踪器对象
        self.training_tracker = NotebookTrainingTracker(state.max_steps, column_names)

    # 在每个训练步骤结束时调用
    def on_step_end(self, args, state, control, **kwargs):
        # 将当前的 epoch 转为整数或保留两位小数的字符串表示
        epoch = int(state.epoch) if int(state.epoch) == state.epoch else f"{state.epoch:.2f}"
        # 更新训练追踪器，包括注释和是否强制更新标志
        self.training_tracker.update(
            state.global_step + 1,
            comment=f"Epoch {epoch}/{state.num_train_epochs}",
            force_update=self._force_next_update,
        )
        # 重置强制更新标志
        self._force_next_update = False

    # 在预测步骤时调用
    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        # 如果评估数据加载器没有长度信息，则返回
        if not has_length(eval_dataloader):
            return
        # 如果预测进度条不存在，则创建它
        if self.prediction_bar is None:
            # 如果训练追踪器存在，则在其基础上创建预测进度条子级
            if self.training_tracker is not None:
                self.prediction_bar = self.training_tracker.add_child(len(eval_dataloader))
            # 否则创建新的 NotebookProgressBar 对象
            else:
                self.prediction_bar = NotebookProgressBar(len(eval_dataloader))
            # 更新进度条
            self.prediction_bar.update(1)
        else:
            # 更新现有预测进度条的值
            self.prediction_bar.update(self.prediction_bar.value + 1)

    # 在预测完成时调用
    def on_predict(self, args, state, control, **kwargs):
        # 如果预测进度条存在，则关闭它
        if self.prediction_bar is not None:
            self.prediction_bar.close()
        # 重置预测进度条对象为 None
        self.prediction_bar = None

    # 在记录日志时调用
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 只有当没有评估时才执行
        if args.evaluation_strategy == IntervalStrategy.NO and "loss" in logs:
            # 设置损失值字典
            values = {"Training Loss": logs["loss"]}
            # 因为不是在 epoch 评估策略下，所以第一列名称一定是 Step
            values["Step"] = state.global_step
            # 将数据行写入训练追踪器
            self.training_tracker.write_line(values)
    # 定义模型评估函数，处理评估时的逻辑和状态
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 如果存在训练追踪器对象
        if self.training_tracker is not None:
            # 初始化值字典，设置默认为未记录
            values = {"Training Loss": "No log", "Validation Loss": "No log"}
            # 遍历状态日志历史记录（倒序）
            for log in reversed(state.log_history):
                # 如果日志中包含"loss"字段
                if "loss" in log:
                    # 记录训练损失
                    values["Training Loss"] = log["loss"]
                    break  # 找到损失值后跳出循环

            # 如果首列为"Epoch"，记录当前 epoch 数值
            if self.first_column == "Epoch":
                values["Epoch"] = int(state.epoch)
            else:
                # 否则记录当前全局步数
                values["Step"] = state.global_step
            
            # 设置指标键名前缀为"eval"
            metric_key_prefix = "eval"
            # 遍历每个指标
            for k in metrics:
                # 如果指标以"_loss"结尾，去除后缀作为指标键名前缀
                if k.endswith("_loss"):
                    metric_key_prefix = re.sub(r"\_loss$", "", k)
            
            # 移除指标中的特定项
            _ = metrics.pop("total_flos", None)
            _ = metrics.pop("epoch", None)
            _ = metrics.pop(f"{metric_key_prefix}_runtime", None)
            _ = metrics.pop(f"{metric_key_prefix}_samples_per_second", None)
            _ = metrics.pop(f"{metric_key_prefix}_steps_per_second", None)
            _ = metrics.pop(f"{metric_key_prefix}_jit_compilation_time", None)
            
            # 遍历剩余的指标项
            for k, v in metrics.items():
                # 将指标键名分割并大写首字母，组成指标名
                splits = k.split("_")
                name = " ".join([part.capitalize() for part in splits[1:]])
                # 如果指标名为"Loss"，修改为"Validation Loss"
                if name == "Loss":
                    name = "Validation Loss"
                # 记录指标值
                values[name] = v
            
            # 将记录的值写入训练追踪器
            self.training_tracker.write_line(values)
            # 移除追踪器的子项
            self.training_tracker.remove_child()
            # 清空预测进度条对象
            self.prediction_bar = None
            # 设置下次更新为强制更新状态
            self._force_next_update = True

    # 定义训练结束处理函数
    def on_train_end(self, args, state, control, **kwargs):
        # 更新训练追踪器状态
        self.training_tracker.update(
            state.global_step,
            comment=f"Epoch {int(state.epoch)}/{state.num_train_epochs}",  # 添加评论，显示当前 epoch 进度
            force_update=True,  # 强制更新标志置为真
        )
        # 清空训练追踪器对象
        self.training_tracker = None
```