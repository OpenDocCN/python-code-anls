# `.\transformers\debug_utils.py`

```py
# 导入collections模块，用于操作Python内置的集合数据类型
import collections

# 从当前包中导入utils模块中的ExplicitEnum、is_torch_available和logging对象
from .utils import ExplicitEnum, is_torch_available, logging

# 如果torch可用，导入torch模块
if is_torch_available():
    import torch

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义一个调试类，用于检测和理解模型在哪些地方开始变得非常大或非常小，以及更重要的是`nan`或`inf`的权重和激活元素
class DebugUnderflowOverflow:
    """
    This debug class helps detect and understand where the model starts getting very large or very small, and more
    importantly `nan` or `inf` weight and activation elements.

    There are 2 working modes:

    1. Underflow/overflow detection (default)
    2. Specific batch absolute min/max tracing without detection

    Mode 1: Underflow/overflow detection

    To activate the underflow/overflow detection, initialize the object with the model :

    ```python
    debug_overflow = DebugUnderflowOverflow(model)
    ```py

    then run the training as normal and if `nan` or `inf` gets detected in at least one of the weight, input or output
    elements this module will throw an exception and will print `max_frames_to_save` frames that lead to this event,
    each frame reporting

    1. the fully qualified module name plus the class name whose `forward` was run
    2. the absolute min and max value of all elements for each module weights, and the inputs and output

    For example, here is the header and the last few frames in detection report for `google/mt5-small` run in fp16
    mixed precision :

    ```
    Detected inf/nan during batch_number=0
    Last 21 forward frames:
    abs min  abs max  metadata
    [...]
                      encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
    2.17e-07 4.50e+00 weight
    1.79e-06 4.65e+00 input[0]
    2.68e-06 3.70e+01 output
                      encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
    8.08e-07 2.66e+01 weight
    1.79e-06 4.65e+00 input[0]
    1.27e-04 2.37e+02 output
                      encoder.block.2.layer.1.DenseReluDense.wo Linear
    1.01e-06 6.44e+00 weight
    0.00e+00 9.74e+03 input[0]
    3.18e-04 6.27e+04 output
                      encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
    1.79e-06 4.65e+00 input[0]
    3.18e-04 6.27e+04 output
                      encoder.block.2.layer.1.dropout Dropout
    3.18e-04 6.27e+04 input[0]
    0.00e+00      inf output
    ```py

    You can see here, that `T5DenseGatedGeluDense.forward` resulted in output activations, whose absolute max value was
    # 创建一个调试工具类，用于检测模型中的浮点数溢出和下溢问题
    class DebugUnderflowOverflow:
        # 初始化调试工具类，接受模型对象以及一些可选参数
        def __init__(self, model, max_frames_to_save=21, trace_batch_nums=[]):
            # 保存模型对象
            self.model = model
            # 设置记录的最大帧数，默认为21
            self.max_frames_to_save = max_frames_to_save
            # 设置要跟踪的批次号列表，默认为空列表
            self.trace_batch_nums = trace_batch_nums
            # 未指定的情况下，设置中止训练的批次号为 None
            self.abort_after_batch_num = None
    
        # 在模型执行前的钩子函数，用于跟踪模型的浮点数情况
        def forward_pre_hook(self, module, inputs):
            # 检查是否达到了中止训练的批次号
            if self.abort_after_batch_num is not None and inputs[-1] >= self.abort_after_batch_num:
                # 如果达到了中止训练的批次号，则抛出异常中止训练
                raise RuntimeError("Aborting training after batch %d" % inputs[-1])
            # 如果当前批次号在跟踪的批次号列表中，或者跟踪的批次号列表为空，则进行跟踪
            if not self.trace_batch_nums or inputs[-1] in self.trace_batch_nums:
                # 获取模型的参数
                params = [p.data for p in self.model.parameters()]
                # 获取绝对最小值和最大值
                min_val = min([p.min() for p in params])
                max_val = max([p.max() for p in params])
                # 将当前批次号和最小最大值信息添加到跟踪记录中
                self.trace.append((inputs[-1], min_val, max_val))
                # 如果跟踪记录的长度超过了设定的最大帧数，则删除最早的记录
                if len(self.trace) > self.max_frames_to_save:
                    del self.trace[0]
    
        # 开始跟踪模型的浮点数情况
        def enable(self, abort_after_batch_num=None):
            # 清空之前的跟踪记录
            self.trace = []
            # 设置中止训练的批次号
            self.abort_after_batch_num = abort_after_batch_num
            # 注册 forward_pre_hook 钩子函数
            self.hook_handle = self.model.register_forward_pre_hook(self.forward_pre_hook)
    
        # 停止跟踪模型的浮点数情况
        def disable(self):
            # 移除 forward_pre_hook 钩子函数
            self.hook_handle.remove()
            # 清空跟踪记录
            self.trace = []
    
        # 获取跟踪记录
        def get_trace(self):
            return self.trace
    # 初始化方法，设置模型、最大保存帧数、跟踪的批次号列表、在某个批次后中止的批次号
    def __init__(self, model, max_frames_to_save=21, trace_batch_nums=[], abort_after_batch_num=None):
        # 设置模型、跟踪的批次号列表、中止的批次号
        self.model = model
        self.trace_batch_nums = trace_batch_nums
        self.abort_after_batch_num = abort_after_batch_num

        # 保存最近的帧以便在遇到inf/nan时立即转储，以提供问题出现的上下文
        self.frames = collections.deque([], max_frames_to_save)
        self.frame = []
        self.batch_number = 0
        self.total_calls = 0
        self.detected_overflow = False
        self.prefix = "                 "

        # 分析模型
        self.analyse_model()

        # 注册前向钩子
        self.register_forward_hook()

    # 保存帧的方法
    def save_frame(self, frame=None):
        # 如果有帧，则扩展帧
        if frame is not None:
            self.expand_frame(frame)
        self.frames.append("\n".join(self.frame))
        self.frame = []  # 开始一个新的帧

    # 扩展帧的方法
    def expand_frame(self, line):
        self.frame.append(line)

    # 跟踪帧的方法
    def trace_frames(self):
        print("\n".join(self.frames))
        self.frames = []

    # 重置保存的帧
    def reset_saved_frames(self):
        self.frames = []

    # 转储保存的帧
    def dump_saved_frames(self):
        print(f"\nDetected inf/nan during batch_number={self.batch_number}")
        print(f"Last {len(self.frames)} forward frames:")
        print(f"{'abs min':8} {'abs max':8} metadata")
        print("\n".join(self.frames))
        print("\n\n")
        self.frames = []

    # 分析模型的方法
    def analyse_model(self):
        # 提取完全限定的模块名称，以便在运行时报告。例如：
        # encoder.block.2.layer.0.SelfAttention.o
        #
        # 对于共享权重，只有第一个共享模块名称将被注册
        self.module_names = {m: name for name, m in self.model.named_modules()}
        # self.longest_module_name = max(len(v) for v in self.module_names.values())

    # 分析变量的方法
    def analyse_variable(self, var, ctx):
        if torch.is_tensor(var):
            self.expand_frame(get_abs_min_max(var, ctx))
            if detect_overflow(var, ctx):
                self.detected_overflow = True
        elif var is None:
            self.expand_frame(f"{'None':>17} {ctx}")
        else:
            self.expand_frame(f"{'not a tensor':>17} {ctx}")

    # 批次开始帧的方法
    def batch_start_frame(self):
        self.expand_frame(f"\n\n{self.prefix} *** Starting batch number={self.batch_number} ***")
        self.expand_frame(f"{'abs min':8} {'abs max':8} metadata")

    # 批次结束帧的方法
    def batch_end_frame(self):
        self.expand_frame(f"{self.prefix} *** Finished batch number={self.batch_number-1} ***\n\n")
    # 创建一个新的调试帧，包含模块名称和类名
    def create_frame(self, module, input, output):
        self.expand_frame(f"{self.prefix} {self.module_names[module]} {module.__class__.__name__}")

        # 遍历模块的参数，分析每个参数
        for name, p in module.named_parameters(recurse=False):
            self.analyse_variable(p, name)

        # 分析输入变量
        if isinstance(input, tuple):
            for i, x in enumerate(input):
                self.analyse_variable(x, f"input[{i}]")
        else:
            self.analyse_variable(input, "input")

        # 分析输出变量
        if isinstance(output, tuple):
            for i, x in enumerate(output):
                # 可能是元组的元组
                if isinstance(x, tuple):
                    for j, y in enumerate(x):
                        self.analyse_variable(y, f"output[{i}][{j}]")
                else:
                    self.analyse_variable(x, f"output[{i}]")
        else:
            self.analyse_variable(output, "output")

        self.save_frame()

    # 注册前向钩子
    def register_forward_hook(self):
        self.model.apply(self._register_forward_hook)

    # 实际注册前向钩子的方法
    def _register_forward_hook(self, module):
        module.register_forward_hook(self.forward_hook)

    # 前向钩子方法
    def forward_hook(self, module, input, output):
        # - input 是一个打包输入的元组（可能不是张量）
        # - output 可能是张量或张量和非张量的元组

        last_frame_of_batch = False

        # 检查是否在跟踪批次号列表中
        trace_mode = True if self.batch_number in self.trace_batch_nums else False
        if trace_mode:
            self.reset_saved_frames()

        if self.total_calls == 0:
            self.batch_start_frame()
        self.total_calls += 1

        # 计算批次号 - 批次完成时将调用批次的第一个前向钩子 - 即它将在最后被调用 - 我们知道这个批次已经完成
        if module == self.model:
            self.batch_number += 1
            last_frame_of_batch = True

        self.create_frame(module, input, output)

        # 如果是批次的最后一个帧
        # if last_frame_of_batch:
        #     self.batch_end_frame()

        if trace_mode:
            self.trace_frames()

        if last_frame_of_batch:
            self.batch_start_frame()

        # 如果检测到溢出并且不是跟踪模式，则转储保存的帧
        if self.detected_overflow and not trace_mode:
            self.dump_saved_frames()

            # 现在我们可以中止，因为继续运行没有意义
            raise ValueError(
                "DebugUnderflowOverflow: 检测到 inf/nan，中止运行。请向上滚动查看此事件之前的激活值。"
            )

        # 如果请求在特定批次后中止，则中止
        if self.abort_after_batch_num is not None and self.batch_number > self.abort_after_batch_num:
            raise ValueError(
                f"DebugUnderflowOverflow: 由于 `abort_after_batch_num={self.abort_after_batch_num}` 参数，中止在 {self.batch_number} 批次后。"
            )
# 定义一个函数，用于获取张量的绝对值的最小值和最大值，并返回格式化后的字符串
def get_abs_min_max(var, ctx):
    # 获取变量的绝对值
    abs_var = var.abs()
    # 返回格式化后的字符串，包括绝对值的最小值和最大值，以及上下文信息
    return f"{abs_var.min():8.2e} {abs_var.max():8.2e} {ctx}"


# 定义一个函数，用于检测张量是否包含任何 `nan` 或 `inf` 的条目
def detect_overflow(var, ctx):
    """
    Report whether the tensor contains any `nan` or `inf` entries.

    This is useful for detecting overflows/underflows and best to call right after the function that did some math that
    modified the tensor in question.

    This function contains a few other helper features that you can enable and tweak directly if you want to track
    various other things.

    Args:
        var: the tensor variable to check
        ctx: the message to print as a context

    Return:
        `True` if `inf` or `nan` was detected, `False` otherwise
    """
    detected = False
    # 如果张量中存在任何 `nan` 条目，则打印上下文信息并将 detected 置为 True
    if torch.isnan(var).any().item():
        detected = True
        print(f"{ctx} has nans")
    # 如果张量中存在任何 `inf` 条目，则打印上下文信息并将 detected 置为 True
    if torch.isinf(var).any().item():
        detected = True
        print(f"{ctx} has infs")

    # 如果需要监视大元素，可以启用以下代码
    if 0:  # and detected:
        # 获取绝对值大于等于100的元素
        n100 = var[torch.ge(var.abs(), 100)]
        if n100.numel() > 0:
            print(f"{ctx}:  n100={n100.numel()}")
        # 获取绝对值大于等于1000的元素
        n1000 = var[torch.ge(var.abs(), 1000)]
        if n1000.numel() > 0:
            print(f"{ctx}: n1000={n1000.numel()}")
        # 获取绝对值大于等于10000的元素
        n10000 = var[torch.ge(var.abs(), 10000)]
        if n10000.numel() > 0:
            print(f"{ctx}: n10000={n10000.numel()}")

    # 如果需要打印变量的最小值和最大值，可以启用以下代码
    if 0:
        print(f"min={var.min():9.2e} max={var.max():9.2e}")

    # 如果需要打印变量的最小值、最大值、方差和均值，可以启用以下代码
    if 0:
        print(f"min={var.min():9.2e} max={var.max():9.2e} var={var.var():9.2e} mean={var.mean():9.2e} ({ctx})")

    # 返回 detected，表示是否检测到 `inf` 或 `nan`
    return detected


# 定义一个枚举类，包含调试选项
class DebugOption(ExplicitEnum):
    UNDERFLOW_OVERFLOW = "underflow_overflow"
    TPU_METRICS_DEBUG = "tpu_metrics_debug"
```