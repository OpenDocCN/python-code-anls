# `.\debug_utils.py`

```py
# 导入collections模块，用于管理特定数据结构的集合
import collections

# 从当前目录下的utils模块中导入ExplicitEnum、is_torch_available和logging对象
from .utils import ExplicitEnum, is_torch_available, logging

# 如果torch可用，则导入torch模块
if is_torch_available():
    import torch

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 定义一个调试类DebugUnderflowOverflow，用于检测和理解模型何时开始变得非常大或非常小，以及重要的nan或inf权重和激活元素
class DebugUnderflowOverflow:
    """
    This debug class helps detect and understand where the model starts getting very large or very small, and more
    importantly `nan` or `inf` weight and activation elements.

    There are 2 working modes:

    1. Underflow/overflow detection (default)
    2. Specific batch absolute min/max tracing without detection

    Mode 1: Underflow/overflow detection

    To activate the underflow/overflow detection, initialize the object with the model :

    ```
    debug_overflow = DebugUnderflowOverflow(model)
    ```

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
    ```

    You can see here, that `T5DenseGatedGeluDense.forward` resulted in output activations, whose absolute max value was

    ```
    """
    # 在此类中定义初始化方法和两种工作模式的详细说明
    pass
    """
        As this module measures absolute `min`/``max` of each weight of the model on every forward it'll slow the training
        down. Therefore remember to turn it off once the debugging needs have been met.
    
        Args:
            model (`nn.Module`):
                The model to debug.
            max_frames_to_save (`int`, *optional*, defaults to 21):
                How many frames back to record
            trace_batch_nums(`List[int]`, *optional*, defaults to `[]`):
                Which batch numbers to trace (turns detection off)
            abort_after_batch_num  (`int``, *optional*):
                Whether to abort after a certain batch number has finished
    """
    # 初始化函数，用于设置对象的初始状态和属性
    def __init__(self, model, max_frames_to_save=21, trace_batch_nums=[], abort_after_batch_num=None):
        self.model = model  # 将传入的模型对象保存到实例属性中
        self.trace_batch_nums = trace_batch_nums  # 保存需要跟踪的批次号列表
        self.abort_after_batch_num = abort_after_batch_num  # 设置在哪个批次号之后终止运行

        # 创建一个LIFO（后进先出）的缓冲区，用于存储帧以便在遇到inf/nan时立即转储，以提供问题发生的上下文
        self.frames = collections.deque([], max_frames_to_save)  # 初始化一个空的固定大小的双端队列
        self.frame = []  # 初始化一个空的帧列表
        self.batch_number = 0  # 初始化批次号为0
        self.total_calls = 0  # 初始化总调用次数为0
        self.detected_overflow = False  # 初始化检测到溢出标志为False
        self.prefix = "                 "  # 初始化前缀字符串

        # 分析模型，可能是提取模型中模块的全限定名以便在运行时报告
        self.analyse_model()

        # 注册前向钩子（hook），用于在模型的前向传播过程中记录信息
        self.register_forward_hook()

    # 保存帧的方法，将当前帧内容转换为字符串并存入缓存队列中
    def save_frame(self, frame=None):
        if frame is not None:
            self.expand_frame(frame)  # 如果有指定帧，则扩展当前帧
        self.frames.append("\n".join(self.frame))  # 将当前帧转换为字符串并添加到帧缓存队列中
        self.frame = []  # 清空当前帧，以便开始新的帧记录

    # 扩展当前帧的方法，将一行文本添加到当前帧中
    def expand_frame(self, line):
        self.frame.append(line)  # 将一行文本添加到当前帧的末尾

    # 跟踪帧的方法，将所有缓存的帧内容打印出来
    def trace_frames(self):
        print("\n".join(self.frames))  # 打印所有缓存的帧内容
        self.frames = []  # 清空帧缓存队列

    # 重置保存的帧的方法，清空所有缓存的帧内容
    def reset_saved_frames(self):
        self.frames = []  # 清空帧缓存队列

    # 转储保存的帧的方法，打印检测到inf/nan时的批次号和最后保存的前向帧信息
    def dump_saved_frames(self):
        print(f"\nDetected inf/nan during batch_number={self.batch_number}")  # 打印检测到inf/nan时的批次号
        print(f"Last {len(self.frames)} forward frames:")  # 打印最后保存的前向帧数量
        print(f"{'abs min':8} {'abs max':8} metadata")  # 打印帧数据表头
        print("\n".join(self.frames))  # 打印所有缓存的前向帧内容
        print("\n\n")  # 打印额外的空行以分隔输出内容
        self.frames = []  # 清空帧缓存队列

    # 分析模型的方法，提取模型中模块的全限定名并保存到实例属性中
    def analyse_model(self):
        # 提取模型中所有模块的全限定名，保存为字典形式，键为模块名，值为全限定名
        self.module_names = {m: name for name, m in self.model.named_modules()}
        # self.longest_module_name = max(len(v) for v in self.module_names.values())

    # 分析变量的方法，根据变量类型进行不同的处理，如打印最大最小值或检测溢出
    def analyse_variable(self, var, ctx):
        if torch.is_tensor(var):  # 如果是张量
            self.expand_frame(get_abs_min_max(var, ctx))  # 获取张量的绝对最小值和最大值，并扩展当前帧
            if detect_overflow(var, ctx):  # 检测张量是否溢出
                self.detected_overflow = True  # 标记检测到溢出
        elif var is None:  # 如果变量为None
            self.expand_frame(f"{'None':>17} {ctx}")  # 扩展当前帧记录为"None"
        else:  # 变量不是张量也不是None
            self.expand_frame(f"{'not a tensor':>17} {ctx}")  # 扩展当前帧记录为"not a tensor"

    # 批次开始时记录帧的方法，记录批次号和开始信息到当前帧中
    def batch_start_frame(self):
        self.expand_frame(f"\n\n{self.prefix} *** Starting batch number={self.batch_number} ***")  # 扩展当前帧记录批次开始信息
        self.expand_frame(f"{'abs min':8} {'abs max':8} metadata")  # 扩展当前帧记录帧数据表头

    # 批次结束时记录帧的方法，记录批次号和结束信息到当前帧中
    def batch_end_frame(self):
        self.expand_frame(f"{self.prefix} *** Finished batch number={self.batch_number-1} ***\n\n")  # 扩展当前帧记录批次结束信息
    def create_frame(self, module, input, output):
        # 扩展调用栈帧，包括模块名称和类名
        self.expand_frame(f"{self.prefix} {self.module_names[module]} {module.__class__.__name__}")

        # 分析模块的参数
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
                # 如果输出是元组，进一步分析内部元素
                if isinstance(x, tuple):
                    for j, y in enumerate(x):
                        self.analyse_variable(y, f"output[{i}][{j}]")
                else:
                    self.analyse_variable(x, f"output[{i}]")
        else:
            self.analyse_variable(output, "output")

        # 保存当前帧信息
        self.save_frame()

    def register_forward_hook(self):
        # 对模型应用前向钩子
        self.model.apply(self._register_forward_hook)

    def _register_forward_hook(self, module):
        # 注册前向钩子到指定模块
        module.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        # - input 是一个打包输入的元组（可能包含非张量）
        # - output 可能是一个张量或者张量和非张量的元组

        last_frame_of_batch = False

        # 判断是否在跟踪批次号中
        trace_mode = True if self.batch_number in self.trace_batch_nums else False
        if trace_mode:
            self.reset_saved_frames()

        # 如果是第一次调用，则开始一个新批次的帧
        if self.total_calls == 0:
            self.batch_start_frame()
        self.total_calls += 1

        # 如果模块是整个模型，增加批次号并标记为批次的最后一帧
        if module == self.model:
            self.batch_number += 1
            last_frame_of_batch = True

        # 创建调用帧
        self.create_frame(module, input, output)

        # 如果是批次的最后一帧，则执行批次结束帧操作
        if last_frame_of_batch:
            self.batch_start_frame()

        # 如果在跟踪模式中，追踪帧
        if trace_mode:
            self.trace_frames()

        # 如果检测到溢出或下溢，并且不在跟踪模式中，则转储保存的帧信息并抛出异常
        if self.detected_overflow and not trace_mode:
            self.dump_saved_frames()
            raise ValueError(
                "DebugUnderflowOverflow: inf/nan detected, aborting as there is no point running further. "
                "Please scroll up above this traceback to see the activation values prior to this event."
            )

        # 如果请求在特定批次之后中止，则抛出异常
        if self.abort_after_batch_num is not None and self.batch_number > self.abort_after_batch_num:
            raise ValueError(
                f"DebugUnderflowOverflow: aborting after {self.batch_number} batches due to"
                f" `abort_after_batch_num={self.abort_after_batch_num}` arg"
            )
# 计算变量的绝对值的最小值和最大值，并返回格式化的字符串
def get_abs_min_max(var, ctx):
    # 计算变量的绝对值
    abs_var = var.abs()
    # 返回格式化的字符串，包括绝对值的最小值和最大值，以及上下文信息 ctx
    return f"{abs_var.min():8.2e} {abs_var.max():8.2e} {ctx}"


# 检测张量变量中是否包含 `nan` 或 `inf` 条目，并打印相关消息
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
    # 检测是否存在 `nan` 条目，若存在则设置 detected 为 True 并打印包含 `nans` 的上下文信息 ctx
    if torch.isnan(var).any().item():
        detected = True
        print(f"{ctx} has nans")
    # 检测是否存在 `inf` 条目，若存在则设置 detected 为 True 并打印包含 `infs` 的上下文信息 ctx
    if torch.isinf(var).any().item():
        detected = True
        print(f"{ctx} has infs")

    # 如果需要监视大的元素，可以启用以下功能
    if 0:  # and detected:
        # 打印绝对值大于等于 100 的元素数量
        n100 = var[torch.ge(var.abs(), 100)]
        if n100.numel() > 0:
            print(f"{ctx}:  n100={n100.numel()}")
        # 打印绝对值大于等于 1000 的元素数量
        n1000 = var[torch.ge(var.abs(), 1000)]
        if n1000.numel() > 0:
            print(f"{ctx}: n1000={n1000.numel()}")
        # 打印绝对值大于等于 10000 的元素数量
        n10000 = var[torch.ge(var.abs(), 10000)]
        if n10000.numel() > 0:
            print(f"{ctx}: n10000={n10000.numel()}")

    # 如果需要打印最小值和最大值，可以启用以下功能
    if 0:
        print(f"min={var.min():9.2e} max={var.max():9.2e}")

    # 如果需要打印最小值、最大值、方差和均值，可以启用以下功能
    if 0:
        print(f"min={var.min():9.2e} max={var.max():9.2e} var={var.var():9.2e} mean={var.mean():9.2e} ({ctx})")

    # 返回是否检测到 `inf` 或 `nan` 的布尔值
    return detected


# 调试选项的枚举类，列出了一些调试选项
class DebugOption(ExplicitEnum):
    # 检测下溢和上溢
    UNDERFLOW_OVERFLOW = "underflow_overflow"
    # TPU 指标调试
    TPU_METRICS_DEBUG = "tpu_metrics_debug"
```