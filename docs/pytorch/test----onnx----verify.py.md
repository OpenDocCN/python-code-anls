# `.\pytorch\test\onnx\verify.py`

```
import difflib  # 导入 difflib 模块，用于执行字符串之间的差异和比较操作
import io  # 导入 io 模块，提供了对 Python 内置的 I/O 操作的支持

import numpy as np  # 导入 NumPy 库，用于数值计算
import onnx  # 导入 ONNX 库，用于处理和操作 ONNX 格式的模型
import onnx.helper  # 导入 ONNX 的辅助函数，用于创建和操作 ONNX 模型中的各种元素

import torch  # 导入 PyTorch 深度学习框架
import torch.jit  # 导入 PyTorch 的 JIT 编译器，用于将 PyTorch 模型编译成 Torch 脚本或 TorchScript
import torch.onnx  # 导入 PyTorch 的 ONNX 导出功能，用于将 PyTorch 模型导出为 ONNX 格式

def colonize(msg, sep=": "):
    # 如果消息为空字符串，则直接返回空字符串
    if not msg:
        return ""
    else:
        # 否则返回消息和指定的分隔符的组合字符串
        return msg + sep

class Errors:
    """
    An error-collecting object which supports error recovery.

    It is intended to be used like a context manager:

    >>> with Errors("Top-level error message") as errs:
    >>>     ...
    """

    def __init__(self, msg, rtol=1e-3, atol=1e-5):
        # 初始化 Errors 类的实例
        self.msg = msg  # 设置错误消息
        self.errors = []  # 初始化错误列表
        self.context = []  # 初始化上下文列表
        self.rtol = rtol  # 设置相对容差
        self.atol = atol  # 设置绝对容差

        # 定义一个内部异常类 ShortCircuit，用于错误的短路处理
        class ShortCircuit(Exception):
            pass

        self.exc_class = ShortCircuit  # 将 ShortCircuit 类赋值给实例变量 exc_class

    def requireAlmostEqual(self, x, y, msg=None):
        """
        Test that x and y are nearly equal (equal within self.rtol
        precision); aborts execution if they are not.
        """
        self.almostEqualAndThen(x, y, msg, self.failWith)

    def checkAlmostEqual(self, x, y, msg=None):
        """
        Test that x and y are nearly equal (equal within self.rtol
        precision), but continue execution even if they are not equal.

        To prevent error cascades, you should remember to call "failIfErrs"
        at some later point in time.
        """
        self.almostEqualAndThen(x, y, msg, self.addErr)

    def almostEqualAndThen(self, x, y, msg, k):
        """
        Helper for implementing "requireAlmostEqual" and "checkAlmostEqual".
        Upon failure, invokes continuation "k" with the error message.

        At the moment, only tests on "numpy.ndarray" are supported.
        """
        # 如果 x 和 y 均为 numpy.ndarray 类型，则使用 assert_allclose 进行数值比较
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            np.testing.assert_allclose(
                x, y, rtol=self.rtol, atol=self.atol, equal_nan=True, verbose=True
            )
        else:
            # 否则抛出运行时异常，表示不支持的几乎相等测试
            raise RuntimeError("Unsupported almost equal test")

    def requireEqual(self, x, y, msg=None):
        """
        Test that x and y are equal; aborts execution if they are not.
        """
        self.equalAndThen(x, y, msg, self.failWith)

    def checkEqual(self, x, y, msg=None):
        """
        Test that x and y are equal, but continue execution even if they are not equal.

        To prevent error cascades, you should remember to call "failIfErrs"
        at some later point in time.
        """
        self.equalAndThen(x, y, msg, self.addErr)
    def equalAndThen(self, x, y, msg, k):
        """
        Helper for implementing "requireEqual" and "checkEqual".  Upon failure,
        invokes continuation "k" with the error message.
        """
        if isinstance(x, onnx.TensorProto) and isinstance(y, onnx.TensorProto):
            # 若 x 和 y 均为 TensorProto 对象，则递归比较它们的名称和数据
            self.equalAndThen(x.name, y.name, msg, k)
            # 使用 numpy 进行数据的比较
            t1 = onnx.numpy_helper.to_array(x)
            t2 = onnx.numpy_helper.to_array(y)
            new_msg = f"{colonize(msg)}In embedded parameter '{x.name}'"
            # 递归比较 TensorProto 对象的数据
            self.equalAndThen(t1, t2, new_msg, k)
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            # 若 x 和 y 均为 numpy 数组，则使用 np.testing.assert_equal 进行比较
            np.testing.assert_equal(x, y)
        else:
            # 若 x 和 y 类型不匹配，则转换为字符串比较
            if x != y:
                # 若字符串较长或包含换行符，则输出详细的比较结果
                sx = str(x)
                sy = str(y)
                if len(sx) > 40 or len(sy) > 40 or "\n" in sx or "\n" in sy:
                    # 较长形式的比较输出
                    l = "=" * 50
                    k(
                        "\n{}The value\n{}\n{}\n{}\n\ndoes not equal\n\n{}\n{}\n{}".format(
                            colonize(msg, ":\n"), l, sx, l, l, sy, l
                        )
                    )
                else:
                    # 简单形式的比较输出
                    k(f"{colonize(msg)}{sx} != {sy}")

    def requireMultiLineEqual(self, x, y, msg=None):
        """
        Test that long, multi-line strings x and y are equal;
        aborts execution if they are not.
        """
        # 调用 multiLineEqualAndThen 方法比较多行字符串 x 和 y
        self.multiLineEqualAndThen(x, y, msg, self.failWith)

    def multiLineEqualAndThen(self, x, y, msg, k):
        """
        Helper for implementing "requireMultiLineEqual".  Upon failure,
        invokes continuation "k" with the error message.
        """
        # 若未提供 msg，则默认为 "Strings are not equal"
        if msg is None:
            msg = "Strings are not equal"
        # 若 x 和 y 不相等，则使用 difflib.ndiff 生成它们的差异信息
        if x != y:
            diff = difflib.ndiff(x.splitlines(True), y.splitlines(True))
            # 调用 continuation "k" 输出差异信息
            k("{}{}".format(colonize(msg, ":\n\n"), "".join(diff)))

    def addErr(self, msg):
        """
        Add an error to the error context, but continue executing.
        """
        # TODO: 将上下文信息作为元数据附加在 msg 中，稍后再决定如何格式化输出
        msg_w_ctx = msg
        for c in reversed(self.context):
            msg += "\n\n  * " + "\n    ".join(c.splitlines())
        # 将错误消息添加到错误列表中
        self.errors.append(msg)

    def fail(self):
        """
        Immediately fail and short-circuit to the next recovery context.

        NB: It is an error to "fail" without having added any errors to
        the error context.
        """
        # 抛出预定义的异常，立即终止执行
        raise self.exc_class

    def failWith(self, msg):
        """
        Add an error to the error context, and then short-circuit.
        """
        # 将错误消息添加到错误列表中
        self.addErr(msg)
        # 立即终止执行
        self.fail()
    def failIfErrs(self):
        """
        如果存在任何错误上下文中的错误，则立即中断执行。

        这用于防止错误级联。
        """
        # 如果错误列表不为空
        if self.errors:
            # 调用 self.fail() 方法来处理错误
            self.fail()

    def recover(self):
        """
        返回一个上下文管理器，可用于在错误发生时进行恢复。示例用法：

        >>> with errs.recover():
        >>>     ...
        """
        # 获取当前对象的引用
        parent_self = self

        class Recover:
            def __enter__(self):
                # 在进入上下文时不执行任何操作
                pass

            def __exit__(self, exc_type, exc_value, traceback):
                # 如果捕获到与预期的异常类型相符合的异常
                if exc_type == parent_self.exc_class:
                    # 返回 True 表示异常已被处理
                    return True

        # 返回定义好的 Recover 类的实例作为上下文管理器
        return Recover()

    def addErrCtxt(self, msg):
        """
        返回一个上下文管理器，用于在代码片段周围添加额外的上下文信息，
        例如错误发生的位置或适用于该区域所有错误的提示。示例用法：

        >>> with errs.addErrCtx("Some text"):
        >>>     ...
        """
        # 获取当前对象的引用
        parent_self = self

        class AddContext:
            def __enter__(self):
                # 将错误上下文信息添加到父对象的上下文列表中
                parent_self.context.append(msg)

            def __exit__(self, exc_type, exc_value, traceback):
                # 在退出上下文时，从父对象的上下文列表中移除最后一个元素
                parent_self.context.pop()

        # 返回定义好的 AddContext 类的实例作为上下文管理器
        return AddContext()

    def __enter__(self):
        # 进入上下文时，直接返回当前对象自身
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # 如果存在错误
        if self.errors:
            # 将错误信息格式化为字符串
            errors_msg = "\n\n".join("ERROR: " + x for x in self.errors)
            final_msg = "{}\n{}\n{}".format(self.msg, "-" * 70, errors_msg)
            # 抛出断言错误，包含详细的错误信息
            raise AssertionError(final_msg)
        # 如果捕获到预期的异常类型
        if exc_type == self.exc_class:
            # 抛出运行时错误，表明 ShortCircuit 被触发，但未记录任何错误
            raise RuntimeError("ShortCircuit was raised, but no errors were recorded")
# 定义一个函数，用于将模型导出为 ONNX 格式，并在指定的 ONNX 后端中导入和验证模型的结果
def verify(
    model,  # 输入参数：要导出为 ONNX 格式的 PyTorch 模型
    args,  # 输入参数：导出模型时的参数
    backend,  # 输入参数：指定的 ONNX 后端
    verbose=False,  # 可选参数：是否显示详细信息，默认为 False
    training=torch.onnx.TrainingMode.EVAL,  # 可选参数：导出模型时的训练模式，默认为 EVAL 模式
    rtol=1e-3,  # 可选参数：相对误差容忍度，默认为 1e-3
    atol=1e-7,  # 可选参数：绝对误差容忍度，默认为 1e-7
    test_args=2,  # 可选参数：测试参数，默认为 2
    do_constant_folding=True,  # 可选参数：是否执行常量折叠，默认为 True
    opset_version=None,  # 可选参数：ONNX 操作集版本号，默认为 None
    keep_initializers_as_inputs=True,  # 可选参数：是否将初始化参数保留为输入，默认为 True
    add_node_names=False,  # 可选参数：是否为节点添加名称，默认为 False
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # 可选参数：操作符导出类型，默认为 ONNX
    input_names=None,  # 可选参数：输入名称列表，默认为 None
    dynamic_axes=None,  # 可选参数：动态轴列表，默认为 None
    remained_onnx_input_idx=None,  # 可选参数：保留的 ONNX 输入索引，默认为 None
):
    """
    Export a model into ONNX, import it into a specified ONNX backend, and then
    on a few random inputs verify that PyTorch and the backend produced the same
    results.  Requires onnx to be installed.

    This function may spuriously fail: some operators are implemented with
    different numerical precision in an ONNX backend, in which case an unstable
    network (e.g., Inception) may blow up these numerical instabilities.  This
    situation is less likely to happen if your model has been trained.  However,
    if this is not the case, you may have found a bug!  Please report it to the
    PyTorch developers.  You can also debug the issue yourself by removing
    suffixes of operators from your model until verification passes.

    For reproducibility, we recommend explicitly setting PyTorch's seed before
    invoking this function.
    """
    # 定义一个函数 _nested_map，用于根据条件 condition 对对象进行映射操作 fn
    def _nested_map(condition, fn, condition_msg=None):
        # 内部函数 _map 对输入的 obj 进行处理
        def _map(obj):
            # 如果 obj 满足条件 condition，则应用 fn 函数
            if condition(obj):
                return fn(obj)
            # 如果 obj 是 None，则直接返回 None
            elif obj is None:
                return None
            # 如果 obj 是 list 或 tuple 类型，则递归地对其元素应用 _map 函数，并返回同类型的对象
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_map(x) for x in obj)
            # 其他情况抛出 ValueError 异常，提示无法处理的类型
            else:
                raise ValueError(
                    "Auto nesting doesn't know how to process "
                    "an input object of type "
                    + torch.typename(obj)
                    + (
                        ". Accepted types: "
                        + condition_msg
                        + ", or lists/tuples of them"
                        if condition_msg
                        else ""
                    )
                )

        return _map
    def _iter_filter(condition, allow_unknown=False, condition_msg=None):
        # 定义一个内部函数 _iter，用于递归遍历对象并根据条件筛选
        def _iter(obj):
            # 如果对象满足条件，则生成该对象
            if condition(obj):
                yield obj
            # 如果对象为 None，则直接返回
            elif obj is None:
                return
            # 如果对象是列表或元组，则递归遍历每个元素
            elif isinstance(obj, (list, tuple)):
                for o in obj:
                    yield from _iter(o)
            # 如果允许未知类型且对象不满足条件，则生成该对象
            elif allow_unknown:
                yield obj
            # 否则抛出异常，说明自动嵌套无法处理该类型的对象
            else:
                raise ValueError(
                    "Auto nesting doesn't know how to process "
                    "an input object of type "
                    + torch.typename(obj)
                    + (
                        ". Accepted types: "
                        + condition_msg
                        + ", or lists/tuples of them"
                        if condition_msg
                        else ""
                    )
                )

        return _iter

    def is_tensor(o):
        # 判断对象是否为 torch.Tensor 类型
        return isinstance(o, torch.Tensor)

    _iter_tensors = _iter_filter(is_tensor, condition_msg="Tensors")

    def randomize_arg(arg):
        # 克隆参数的数据，并进行随机化处理
        new_data = arg.data.clone()
        # 对于浮点数类型的张量，进行均匀随机初始化
        if arg.is_floating_point():
            new_data.uniform_()
        return torch.autograd.Variable(new_data, requires_grad=arg.requires_grad)

    randomize_args = _nested_map(is_tensor, randomize_arg)

    def backend_args(args):
        # TODO: onnx 应当接受可迭代对象
        # 将输入参数 args 中的每个 Tensor 数据转换为 numpy 数组，并移到 CPU 上处理
        return tuple(v.data.cpu().numpy() for v in _iter_tensors(args))

    def load_bytes(b):
        # 重置字节流的指针位置到起始处
        b.seek(0)
        # 使用 onnx 库加载字节流中的数据
        x = onnx.load(b)
        # 移除文档字符串中的堆栈跟踪信息，以便进行比较
        onnx.helper.strip_doc_string(x)
        return x

    # 处理将单个 Tensor 作为参数的常见情况
    if isinstance(args, torch.Tensor):
        args = (args,)
```