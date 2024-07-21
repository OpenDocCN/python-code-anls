# `.\pytorch\test\test_functionalization.py`

```
# Owner(s): ["module: codegen"]

# 导入单元测试模块
import unittest
# 导入上下文管理器的空上下文
from contextlib import nullcontext

# 导入 torch 库
import torch
# 导入 torch 内部的 Python 调度模块
from torch._dispatch.python import (
    enable_crossref_functionalize,
    enable_python_dispatcher,
)
# 导入 torch._subclasses.functional_tensor 模块
from torch._subclasses.functional_tensor import (
    dispatch_functionalize,
    FunctionalTensor,
    FunctionalTensorMode,
)
# 导入 torch.fx.experimental.proxy_tensor 模块
from torch.fx.experimental.proxy_tensor import make_fx
# 导入 torch.fx.passes.reinplace 模块
from torch.fx.passes.reinplace import reinplace
# 导入 torch.multiprocessing.reductions 模块
from torch.multiprocessing.reductions import StorageWeakRef
# 导入 torch.testing._internal.common_utils 模块
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfail_inherited_tests,
)
# 导入 torch.testing._internal.logging_tensor 模块
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensor
# 导入 torch.utils._pytree 模块
from torch.utils import _pytree as pytree
# 导入 torch.utils._pytree 中的 tree_map_only 函数
from torch.utils._pytree import tree_map_only


# 检查两个 Tensor 是否共享存储
def are_aliased(x, y):
    x_storage = StorageWeakRef(x.storage())
    y_storage = StorageWeakRef(y.storage())
    return x_storage == y_storage


# 函数用于将普通 Tensor 转换为功能化 Tensor，并保留 requires_grad 属性
# 使用 tree_map_only 对输入中的所有 Tensor 进行转换
def _functionalize(
    f, *, reapply_views: bool, crossref: bool, skip_input_mutations: bool = False
):
    # 将普通 Tensor 转换为功能化 Tensor 的内部函数
    def to_fun(t: torch.Tensor):
        func_t = torch._to_functional_tensor(t)
        func_t.requires_grad = t.requires_grad
        return func_t

    # 包装函数，实现功能化
    def wrapped(*inputs):
        ctx = nullcontext()
        if crossref:
            ctx = enable_crossref_functionalize()
        with ctx:
            # 使用 tree_map_only 将输入中的所有 Tensor 转换为功能化 Tensor
            inputs_functional = tree_map_only(torch.Tensor, to_fun, inputs)
            # 开启功能化
            torch._enable_functionalization(reapply_views=reapply_views)
            try:
                # 调用原始函数进行计算
                out = f(*inputs_functional)
            finally:
                # 关闭功能化
                torch._disable_functionalization()

            # 获取扁平化的输入和功能化后的输入
            flat_inputs = pytree.tree_leaves(inputs)
            flat_inputs_functional = pytree.tree_leaves(inputs_functional)

            # 同步功能化后的输入和恢复为普通 Tensor
            for inpt, input_functional in zip(flat_inputs, flat_inputs_functional):
                torch._sync(input_functional)
                inpt_new = torch._from_functional_tensor(input_functional)
                if inpt_new is not inpt and not skip_input_mutations:
                    # 对输入进行更新，解决功能化的不足之处
                    if inpt_new.shape == inpt.shape:
                        inpt.copy_(inpt_new)

            # 同步输出，并恢复为普通 Tensor
            tree_map_only(torch.Tensor, torch._sync, out)
            out_unwrapped = tree_map_only(
                torch.Tensor, torch._from_functional_tensor, out
            )
            return out_unwrapped

    return wrapped


# 根据条件跳过测试，若测试使用了 TorchDynamo
@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO, "https://github.com/pytorch/pytorch/issues/81457"
)
# 定义功能化测试类，继承自 TestCase
class TestFunctionalization(TestCase):
    # 是否开启交叉引用
    crossref = False
    # 定义一个方法用于获取函数的日志
    def get_logs(self, func, *inpts, reapply_views=False, run_reinplace=False):
        # 复制输入参数，并仅保留包含torch.Tensor类型的元素
        inpts_clone = tree_map_only(torch.Tensor, torch.clone, inpts)
        # 将函数转换为FX，并进行功能化处理，如果需要重新应用视图，则跨引用对象为self.crossref
        traced_f = make_fx(
            _functionalize(func, reapply_views=reapply_views, crossref=self.crossref)
        )(*inpts)
        # 如果需要运行重新替换操作，则在复制的输入参数上运行重新替换函数
        if run_reinplace:
            traced_f = reinplace(traced_f, *inpts_clone)
        # 返回FX对象的代码属性
        return traced_f.code

    # 定义一个方法用于断言功能化的正确性
    def assert_functionalization(
        self, func, *inpts, reapply_views=False, mutated_input_metadata=False
    ):
        # 对输入参数进行三次复制
        clones1 = tree_map_only(torch.Tensor, torch.clone, inpts)
        clones2 = tree_map_only(torch.Tensor, torch.clone, inpts)
        clones3 = tree_map_only(torch.Tensor, torch.clone, inpts)

        # 比较原始函数和功能化后函数的输出（以及变异的输入）
        out_ref = func(*inpts)
        out_functional = _functionalize(
            func, reapply_views=reapply_views, crossref=self.crossref
        )(*clones1)

        # 只有在reapply_views=True时，重新替换操作才有效
        functional_func = make_fx(
            _functionalize(func, reapply_views=True, crossref=self.crossref)
        )(*clones2)
        reinplace_func = reinplace(functional_func, *clones2)

        # 注意：目前需要在此处传入新的输入参数，因为make_fx会直接修改你用于追踪的输入。
        # 一旦这个问题解决了，我们可以清理这部分代码。
        out_reinplace = reinplace_func(*clones3)

        # functionalize()存在的缺陷：输入元数据的变异不能正确传播，因此我们需要跳过这里的检查，用于测试这种情况。
        if not mutated_input_metadata:
            # 展平输入参数列表和复制后的列表，用于逐个比较
            flat_inpts = pytree.tree_leaves(inpts)
            flat_clones1 = pytree.tree_leaves(clones1)
            flat_clones3 = pytree.tree_leaves(clones3)
            for inpt, input_clone, input_clone3 in zip(
                flat_inpts, flat_clones1, flat_clones3
            ):
                # 断言输入参数和其克隆的相等性，检查输入的变化是否发生
                self.assertEqual(
                    inpt, input_clone
                )
                self.assertEqual(inpt, input_clone3)

        # 处理多张量输出的测试情况
        if isinstance(out_ref, tuple):
            out_refs, out_functionals, out_reinplaces = (
                list(out_ref),
                list(out_functional),
                list(out_reinplace),
            )
        else:
            out_refs, out_functionals, out_reinplaces = (
                [out_ref],
                [out_functional],
                [out_reinplace],
            )

        # 逐个比较原始函数输出、功能化后函数输出以及重新替换后函数输出的一致性
        for out_ref_, out_functional_, out_reinplace_ in zip(
            out_refs, out_functionals, out_reinplaces
        ):
            self.assertEqual(out_ref_, out_functional_)
            self.assertEqual(out_ref_, out_reinplace_)
    def test_save_for_backwards_segfault(self):
        # 创建 LoggingTensor 对象，使用随机初始化的数据创建 2x2 的张量
        inp = torch._to_functional_tensor(
            LoggingTensor(torch.randn(2, 2))
        ).requires_grad_(True)
        # 对张量应用指数函数
        inp.exp()

    def test_multiple_views_of_same_base(self):
        def f(x):
            # 创建视图 y 和 z，均对输入张量 x 进行形状重塑
            y = x.view(-1)
            z = x.view(-1)
            # 在原地对输入张量 x 的每个元素加 1
            x.add_(1)
            # y 应当已经被更新
            y2 = y + 1
            # z 也应当已经被更新
            z2 = z + 1
            return z2

        # 对函数 f 进行功能化处理，传入一个全为 1 的张量作为输入
        self.assert_functionalization(f, torch.ones(4))

    def test_freeze(self):
        def f(x):
            # 克隆输入张量 x 到 y
            y = x.clone()
            # 从 y 中获取第一个元素 z
            z = y[0]
            # 冻结 y
            torch._freeze_functional_tensor(y)
            # 在输入张量 x 上原地加 1，应当抛出 RuntimeError 异常
            self.assertRaises(RuntimeError, lambda: y.add_(1))
            # 在 z 上加 1，也应当抛出 RuntimeError 异常
            self.assertRaises(RuntimeError, lambda: z.add_(1))
            return z

        # 对函数 f 进行功能化处理，传入一个 3x3 的全为 1 的张量作为输入
        _functionalize(f, reapply_views=True, crossref=self.crossref)(torch.ones(3, 3))

    def test_copy_stride_mismatch(self):
        def f(x):
            # 创建一个空的张量 y，使用指定的尺寸和步长
            y = torch.empty_strided((2, 2), (5, 1))
            # 将输入张量 x 的数据复制到 y 中
            y.copy_(x)
            return y

        # 对函数 f 进行功能化处理，传入一个 2x2 的全为 1 的张量作为输入
        r = _functionalize(f, reapply_views=True, crossref=self.crossref)(
            torch.ones(2, 2)
        )
        # 断言返回张量 r 的步长与指定的一致
        self.assertEqual(r.stride(), (5, 1))

    def test_set_(self):
        def f(x):
            # 创建一个全为 1 的张量 y
            y = torch.ones(2)
            # 使用 x 的存储设置张量 y
            y.set_(x.storage())
            return y

        # 对函数 f 进行功能化处理，传入一个全为 1 的张量作为输入
        r = _functionalize(f, reapply_views=True, crossref=False)(torch.ones(2))
        # 断言返回张量 r 的设备为 CPU
        self.assertEqual(str(r.device), "cpu")

    def test_advanced_indexing(self):
        def f():
            # 创建一个全为 0 的 3x3 张量 x
            x = torch.zeros(3, 3)
            # 创建一个索引张量 idx，包含一个值为 0 的元素
            idx = torch.tensor([0])
            # 创建一个全为 1 的 3x1 张量 val
            val = torch.ones(3, 1)
            # 使用高级索引将 val 的值赋给 x 的列（所有行中的指定列）
            x[:, idx] = val
            return x

        # 断言函数 f 的功能化输出
        self.assert_functionalization(f)

    def test_view_clone_view_inplace(self):
        def f(input):
            # 定义形状为 [1, 1024, 128, 128] 的张量 shape
            shape = [1, 1024, 128, 128]
            # 将输入张量 input 进行形状重塑为 input_reshaped
            input_reshaped = input.view(shape)
            # 克隆 input_reshaped 到 out
            out = input_reshaped.clone()
            # 将 out 再次视图为与 input 相同的形状，并在原地应用 ReLU 函数
            r = out.view(input.shape)
            r.relu_()
            return r

        # 定义函数 g，调用 f，并对梯度进行跟踪和回溯
        def g(x):
            loss = f(x).sum()
            import torch.fx.traceback as fx_traceback
            from torch._functorch.aot_autograd import (
                setup_stacktrace_preservation_hooks,
            )

            setup_stacktrace_preservation_hooks([loss.grad_fn])
            with fx_traceback.preserve_node_meta():
                loss.backward()
            return x.grad

        # 使用异常检测模式运行函数 g，传入一个形状为 16x64x128x128 的张量作为输入
        with torch.autograd.detect_anomaly(check_nan=False):
            logs = self.get_logs(g, torch.ones(16, 64, 128, 128, requires_grad=True))
        # 断言 logs 与预期的内联文本相符
        self.assertExpectedInline(
            logs,
            """\
# 定义一个方法 `forward`，接受一个参数 `arg0_1`
def forward(self, arg0_1):
    # 使用 torch 操作符创建一个全为 1 的张量，形状为 [4, 2]，默认在 CPU 上，不固定内存
    ones = torch.ops.aten.ones.default([4, 2], device=device(type='cpu'), pin_memory=False)
    # 使用 torch 操作符复制输入参数 `arg0_1`，并指定新的形状为 [4, 2]
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2])
    # 使用 torch.ops.aten.add.Tensor 函数对 view_copy 和 ones 进行张量相加，结果赋给 add
    add = torch.ops.aten.add.Tensor(view_copy, ones);  view_copy = ones = None
    # 使用 torch.ops.aten.view_copy.default 函数对 add 张量进行视图复制，形状设置为 [4, 2]，结果赋给 view_copy_1
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None
    # 使用 torch.ops.aten.view_copy.default 函数对 view_copy_1 张量进行视图复制，形状设置为 [4, 2]，结果赋给 view_copy_2
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [4, 2])
    # 使用 torch.ops.aten.mul.Tensor 函数对 view_copy_1 张量进行自身的逐元素平方操作，结果赋给 mul
    mul = torch.ops.aten.mul.Tensor(view_copy_1, view_copy_1)
    # 使用 torch.ops.aten.copy_.default 函数将 arg0_1 张量复制到 view_copy_1 张量，结果赋给 copy_
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_copy_1);  arg0_1 = view_copy_1 = None
    # 返回 view_copy_2 张量作为函数结果
    return view_copy_2
def forward(self, arg0_1):
    # 创建一个形状为 [4, 2] 的全1张量，设备为CPU，不固定在内存中
    ones = torch.ops.aten.ones.default([4, 2], device=device(type='cpu'), pin_memory=False)
    
    # 对输入张量 arg0_1 进行形状变换，变换为 [4, 2] 的视图
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2]);  arg0_1 = None
    
    # 创建一个空张量，未指定大小和形状
    empty = torch.ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
    
    # 将 view_copy 和 ones 相加，得到张量 add
    add = torch.ops.aten.add.Tensor(view_copy, ones);  view_copy = ones = None
    
    # 对 add 张量进行元素级乘法，得到张量 mul
    mul = torch.ops.aten.mul.Tensor(add, add);  add = None
    
    # 返回 mul 张量作为结果
    return mul
    # 使用 Torch 操作符 `torch.ops.aten.aminmax.default` 对 `arg0_1` 进行计算，指定 `dim = 0`
    aminmax = torch.ops.aten.aminmax.default(arg0_1, dim=0);  arg0_1 = None
    
    # 从 aminmax 结果中获取索引为 0 的元素
    getitem = aminmax[0]
    
    # 从 aminmax 结果中获取索引为 1 的元素
    getitem_1 = aminmax[1];  aminmax = None
    
    # 返回获取到的第一个元素
    return getitem
    
    
    
            )
    
            # 调用 self.get_logs 方法获取重新放置的日志
            reinplaced_logs = self.get_logs(
                f,
                torch.arange(8, dtype=torch.float32),
                reapply_views=True,
                run_reinplace=True,
            )
    
            # 断言重新放置的日志与期望的内联日志一致
            self.assertExpectedInline(
                reinplaced_logs,
                """\
    
    
    
    注释：
# 定义一个方法 `forward`，接受一个参数 `arg0_1`
def forward(self, arg0_1):
    # 创建一个空的张量，形状为 [4]，在 CPU 上分配内存
    empty = torch.ops.aten.empty.memory_format([4], device=device(type='cpu'), pin_memory=False)
    # 创建另一个空的张量，形状也为 [4]，在 CPU 上分配内存
    empty_1 = torch.ops.aten.empty.memory_format([4], device=device(type='cpu'), pin_memory=False)
    # 对输入张量 arg0_1 沿着第一个维度计算最小和最大值，返回元组 aminmax
    aminmax = torch.ops.aten.aminmax.default(arg0_1, dim=0); arg0_1 = None
    # 从 aminmax 元组中获取第一个元素
    getitem = aminmax[0]
    # 从 aminmax 元组中获取第二个元素
    getitem_1 = aminmax[1]; aminmax = None
    # 返回第一个元素 getitem
    return getitem
        def f(x):
            # 定义函数 f，接受参数 x
            # 测试不是视图的情况下的原地操作情况
            # 这种情况值得检查，因为张量将具有一个空的 ViewMeta 栈，需要特殊处理。
            # 创建一个临时张量 tmp，其值全为 1，形状为 (4, 2)
            tmp = torch.ones(4, 2)
            # 将参数 x 变形为形状为 (4, 2) 的张量 y
            y = x.view(4, 2)
            # 将临时张量 tmp 原地加到参数 x 上
            x.add_(tmp)
            # 返回变形后的张量 y
            return y

        # 调用 assert_functionalization 方法，测试函数 f 的功能化
        self.assert_functionalization(f, torch.ones(4, 2))
        # 获取函数 f 在参数为 torch.ones(4, 2) 时的日志
        logs = self.get_logs(f, torch.ones(4, 2))
        # 使用 assertExpectedInline 方法验证日志的预期输出
        self.assertExpectedInline(
            logs,
            """
# 定义一个方法 forward，接受参数 arg0_1
def forward(self, arg0_1):
    # 使用 torch.ops.aten.ones.default 创建一个全为 1 的张量 ones，形状为 [4, 2]，在 CPU 上，不固定在内存中
    ones = torch.ops.aten.ones.default([4, 2], device=device(type='cpu'), pin_memory=False)
    # 使用 torch.ops.aten.view_copy.default 复制并返回 arg0_1 的视图，形状为 [4, 2]
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2])
    # 使用 torch.ops.aten.add.Tensor 将 ones 添加到 arg0_1，结果保存在 add 中；将 ones 置为 None
    add = torch.ops.aten.add.Tensor(arg0_1, ones);  ones = None
    # 使用 torch.ops.aten.copy_.default 将 add 的值拷贝给 arg0_1，将 arg0_1 置为 None
    copy_ = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = None
    # 使用 torch.ops.aten.view_copy.default 复制并返回 add 的视图，形状为 [4, 2]，将 add 置为 None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None
    # 返回 view_copy_1，即最终的视图
    return view_copy_1
    # 调用 TorchScript 中的 aten 操作的 copy_ 方法，传入参数 arg0_1 和 as_strided_scatter，并将结果赋给 copy_
    copy_ = torch.ops.aten.copy_.default(arg0_1, as_strided_scatter);  arg0_1 = None
    # 返回变量 as_strided_scatter
    return as_strided_scatter
    """
        )

        # NB: 即使 reapply_views=True，我们仍然期望看到 scatter 操作
        # 获取使用指定参数运行函数 f 后的日志信息，传入参数 torch.ones(2, 2)，同时设置 reapply_views=True，run_reinplace=False
        reinplaced_logs = self.get_logs(
            f, torch.ones(2, 2), reapply_views=True, run_reinplace=False
        )
        # 断言实际日志与期望的内联日志一致
        self.assertExpectedInline(
            reinplaced_logs,
            """\
def forward(self, arg0_1):
    # 使用 torch.ops.aten.as_strided.default 操作创建一个新的张量，根据给定的参数
    as_strided = torch.ops.aten.as_strided.default(arg0_1, [2], [2], 1)
    # 使用 torch.ops.aten.add.Tensor 操作对两个张量进行加法操作，结果存储在 add 中
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    # 使用 torch.ops.aten.as_strided_scatter.default 操作对输入的张量进行写入操作，并返回更新后的张量
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(arg0_1, add, [2], [2], 1);  add = None
    # 使用 torch.ops.aten.as_strided.default 操作创建一个新的张量，根据给定的参数
    as_strided_1 = torch.ops.aten.as_strided.default(as_strided_scatter, [2], [2], 1)
    # 使用 torch.ops.aten.copy_.default 操作复制一个张量并返回结果
    copy_ = torch.ops.aten.copy_.default(arg0_1, as_strided_scatter);  arg0_1 = None
    # 返回更新后的张量
    return as_strided_scatter
    # 使用 PyTorch 的 aten 操作执行张量的乘法，结果保存在 mul 变量中
    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
    # 返回乘法结果
    return mul
    """
        )
        
        # 获取经过替换的日志记录，重新应用视图，并运行替换操作
        reinplaced_logs = self.get_logs(
            f, torch.ones(2, 2), reapply_views=True, run_reinplace=True
        )
        # 断言内联的期望结果
        self.assertExpectedInline(
            reinplaced_logs,
            """\






注释：
def forward(self, arg0_1):
    # 创建一个包含两个元素的张量，值为1，设备类型为CPU，不使用固定内存
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    # 对输入张量进行克隆操作
    clone = torch.ops.aten.clone.default(arg0_1)
    # 提取克隆张量的对角线元素
    diagonal = torch.ops.aten.diagonal.default(clone)
    # 将提取的对角线元素与ones张量进行加法操作
    add = torch.ops.aten.add.Tensor(diagonal, ones);  diagonal = ones = None
    # 再次提取克隆张量的对角线元素
    diagonal_1 = torch.ops.aten.diagonal.default(clone);  clone = None
    # 对输入张量进行自身元素的逐元素乘法
    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
    # 返回乘法结果
    return mul
    def test_split(self):
        def f(x):
            # 定义一个函数 f，接收参数 x
            # 创建一个临时张量，所有元素为1
            tmp = torch.ones(2)
            # 对输入张量 x 进行分割，得到两个子张量 y1 和 y2
            y1, y2 = x.split(2)
            # 取 y2 的对角线元素作为 y3
            y3 = y2.diagonal()
            # 将临时张量 tmp 加到 y3 上
            y3.add_(tmp)
            # 计算输入张量 x 的元素平方
            z = x * x
            # 返回结果 y3
            return y3

        # 使用 assert_functionalization 方法验证函数 f 的功能
        self.assert_functionalization(f, torch.ones(4, 2))
        # 获取函数 f 在输入张量 torch.ones(4, 2) 上的日志
        logs = self.get_logs(f, torch.ones(4, 2))
        # 使用 assertExpectedInline 方法比较日志和预期输出
        self.assertExpectedInline(
            logs,
            """\
def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    # 使用 torch 操作创建一个包含两个元素的张量全为 1，设备为 CPU，不使用内存固定
    split_copy = torch.ops.aten.split_copy.Tensor(arg0_1, 2)
    # 使用 torch 操作对输入张量进行分割复制操作，分成两部分
    getitem = split_copy[0]
    getitem_1 = split_copy[1];  split_copy = None
    # 获取分割复制后的第一部分和第二部分，并释放 split_copy 变量
    diagonal_copy = torch.ops.aten.diagonal_copy.default(getitem_1);  getitem_1 = None
    # 对第二部分进行对角线复制操作，得到新的张量，释放 getitem_1 变量
    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None
    # 将对角线复制得到的张量与全为 1 的张量相加，释放 diagonal_copy 和 ones 变量
    split_copy_1 = torch.ops.aten.split_copy.Tensor(arg0_1, 2)
    # 再次对输入张量进行分割复制操作，分成两部分
    getitem_2 = split_copy_1[0]
    getitem_3 = split_copy_1[1];  split_copy_1 = None
    # 获取分割复制后的第一部分和第二部分，并释放 split_copy_1 变量
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(getitem_3, add);  getitem_3 = add = None
    # 使用对角线散射操作将 add 和第二部分进行更新散射操作，释放 getitem_3 和 add 变量
    slice_scatter = torch.ops.aten.slice_scatter.default(arg0_1, diagonal_scatter, 0, 2, 4);  diagonal_scatter = None
    # 使用切片散射操作将输入张量和对角线散射结果在指定维度上进行切片散射，释放 diagonal_scatter 变量
    split_copy_2 = torch.ops.aten.split_copy.Tensor(slice_scatter, 2)
    # 对切片散射结果进行再次分割复制操作，分成两部分
    getitem_4 = split_copy_2[0]
    getitem_5 = split_copy_2[1];  split_copy_2 = None
    # 获取分割复制后的第一部分和第二部分，并释放 split_copy_2 变量
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(getitem_5);  getitem_5 = None
    # 对第二部分进行对角线复制操作，得到新的张量，释放 getitem_5 变量
    mul = torch.ops.aten.mul.Tensor(slice_scatter, slice_scatter)
    # 使用乘法操作计算切片散射结果与自身的乘积
    copy_ = torch.ops.aten.copy_.default(arg0_1, slice_scatter);  arg0_1 = slice_scatter = None
    # 使用复制操作将切片散射结果复制给输入张量，并释放 arg0_1 和 slice_scatter 变量
    return diagonal_copy_1
    # 返回最终的对角线复制结果
    def test_split_with_sizes(self):
        def f(x):
            # 定义一个函数 f，接受参数 x
            # 在此处创建一个临时张量 tmp，所有元素为1
            tmp = torch.ones(2)
            # 使用 x 的 split_with_sizes 方法，按给定大小拆分成多个张量 y1 和 y2
            y1, y2 = x.split_with_sizes([2, 2])
            # 获取 y1 的对角线元素，存储在 y3 中
            y3 = y1.diagonal()
            # 将 tmp 加到 y3 中，使用原地加法
            y3.add_(tmp)
            # 计算张量 x 的平方，存储在 z 中
            z = x * x
            # 返回计算得到的 y3
            return y3

        # 调用 assert_functionalization 方法，验证函数 f 在给定输入下的功能化结果
        self.assert_functionalization(f, torch.ones(4, 2))
        # 获取函数 f 在给定输入下的日志
        logs = self.get_logs(f, torch.ones(4, 2))
        # 使用 assertExpectedInline 方法，验证 logs 是否符合预期输出
        self.assertExpectedInline(
            logs,
            """\
def forward(self, arg0_1):
    # 创建一个包含两个元素的张量，值为1，设备为CPU，不固定在内存中
    ones = torch.ops.aten.ones.default([2], device=device(type='cpu'), pin_memory=False)
    
    # 将输入张量按指定大小分割成多个张量，并复制结果
    split_with_sizes_copy = torch.ops.aten.split_with_sizes_copy.default(arg0_1, [2, 2])
    
    # 获取分割结果的第一个张量
    getitem = split_with_sizes_copy[0]
    
    # 获取分割结果的第二个张量，并清空中间变量
    getitem_1 = split_with_sizes_copy[1]; split_with_sizes_copy = None
    
    # 对第一个张量进行对角线复制操作
    diagonal_copy = torch.ops.aten.diagonal_copy.default(getitem); getitem = None
    
    # 将对角线复制结果与张量 ones 相加
    add = torch.ops.aten.add.Tensor(diagonal_copy, ones); diagonal_copy = ones = None
    
    # 再次按指定大小分割输入张量，并复制结果
    split_with_sizes_copy_1 = torch.ops.aten.split_with_sizes_copy.default(arg0_1, [2, 2])
    
    # 获取第一个分割结果张量
    getitem_2 = split_with_sizes_copy_1[0]
    
    # 获取第二个分割结果张量，并清空中间变量
    getitem_3 = split_with_sizes_copy_1[1]; split_with_sizes_copy_1 = None
    
    # 使用对角线散开操作将第一个分割结果张量与之前得到的 add 结果相结合
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(getitem_2, add); getitem_2 = add = None
    
    # 使用切片散开操作，将 diagonal_scatter 散开到输入张量的指定位置
    slice_scatter = torch.ops.aten.slice_scatter.default(arg0_1, diagonal_scatter, 0, 0, 2); diagonal_scatter = None
    
    # 再次按指定大小分割切片散开后的张量，并复制结果
    split_with_sizes_copy_2 = torch.ops.aten.split_with_sizes_copy.default(slice_scatter, [2, 2])
    
    # 获取第一个分割结果张量
    getitem_4 = split_with_sizes_copy_2[0]
    
    # 获取第二个分割结果张量，并清空中间变量
    getitem_5 = split_with_sizes_copy_2[1]; split_with_sizes_copy_2 = None
    
    # 对第一个分割结果张量进行对角线复制操作
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(getitem_4); getitem_4 = None
    
    # 使用张量乘法，计算切片散开张量的平方
    mul = torch.ops.aten.mul.Tensor(slice_scatter, slice_scatter)
    
    # 返回对角线复制后的结果张量
    return diagonal_copy_1
    # 调用 torch 库中的 aten 模块下的 copy_ 函数，使用默认参数 arg0_1 和 slice_scatter 进行复制操作，并清空这两个变量
    copy_ = torch.ops.aten.copy_.default(arg0_1, slice_scatter);  arg0_1 = slice_scatter = None
    # 返回 diagonal_1 变量的值
    return diagonal_1
    """,
        )  # noqa: B950

    def test_slice(self):
        # 定义内部函数 f，接受参数 x
        def f(x):
            # 创建临时变量 tmp，并将其初始化为包含四个 1 的张量
            tmp = torch.ones(4)
            # 将 x 张量进行转置，交换维度 0 和 1
            x.transpose_(1, 0)
            # 提取 x 张量的第 0 到 1 行（即第 0 行和第 1 行），存储在变量 y 中
            y = x[0:2]
            # 将 tmp 张量加到 y 中，即对 y 中的每个元素都加上对应的 tmp 元素
            y.add_(tmp)
            # 返回经过修改后的 x 张量
            return x

        # 调用 assert_functionalization 方法验证函数 f 的功能，输入参数为一个 4x2 的全 1 张量，同时启用输入参数元数据的变异
        self.assert_functionalization(f, torch.ones(4, 2), mutated_input_metadata=True)
        # 获取函数 f 在输入为一个 4x2 的全 1 张量时的日志信息
        logs = self.get_logs(f, torch.ones(4, 2))
        # 使用 assertExpectedInline 方法断言获取的日志 logs 是否与期望输出匹配
        self.assertExpectedInline(
            logs,
            """\
    # 使用 torch.ops.aten.ones.default 创建一个包含四个元素的张量，所有元素均为1，
    # 设备类型为CPU，不启用内存固定
    ones = torch.ops.aten.ones.default([4], device=device(type='cpu'), pin_memory=False)

    # 使用 torch.ops.aten.transpose_copy.int 对输入张量 arg0_1 进行转置操作
    transpose_copy = torch.ops.aten.transpose_copy.int(arg0_1, 1, 0)

    # 使用 torch.ops.aten.slice_copy.Tensor 对转置后的张量 transpose_copy 进行切片操作，
    # 保留索引为0到1的元素，返回一个新的张量，同时释放 transpose_copy 的引用
    slice_copy = torch.ops.aten.slice_copy.Tensor(transpose_copy, 0, 0, 2); transpose_copy = None

    # 使用 torch.ops.aten.add.Tensor 将 ones 和 slice_copy 相加，返回相加后的张量，
    # 同时释放 slice_copy 和 ones 的引用
    add = torch.ops.aten.add.Tensor(slice_copy, ones); slice_copy = ones = None

    # 再次使用 torch.ops.aten.transpose_copy.int 对输入张量 arg0_1 进行转置操作，
    # 返回转置后的新张量，同时释放 arg0_1 的引用
    transpose_copy_1 = torch.ops.aten.transpose_copy.int(arg0_1, 1, 0); arg0_1 = None

    # 使用 torch.ops.aten.slice_scatter.default 将 add 散射到 transpose_copy_1 中，
    # 操作的维度为0，索引从0开始到2结束，返回新的张量，同时释放 transpose_copy_1 和 add 的引用
    slice_scatter = torch.ops.aten.slice_scatter.default(transpose_copy_1, add, 0, 0, 2); transpose_copy_1 = add = None

    # 使用 torch.ops.aten.transpose_copy.int 对 slice_scatter 进行转置操作，
    # 返回转置后的新张量，同时释放 slice_scatter 的引用
    transpose_copy_2 = torch.ops.aten.transpose_copy.int(slice_scatter, 1, 0); slice_scatter = None

    # 再次使用 torch.ops.aten.transpose_copy.int 对 transpose_copy_2 进行转置操作，
    # 返回转置后的新张量，同时释放 transpose_copy_2 的引用
    transpose_copy_3 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0)

    # 使用 torch.ops.aten.slice_copy.Tensor 对 transpose_copy_3 进行切片操作，
    # 保留索引为0到1的元素，返回一个新的张量，同时释放 transpose_copy_3 的引用
    slice_copy_1 = torch.ops.aten.slice_copy.Tensor(transpose_copy_3, 0, 0, 2); transpose_copy_3 = None

    # 再次使用 torch.ops.aten.transpose_copy.int 对 transpose_copy_2 进行转置操作，
    # 返回转置后的新张量，同时释放 transpose_copy_2 的引用
    transpose_copy_4 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0); transpose_copy_2 = None

    # 返回最后一个转置后的张量 transpose_copy_4
    return transpose_copy_4
    # 调用 Torch 的 aten.add.Tensor 操作，将 select_copy 和 ones 张量相加，结果存入 add 变量；然后释放 select_copy 和 ones 变量
    add = torch.ops.aten.add.Tensor(select_copy, ones);  select_copy = ones = None
    # 调用 Torch 的 aten.transpose_copy.int 操作，对 arg0_1 张量进行转置，结果存入 transpose_copy_1 变量；然后释放 arg0_1 变量
    transpose_copy_1 = torch.ops.aten.transpose_copy.int(arg0_1, 1, 0);  arg0_1 = None
    # 调用 Torch 的 aten.select_scatter.default 操作，在 transpose_copy_1 上执行 scatter 操作，结果存入 select_scatter 变量；然后释放 transpose_copy_1 和 add 变量
    select_scatter = torch.ops.aten.select_scatter.default(transpose_copy_1, add, 0, 0);  transpose_copy_1 = add = None
    # 调用 Torch 的 aten.transpose_copy.int 操作，对 select_scatter 张量进行转置，结果存入 transpose_copy_2 变量；然后释放 select_scatter 变量
    transpose_copy_2 = torch.ops.aten.transpose_copy.int(select_scatter, 1, 0);  select_scatter = None
    # 调用 Torch 的 aten.transpose_copy.int 操作，再次对 transpose_copy_2 张量进行转置，结果存入 transpose_copy_3 变量；然后释放 transpose_copy_2 变量
    transpose_copy_3 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0)
    # 调用 Torch 的 aten.select_copy.int 操作，从 transpose_copy_3 中选择索引为 0 的维度，结果存入 select_copy_1 变量；然后释放 transpose_copy_3 变量
    select_copy_1 = torch.ops.aten.select_copy.int(transpose_copy_3, 0, 0);  transpose_copy_3 = None
    # 调用 Torch 的 aten.transpose_copy.int 操作，对 transpose_copy_2 张量进行再次转置，结果存入 transpose_copy_4 变量；然后释放 transpose_copy_2 变量
    transpose_copy_4 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0);  transpose_copy_2 = None
    # 返回最终结果 transpose_copy_4
    return transpose_copy_4
def forward(self, arg0_1):
    # 创建一个包含四个元素的全为1的张量，设备为 CPU，不使用内存固定
    ones = torch.ops.aten.ones.default([4], device = device(type='cpu'), pin_memory = False)
    # 将输入张量 arg0_1 进行转置操作，交换维度 0 和 1
    transpose = torch.ops.aten.transpose.int(arg0_1, 1, 0)
    # 从转置后的张量中选择索引为 0 的维度，并复制到 select 中，然后释放 transpose
    select = torch.ops.aten.select.int(transpose, 0, 0);  transpose = None
    # 将 ones 张量和 select 张量相加，得到 add 结果，释放 ones 和 select
    add = torch.ops.aten.add.Tensor(select, ones);  select = ones = None
    # 再次对输入张量 arg0_1 进行转置操作，交换维度 0 和 1，并释放 arg0_1
    transpose_1 = torch.ops.aten.transpose.int(arg0_1, 1, 0);  arg0_1 = None
    # 使用转置后的张量 transpose_1 和 add 张量执行 select_scatter 操作，释放 transpose_1 和 add
    select_scatter = torch.ops.aten.select_scatter.default(transpose_1, add, 0, 0);  transpose_1 = add = None
    # 对 select_scatter 张量再次进行转置操作，交换维度 0 和 1，并释放 select_scatter
    transpose_2 = torch.ops.aten.transpose.int(select_scatter, 1, 0);  select_scatter = None
    # 对 transpose_2 张量进行两次转置操作，交换维度 0 和 1，并释放 transpose_2
    transpose_3 = torch.ops.aten.transpose.int(transpose_2, 1, 0)
    # 从 transpose_3 张量中选择索引为 0 的维度，并释放 transpose_3
    select_1 = torch.ops.aten.select.int(transpose_3, 0, 0);  transpose_3 = None
    # 对 transpose_2 张量再次进行转置操作，交换维度 0 和 1，并释放 transpose_2
    transpose_4 = torch.ops.aten.transpose.int(transpose_2, 1, 0);  transpose_2 = None
    # 返回 transpose_4 张量作为函数的输出结果
    return transpose_4
    # 使用 torch.aten 操作将 transpose 解绑（unbind），将结果赋给 unbind，并将 transpose 置为 None
    unbind = torch.ops.aten.unbind.int(transpose);  transpose = None
    # 从 unbind 中获取第一个元素，赋给 getitem
    getitem = unbind[0]
    # 从 unbind 中获取第二个元素，赋给 getitem_1，并将 unbind 置为 None
    getitem_1 = unbind[1];  unbind = None
    # 使用 torch.aten 操作将 getitem 和 ones 相加，结果赋给 add，并将 getitem 和 ones 置为 None
    add = torch.ops.aten.add.Tensor(getitem, ones);  getitem = ones = None
    # 使用 torch.aten 操作对 arg0_1 进行转置操作，结果赋给 transpose_1，并将 arg0_1 置为 None
    transpose_1 = torch.ops.aten.transpose.int(arg0_1, 1, 0);  arg0_1 = None
    # 使用 torch.aten 操作进行 select_scatter 操作，结果赋给 select_scatter，并将 transpose_1 和 add 置为 None
    select_scatter = torch.ops.aten.select_scatter.default(transpose_1, add, 0, 0);  transpose_1 = add = None
    # 使用 torch.aten 操作对 select_scatter 进行转置操作，结果赋给 transpose_2，并将 select_scatter 置为 None
    transpose_2 = torch.ops.aten.transpose.int(select_scatter, 1, 0);  select_scatter = None
    # 使用 torch.aten 操作对 transpose_2 进行转置操作，结果赋给 transpose_3，并将 transpose_2 置为 None
    transpose_3 = torch.ops.aten.transpose.int(transpose_2, 1, 0)
    # 使用 torch.aten 操作将 transpose_3 解绑（unbind），将结果赋给 unbind_1，并将 transpose_3 置为 None
    unbind_1 = torch.ops.aten.unbind.int(transpose_3);  transpose_3 = None
    # 从 unbind_1 中获取第一个元素，赋给 getitem_2
    getitem_2 = unbind_1[0]
    # 从 unbind_1 中获取第二个元素，赋给 getitem_3，并将 unbind_1 置为 None
    getitem_3 = unbind_1[1];  unbind_1 = None
    # 使用 torch.aten 操作对 transpose_2 进行转置操作，结果赋给 transpose_4，并将 transpose_2 置为 None
    transpose_4 = torch.ops.aten.transpose.int(transpose_2, 1, 0);  transpose_2 = None
    # 返回 transpose_4
    return transpose_4
def forward(self, arg0_1):
    # 使用 torch 操作符进行视图复制，默认视图为 [8]
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [8])
    # 使用 torch 操作符创建一个包含 [0, 1, 2, 3] 的张量，默认在 CPU 上，不固定内存
    arange = torch.ops.aten.arange.default(4, device = device(type='cpu'), pin_memory = False)
    # 使用 torch 操作符创建一个包含 [0.0, 1.0, 2.0, 3.0] 的张量，默认为 float32 类型，在 CPU 上，不固定内存
    arange_1 = torch.ops.aten.arange.default(4, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    # 使用 torch 操作符在 view_copy 中的 arange 索引位置处放入 arange_1 的数据，并清除变量引用
    index_put = torch.ops.aten.index_put.default(view_copy, [arange], arange_1);  view_copy = arange = arange_1 = None
    # 使用 torch 操作符对 index_put 进行视图复制，默认视图为 [4, 2]，并清除 index_put 的引用
    view_copy_1 = torch.ops.aten.view_copy.default(index_put, [4, 2]);  index_put = None
    # 使用 torch 操作符对 view_copy_1 进行视图复制，默认视图为 [8]
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [8])
    # 使用 torch 操作符对 arg0_1 进行复制到 view_copy_1，并清除 arg0_1 和 view_copy_1 的引用
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_copy_1);  arg0_1 = view_copy_1 = None
    # 返回最终的视图复制结果 view_copy_2
    return view_copy_2
    # 调用 torch.ops.aten.ge.Scalar 操作，将 clone 的值与 0 比较并赋给 ge；同时将 clone 置为 None
    ge = torch.ops.aten.ge.Scalar(clone, 0);  clone = None
    # 调用 torch.ops.aten._to_copy.default 操作，复制 ge 的值到 _to_copy，指定 dtype 为 torch.float32，layout 为 torch.strided；同时将 ge 置为 None
    _to_copy = torch.ops.aten._to_copy.default(ge, dtype=torch.float32, layout=torch.strided);  ge = None
    # 返回 _to_copy
    return _to_copy
    """,
        )  # noqa: B950

@skipIfTorchDynamo("Test does not work with TorchDynamo")
def test_metadata_change_out_op(self):
    def f(t, y):
        # 创建一个全为 1 的张量 out_1
        out_1 = torch.ones(1)
        # 使用 torch.add 函数将 t 和 y 相加，并将结果写入 out_1 中
        return torch.add(t, y, out=out_1)

    # 创建输入张量 inpt1 和 inpt2，分别赋值为 torch.tensor([1]) 和 torch.tensor([1])
    inpt1, inpt2 = torch.tensor([1]), torch.tensor([1])
    # 将 inpt1 和 inpt2 转换为函数式张量 inpt1_func 和 inpt2_func
    inpt1_func, inpt2_func = torch._to_functional_tensor(inpt1), torch._to_functional_tensor(inpt2)

    # 调用函数 f，并将结果保存在 out_ref 中
    out_ref = f(inpt1, inpt2)
    # 启用张量函数化，重新应用视图
    torch._enable_functionalization(reapply_views=True)
    try:
        # 使用函数式张量调用函数 f，并将结果保存在 out_functional 中
        out_functional = f(inpt1_func, inpt2_func)
    finally:
        # 禁用张量函数化
        torch._disable_functionalization()
    # 断言 out_ref 与从函数式张量 out_functional 转回的张量相等
    self.assertEqual(out_ref, torch._from_functional_tensor(out_functional))

def test_only_one_view(self):
    def f(x):
        # 这个测试确保在跟踪中没有不必要的视图。
        # 如果输入未被修改，我们无需重新生成它，因此输出跟踪中应该有总共 1 个操作。
        return x.view(4, 2)

    # 获取函数 f 的日志记录，输入为全为 1 的 4x2 张量
    logs = self.get_logs(f, torch.ones(4, 2))
    # 断言日志记录与预期的内联字符串相等
    self.assertExpectedInline(
        logs,
        """\
def forward(self, arg0_1):
    # 调用 torch.ops.aten.view_copy.default 方法，将输入张量 arg0_1 进行形状变换，得到新的张量 view_copy
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2]);  arg0_1 = None
    # 返回新的张量 view_copy
    return view_copy
    unsqueeze_copy_3 = torch.ops.aten.unsqueeze_copy.default(transpose_copy_3, 0);  transpose_copy_3 = None
    squeeze_copy_3 = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_3);  unsqueeze_copy_3 = None
    split_copy_1 = torch.ops.aten.split_copy.Tensor(squeeze_copy_3, 2);  squeeze_copy_3 = None
    getitem_2 = split_copy_1[0]
    getitem_3 = split_copy_1[1];  split_copy_1 = None
    select_copy = torch.ops.aten.select_copy.int(view_copy_1, 0, 0);  view_copy_1 = None
    view_copy_8 = torch.ops.aten.view_copy.default(getitem_2, [4])
    view_copy_9 = torch.ops.aten.view_copy.default(view_copy_5, [8])
    view_copy_10 = torch.ops.aten.view_copy.default(view_copy_9, [2, 4]);  view_copy_9 = None
    select_copy_1 = torch.ops.aten.select_copy.int(view_copy_10, 0, 0);  view_copy_10 = None
    view_copy_11 = torch.ops.aten.view_copy.default(view_copy_5, [8]);  view_copy_5 = None
    view_copy_12 = torch.ops.aten.view_copy.default(view_copy_11, [2, 4]);  view_copy_11 = None
    transpose_copy_4 = torch.ops.aten.transpose_copy.int(view_copy_12, 1, 0);  view_copy_12 = None
    unsqueeze_copy_4 = torch.ops.aten.unsqueeze_copy.default(transpose_copy_4, 0);  transpose_copy_4 = None
    squeeze_copy_4 = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_4);  unsqueeze_copy_4 = None
    split_copy_2 = torch.ops.aten.split_copy.Tensor(squeeze_copy_4, 2);  squeeze_copy_4 = None
    getitem_4 = split_copy_2[0]
    getitem_5 = split_copy_2[1];  split_copy_2 = None
    view_copy_13 = torch.ops.aten.view_copy.default(getitem_4, [4]);  getitem_4 = None
    add_2 = torch.ops.aten.add.Tensor(select_copy_1, view_copy_13);  select_copy_1 = view_copy_13 = None
    return getitem_2
    
    
    注释：
    
    
    unsqueeze_copy_3 = torch.ops.aten.unsqueeze_copy.default(transpose_copy_3, 0);  transpose_copy_3 = None
    # 对 transpose_copy_3 进行在第0维度上的unsqueeze操作，并将结果赋给unsqueeze_copy_3，然后清空transpose_copy_3
    
    squeeze_copy_3 = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_3);  unsqueeze_copy_3 = None
    # 对 unsqueeze_copy_3 执行squeeze操作，得到squeeze_copy_3，并清空unsqueeze_copy_3
    
    split_copy_1 = torch.ops.aten.split_copy.Tensor(squeeze_copy_3, 2);  squeeze_copy_3 = None
    # 对 squeeze_copy_3 进行按照第2维度分割操作，得到一个包含两个Tensor的列表split_copy_1，并清空squeeze_copy_3
    
    getitem_2 = split_copy_1[0]
    getitem_3 = split_copy_1[1];  split_copy_1 = None
    # 获取 split_copy_1 中的第一个和第二个Tensor，分别赋给getitem_2和getitem_3，并清空split_copy_1
    
    select_copy = torch.ops.aten.select_copy.int(view_copy_1, 0, 0);  view_copy_1 = None
    # 对 view_copy_1 进行选择操作，选取第0维度上的第0个元素，并清空view_copy_1
    
    view_copy_8 = torch.ops.aten.view_copy.default(getitem_2, [4])
    # 对 getitem_2 进行view操作，reshape成形状为[4]的Tensor
    
    view_copy_9 = torch.ops.aten.view_copy.default(view_copy_5, [8])
    # 对 view_copy_5 进行view操作，reshape成形状为[8]的Tensor
    
    view_copy_10 = torch.ops.aten.view_copy.default(view_copy_9, [2, 4]);  view_copy_9 = None
    # 对 view_copy_9 进行view操作，reshape成形状为[2, 4]的Tensor，并清空view_copy_9
    
    select_copy_1 = torch.ops.aten.select_copy.int(view_copy_10, 0, 0);  view_copy_10 = None
    # 对 view_copy_10 进行选择操作，选取第0维度上的第0个元素，并清空view_copy_10
    
    view_copy_11 = torch.ops.aten.view_copy.default(view_copy_5, [8]);  view_copy_5 = None
    # 对 view_copy_5 进行view操作，reshape成形状为[8]的Tensor，并清空view_copy_5
    
    view_copy_12 = torch.ops.aten.view_copy.default(view_copy_11, [2, 4]);  view_copy_11 = None
    # 对 view_copy_11 进行view操作，reshape成形状为[2, 4]的Tensor，并清空view_copy_11
    
    transpose_copy_4 = torch.ops.aten.transpose_copy.int(view_copy_12, 1, 0);  view_copy_12 = None
    # 对 view_copy_12 进行转置操作，交换第1和第0维，并清空view_copy_12
    
    unsqueeze_copy_4 = torch.ops.aten.unsqueeze_copy.default(transpose_copy_4, 0);  transpose_copy_4 = None
    # 对 transpose_copy_4 进行在第0维度上的unsqueeze操作，并将结果赋给unsqueeze_copy_4，然后清空transpose_copy_4
    
    squeeze_copy_4 = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_4);  unsqueeze_copy_4 = None
    # 对 unsqueeze_copy_4 执行squeeze操作，得到squeeze_copy_4，并清空unsqueeze_copy_4
    
    split_copy_2 = torch.ops.aten.split_copy.Tensor(squeeze_copy_4, 2);  squeeze_copy_4 = None
    # 对 squeeze_copy_4 进行按照第2维度分割操作，得到一个包含两个Tensor的列表split_copy_2，并清空squeeze_copy_4
    
    getitem_4 = split_copy_2[0]
    getitem_5 = split_copy_2[1];  split_copy_2 = None
    # 获取 split_copy_2 中的第一个和第二个Tensor，分别赋给getitem_4和getitem_5，并清空split_copy_2
    
    view_copy_13 = torch.ops.aten.view_copy.default(getitem_4, [4]);  getitem_4 = None
    # 对 getitem_4 进行view操作，reshape成形状为[4]的Tensor，并清空getitem_4
    
    add_2 = torch.ops.aten.add.Tensor(select_copy_1, view_copy_13);  select_copy_1 = view_copy_13 = None
    # 对 select_copy_1 和 view_copy_13 执行张量相加操作，并清空这两个变量
    
    return getitem_2
    # 返回getitem_2
def forward(self, arg0_1):
    # 创建一个 2x2 的张量，所有元素为1，存储在CPU上，不固定在内存中
    ones = torch.ops.aten.ones.default([2, 2], device=device(type='cpu'), pin_memory=False)
    # 对输入张量 arg0_1 执行张量加法操作，结果存储在 add 中；清空 arg0_1 引用
    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1); arg0_1 = None
    # 将 add 张量重塑为长度为8的一维张量
    view = torch.ops.aten.view.default(add, [8])
    # 将 view 张量重塑为 2x4 的张量；清空 view 引用
    view_1 = torch.ops.aten.view.default(view, [2, 4]); view = None
    # 将 view_1 张量进行转置操作，交换维度1和维度0
    transpose = torch.ops.aten.transpose.int(view_1, 1, 0)
    # 在 transpose 张量的维度0上增加一个维度；清空 transpose 引用
    unsqueeze = torch.ops.aten.unsqueeze.default(transpose, 0); transpose = None
    # 压缩 squeeze 张量的所有维度为1
    squeeze = torch.ops.aten.squeeze.default(unsqueeze); unsqueeze = None
    # 将 squeeze 张量在维度2上分割成两个张量
    split = torch.ops.aten.split.Tensor(squeeze, 2); squeeze = None
    # 获取 split 中索引为0的张量
    getitem = split[0]
    # 获取 split 中索引为1的张量；清空 split 引用
    getitem_1 = split[1]; split = None
    # 将 getitem 张量与 ones 张量进行张量加法操作，结果存储在 add_1 中；清空 getitem 和 ones 引用
    add_1 = torch.ops.aten.add_.Tensor(getitem, ones); getitem = ones = None
    # 将 add 张量重塑为长度为8的一维张量；清空 add 引用
    view_2 = torch.ops.aten.view.default(add, [8]); add = None
    # 将 view_2 张量重塑为 2x4 的张量；清空 view_2 引用
    view_3 = torch.ops.aten.view.default(view_2, [2, 4]); view_2 = None
    # 将 view_3 张量进行转置操作，交换维度1和维度0；清空 view_3 引用
    transpose_1 = torch.ops.aten.transpose.int(view_3, 1, 0); view_3 = None
    # 在 transpose_1 张量的维度0上增加一个维度；清空 transpose_1 引用
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(transpose_1, 0); transpose_1 = None
    # 压缩 squeeze_1 张量的所有维度为1
    squeeze_1 = torch.ops.aten.squeeze.default(unsqueeze_1); unsqueeze_1 = None
    # 在 squeeze_1 张量的维度0上增加一个维度；清空 squeeze_1 引用
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(squeeze_1, 0); squeeze_1 = None
    # 压缩 squeeze_2 张量的维度0，清空 unsqueeze_2 引用
    squeeze_2 = torch.ops.aten.squeeze.dim(unsqueeze_2, 0); unsqueeze_2 = None
    # 将 squeeze_2 张量进行转置操作，交换维度1和维度0；清空 squeeze_2 引用
    transpose_2 = torch.ops.aten.transpose.int(squeeze_2, 1, 0); squeeze_2 = None
    # 将 transpose_2 张量重塑为长度为8的一维张量；清空 transpose_2 引用
    view_4 = torch.ops.aten.view.default(transpose_2, [8]); transpose_2 = None
    # 将 view_4 张量重塑为 4x2 的张量；清空 view_4 引用
    view_5 = torch.ops.aten.view.default(view_4, [4, 2]); view_4 = None
    # 将 view_5 张量重塑为长度为8的一维张量
    view_6 = torch.ops.aten.view.default(view_5, [8])
    # 将 view_6 张量重塑为 2x4 的张量；清空 view_6 引用
    view_7 = torch.ops.aten.view.default(view_6, [2, 4]); view_6 = None
    # 将 view_7 张量进行转置操作，交换维度1和维度0；清空 view_7 引用
    transpose_3 = torch.ops.aten.transpose.int(view_7, 1, 0); view_7 = None
    # 在 transpose_3 张量的维度0上增加一个维度；清空 transpose_3 引用
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(transpose_3, 0); transpose_3 = None
    # 压缩 squeeze_3 张量的所有维度为1
    squeeze_3 = torch.ops.aten.squeeze.default(unsqueeze_3); unsqueeze_3 = None
    # 将 squeeze_3 张量在维度2上分割成两个张量；清空 squeeze_3 引用
    split_1 = torch.ops.aten.split.Tensor(squeeze_3, 2); squeeze_3 = None
    # 获取 split_1 中索引为0的张量
    getitem_2 = split_1[0]
    # 获取 split_1 中索引为1的张量；清空 split_1 引用
    getitem_3 = split_1[1]; split_1 = None
    # 在 view_1 张量的第0维上选择索引为0的切片；清空 view_1 引用
    select = torch.ops.aten.select.int(view_1, 0, 0); view_1 = None
    # 对 getitem_2 张量进行克隆操作，采用连续内存格式；清空 getitem_2 引用
    clone = torch.ops.aten.clone.default(getitem_2, memory_format=torch.contiguous_format)
    # 对 clone 张量执行不安全的重塑操作，重塑为长度为4的一维张量；清空 clone 引用
    _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [4]); clone = None
    # 将 view_5 张量重塑为长度为8的一维张量；清空 view_5 引用
    view_8 = torch.ops.aten.view.default(view_5, [8]); view_5 = None
    # 将 view_8 张量重塑为 2x4 的张量；清空 view_8 引用
    view_9 = torch.ops.aten.view.default(view_8, [2, 4]); view_8 = None
    # 在 view_9 张量的第0维上选择索引为0的切片；清空 view_9 引用
    select_1 = torch.ops.aten.select.int(view_9, 0, 0); view_9 = None
    # 将 select_1 张量与 _unsafe_view 张量进行张量加法操作；清空 select_1 和 _unsafe_view 引用
    add_2 = torch.ops.aten.add.Tensor(select_1, _unsafe_view); select_1 = _unsafe_view = None
    # 返回 getitem_2 张量
    return getitem_2
    # 定义内部函数 f，接受参数 x
    def f(x):
        # 创建一个全为 1 的张量 tmp，形状为 (4, 2)
        tmp = torch.ones(4, 2)
        # 对输入张量 x 进行视图重塑为 (4, 2)，保存到 y 中
        y = x.view(4, 2)
        # 将 tmp 加到 y 上，使用原地操作 add_
        y.add_(tmp)
        # 计算 x 的平方，保存到 z 中
        z = x * x
        # 返回 y
        return y

    # 调用 assert_functionalization 函数，验证 f 函数在 reapply_views=True 下的行为
    self.assert_functionalization(f, torch.ones(4, 2), reapply_views=True)
    # 获取 f 函数在 reapply_views=True 下的日志信息，保存到 logs 中
    logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True)
    # 断言 logs 中的实际输出与给定的期望输出字符串相符
    self.assertExpectedInline(
        logs,
        """\
# 定义了一个名为 forward 的方法，接受一个参数 arg0_1
def forward(self, arg0_1):
    # 创建一个尺寸为 [4, 2] 的全 1 张量 ones，设备为 CPU，不使用内存固定
    ones = torch.ops.aten.ones.default([4, 2], device=device(type='cpu'), pin_memory=False)
    # 使用 torch.ops.aten.view.default 方法对 arg0_1 进行形状变换，变换成 [4, 2] 的张量 view
    view = torch.ops.aten.view.default(arg0_1, [4, 2])
    # 将 ones 张量加到 view 张量上，并将 view 和 ones 设为 None
    add = torch.ops.aten.add.Tensor(view, ones); view = ones = None
    # 对 add 张量进行形状变换，变换成 [4, 2] 的张量 view_1，同时将 add 设为 None
    view_1 = torch.ops.aten.view.default(add, [4, 2]); add = None
    # 对 view_1 张量再次进行形状变换，变换成 [4, 2] 的张量 view_2
    view_2 = torch.ops.aten.view.default(view_1, [4, 2])
    # 将 view_1 张量与自身相乘，得到 [4, 2] 的张量 mul
    mul = torch.ops.aten.mul.Tensor(view_1, view_1)
    # 将 view_1 张量的内容拷贝到 arg0_1 张量上，并将 arg0_1 和 view_1 设为 None
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_1); arg0_1 = view_1 = None
    # 返回形状变换后的张量 view_2
    return view_2
        )
        # 调用 self.get_logs 方法，传入参数 f 作为文件名，torch.ones(2) 作为权重向量，
        # 同时开启重新应用视图和运行替换操作的标志
        reinplaced_logs = self.get_logs(
            f, torch.ones(2), reapply_views=True, run_reinplace=True
        )
        # 使用 self.assertExpectedInline 方法来断言 reinplaced_logs 的期望输出，
        # 期望的输出是一个空字符串，用于验证替换后的日志或内容是否符合预期
        self.assertExpectedInline(
            reinplaced_logs,
            """\
# 定义一个方法 `forward`，接收一个参数 `arg0_1`
def forward(self, arg0_1):
    # 使用 torch.ops.aten.zeros.default 创建一个 2x2 的零张量 `zeros`，
    # 指定设备为 CPU，并且不使用固定内存
    zeros = torch.ops.aten.zeros.default([2, 2], device=device(type='cpu'), pin_memory=False)
    
    # 从 `zeros` 中提取对角线元素，存入 `diagonal`
    diagonal = torch.ops.aten.diagonal.default(zeros)
    
    # 使用 `torch.ops.aten.copy.default` 复制 `diagonal` 到 `arg0_1`，
    # 并将 `diagonal` 置为 None
    copy = torch.ops.aten.copy.default(diagonal, arg0_1);  diagonal = None
    
    # 从 `zeros` 中再次提取对角线元素，存入 `diagonal_1`
    diagonal_1 = torch.ops.aten.diagonal.default(zeros)
    
    # 使用 `torch.ops.aten.add.Tensor` 将 `diagonal_1` 与 `arg0_1` 相加，
    # 并将 `diagonal_1` 和 `arg0_1` 置为 None
    add = torch.ops.aten.add.Tensor(diagonal_1, arg0_1);  diagonal_1 = arg0_1 = None
    
    # 从 `zeros` 中再次提取对角线元素，存入 `diagonal_2`，
    # 并将 `zeros` 置为 None
    diagonal_2 = torch.ops.aten.diagonal.default(zeros);  zeros = None
    
    # 返回 `diagonal_2` 作为方法 `forward` 的结果
    return diagonal_2
    # 使用 torch.ops.aten.add.Tensor 函数将 diagonal_copy_1 和 arg0_1 张量相加，并将结果赋给 add
    add = torch.ops.aten.add.Tensor(diagonal_copy_1, arg0_1);  diagonal_copy_1 = arg0_1 = None
    # 使用 torch.ops.aten.diagonal_scatter.default 函数将 diagonal_scatter 和 add 张量进行对角散射操作，并将结果赋给 diagonal_scatter_1
    diagonal_scatter_1 = torch.ops.aten.diagonal_scatter.default(diagonal_scatter, add);  diagonal_scatter = add = None
    # 使用 torch.ops.aten.diagonal_copy.default 函数将 diagonal_scatter_1 张量进行默认对角复制操作，并将结果赋给 diagonal_copy_2
    diagonal_copy_2 = torch.ops.aten.diagonal_copy.default(diagonal_scatter_1);  diagonal_scatter_1 = None
    # 返回最终的对角复制结果张量 diagonal_copy_2
    return diagonal_copy_2
# 定义一个名为 forward 的方法，接受 self 和 arg0_1 作为参数
def forward(self, arg0_1):
    # 使用 torch.ops.aten.zeros.default 创建一个形状为 [2, 2] 的零张量，存储在 zeros 变量中
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    # 使用 torch.ops.aten.diagonal.default 获取 zeros 的对角线元素，存储在 diagonal 变量中
    diagonal = torch.ops.aten.diagonal.default(zeros)
    # 使用 torch.ops.aten.copy_.default 将 diagonal 复制给 arg0_1，然后将 diagonal 置为 None
    copy = torch.ops.aten.copy_.default(diagonal, arg0_1);  diagonal = None
    # 再次使用 torch.ops.aten.diagonal.default 获取 zeros 的对角线元素，存储在 diagonal_1 变量中
    diagonal_1 = torch.ops.aten.diagonal.default(zeros)
    # 使用 torch.ops.aten.add_.Tensor 将 diagonal_1 和 arg0_1 相加，结果存储在 add 变量中，然后将 diagonal_1 和 arg0_1 置为 None
    add = torch.ops.aten.add_.Tensor(diagonal_1, arg0_1);  diagonal_1 = arg0_1 = None
    # 再次使用 torch.ops.aten.diagonal.default 获取 zeros 的对角线元素，存储在 diagonal_2 变量中，然后将 zeros 置为 None
    diagonal_2 = torch.ops.aten.diagonal.default(zeros);  zeros = None
    # 返回 diagonal_2 变量的值作为方法的结果
    return diagonal_2
    # 定义测试函数 test_fill_，该函数是一个嵌套函数 f 的包装器
    def test_fill_(self):
        # 定义函数 f，接收参数 x
        def f(x):
            # 计算变量 y，将 x 加上自身得到 y
            y = x + x
            # 获取 y 的对角线元素组成的张量 z
            z = y.diagonal()
            # 将张量 z 中的所有元素填充为 0
            z.fill_(0)
            # 返回修改后的张量 y
            return y

        # 调用测试工具的 assert_functionalization 方法，验证函数 f 的功能
        self.assert_functionalization(f, torch.ones(2, 2))
        # 获取函数 f 的执行日志
        logs = self.get_logs(f, torch.ones(2, 2))
        # 使用 assertExpectedInline 方法验证日志内容是否符合预期
        self.assertExpectedInline(
            logs,
            """\
def forward(self, arg0_1):
    # 调用 torch 库中的 add 函数，将 arg0_1 与自身相加，结果保存在 add 中；清空 arg0_1 的引用
    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1);  arg0_1 = None
    # 对 add 张量进行对角线复制操作，结果保存在 diagonal_copy 中
    diagonal_copy = torch.ops.aten.diagonal_copy.default(add)
    # 使用 fill_ 操作，在 diagonal_copy 上填充值 0，结果保存在 fill 中；清空 diagonal_copy 的引用
    fill = torch.ops.aten.fill_.Scalar(diagonal_copy, 0);  diagonal_copy = None
    # 使用 diagonal_scatter 操作，在 add 上应用 fill，结果保存在 diagonal_scatter 中；清空 add 和 fill 的引用
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(add, fill);  add = fill = None
    # 对 diagonal_scatter 张量进行对角线复制操作，结果保存在 diagonal_copy_1 中
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter)
    # 返回 diagonal_scatter 张量作为结果
    return diagonal_scatter
    # 使用 torch.ops.aten.add.Tensor 函数将 as_strided_copy_3 和标量 1 相加，并将结果赋给 add_2
    add_2 = torch.ops.aten.add.Tensor(as_strided_copy_3, 1);  as_strided_copy_3 = None
    # 返回 add_2 的值作为函数的结果
    return add_2
def forward(self, arg0_1):
    # 使用 torch.ops.aten.add.Tensor 函数将 arg0_1 和标量 1 相加，结果存入 add，arg0_1 被置为 None
    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
    # 使用 torch.ops.aten.resize.default 函数将 add 张量 resize 成大小为 [5, 5] 的张量，add 被置为 None
    resize = torch.ops.aten.resize.default(add, [5, 5]);  add = None
    # 使用 torch.ops.aten.view_copy.default 函数将 resize 张量按照 [25] 形状重新视图化，resize 被置为 None
    view_copy = torch.ops.aten.view_copy.default(resize, [25]);  resize = None
    # 使用 torch.ops.aten.fill.Scalar 函数将 view_copy 张量的所有元素填充为标量 1，view_copy 被置为 None
    fill = torch.ops.aten.fill.Scalar(view_copy, 1);  view_copy = None
    # 使用 torch.ops.aten.view_copy.default 函数将 fill 张量按照 [5, 5] 形状重新视图化，fill 被置为 None
    view_copy_1 = torch.ops.aten.view_copy.default(fill, [5, 5]);  fill = None
    # 使用 torch.ops.aten.view_copy.default 函数将 view_copy_1 张量按照 [25] 形状重新视图化
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [25])
    # 使用 torch.ops.aten.add.Tensor 函数将 view_copy_1 张量和标量 1 相加，结果存入 add_1
    add_1 = torch.ops.aten.add.Tensor(view_copy_1, 1)
    # 返回两个张量 view_copy_1 和 add_1
    return (view_copy_1, add_1)
        )
        # 调用 self.get_logs 方法，传入文件 f、全为 1 的 8x2 的张量作为参数，
        # 并设置 reapply_views=True 和 run_reinplace=True
        reinplaced_logs = self.get_logs(
            f, torch.ones(8, 2), reapply_views=True, run_reinplace=True
        )
        # 使用 self.assertExpectedInline 方法断言 reinplaced_logs 的预期输出，
        # 传入预期输出字符串作为参数
        self.assertExpectedInline(
            reinplaced_logs,
            """\
# 定义一个方法 `forward`，接收一个参数 `arg0_1`
def forward(self, arg0_1):
    # 调用 ATen 操作 `add`，将 `arg
    # 定义一个测试方法，用于测试在非输入情况下的索引变异
    def test_index_mutation_on_non_input(self):
        # 定义一个函数 f，接受参数 x
        def f(x):
            # 创建一个长度为 10 的全零张量 tmp
            tmp = torch.zeros(10)
            # 将 tmp 中索引为 5 的元素设置为 1
            tmp[5].fill_(1)
            # 返回更新后的 tmp
            return tmp

        # 调用 assert_functionalization 方法，测试函数 f 的功能
        self.assert_functionalization(f, torch.ones(2))
        # 获取函数 f 执行时的日志
        logs = self.get_logs(f, torch.ones(2))
        # 断言日志内容符合预期
        self.assertExpectedInline(
            logs,
            """\
def forward(self, arg0_1):
    # 使用 torch.ops.aten.zeros.default 函数创建一个大小为 [10] 的张量，初始化为零
    zeros = torch.ops.aten.zeros.default([10], device = device(type='cpu'), pin_memory = False)
    # 使用 torch.ops.aten.select_copy.int 函数从 zeros 张量中选择部分数据复制到 select_copy 中
    select_copy = torch.ops.aten.select_copy.int(zeros, 0, 5)
    # 使用 torch.ops.aten.fill.Scalar 函数将 select_copy 张量中的数据填充为标量 1
    fill = torch.ops.aten.fill.Scalar(select_copy, 1);  select_copy = None
    # 使用 torch.ops.aten.select_scatter.default 函数将 fill 张量的数据散列到 zeros 张量中指定的位置
    select_scatter = torch.ops.aten.select_scatter.default(zeros, fill, 0, 5);  zeros = fill = None
    # 使用 torch.ops.aten.select_copy.int 函数从 select_scatter 张量中选择部分数据复制到 select_copy_1 中
    select_copy_1 = torch.ops.aten.select_copy.int(select_scatter, 0, 5)
    # 返回 select_scatter 张量作为 forward 方法的输出
    return select_scatter


```    
        )  # noqa: B950

        reinplaced_logs = self.get_logs(
            f, torch.ones(2), reapply_views=True, run_reinplace=True
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    # 使用 torch.ops.aten.zeros.default 函数创建一个大小为 [10] 的张量，初始化为零
    zeros = torch.ops.aten.zeros.default([10], device = device(type='cpu'), pin_memory = False)
    # 使用 torch.ops.aten.select.int 函数从 zeros 张量中选择部分数据复制到 select 中
    select = torch.ops.aten.select.int(zeros, 0, 5)
    # 使用 torch.ops.aten.fill_.Scalar 函数将 select 张量中的数据填充为标量 1
    fill = torch.ops.aten.fill_.Scalar(select, 1);  select = None
    # 使用 torch.ops.aten.select.int 函数从 zeros 张量中选择部分数据复制到 select_1 中
    select_1 = torch.ops.aten.select.int(zeros, 0, 5)
    # 返回 zeros 张量作为 forward 方法的输出
    return zeros
    """,
        )

    def test_instance_norm(self):
        size = 100

        def f(x, running_mean, running_var):
            with enable_python_dispatcher():
                return torch.instance_norm(
                    x,
                    None,
                    None,
                    running_mean,
                    running_var,
                    use_input_stats=True,
                    momentum=0.1,
                    eps=1e-5,
                    cudnn_enabled=False,
                )

        self.assert_functionalization(
            f, torch.randn(20, size, 35, 45), torch.zeros(size), torch.ones(size)
        )
        # On Windows, for instance_norm, the alias_copy's are reordered to come right before they need to be used
        # whereas on other platforms, the alias_copy's are before the view_copy's.
        # e.g., the alias_copy after the getitem_4 assignment would be moved to be right before the copy assignment.
        if not IS_WINDOWS:
            logs = self.get_logs(
                f, torch.randn(20, size, 35, 45), torch.zeros(size), torch.ones(size)
            )
            self.assertExpectedInline(
                logs,
                """\



def forward(self, arg0_1, arg1_1, arg2_1):
    # 使用 torch.ops.aten.repeat.default 函数复制 arg1_1 张量的数据 20 次
    repeat = torch.ops.aten.repeat.default(arg1_1, [20])
    # 使用 torch.ops.aten.repeat.default 函数复制 arg2_1 张量的数据 20 次
    repeat_1 = torch.ops.aten.repeat.default(arg2_1, [20])
    # 使用 torch.ops.aten.view_copy.default 函数创建一个与 arg0_1 形状相同的视图
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [1, 2000, 35, 45]);  arg0_1 = None
    # 使用 torch.ops.aten.empty.memory_format 函数创建一个空张量
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
    # 使用 torch.ops.aten._native_batch_norm_legit_functional.default 函数进行批量归一化操作
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(view_copy, None, None, repeat, repeat_1, True, 0.1, 1e-05);  view_copy = repeat = repeat_1 = None
    # 从 _native_batch_norm_legit_functional 结果中获取不同的项
    getitem = _native_batch_norm_legit_functional[0]
    getitem_1 = _native_batch_norm_legit_functional[1]
    getitem_2 = _native_batch_norm_legit_functional[2]
    # 从 _native_batch_norm_legit_functional 列表中获取第三个元素
    getitem_3 = _native_batch_norm_legit_functional[3]
    # 从 _native_batch_norm_legit_functional 列表中获取第四个元素，并将 _native_batch_norm_legit_functional 置为 None
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    # 使用 torch.ops.aten.alias_copy.default 方法复制 arg1_1 的值
    alias_copy = torch.ops.aten.alias_copy.default(arg1_1)
    # 使用 torch.ops.aten.view_copy.default 方法将 getitem_3 视图变换为 [20, 100]
    view_copy_1 = torch.ops.aten.view_copy.default(getitem_3, [20, 100])
    # 使用 torch.ops.aten.view_copy.default 方法将 getitem_3 视图变换为 [20, 100]，并将 getitem_3 置为 None
    view_copy_2 = torch.ops.aten.view_copy.default(getitem_3, [20, 100]);  getitem_3 = None
    # 使用 torch.ops.aten.mean.dim 方法计算 view_copy_2 在维度 [0] 上的平均值，并将 view_copy_2 置为 None
    mean = torch.ops.aten.mean.dim(view_copy_2, [0]);  view_copy_2 = None
    # 使用 torch.ops.aten.copy.default 方法复制 alias_copy 和 mean 的值，并将它们置为 None
    copy = torch.ops.aten.copy.default(alias_copy, mean);  alias_copy = mean = None
    # 使用 torch.ops.aten.alias_copy.default 方法复制 copy 的值
    alias_copy_1 = torch.ops.aten.alias_copy.default(copy);  copy = None
    # 使用 torch.ops.aten.alias_copy.default 方法复制 arg2_1 的值
    alias_copy_2 = torch.ops.aten.alias_copy.default(arg2_1)
    # 使用 torch.ops.aten.alias_copy.default 方法复制 getitem_4 的值
    alias_copy_3 = torch.ops.aten.alias_copy.default(getitem_4)
    # 使用 torch.ops.aten.view_copy.default 方法将 getitem_4 视图变换为 [20, 100]
    view_copy_3 = torch.ops.aten.view_copy.default(getitem_4, [20, 100])
    # 使用 torch.ops.aten.view_copy.default 方法将 getitem_4 视图变换为 [20, 100]，并将 getitem_4 置为 None
    view_copy_4 = torch.ops.aten.view_copy.default(getitem_4, [20, 100]);  getitem_4 = None
    # 使用 torch.ops.aten.mean.dim 方法计算 view_copy_4 在维度 [0] 上的平均值，并将 view_copy_4 置为 None
    mean_1 = torch.ops.aten.mean.dim(view_copy_4, [0]);  view_copy_4 = None
    # 使用 torch.ops.aten.copy.default 方法复制 alias_copy_3 和 mean_1 的值，并将它们置为 None
    copy_1 = torch.ops.aten.copy.default(alias_copy_3, mean_1);  alias_copy_3 = mean_1 = None
    # 使用 torch.ops.aten.alias_copy.default 方法复制 copy_1 的值
    alias_copy_4 = torch.ops.aten.alias_copy.default(copy_1);  copy_1 = None
    # 使用 torch.ops.aten.alias_copy.default 方法复制 alias_copy_4 的值
    alias_copy_5 = torch.ops.aten.alias_copy.default(alias_copy_4)
    # 使用 torch.ops.aten.view_copy.default 方法将 getitem 视图变换为 [20, 100, 35, 45]，并将 getitem 置为 None
    view_copy_5 = torch.ops.aten.view_copy.default(getitem, [20, 100, 35, 45]);  getitem = None
    # 使用 torch.ops.aten.copy_.default 方法复制 arg1_1 和 alias_copy_1 的值，并将 arg1_1 和 alias_copy_1 置为 None
    copy_ = torch.ops.aten.copy_.default(arg1_1, alias_copy_1);  arg1_1 = alias_copy_1 = None
    # 使用 torch.ops.aten.copy_.default 方法复制 arg2_1 和 alias_copy_4 的值，并将 arg2_1 和 alias_copy_4 置为 None
    copy__1 = torch.ops.aten.copy_.default(arg2_1, alias_copy_4);  arg2_1 = alias_copy_4 = None
    # 返回 view_copy_5
    return view_copy_5
def forward(self, arg0_1, arg1_1, arg2_1):
    # 使用 torch.ops.aten.repeat.default 函数将 arg1_1 重复 20 次
    repeat = torch.ops.aten.repeat.default(arg1_1, [20])
    # 使用 torch.ops.aten.repeat.default 函数将 arg2_1 重复 20 次
    repeat_1 = torch.ops.aten.repeat.default(arg2_1, [20])
    # 使用 torch.ops.aten.view.default 函数对 arg0_1 进行形状重塑为 [1, 2000, 35, 45]，并清空 arg0_1
    view = torch.ops.aten.view.default(arg0_1, [1, 2000, 35, 45]);  arg0_1 = None
    # 使用 torch.ops.aten.empty.memory_format 函数创建一个空的 uint8 类型张量
    empty = torch.ops.aten.empty.memory_format([0], dtype=torch.uint8, layout=torch.strided, device=device(type='cpu'))
    # 使用 torch.ops.aten._native_batch_norm_legit_functional.default 函数执行批量归一化操作
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(view, None, None, repeat, repeat_1, True, 0.1, 1e-05);  view = repeat = repeat_1 = None
    # 从 _native_batch_norm_legit_functional 结果中获取多个张量
    getitem = _native_batch_norm_legit_functional[0]
    getitem_1 = _native_batch_norm_legit_functional[1]
    getitem_2 = _native_batch_norm_legit_functional[2]
    getitem_3 = _native_batch_norm_legit_functional[3]
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    # 使用 torch.ops.aten.alias.default 函数创建别名张量
    alias = torch.ops.aten.alias.default(arg1_1)
    # 使用 torch.ops.aten.view.default 函数将 getitem_3 的形状重塑为 [20, 100]，并清空 getitem_3
    view_1 = torch.ops.aten.view.default(getitem_3, [20, 100])
    # 再次使用 torch.ops.aten.view.default 函数将 getitem_3 的形状重塑为 [20, 100]，并清空 getitem_3
    view_2 = torch.ops.aten.view.default(getitem_3, [20, 100]);  getitem_3 = None
    # 使用 torch.ops.aten.mean.dim 函数计算 view_2 在维度 0 上的均值，并清空 view_2
    mean = torch.ops.aten.mean.dim(view_2, [0]);  view_2 = None
    # 使用 torch.ops.aten.copy.default 函数复制 alias 到 mean，并清空 alias 和 mean
    copy = torch.ops.aten.copy.default(alias, mean);  alias = mean = None
    # 使用 torch.ops.aten.alias.default 函数创建别名张量 alias_1，并清空 copy
    alias_1 = torch.ops.aten.alias.default(copy);  copy = None
    # 使用 torch.ops.aten.alias.default 函数创建别名张量 alias_2
    alias_2 = torch.ops.aten.alias.default(alias_1)
    # 使用 torch.ops.aten.alias.default 函数创建别名张量 alias_3
    alias_3 = torch.ops.aten.alias.default(arg2_1)
    # 使用 torch.ops.aten.view.default 函数将 getitem_4 的形状重塑为 [20, 100]，并清空 getitem_4
    view_3 = torch.ops.aten.view.default(getitem_4, [20, 100])
    # 再次使用 torch.ops.aten.view.default 函数将 getitem_4 的形状重塑为 [20, 100]，并清空 getitem_4
    view_4 = torch.ops.aten.view.default(getitem_4, [20, 100]);  getitem_4 = None
    # 使用 torch.ops.aten.mean.dim 函数计算 view_4 在维度 0 上的均值，并清空 view_4
    mean_1 = torch.ops.aten.mean.dim(view_4, [0]);  view_4 = None
    # 使用 torch.ops.aten.copy.default 函数复制 alias_3 到 mean_1，并清空 alias_3 和 mean_1
    copy_1 = torch.ops.aten.copy.default(alias_3, mean_1);  alias_3 = mean_1 = None
    # 使用 torch.ops.aten.alias.default 函数创建别名张量 alias_4，并清空 copy_1
    alias_4 = torch.ops.aten.alias.default(copy_1);  copy_1 = None
    # 使用 torch.ops.aten.alias.default 函数创建别名张量 alias_5
    alias_5 = torch.ops.aten.alias.default(alias_4)
    # 使用 torch.ops.aten.view.default 函数将 getitem 的形状重塑为 [20, 100, 35, 45]，并清空 getitem
    view_5 = torch.ops.aten.view.default(getitem, [20, 100, 35, 45]);  getitem = None
    # 使用 torch.ops.aten.copy_.default 函数复制 alias_1 到 arg1_1，并清空 arg1_1 和 alias_1
    copy_ = torch.ops.aten.copy_.default(arg1_1, alias_1);  arg1_1 = alias_1 = None
    # 使用 torch.ops.aten.copy_.default 函数复制 alias_4 到 arg2_1，并清空 arg2_1 和 alias_4
    copy__1 = torch.ops.aten.copy_.default(arg2_1, alias_4);  arg2_1 = alias_4 = None
    # 返回形状重塑后的张量 view_5
    return view_5
    # 定义一个测试用例函数 test_batch_norm
    def test_batch_norm(self):
        
        # 定义一个内部函数 f，用于进行批量归一化操作
        def f(x, running_mean, running_var):
            # 启用 Python 调度器上下文管理器
            with enable_python_dispatcher():
                # 调用 torch.batch_norm 函数进行批量归一化处理
                return torch.batch_norm(
                    x, None, None, running_mean, running_var, True, 0.1, 1e-5, False
                )

        # 调用 self.assert_functionalization 方法，验证函数 f 的功能
        self.assert_functionalization(
            f, torch.randn(20, 100, 35, 45), torch.zeros(100), torch.ones(100)
        )
        
        # 获取函数 f 的日志信息
        logs = self.get_logs(
            f, torch.randn(20, 100, 35, 45), torch.zeros(100), torch.ones(100)
        )
        
        # 使用 self.assertExpectedInline 方法，验证日志信息是否符合预期
        self.assertExpectedInline(
            logs,
            """\
# 定义一个方法 forward，用于模型前向传播
def forward(self, arg0_1, arg1_1, arg2_1):
    # 创建一个空的 torch 张量，用于存储结果，设备为 CPU
    empty = torch.ops.aten.empty.memory_format([0], dtype=torch.uint8, layout=torch.strided, device=device(type='cpu'))
    # 调用 torch 的 C++ 函数 _native_batch_norm_legit_functional.default 进行批归一化操作
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(arg0_1, None, None, arg1_1, arg2_1, True, 0.1, 1e-05); arg0_1 = None
    # 获取批归一化操作的结果的第一个元素
    getitem = _native_batch_norm_legit_functional[0]
    # 获取批归一化操作的结果的第二个元素
    getitem_1 = _native_batch_norm_legit_functional[1]
    # 获取批归一化操作的结果的第三个元素
    getitem_2 = _native_batch_norm_legit_functional[2]
    # 获取批归一化操作的结果的第四个元素
    getitem_3 = _native_batch_norm_legit_functional[3]
    # 获取批归一化操作的结果的第五个元素
    getitem_4 = _native_batch_norm_legit_functional[4]; _native_batch_norm_legit_functional = None
    # 使用 torch 的 copy_ 方法复制第三个元素的内容到 arg1_1 中，并清空相关变量
    copy_ = torch.ops.aten.copy_.default(arg1_1, getitem_3); arg1_1 = getitem_3 = None
    # 使用 torch 的 copy_ 方法复制第四个元素的内容到 arg2_1 中，并清空相关变量
    copy__1 = torch.ops.aten.copy_.default(arg2_1, getitem_4); arg2_1 = getitem_4 = None
    # 返回批归一化操作结果的第一个元素
    return getitem
    def test_python_functionalization(self):
        # 定义一个内部函数 f，接受一个张量 x，并对其进行重塑、乘以2、加1的操作，返回处理后的张量
        def f(x):
            x_view = x.view(-1)  # 将张量 x 重塑为一维张量
            x.mul_(2)  # 将张量 x 中的所有元素乘以2
            return x_view + 1  # 返回重塑后的张量加1的结果

        # 定义一个函数 f_functionalized，接受一个张量 x，用 FunctionalTensor 进行包装后调用 f 函数处理，
        # 并最终将结果转换为普通张量返回
        def f_functionalized(x):
            # 注释区域说明如何在使用 python FunctionalTensor 和 FunctionalTensorMode 时禁用 functionalization
            # FunctionalTensor 是一个包装张量，包含一个内部的 FunctionalTensorWrapper
            # 由于内部张量的 keyset 中包含 `DispatchKey.Functionalize`，所以默认情况下，FunctionalTensor 也会继承相同的 keyset
            # 由于没有简单的方法直接从 python 中变更张量的 keyset，因此在此全局禁用 functionalization 更为简单
            maybe_disable = torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
            )
            with maybe_disable, FunctionalTensorMode():
                x_wrapped = FunctionalTensor.to_functional(x)  # 使用 FunctionalTensor 包装输入张量 x
                out_wrapped = f(x_wrapped)  # 对包装后的张量应用函数 f
            out_unwrapped = out_wrapped.elem  # 获取包装后结果的元素
            torch._sync(out_unwrapped)  # 同步张量 out_unwrapped
            return torch._from_functional_tensor(out_unwrapped)  # 将 out_unwrapped 转换为普通张量返回

        # 创建一个非叶节点张量 x，具有梯度信息
        x = torch.randn(2, requires_grad=True) + 1
        fx_g = make_fx(f_functionalized)(x)  # 对 f_functionalized 函数和输入 x 进行处理
        # 注意：由于视图重播，下面的 view_1 预期是存在的（虽然未使用）。AOTAutograd 运行 DCE（死代码消除）传递，稍后将删除此类节点。
        self.assertExpectedInline(
            fx_g.code.strip(),
            """\
# 定义一个类方法 `forward`，接受参数 `self` 和 `x_1`
def forward(self, x_1):
    # 使用 Torch 操作 `aten.view.default` 对输入张量 `x_1` 进行视图重塑，将其展开为一维数组
    view = torch.ops.aten.view.default(x_1, [-1])
    # 使用 Torch 操作 `aten.mul.Tensor` 对输入张量 `x_1` 中的每个元素乘以2，然后将 `x_1` 设为 `None`
    mul = torch.ops.aten.mul.Tensor(x_1, 2);  x_1 = None
    # 使用 Torch 操作 `aten.view.default` 对乘法结果 `mul` 进行视图重塑，将其展开为一维数组
    view_1 = torch.ops.aten.view.default(mul, [-1])
    # 再次使用 Torch 操作 `aten.view.default` 对乘法结果 `mul` 进行视图重塑，将其展开为一维数组，然后将 `mul` 设为 `None`
    view_2 = torch.ops.aten.view.default(mul, [-1]);  mul = None
    # 使用 Torch 操作 `aten.add.Tensor` 对视图 `view_2` 中的每个元素加1，然后将 `view_2` 设为 `None`
    add = torch.ops.aten.add.Tensor(view_2, 1);  view_2 = None
    # 返回加法结果 `add`
    return add
        # 定义测试函数 test_python_functionalization_conj
        def test_python_functionalization_conj(self):
            # 定义内部函数 f，接受一个参数 x
            def f(x):
                # 将 x 的克隆进行共轭操作，并将结果赋给 y
                y = x.clone().conj()
                # 将 y 的值乘以2
                y.mul_(2)
                # 返回 y 解析共轭后的结果视图
                return torch.view_as_real(y.resolve_conj())

            # 创建一个形状为 (4,)，数据类型为 complex64 的张量 x
            x = torch.randn(4, dtype=torch.complex64)
            # 调用函数 f，得到参考输出 out_ref
            out_ref = f(x)
            # 使用 dispatch_functionalize 包装函数 f，并对 x 进行调用，得到测试输出 out_test
            out_test = dispatch_functionalize(f)(x)
            # 使用 _functionalize 包装函数 f，并进行调用，得到测试输出 out_test_cpp
            out_test_cpp = _functionalize(
                f, reapply_views=True, crossref=False, skip_input_mutations=True
            )(x)
            # 断言参考输出与测试输出 out_ref 相等
            self.assertEqual(out_ref, out_test)
            # 断言测试输出 out_test 与 out_test_cpp 相等
            self.assertEqual(out_test, out_test_cpp)
            # 使用 make_fx 函数对 dispatch_functionalize(f) 包装后的结果进行调用，得到 fx_g
            fx_g = make_fx(dispatch_functionalize(f))(x)
            # 使用 make_fx 函数对 _functionalize(f) 包装后的结果进行调用，得到 fx_g_cpp
            fx_g_cpp = make_fx(
                _functionalize(
                    f, reapply_views=True, crossref=False, skip_input_mutations=True
                )
            )(x)
            # 断言 fx_g 的代码行与预期结果一致，忽略首尾空白字符
            self.assertExpectedInline(
                fx_g.code.strip(),
                """\
# 定义一个方法 `forward`，接受一个参数 `arg0_1`
def forward(self, arg0_1):
    # 使用 `torch.ops.aten.clone.default` 方法克隆 `arg0_1`，并清空 `arg0_1`
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    # 对克隆的张量应用共轭操作 `_conj`，并清空 `clone`
    _conj = torch.ops.aten._conj.default(clone);  clone = None
    # 再次克隆 `_conj`，赋值给 `clone_1`
    clone_1 = torch.ops.aten.clone.default(_conj)
    # 使用 `torch.ops.aten.mul.Tensor` 方法将 `clone_1` 乘以标量 2，清空 `clone_1` 和 `mul`
    mul = torch.ops.aten.mul.Tensor(clone_1, 2);  clone_1 = None; mul = None
    # 再次克隆 `_conj`，并应用 `torch.ops.aten.copy.default` 方法将其复制给 `copy`
    clone_2 = torch.ops.aten.clone.default(_conj);  _conj = None
    copy = torch.ops.aten.copy.default(clone_2, mul);  clone_2 = mul = None
    # 对 `copy` 应用共轭操作 `_conj_1`，并清空 `copy`
    _conj_1 = torch.ops.aten._conj.default(copy);  copy = None
    # 对 `_conj_1` 再次应用共轭操作 `_conj_2`，并清空 `_conj_1`
    _conj_2 = torch.ops.aten._conj.default(_conj_1);  _conj_1 = None
    # 再次克隆 `_conj_2`，赋值给 `clone_3`
    clone_3 = torch.ops.aten.clone.default(_conj_2);  _conj_2 = None
    # 对 `clone_3` 应用 `torch.ops.aten.view_as_real.default` 方法，返回结果并清空 `clone_3`
    view_as_real = torch.ops.aten.view_as_real.default(clone_3);  clone_3 = None
    # 返回 `view_as_real` 张量
    return view_as_real
    # 定义测试方法，测试 Python 函数式化的提升
    def test_python_functionalization_lift_fresh(self):
        # 定义内部函数 f，接受参数 x，创建临时张量 tmp，返回 tmp 加上 x 的结果
        def f(x):
            tmp = torch.tensor([0.0])
            return tmp + x

        # 生成一个包含四个随机数的张量 x
        x = torch.randn(4)
        # 调用 f 函数计算参考结果 out_ref
        out_ref = f(x)
        # 使用 dispatch_functionalize 函数将 f 函数进行函数式化处理，并计算结果 out_test
        out_test = dispatch_functionalize(f)(x)
        # 使用 _functionalize 函数处理 f 函数，生成 C++ 代码并计算结果 out_test_cpp
        out_test_cpp = _functionalize(
            f, reapply_views=True, crossref=False, skip_input_mutations=True
        )(x)
        # 断言 out_ref 和 out_test 相等
        self.assertEqual(out_ref, out_test)
        # 断言 out_ref 和 out_test_cpp 相等
        self.assertEqual(out_ref, out_test_cpp)
        # 使用 make_fx 函数将 dispatch_functionalize 处理后的 f 函数再次进行函数式化，并计算结果 fx_g
        fx_g = make_fx(dispatch_functionalize(f))(x)
        # 使用 make_fx 函数将 _functionalize 处理后的 f 函数再次进行函数式化，并计算结果 fx_g_cpp
        fx_g_cpp = make_fx(
            _functionalize(
                f, reapply_views=True, crossref=False, skip_input_mutations=True
            )
        )(x)
        # 断言 fx_g 生成的代码符合期望的内联格式
        self.assertExpectedInline(
            fx_g.code.strip(),
            """\
# 定义一个方法 `forward`，接受一个参数 `arg0_1`，用于执行前向传播操作
def forward(self, arg0_1):
    # 获取对象属性 `_tensor_constant0` 的值，并赋给 `_tensor_constant0`
    _tensor_constant0 = self._tensor_constant0
    # 调用 Torch 的 ATen 库中的 `lift_fresh_copy` 函数，对 `_tensor_constant0` 进行操作并返回结果给 `lift_fresh_copy`，同时置 `_tensor_constant0` 为 `None`
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    # 调用 Torch 的 ATen 库中的 `add` 函数，将 `lift_fresh_copy` 和 `arg0_1` 相加，结果赋给 `add`，同时将 `lift_fresh_copy` 和 `arg0_1` 置为 `None`
    add = torch.ops.aten.add.Tensor(lift_fresh_copy, arg0_1);  lift_fresh_copy = arg0_1 = None
    # 返回相加后的结果 `add`
    return add
```