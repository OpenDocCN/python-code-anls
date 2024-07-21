# `.\pytorch\test\test_fx_reinplace_pass.py`

```py
# 导入必要的模块和函数
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.fx.passes.reinplace import reinplace
from torch.fx.experimental.proxy_tensor import make_fx

try:
    # 尝试导入函数化模块，标记是否成功导入
    from functorch.experimental import functionalize
    HAS_FUNCTIONALIZATION = True
except Exception as e:
    # 导入失败时设置标记为False
    HAS_FUNCTIONALIZATION = False

# 定义测试类，继承自TestCase
class TestReinplacePass(TestCase):

    # 第一个测试函数：测试基本的reinplace功能
    def test_reinplace_basic(self):
        # 定义一个函数f，对输入张量进行克隆并执行加法操作
        def f(x):
            a = x.clone()
            b = a.add(1)
            return b

        # 准备输入数据：全1张量
        inpt = torch.ones(2)
        # 使用make_fx将f函数转换为FX图，并进行reinplace处理
        f2 = reinplace(make_fx(f)(inpt), inpt)
        # 期望输出：调用f函数得到的结果
        expected_out = f(inpt)
        # 实际输出：调用经过reinplace处理后的f2函数得到的结果
        actual_out = f2(inpt)
        # 断言：比较期望输出和实际输出是否一致
        self.assertEqual(actual_out, expected_out)
        # 断言：比较f2的代码生成是否与预期一致
        self.assertExpectedInline(f2.code, """\



def forward(self, x_1):
    clone = torch.ops.aten.clone.default(x_1);  x_1 = None
    add = torch.ops.aten.add_.Tensor(clone, 1)
    return clone
    """)


    # 第二个测试函数：测试带视图操作的reinplace功能
    def test_reinplace_with_view(self):
        # 定义一个函数f，对输入张量进行克隆并执行视图和加法操作
        def f(x):
            a = x.clone()
            a_view = a.view(-1)
            # 不应对第一个add()进行reinplace，因为稍后程序中会重用a的别名
            b = a.add(1)
            # 第二个add()可以进行reinplace
            c = a_view.add(1)
            return c

        # 准备输入数据：全1张量
        inpt = torch.ones(2)
        # 使用make_fx将f函数转换为FX图，并进行reinplace处理
        f2 = reinplace(make_fx(f)(inpt), inpt)
        # 期望输出：调用f函数得到的结果
        expected_out = f(inpt)
        # 实际输出：调用经过reinplace处理后的f2函数得到的结果
        actual_out = f2(inpt)
        # 断言：比较期望输出和实际输出是否一致
        self.assertEqual(actual_out, expected_out)
        # 断言：比较f2的代码生成是否与预期一致
        self.assertExpectedInline(f2.code, """\



def forward(self, x_1):
    clone = torch.ops.aten.clone.default(x_1);  x_1 = None
    view = torch.ops.aten.view.default(clone, [-1])
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    add_1 = torch.ops.aten.add_.Tensor(view, 1)
    return view
    """)

    # 第三个测试函数：测试带不同元数据的reinplace功能
    def test_reinplace_different_metadata(self):
        # 定义一个函数f，对输入张量进行克隆并执行加法和比较操作
        def f(a_):
            a = a_.clone()
            b = a + 1
            # 不应对.ge()进行reinplace，因为需要将"b"从浮点张量调整为布尔张量
            c = torch.ge(b, a)
            return c

        # 准备输入数据：全1张量
        inpt = torch.ones(4)
        # 使用make_fx将f函数转换为FX图，并进行reinplace处理
        f2 = reinplace(make_fx(f)(inpt), inpt)
        # 期望输出：调用f函数得到的结果
        expected_out = f(inpt)
        # 实际输出：调用经过reinplace处理后的f2函数得到的结果
        actual_out = f2(inpt)
        # 断言：比较期望输出和实际输出是否一致
        self.assertEqual(actual_out, expected_out)
        # 断言：比较f2的代码生成是否与预期一致
        # .ge()操作不应该被reinplace
        self.assertExpectedInline(f2.code, """\



def forward(self, a__1):
    clone = torch.ops.aten.clone.default(a__1);  a__1 = None
    add = torch.ops.aten.add.Tensor(clone, 1)
    ge = torch.ops.aten.ge.Tensor(add, clone);  add = clone = None
    return ge
    """)
    def test_reinplace_overlapping_memory(self):
        # 定义内部函数 f，接受一个参数 a_，克隆该参数并赋值给 a
        def f(a_):
            a = a_.clone()
            # 将 a 扩展为一个新的张量 b，大小为 (4, 4)
            b = a.expand(4, 4)
            # 由于 b 具有重叠的内存，无法直接就地替换
            # 执行加法操作，生成新的张量 c
            c = b.add(1)
            return c
        # 创建一个包含全为 1 的张量 inpt
        inpt = torch.ones(1)
        # 使用 make_fx 函数创建函数 f 的 TorchScript 版本，并应用在输入张量 inpt 上，得到 f2
        f2 = reinplace(make_fx(f)(inpt), inpt)
        # 期望的输出是调用 f 函数的结果
        expected_out = f(inpt)
        # 实际的输出是调用 f2 函数的结果
        actual_out = f2(inpt)
        # 断言实际输出等于期望输出
        self.assertEqual(actual_out, expected_out)
        # 断言 f2 函数的代码与给定字符串的内联版本匹配
        self.assertExpectedInline(f2.code, """
def forward(self, a__1):
    # 使用 torch.ops.aten.clone.default 方法对输入张量进行克隆操作，并清空原始输入变量
    clone = torch.ops.aten.clone.default(a__1);  a__1 = None
    # 使用 torch.ops.aten.expand.default 方法对克隆的张量进行扩展操作，扩展为 4x4 的张量，并清空克隆变量
    expand = torch.ops.aten.expand.default(clone, [4, 4]);  clone = None
    # 使用 torch.ops.aten.add.Tensor 方法在扩展后的张量上加法操作，加 1，并清空扩展变量
    add = torch.ops.aten.add.Tensor(expand, 1);  expand = None
    # 返回加法操作后的张量
    return add
        def f(a_):
            # 定义函数 f，用于操作输入张量 a_
            # for now, don't test mutations to inputs
            # 复制输入张量 a_，避免对输入的直接改变
            a = a_.clone()
            # 取张量 a 的第二列并赋值给 b
            b = a[:, 1]
            # 取 b 的第二个元素并赋值给 c
            c = b[1]
            # 对 c 增加 1，直接在原地修改 c 的值
            c.add_(1)
            # 返回处理后的张量 a
            return a

        # 如果没有函数化支持，退出测试
        if not HAS_FUNCTIONALIZATION:
            return

        # 创建一个全为 1 的 4x4 的张量作为输入
        inpt = torch.ones(4, 4)
        # 将函数 f 函数化并应用到输入张量上，得到函数化后的新函数 f2
        f2 = reinplace(make_fx(functionalize(f))(inpt), inpt)
        # 计算预期输出，即应用函数 f 到输入张量上的结果
        expected_out = f(inpt)
        # 计算实际输出，即应用函数 f2 到输入张量上的结果
        actual_out = f2(inpt)
        # 断言实际输出与预期输出相等
        self.assertEqual(actual_out, expected_out)
        # 断言函数 f2 的代码与预期的内联代码相等
        self.assertExpectedInline(f2.code, """\
def forward(self, a__1):
    # 使用 torch.ops.aten.clone.default 方法克隆输入张量 a__1
    clone = torch.ops.aten.clone.default(a__1);  a__1 = None
    # 使用 torch.ops.aten.slice.Tensor 方法对克隆张量进行切片操作
    slice_1 = torch.ops.aten.slice.Tensor(clone, 0, 0, 9223372036854775807)
    # 使用 torch.ops.aten.select.int 方法在切片结果上选择特定索引
    select = torch.ops.aten.select.int(slice_1, 1, 1);  slice_1 = None
    # 使用 torch.ops.aten.select.int 方法在选择结果上进行进一步选择
    select_1 = torch.ops.aten.select.int(select, 0, 1);  select = None
    # 使用 torch.ops.aten.add_.Tensor 方法在选择结果上进行加法操作
    add = torch.ops.aten.add_.Tensor(select_1, 1);  select_1 = None
    # 使用 torch.ops.aten.as_strided.default 方法创建一个新视图张量
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [4], 1);  clone = None
    # 返回新创建的视图张量
    return as_strided
        def test_reinplace_scatter_twice_with_different_view_op_invalid(self):
            # 定义测试函数，测试在不同视图操作无效的情况下的重复散布更新
            def f(a_):
                # 复制输入张量a_
                a = a_.clone()
                # 获取a的第二列视图b
                b = a[:, 1]
                # 获取b的第二个元素c
                c = b[1]
                # 对c进行加1操作，生成更新后的c_updated
                c_updated = c.add(1)
                # 创建b的等效视图good_mirror_of_b
                good_mirror_of_b = a.as_strided((4,), (4,), 1)
                # select_scatter的第一个参数是b的等效视图，但是下面的select_scatter调用尝试
                # 将c_updated放入"b"的一个与当前"c"所占用不同的切片中。
                b_updated = torch.select_scatter(good_mirror_of_b, c_updated, 0, 0)
                return b_updated

            # 创建4x4的全1张量作为输入
            inpt = torch.ones(4, 4)
            # 使用make_fx和reinplace对f进行转换和就地操作
            f2 = reinplace(make_fx(f)(inpt), inpt)
            # 期望的输出是对输入inpt调用f的结果
            expected_out = f(inpt)
            # 实际输出是对转换后的f2对输入inpt的调用结果
            actual_out = f2(inpt)
            # 断言实际输出等于期望输出
            self.assertEqual(actual_out, expected_out)
            # 断言f2的代码内容符合内联期望值
            self.assertExpectedInline(f2.code, """\
# 定义一个名为 forward 的方法，接受参数 a__1
def forward(self, a__1):
    # 使用 torch.ops.aten.clone.default 克隆输入参数 a__1
    clone = torch.ops.aten.clone.default(a__1);  a__1 = None
    # 对克隆的张量进行切片操作，选择从索引 0 到最大索引，即全张量切片
    slice_1 = torch.ops.aten.slice.Tensor(clone, 0, 0, 9223372036854775807)
    # 在切片结果上选择第一维度索引为 1 的部分
    select = torch.ops.aten.select.int(slice_1, 1, 1);  slice_1 = None
    # 在上一步选择的结果上再选择第零维度索引为 1 的部分
    select_1 = torch.ops.aten.select.int(select, 0, 1);  select = None
    # 将上述选择结果加上常数 1
    add = torch.ops.aten.add.Tensor(select_1, 1);  select_1 = None
    # 使用 as_strided 将克隆的张量视图重新安排为指定形状和步幅
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [4], 0);  clone = None
    # 在重新排列后的张量视图上选择第零维度索引为 0 的部分
    select_int = torch.ops.aten.select.int(as_strided, 0, 1)
    # 将 add 的结果复制到上一步选择的结果中
    copy__default = torch.ops.aten.copy_.default(select_int, add);  select_int = add = None
    # 返回重新排列后的张量视图
    return as_strided
    # 定义一个测试函数，测试索引变异
    def test_reinplace_index_mutation(self):
        # 定义一个函数f，创建一个4x4x4的全零张量a，然后将a的第二列及之后的元素替换为4x2x4的全一张量
        def f():
            a = torch.zeros(4, 4, 4)
            a[:, 2:] = torch.ones(4, 2, 4)
            return a

        # 如果没有函数化功能，则直接返回
        if not HAS_FUNCTIONALIZATION:
            return
        # 将函数f进行函数化处理，并使用reinplace进行替换
        f2 = reinplace(make_fx(functionalize(f))())
        # 期望输出为函数f的结果
        expected_out = f()
        # 实际输出为经过reinplace处理后的函数f2的结果
        actual_out = f2()
        # 断言实际输出与期望输出相等
        self.assertEqual(actual_out, expected_out)
        # 断言函数f2的代码与预期的内联代码相等
        self.assertExpectedInline(f2.code, """\
# 定义一个方法，用于执行前向计算
def forward(self):
    # 创建一个形状为 [4, 4, 4] 的全零张量，分配在 CPU 上，不固定在内存中
    zeros = torch.ops.aten.zeros.default([4, 4, 4], device=device(type='cpu'), pin_memory=False)
    # 创建一个形状为 [4, 2, 4] 的全一张量，分配在 CPU 上，不固定在内存中
    ones = torch.ops.aten.ones.default([4, 2, 4], device=device(type='cpu'), pin_memory=False)
    # 对 zeros 张量进行切片操作，从第 0 维度开始，截取到 9223372036854775807（Python 中的最大整数）
    slice_1 = torch.ops.aten.slice.Tensor(zeros, 0, 0, 9223372036854775807)
    # 对 slice_1 张量进行切片操作，从第 1 维度开始，截取到 9223372036854775807，然后释放 slice_1 引用
    slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 2, 9223372036854775807);  slice_1 = None
    # 使用 torch 的原位复制函数将 slice_2 的内容复制到 ones 张量上，并释放 slice_2 和 ones 引用
    copy = torch.ops.aten.copy_.default(slice_2, ones);  slice_2 = ones = None
    # 对 zeros 张量进行切片操作，从第 0 维度开始，截取到 9223372036854775807
    slice_3 = torch.ops.aten.slice.Tensor(zeros, 0, 0, 9223372036854775807)
    # 对 zeros 张量进行切片操作，从第 0 维度开始，截取到 9223372036854775807
    slice_4 = torch.ops.aten.slice.Tensor(zeros, 0, 0, 9223372036854775807)
    # 对 slice_4 张量进行切片操作，从第 1 维度开始，截取到 9223372036854775807，然后释放 slice_4 引用
    slice_5 = torch.ops.aten.slice.Tensor(slice_4, 1, 2, 9223372036854775807);  slice_4 = None
    # 返回全零张量 zeros
    return zeros
```