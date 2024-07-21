# `.\pytorch\test\dynamo\test_input_attr_tracking.py`

```py
# Owner(s): ["module: dynamo"]
# flake8: noqa

# 导入 torch 库及其私有模块
import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import (
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    normalize_gm,
)

# 定义测试类 TestInputAttrTracking，继承自 torch._dynamo.test_case.TestCase
class TestInputAttrTracking(torch._dynamo.test_case.TestCase):

    # 测试函数，测试在张量上的属性访问
    def test_tensor_property_on_tensor(self):
        # 定义一个函数 fn，接受参数 x，并返回 x 与 x.y 的乘积
        def fn(x):
            return x * x.y

        # 创建一个随机张量 x_
        x_ = torch.randn([2, 2])
        # 创建一个随机张量 y_
        y_ = torch.randn([2, 2])
        # 将 y_ 赋值给 x_ 的属性 y
        x_.y = y_

        # 在非优化状态下执行函数 fn，获取结果
        eager_result = fn(x_)

        # 定义图变量
        graph = None

        # 定义一个函数 grab_graph_backend，用于捕获图信息
        def grab_graph_backend(gm, inps):
            nonlocal graph
            graph = gm
            return gm

        # 使用 torch._dynamo.optimize 优化 fn 函数，开启无 Python 模式
        fn = torch._dynamo.optimize(grab_graph_backend, nopython=True)(fn)
        # 在优化状态下执行函数 fn，获取结果
        compile_result = fn(x_)
        # 断言优化前后的结果相等
        self.assertEqual(eager_result, compile_result)

        # 统计占位符节点的数量
        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                placeholder_cnt += 1

        # 断言占位符节点数量为 2
        # 我们希望确保 y 被提升为输入！
        self.assertEqual(placeholder_cnt, 2)

    # 测试函数，测试在张量上分配属性
    def test_tensor_property_assigned_on_tensor(self):
        # 定义一个函数 fn，接受参数 x 和 y，并将 y 分配给 x.y，然后返回 x 与 x.y 的乘积
        def fn(x, y):
            x.y = y
            return x * x.y

        # 创建一个随机张量 x_
        x_ = torch.randn([2, 2])
        # 创建一个随机张量 y_
        y_ = torch.randn([2, 2])

        # 在非优化状态下执行函数 fn，获取结果
        eager_result = fn(x_, y_)

        # 定义图变量
        graph = None

        # 定义一个函数 grab_graph_backend，用于捕获图信息
        def grab_graph_backend(gm, inps):
            nonlocal graph
            graph = gm
            return gm

        # 使用 torch._dynamo.optimize 优化 fn 函数，开启无 Python 模式
        fn = torch._dynamo.optimize(grab_graph_backend, nopython=True)(fn)
        # 在优化状态下执行函数 fn，获取结果
        compile_result = fn(x_, y_)
        # 断言优化前后的结果相等
        self.assertEqual(eager_result, compile_result)

        # 统计占位符节点的数量
        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                placeholder_cnt += 1

        # 断言占位符节点数量为 2
        # y 已经是一个输入
        self.assertEqual(placeholder_cnt, 2)

    # 测试函数，测试在张量上分配常量属性
    def test_const_property_on_tensor(self):
        # 定义一个函数 fn，接受参数 x，并返回 x 与 x.y 的乘积
        def fn(x):
            return x * x.y

        # 创建一个随机张量 x_
        x_ = torch.randn([2, 2])
        # 创建一个常量 y_
        y_ = 4
        # 将 y_ 赋值给 x_ 的属性 y
        x_.y = y_

        # 在非优化状态下执行函数 fn，获取结果
        eager_result = fn(x_)

        # 定义图变量
        graph = None

        # 定义一个函数 grab_graph_backend，用于捕获图信息
        def grab_graph_backend(gm, inps):
            nonlocal graph
            graph = gm
            return gm

        # 使用 torch._dynamo.optimize 优化 fn 函数，开启无 Python 模式
        fn = torch._dynamo.optimize(grab_graph_backend, nopython=True)(fn)
        # 在优化状态下执行函数 fn，获取结果
        compile_result = fn(x_)
        # 断言优化前后的结果相等
        self.assertEqual(eager_result, compile_result)

        # 统计占位符节点的数量
        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                placeholder_cnt += 1

        # 断言占位符节点数量为 1
        # 我们希望确保这不会将 y 提升为输入，因为它是一个常量
        self.assertEqual(placeholder_cnt, 1)

    # 测试函数，测试在张量上分配常量属性
    def test_const_property_assigned_on_tensor(self):
        # 定义一个函数 fn，接受参数 x 和 y，并将 y 分配给 x.y，然后返回 x 与 x.y 的乘积
        def fn(x, y):
            x.y = y
            return x * x.y

        # 创建一个随机张量 x_
        x_ = torch.randn([2, 2])
        # 创建一个常量 y_
        y_ = 4

        # 在非优化状态下执行函数 fn，获取结果
        eager_result = fn(x_, y_)

        # 使用 "eager" 模式优化 fn 函数，开启无 Python 模式
        fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 在优化状态下执行函数 fn，获取结果
        compile_result = fn(x_, y_)
        # 断言优化前后的结果相等
        self.assertEqual(eager_result, compile_result)
    # 测试函数，用于验证在张量类型更改时正确分配属性
    def test_guards_correctly_property_assigned_on_tensor_type_change(self):
        # 定义内部函数fn，接受两个参数x和y
        def fn(x, y):
            # 给x对象分配属性y
            x.y = y
            # 返回x乘以其属性y的结果
            return x * x.y
    
        # 创建一个形状为[2, 2]的随机张量x_
        x_ = torch.randn([2, 2])
    
        # 优化fn函数，使用"eager"模式，禁用nopython优化
        fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 编译并计算常数参数4的结果
        compile_result_const = fn(x_, 4)
        # 断言编译结果与x_乘以4的结果相等
        self.assertEqual(compile_result_const, x_ * 4)
    
        # 创建一个形状为[2, 2]的随机张量y
        y = torch.randn([2, 2])
        # 编译并计算张量参数y的结果
        compile_result_tensor = fn(x_, y)
        # 断言编译结果与x_乘以y的结果相等
        self.assertEqual(compile_result_tensor, x_ * y)
    
    # 测试函数，用于验证在张量类型更改时正确分配属性（使用"inductor"模式）
    def test_guards_correctly_property_assigned_on_tensor_type_change_inductor(self):
        # 定义内部函数fn，接受两个参数x和y
        def fn(x, y):
            # 给x对象分配属性y
            x.y = y
            # 返回x乘以其属性y的结果
            return x * x.y
    
        # 创建一个形状为[2, 2]的随机张量x_
        x_ = torch.randn([2, 2])
    
        # 优化fn函数，使用"inductor"模式，禁用nopython优化
        fn = torch._dynamo.optimize("inductor", nopython=True)(fn)
        # 编译并计算常数参数4的结果
        compile_result_const = fn(x_, 4)
        # 断言编译结果与x_乘以4的结果相等
        self.assertEqual(compile_result_const, x_ * 4)
    
        # 创建一个形状为[2, 2]的随机张量y
        y = torch.randn([2, 2])
        # 编译并计算张量参数y的结果
        compile_result_tensor = fn(x_, y)
        # 断言编译结果与x_乘以y的结果相等
        self.assertEqual(compile_result_tensor, x_ * y)
    # 定义测试方法，测试复杂属性访问是否能够在没有图中断的情况下执行
    def test_complex_attr_access_without_graph_breaks(self):
        # 定义内部函数 fn，接受三个参数 x, y, z
        def fn(x, y, z):
            # 对列表 x 中的每个元素 t 进行迭代
            for t in x:
                # 设置 t 对象的属性 y 为参数 y 的值
                t.y = y
                # 设置 t 对象的属性 z 为 y 乘以 z 的值
                t.z = y * z

            # 初始化新的变量 new_y 和 new_z 为 1
            new_y = 1
            new_z = 1
            # 再次对列表 x 中的每个元素 t 进行迭代
            for t in x:
                # 更新 new_y 为 t 对象的属性 y 与 new_y 的乘积
                new_y = t.y * new_y
                # 更新 new_z 为 t 对象的属性 z 与 new_z 的乘积
                new_z = t.z * new_z

            # 返回计算后的 new_y 和 new_z 值
            return new_y, new_z

        # 创建三个 2x2 大小的张量 x_0, x_1, x_2
        x_0 = torch.randn([2, 2])
        x_1 = torch.randn([2, 2])
        x_2 = torch.randn([2, 2])
        # 将这些张量放入列表 x 中
        x = [x_0, x_1, x_2]

        # 创建一个 2x2 大小的张量 y
        y = torch.randn([2, 2])
        # 设置一个整数值 z
        z = 5

        # 使用 fn 函数计算 eager_result
        eager_result = fn(x, y, z)

        # 创建 CompileCounter 对象 counter
        counter = CompileCounter()
        # 使用 torch._dynamo.optimize 进行优化，并执行优化后的 fn 函数
        fn = torch._dynamo.optimize(counter, nopython=True)(fn)

        # 使用优化后的 fn 函数计算 compile_result
        compile_result = fn(x, y, z)

        # 断言优化前后的结果应该一致
        self.assertEqual(compile_result, eager_result)
        # 断言编译计数器的帧数为 1
        self.assertEqual(counter.frame_count, 1)
        # 断言编译计数器的操作数为 9
        self.assertEqual(counter.op_count, 9)
        # 以下是一个用于参考的图形化描述，展示了操作的顺序和参数
        #         -------------  ------  -----------------------  ------------------------------------  --------
        # placeholder    l_y_    L_y_                     ()                                    {}
        # call_function  mul     <built-in function mul>  (l_y_, 5)                             {}
        # call_function  mul_1   <built-in function mul>  (l_y_, 5)                             {}
        # call_function  mul_2   <built-in function mul>  (l_y_, 5)                             {}
        # call_function  mul_3   <built-in function mul>  (l_y_, 1)                             {}
        # call_function  mul_4   <built-in function mul>  (mul, 1)                              {}
        # call_function  mul_5   <built-in function mul>  (l_y_, mul_3)                         {}
        # call_function  mul_6   <built-in function mul>  (mul_1, mul_4)                        {}
        # call_function  mul_7   <built-in function mul>  (l_y_, mul_5)                         {}
        # call_function  mul_8   <built-in function mul>  (mul_2, mul_6)                        {}
        # output         output  output                   ((mul_7, mul_8, mul, mul_1, mul_2),)  {}
    # 定义一个测试函数，测试复杂属性访问和图形断点
    def test_complex_attr_access_with_graph_breaks(self):
        # 定义一个内部函数fn，接受参数x, y, z
        def fn(x, y, z):
            # 遍历x中的每个元素t
            for t in x:
                # 设置t的属性y为y的值
                t.y = y
                # 设置t的属性z为y和z的乘积
                t.z = y * z

            # 打印调试信息 "Break!"
            print("Break!")

            # 初始化new_y和new_z为1
            new_y = 1
            new_z = 1
            # 再次遍历x中的每个元素t
            for t in x:
                # 更新new_y为t.y和new_y的乘积
                new_y = t.y * new_y
                # 更新new_z为t.z和new_z的乘积
                new_z = t.z * new_z

            # 返回更新后的new_y和new_z
            return new_y, new_z

        # 生成一个形状为[2, 2]的随机张量x_0
        x_0 = torch.randn([2, 2])
        # 生成一个形状为[2, 2]的随机张量x_1
        x_1 = torch.randn([2, 2])
        # 生成一个形状为[2, 2]的随机张量x_2
        x_2 = torch.randn([2, 2])
        # 将x_0, x_1, x_2组成一个列表x
        x = [x_0, x_1, x_2]

        # 生成一个形状为[2, 2]的随机张量y
        y = torch.randn([2, 2])
        # 设置z为5
        z = 5

        # 使用未优化的fn函数计算eager_result
        eager_result = fn(x, y, z)

        # 初始化CompileCounter对象counter
        counter = CompileCounter()
        # 优化fn函数，并赋值给fn
        fn = torch._dynamo.optimize(counter, nopython=False)(fn)

        # 使用优化后的fn函数计算compile_result
        compile_result = fn(x, y, z)
        # 断言compile_result等于eager_result
        self.assertEqual(compile_result, eager_result)
        # 断言frame_count等于2
        self.assertEqual(counter.frame_count, 2)
        # 断言op_count等于9
        self.assertEqual(counter.op_count, 9)
        # 输出用于参考的计算图信息
        # -------------  ------  -----------------------  ----------------------  --------
        # placeholder    l_y_    L_y_                     ()                      {}
        # call_function  mul     <built-in function mul>  (l_y_, 5)               {}
        # call_function  mul_1   <built-in function mul>  (l_y_, 5)               {}
        # call_function  mul_2   <built-in function mul>  (l_y_, 5)               {}
        # output         output  output                   ((mul, mul_1, mul_2),)  {}
        # [GRAPH BREAK!]
        # -------------  -------  -----------------------  -----------------  --------
        # placeholder    l_x_0_y  L_x_0_y                  ()                 {}
        # placeholder    l_x_0_z  L_x_0_z                  ()                 {}
        # placeholder    l_x_1_y  L_x_1_y                  ()                 {}
        # placeholder    l_x_1_z  L_x_1_z                  ()                 {}
        # placeholder    l_x_2_y  L_x_2_y                  ()                 {}
        # placeholder    l_x_2_z  L_x_2_z                  ()                 {}
        # call_function  mul      <built-in function mul>  (l_x_0_y, 1)       {}
        # call_function  mul_1    <built-in function mul>  (l_x_0_z, 1)       {}
        # call_function  mul_2    <built-in function mul>  (l_x_1_y, mul)     {}
        # call_function  mul_3    <built-in function mul>  (l_x_1_z, mul_1)   {}
        # call_function  mul_4    <built-in function mul>  (l_x_2_y, mul_2)   {}
        # call_function  mul_5    <built-in function mul>  (l_x_2_z, mul_3)   {}
        # output         output   output                   ((mul_4, mul_5),)  {}
    # 定义一个测试方法，用于测试在内联重构中复杂属性访问的情况
    def test_complex_attr_access_with_inline_reconstruct(self):
        # 定义内联测试函数，接受参数 x, y, z，并打印 "f"，返回 x.a + y.a + z.a 的结果
        def inline_test_fn(x, y, z):
            print("f")
            return x.a + y.a + z.a

        # 定义函数 fn，接受参数 x, y, z，分别给 x.a, y.a, z.a 赋值为 1, 2, 3
        def fn(x, y, z):
            x.a = 1
            y.a = 2
            z.a = 3

            # 调用内联测试函数，计算并返回 x, y 乘以内联测试函数结果的值
            mult = inline_test_fn(x, y, z)
            y = y * mult
            x = x * mult
            return x, y

        # 生成随机的 2x2 Tensor：x, y, z
        x = torch.randn([2, 2])
        y = torch.randn([2, 2])
        z = torch.randn([2, 2])

        # 在 eager 模式下执行 fn 函数，记录结果
        eager_result = fn(x, y, z)

        # 初始化一个编译计数器
        counter = CompileCounter()

        # 使用 Torch 内部优化函数对 fn 进行编译，关闭 JIT 编译模式（nopython=False）
        fn = torch._dynamo.optimize(counter, nopython=False)(fn)

        # 在编译模式下执行优化后的 fn 函数，记录结果
        compile_result = fn(x, y, z)

        # 断言编译结果与 eager 模式结果相等
        self.assertEqual(compile_result, eager_result)
        # 断言编译计数器中的帧数为 1
        self.assertEqual(counter.frame_count, 1)
        # 断言编译计数器中的操作数为 2
        self.assertEqual(counter.op_count, 2)

        # 给出的注释是关于该测试中的图形表示，供参考
        # 图形参考
        # __compiled_fn_2 <eval_with_key>.0 opcode         name    target                   args             kwargs
        # -------------  ------  -----------------------  ---------------  --------
        # placeholder    l_x_    L_x_                     ()               {}
        # placeholder    l_y_    L_y_                     ()               {}
        # call_function  mul     <built-in function mul>  (l_y_, 6)        {}
        # call_function  mul_1   <built-in function mul>  (l_x_, 6)        {}
        # output         output  output                   ((mul_1, mul),)  {}

    # 定义一个测试方法，用于测试在输入张量上设置数据的情况
    def test_set_data_on_input_tensor(self):
        # 定义函数 fn，接受参数 x, y，将 y.data 赋值给 x.data，根据 x, y 的尺寸返回对应操作的结果
        def fn(x, y):
            x.data = y.data
            if x.size() == y.size():
                return x * y
            else:
                return y * y

        # 生成随机的 5x5 和 2x2 Tensor：x, y
        x = torch.randn([5, 5])
        y = torch.randn([2, 2])

        # 在 eager 模式下执行 fn 函数，记录结果
        eager_result = fn(x, y)

        # 初始化一个 EagerAndRecordGraphs 实例
        eager_and_record = EagerAndRecordGraphs()

        # 初始化一个带后端的编译计数器
        counter = CompileCounterWithBackend(eager_and_record)

        # 使用 Torch 内部优化函数对 fn 进行编译，开启 JIT 编译模式（nopython=True）
        fn = torch._dynamo.optimize(counter, nopython=True)(fn)

        # 在编译模式下执行优化后的 fn 函数，记录结果
        compile_result = fn(x, y)

        # 获取编译图形的可读字符串并标准化
        graph = eager_and_record.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        # 断言编译结果与 eager 模式结果相等
        self.assertEqual(compile_result, eager_result)
        # 断言编译计数器中的帧数为 1
        self.assertEqual(counter.frame_count, 1)
        # 断言编译计数器中的操作数为 6
        self.assertEqual(counter.op_count, 6)
        # 断言生成的图形表示与预期的一致
        self.assertExpectedInline(
            actual,
            """\
# 定义一个名为 GraphModule 的类，继承自 torch.nn.Module
class GraphModule(torch.nn.Module):
    
    # 定义前向传播函数 forward，接受两个参数 L_y_ 和 L_x_，类型为 f32[2, 2]
    def forward(self, L_y_: "f32[2, 2]", L_x_: "f32[2, 2]"):
        
        # 将输入参数 L_y_ 和 L_x_ 分别赋值给 l_y_ 和 l_x_
        l_y_ = L_y_
        l_x_ = L_x_

        # 对 l_y_ 进行 detach 操作，生成一个新的 Tensor
        detach: "f32[2, 2]" = l_y_.detach()

        # 禁用梯度计算，并将 l_x_ 的值设置为 detach 的值，然后将 detach 置为 None
        _set_grad_enabled = torch._C._set_grad_enabled(False)
        set_: "f32[2, 2]" = torch_Tensor_set_(l_x_, detach);  detach = None

        # 再次启用梯度计算
        _set_grad_enabled_1 = torch._C._set_grad_enabled(True)

        # 使用内部函数 torch__dynamo_variables_builtin__lower_version_count_by_1 对 set_ 进行操作，然后将 set_ 置为 None
        _lower_version_count_by_1 = torch__dynamo_variables_builtin__lower_version_count_by_1(set_);  set_ = None

        # 计算 l_x_ 和 l_y_ 的元素级乘积，然后将 l_x_ 和 l_y_ 置为 None
        mul: "f32[2, 2]" = l_x_ * l_y_;  l_x_ = l_y_ = None
        
        # 返回结果，以元组形式包裹在括号中
        return (mul,)
    # 定义一个测试方法，用于在自定义类输入张量时设置数据
    def test_set_data_on_user_defined_class_input_tensor(self):
        # 定义一个用户自定义的类 MyUserDefinedClass
        class MyUserDefinedClass:
            # 初始化方法，接收参数 x 和 y
            def __init__(self, x, y):
                self.x = x  # 将参数 x 赋值给实例变量 self.x
                self.y = y  # 将参数 y 赋值给实例变量 self.y

            # 执行一些 setattr 操作的方法
            def do_some_setattr_stuff(self):
                self.z = x * y  # 计算 x 和 y 的乘积，并将结果赋值给 self.z
                self.a = x + x  # 计算 x 的两倍，并将结果赋值给 self.a
                return self.z * self.a  # 返回 self.z 和 self.a 的乘积作为结果

        x = torch.randn([5, 5])  # 生成一个大小为 [5, 5] 的随机张量 x
        y = torch.randn([5, 5])  # 生成一个大小为 [5, 5] 的随机张量 y
        mudc_1 = MyUserDefinedClass(x, y)  # 创建 MyUserDefinedClass 的实例 mudc_1，传入 x 和 y

        eager_result = mudc_1.do_some_setattr_stuff()  # 调用实例 mudc_1 的 do_some_setattr_stuff 方法，并得到结果

        counter = CompileCounter()  # 创建一个 CompileCounter 实例 counter

        mudc_2 = MyUserDefinedClass(x, y)  # 创建 MyUserDefinedClass 的另一个实例 mudc_2，传入 x 和 y
        # 使用 torch._dynamo.optimize 进行优化，设置 nopython=True，将 mudc_2.do_some_setattr_stuff 方法传入进行优化
        do_some_setattr_stuff = torch._dynamo.optimize(counter, nopython=True)(
            mudc_2.do_some_setattr_stuff
        )

        compile_result = do_some_setattr_stuff()  # 执行优化后的 do_some_setattr_stuff 方法，并得到结果
        self.assertEqual(compile_result, eager_result)  # 断言优化后的结果与非优化结果相等
        self.assertEqual(counter.frame_count, 1)  # 断言编译器统计的帧数为 1
        self.assertEqual(counter.op_count, 3)  # 断言编译器统计的操作数为 3
        # 提供一个用于参考的图形表示，显示编译后的执行流程
        #  __compiled_fn_0 <eval_with_key>.0 opcode         name    target                   args                  kwargs
        # -------------  ------  -----------------------  --------------------  --------
        # placeholder    l_x_    L_x_                     ()                    {}
        # placeholder    l_y_    L_y_                     ()                    {}
        # call_function  mul     <built-in function mul>  (l_x_, l_y_)          {}
        # call_function  add     <built-in function add>  (l_x_, l_x_)          {}
        # call_function  mul_1   <built-in function mul>  (mul, add)            {}
        # output         output  output                   ((mul_1, mul, add),)  {}
# 如果这个脚本被直接执行（而不是作为模块被导入），则执行以下代码
if __name__ == "__main__":
    # 从torch._dynamo.test_case模块中导入run_tests函数
    from torch._dynamo.test_case import run_tests

    # 运行导入的run_tests函数，通常用于执行测试用例
    run_tests()
```