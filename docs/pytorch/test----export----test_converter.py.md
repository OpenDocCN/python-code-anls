# `.\pytorch\test\export\test_converter.py`

```py
# Owner(s): ["oncall: export"]

import unittest  # 导入单元测试模块
from collections import OrderedDict  # 导入有序字典模块
from typing import Dict, List, Tuple, Union  # 导入类型提示相关模块

import torch  # 导入PyTorch库

import torch.utils._pytree as pytree  # 导入PyTorch的内部模块

from torch._dynamo.test_case import TestCase  # 导入测试用例基类
from torch._export.converter import TS2EPConverter  # 导入转换器
from torch.export import ExportedProgram  # 导入导出程序类
from torch.testing._internal.common_utils import run_tests  # 导入测试辅助函数

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")  # 装饰器，若CUDA可用则跳过测试


class TestConverter(TestCase):  # 定义测试转换器的测试类，继承自TestCase

    def _check_equal_ts_ep_converter(  # 定义检查TS和EP转换器相等性的方法
        self,
        M,  # 模型M
        inp,  # 输入
        option: Union[List[str]] = None,  # 转换选项，默认为["trace", "script"]
        check_persistent=False,  # 是否检查持久性
        lifted_tensor_constants=None,  # 提升的张量常量
    ) -> ExportedProgram:  # 返回类型为ExportedProgram的对象
        # 默认情况下，测试jit.trace和jit.script两种模式
        if option is None:
            option = ["trace", "script"]

        if check_persistent:
            num_iterations = 10  # 持久性检查时迭代次数为10
        else:
            num_iterations = 1  # 否则为1次迭代

        ep_list = []  # 初始化ExportedProgram对象列表
        for opt in option:  # 遍历选项列表
            if opt == "script":  # 如果选项是"script"
                # 分离两个模型以测试非功能效果
                if check_persistent:
                    original_ts_model = torch.jit.script(M())  # 使用torch.jit.script对模型M实例化
                    ts_model = torch.jit.script(M())  # 使用torch.jit.script对模型M实例化
                    eager_model = M()  # 获取模型M的实例
                else:
                    original_ts_model = torch.jit.script(M)  # 使用torch.jit.script对模型M进行脚本化
                    ts_model = torch.jit.script(M)  # 使用torch.jit.script对模型M进行脚本化
                    eager_model = M  # 获取模型M

            elif opt == "trace":  # 如果选项是"trace"
                if check_persistent:
                    original_ts_model = torch.jit.trace(M(), inp)  # 使用torch.jit.trace对模型M和输入inp进行追踪
                    ts_model = torch.jit.trace(M(), inp)  # 使用torch.jit.trace对模型M和输入inp进行追踪
                    eager_model = M()  # 获取模型M的实例
                else:
                    original_ts_model = torch.jit.trace(M, inp)  # 使用torch.jit.trace对模型M和输入inp进行追踪
                    ts_model = torch.jit.trace(M, inp)  # 使用torch.jit.trace对模型M和输入inp进行追踪
                    eager_model = M  # 获取模型M

            else:
                raise RuntimeError(f"Unrecognized mode for torch.jit: {opt}")  # 抛出运行时错误，不识别的torch.jit模式

            ep = TS2EPConverter(ts_model, inp).convert()  # 使用TS2EPConverter将ts_model和inp转换为ExportedProgram对象
            ep_list.append(ep)  # 将转换后的ExportedProgram对象添加到ep_list中

            for _ in range(num_iterations):  # 根据迭代次数执行以下操作
                orig_out, _ = pytree.tree_flatten(original_ts_model(*inp))  # 对原始模型的输出进行扁平化处理
                ep_out, _ = pytree.tree_flatten(ep.module()(*inp))  # 对转换后模型的输出进行扁平化处理

                # 检查模型
                if isinstance(eager_model, torch.nn.Module):  # 如果eager_model是torch.nn.Module的实例
                    expected_state_dict = OrderedDict()  # 创建有序字典对象
                    expected_state_dict.update(ts_model.state_dict())  # 更新模型ts_model的状态字典
                    if lifted_tensor_constants:
                        expected_state_dict.update(lifted_tensor_constants)  # 更新提升的张量常量
                    self.assertEqual(
                        ep.state_dict.keys(),  # 断言：验证ExportedProgram对象的状态字典键
                        expected_state_dict.keys(),  # 预期的状态字典键
                    )

                # 检查结果
                self._check_tensor_list_equal(ep_out, orig_out)  # 调用TestCase中的检查张量列表相等性的方法
        return ep_list  # 返回ExportedProgram对象列表
    # 检查两个张量列表是否相等
    def _check_tensor_list_equal(self, xs: List[torch.Tensor], ys: List[torch.Tensor]):
        # 使用断言检查两个列表的长度是否相等
        self.assertEqual(len(xs), len(ys))
        # 遍历两个张量列表中的对应元素
        for x, y in zip(xs, ys):
            # 如果 x 和 y 都是 torch.Tensor 类型
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                # 使用断言检查两个张量的形状是否相等
                self.assertEqual(x.shape, y.shape)
                # 使用断言检查两个张量是否在数值上相近
                self.assertTrue(torch.allclose(x, y))
            else:
                # 如果 x 和 y 类型不同，使用断言检查它们是否相等
                self.assertEqual(type(x), type(y))
                self.assertEqual(x, y)

    # 测试张量到输出结构的转换器是否正常工作（基本情况）
    def test_ts2ep_converter_basic(self):
        # 单个输出的模型类
        class MSingle(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        # 多个输出的模型类
        class MMulti(torch.nn.Module):
            def forward(self, x, y):
                x = x.cos() + 1
                y = y.sin() - 1
                return x, y

        # 输入张量
        inp = (torch.ones(1, 3), torch.ones(1, 3))
        # 使用自定义函数检查转换器是否将输入张量与模型输出等效地转换
        self._check_equal_ts_ep_converter(MSingle(), inp)
        self._check_equal_ts_ep_converter(MMulti(), inp)

    # 测试张量到输出结构的转换器是否正常工作（输出为容器的情况）
    def test_ts2ep_converter_container_output(self):
        # 输出为列表的模型类
        class MOutputList(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                a = x * x
                b = y + y
                return [a, b]

        # 输出为元组的模型类
        class MOutputTuple(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                a = x * x
                b = y + y
                return (a, b)

        # 输出为字典的模型类
        class MOutputDict(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                a = x * x
                b = y + y
                return {"data": {"mul": a, "add": b}}

        # 输入张量
        inp = (torch.tensor(4), torch.tensor(4))

        # 使用自定义函数检查转换器是否将输入张量与模型输出等效地转换
        # 对于输出为列表的模型类，使用脚本模式进行检查
        self._check_equal_ts_ep_converter(MOutputList(), inp, ["script"])
        # 对于输出为元组的模型类，直接检查
        self._check_equal_ts_ep_converter(MOutputTuple(), inp)
        # 对于输出为字典的模型类，使用脚本模式进行检查
        self._check_equal_ts_ep_converter(MOutputDict(), inp, ["script"])

    # 测试 torch.Tensor 的 dim 方法
    def test_aten_dim(self):
        # 获取输入张量的维度数，并返回一个维度数相同的张量
        class Module(torch.nn.Module):
            def forward(self, x):
                num_dim = x.dim()
                return torch.ones(num_dim)

        # 输入张量
        inp = (torch.ones(1, 3),)
        # 使用自定义函数检查转换器是否将输入张量与模型输出等效地转换
        self._check_equal_ts_ep_converter(Module(), inp)
    def test_aten_len(self):
        # 定义一个继承自torch.nn.Module的内部类Module，用于测试aten::len.Tensor
        class Module(torch.nn.Module):
            # 定义Module类的前向传播函数，接收一个torch.Tensor类型的参数x
            def forward(self, x: torch.Tensor):
                # 获取x的长度
                length = len(x)
                # 返回一个长度为length的全1张量
                return torch.ones(length)

        # 创建一个包含torch.ones(2, 3)的元组inp
        inp = (torch.ones(2, 3),)
        # 调用self._check_equal_ts_ep_converter方法，验证Module实例与inp是否相等
        self._check_equal_ts_ep_converter(Module(), inp)

        # 定义一个继承自torch.nn.Module的内部类Module，用于测试aten::len.t
        class Module(torch.nn.Module):
            # 定义Module类的前向传播函数，接收一个List[int]类型的参数x
            def forward(self, x: List[int]):
                # 获取x的长度
                length = len(x)
                # 返回一个长度为length的全1张量
                return torch.ones(length)

        # 创建一个包含[1, 2, 3]的列表inp
        inp = ([1, 2, 3],)
        # 调用self._check_equal_ts_ep_converter方法，验证Module实例与inp是否相等，并使用脚本模式
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        # 定义一个继承自torch.nn.Module的内部类Module，用于测试aten::len.Dict_int
        class Module(torch.nn.Module):
            # 定义Module类的前向传播函数，接收一个Dict[int, str]类型的参数x
            def forward(self, x: Dict[int, str]):
                # 获取x的长度
                length = len(x)
                # 返回一个长度为length的全1张量
                return torch.ones(length)

        # 创建一个包含{1: "a", 2: "b", 3: "c"}的字典inp
        inp = ({1: "a", 2: "b", 3: "c"},)
        # 调用self._check_equal_ts_ep_converter方法，验证Module实例与inp是否相等，并使用脚本模式
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        # 定义一个继承自torch.nn.Module的内部类Module，用于测试aten::len.Dict_bool
        class Module(torch.nn.Module):
            # 定义Module类的前向传播函数，接收一个Dict[bool, str]类型的参数x
            def forward(self, x: Dict[bool, str]):
                # 获取x的长度
                length = len(x)
                # 返回一个长度为length的全1张量
                return torch.ones(length)

        # 创建一个包含{True: "a", False: "b"}的字典inp
        inp = ({True: "a", False: "b"},)
        # 调用self._check_equal_ts_ep_converter方法，验证Module实例与inp是否相等，并使用脚本模式
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        # 定义一个继承自torch.nn.Module的内部类Module，用于测试aten::len.Dict_float
        class Module(torch.nn.Module):
            # 定义Module类的前向传播函数，接收一个Dict[float, str]类型的参数x
            def forward(self, x: Dict[float, str]):
                # 获取x的长度
                length = len(x)
                # 返回一个长度为length的全1张量
                return torch.ones(length)

        # 创建一个包含{1.2: "a", 3.4: "b"}的字典inp
        inp = ({1.2: "a", 3.4: "b"},)
        # 调用self._check_equal_ts_ep_converter方法，验证Module实例与inp是否相等，并使用脚本模式
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        # 定义一个继承自torch.nn.Module的内部类Module，用于测试aten::len.Dict_Tensor
        class Module(torch.nn.Module):
            # 定义Module类的前向传播函数，接收一个Dict[torch.Tensor, str]类型的参数x
            def forward(self, x: Dict[torch.Tensor, str]):
                # 获取x的长度
                length = len(x)
                # 返回一个长度为length的全1张量
                return torch.ones(length)

        # 创建一个包含{torch.zeros(2, 3): "a", torch.ones(2, 3): "b"}的字典inp
        inp = ({torch.zeros(2, 3): "a", torch.ones(2, 3): "b"},)
        # 调用self._check_equal_ts_ep_converter方法，验证Module实例与inp是否相等，并使用脚本模式
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        # aten::len.str 和 aten::len.Dict_str 不受支持
        # 因为torch._C._jit_flatten不支持str类型
        # inp = ("abcdefg",)
        # self._check_equal_ts_ep_converter(Module(), inp)
        # inp = ({"a": 1, "b": 2},)
        # self._check_equal_ts_ep_converter(Module(), inp)
    # 定义一个名为 test_prim_min 的测试方法
    def test_prim_min(self):
        # 定义一个内嵌的 Module 类，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 定义 forward 方法，接受两个参数 x 和 y，返回一个 torch.Tensor
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # 计算输入张量 x 和 y 的长度
                x_len = len(x)
                y_len = len(y)

                # 使用 prim::min.int 操作，计算 x_len 和 y_len 的最小值
                len_int = min(x_len, y_len)

                # 使用 prim::min.float 操作，计算 x_len * 2.0 和 y_len * 2.0 的最小值，并转换为整数
                len_float = int(min(x_len * 2.0, y_len * 2.0))

                # 使用 prim::min.self_int 操作，计算列表 [x_len, y_len] 的最小值
                len_self_int = min([x_len, y_len])

                # 使用 prim::min.self_float 操作，计算列表 [x_len * 2.0, y_len * 2.0] 的最小值，并转换为整数
                len_self_float = int(min([x_len * 2.0, y_len * 2.0]))

                # 使用 prim::min.float_int 操作，计算 x_len * 2.0 和 y_len 的最小值，并转换为整数
                len_float_int = int(min(x_len * 2.0, y_len))

                # 使用 prim::min.int_float 操作，计算 x_len 和 y_len * 2.0 的最小值，并转换为整数
                len_int_float = int(min(x_len, y_len * 2.0))

                # 返回一个张量，该张量包含所有计算出的长度的和
                return torch.ones(
                    len_int
                    + len_float
                    + len_self_int
                    + len_self_float
                    + len_float_int
                    + len_int_float
                )

        # 定义输入 inp，包含一个随机张量和一个随机张量的元组
        inp = (torch.randn(10, 2), torch.randn(5))
        # 调用 _check_equal_ts_ep_converter 方法，将 Module 实例和输入作为参数传入
        self._check_equal_ts_ep_converter(Module(), inp)

    # 定义一个名为 test_aten___getitem___list 的测试方法
    def test_aten___getitem___list(self):
        # 定义一个内嵌的 Module 类，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 定义 forward 方法，接受一个参数 x
            def forward(self, x):
                # 使用 torch.split 将输入张量 x 按照长度为 2 进行分割，返回一个列表
                y = torch.split(x, 2)
                # 返回列表 y 的第一个元素
                return y[0]

        # 定义输入 inp，包含一个形状为 (3, 2) 的随机张量的元组
        inp = (torch.rand((3, 2)),)
        # 调用 _check_equal_ts_ep_converter 方法，将 Module 实例和输入作为参数传入
        self._check_equal_ts_ep_converter(Module(), inp)

    # 定义一个名为 test_aten___getitem___dict 的测试方法
    def test_aten___getitem___dict(self):
        # 定义一个内嵌的 Module 类，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 定义 forward 方法，接受一个参数 x
            def forward(self, x):
                # 使用 torch.split 将输入张量 x 按照长度为 2 进行分割，返回一个列表
                y = torch.split(x, 2)
                # 构建四个字典，每个字典的值分别为列表 y 的不同元素
                d_int = {0: y[0], 1: y[1]}
                d_str = {"0": y[0], "1": y[1]}
                d_bool = {True: y[0], False: y[1]}
                d_float = {0.1: y[0], 2.3: y[1]}
                # 返回四个字典中的不同键对应的值
                return d_int[0], d_str["0"], d_bool[True], d_float[0.1]

        # 定义输入 inp，包含一个形状为 (3, 2) 的随机张量的元组
        inp = (torch.rand((3, 2)),)
        # 调用 _check_equal_ts_ep_converter 方法，将 Module 实例和输入作为参数传入
        self._check_equal_ts_ep_converter(Module(), inp)

    # 定义一个名为 test_prim_device 的测试方法
    def test_prim_device(self):
        # 定义一个内嵌的 Module 类，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 定义 forward 方法，接受一个参数 x
            def forward(self, x):
                # 获取输入张量 x 的设备信息
                device = x.device
                # 返回一个在指定设备上创建的形状为 (2, 3) 的全 1 张量
                return torch.ones(2, 3, device=device)

        # 定义输入 inp，包含一个形状为 (3, 4) 的随机张量的元组
        inp = (torch.rand(3, 4),)
        # 调用 _check_equal_ts_ep_converter 方法，将 Module 实例和输入作为参数传入
        self._check_equal_ts_ep_converter(Module(), inp)

    # 定义一个名为 test_prim_device_cuda 的测试方法，使用 @requires_cuda 装饰器
    @requires_cuda
    def test_prim_device_cuda(self):
        # 定义一个内嵌的 Module 类，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 定义 forward 方法，接受一个参数 x
            def forward(self, x):
                # 获取输入张量 x 的设备信息
                device = x.device
                # 返回一个在指定 CUDA 设备上创建的形状为 (2, 3) 的全 1 张量
                return torch.ones(2, 3, device=device)

        # 定义输入 inp，包含一个形状为 (3, 4) 的随机张量，并指定为 CUDA 设备 "cuda:0"
        inp = (torch.rand((3, 4), device="cuda:0"),)
        # 调用 _check_equal_ts_ep_converter 方法，将 Module 实例和输入作为参数传入
        self._check_equal_ts_ep_converter(Module(), inp)
    # 定义一个名为 test_prim_dtype 的测试方法
    def test_prim_dtype(self):
        # 定义一个内部类 Module，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 实现 Module 类的前向传播方法
            def forward(self, x):
                # 获取输入张量 x 的数据类型
                dtype = x.dtype
                # 返回一个形状为 (2, 3) 的张量，数据类型与输入张量 x 一致
                return torch.ones(2, 3, dtype=dtype)

        # 针对不同的数据类型 dtype 进行测试
        for dtype in [
            torch.float32,
            torch.double,
        ]:
            # 创建一个输入元组，包含一个随机张量，数据类型为当前循环中的 dtype
            inp = (torch.rand((3, 4), dtype=dtype),)
            # 调用 _check_equal_ts_ep_converter 方法，验证 Module 实例在输入 inp 上的输出是否符合预期
            self._check_equal_ts_ep_converter(Module(), inp)

        # 针对另一组数据类型 dtype 进行测试
        for dtype in [
            torch.uint8,
            torch.int8,
            torch.int32,
        ]:
            # 创建一个输入元组，包含一个在指定数据类型下的随机整数张量
            inp = (torch.randint(high=128, size=(3, 4), dtype=dtype),)
            # 调用 _check_equal_ts_ep_converter 方法，验证 Module 实例在输入 inp 上的输出是否符合预期
            self._check_equal_ts_ep_converter(Module(), inp)

    # 定义一个名为 test_convert_if_basic 的测试方法
    def test_convert_if_basic(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 实现 M 类的前向传播方法，接受两个输入张量 x 和 y
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # 如果 x 为真值
                if x:
                    # 返回 y 乘以 y 的结果
                    return y * y
                else:
                    # 返回 y 加上 y 的结果
                    return y + y

        # 创建一个输入元组，包含一个布尔张量和一个标量张量
        inp = (torch.tensor(True), torch.tensor(4))
        # 调用 _check_equal_ts_ep_converter 方法，验证 M 实例在输入 inp 上的输出是否符合预期
        ep_list = self._check_equal_ts_ep_converter(M(), inp)

        # 遍历除第一个外的 ep_list 元素
        for ep in ep_list[1:]:
            # 断言通过，验证 M 类在给定输入 (False, 4) 下的输出与预期结果相等
            torch.testing.assert_close(
                ep.module()(torch.tensor(False), torch.tensor(4)),
                M()(torch.tensor(False), torch.tensor(4)),
            )

    # 定义一个名为 test_convert_if_multiple_out 的测试方法
    def test_convert_if_multiple_out(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 定义一个返回两个张量元组的方法 true_fn
            def true_fn(self, y, z):
                return (z * z, z + z)

            # 定义一个返回两个张量元组的方法 false_fn
            def false_fn(self, y, z):
                return (y * y * y, y + y)

            # 实现 M 类的前向传播方法，接受两个输入张量 x 和 y
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # 计算 y 的平方赋值给 z
                z = y * y

                # 如果 x 为真值
                if x:
                    # 调用 true_fn 方法处理输入 y 和 z，得到结果元组 res
                    res = self.true_fn(y, z)
                else:
                    # 调用 false_fn 方法处理输入 y 和 z，得到结果元组 res
                    res = self.false_fn(y, z)

                # 返回结果元组 res 的两个张量元素相加的结果
                return res[0] + res[1]

        # 创建一个输入元组，包含一个布尔张量和一个标量张量
        inp = (torch.tensor(True), torch.tensor(4))
        # 调用 _check_equal_ts_ep_converter 方法，验证 M 实例在输入 inp 上的输出是否符合预期
        ep_list = self._check_equal_ts_ep_converter(M(), inp)

        # 遍历除第一个外的 ep_list 元素
        for ep in ep_list[1:]:
            # 断言通过，验证 M 类在给定输入 (False, 4) 下的输出与预期结果相等
            torch.testing.assert_close(
                ep.module()(torch.tensor(False), torch.tensor(4)),
                M()(torch.tensor(False), torch.tensor(4)),
            )

    # 定义一个名为 test_profiler__record_function 的测试方法
    def test_profiler__record_function(self):
        # 定义一个内部类 Module，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 实现 Module 类的前向传播方法，接受一个张量输入 x，并返回一个张量
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 记录进入名为 "foo" 的记录函数，获取记录句柄 handle
                handle = torch.ops.profiler._record_function_enter_new("foo", None)
                # 计算输入张量 x 的每个元素乘以 2，再加上 4，结果赋值给 y
                y = x * 2 + 4
                # 根据记录句柄 handle 结束记录函数
                torch.ops.profiler._record_function_exit(handle)
                # 返回计算结果张量 y
                return y

        # 创建一个形状为 (10, 10) 的随机张量 x
        x = torch.randn(10, 10)
        # 调用 _check_equal_ts_ep_converter 方法，验证 Module 实例在输入 x 上的输出是否符合预期
        self._check_equal_ts_ep_converter(Module(), (x,))

    # 定义一个名为 test_aten_floordiv 的测试方法
    def test_aten_floordiv(self):
        # 定义一个内部类 Module，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 实现 Module 类的前向传播方法，接受一个张量输入 x，并返回一个张量
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 对输入张量 x 中的每个元素执行整数除以 2 的操作，返回结果张量
                return x // 2

        # 创建一个形状为 (10, 10) 的随机张量 x
        x = torch.randn(10, 10)
        # 调用 _check_equal_ts_ep_converter 方法，验证 Module 实例在输入 x 上的输出是否符合预期
        self._check_equal_ts_ep_converter(Module(), (x,))
    # 定义一个测试方法，用于测试 torch 模块中的 aten.__is__ 功能
    def test_aten___is__(self):
        # 定义一个继承自 torch.nn.Module 的内部类 Module
        class Module(torch.nn.Module):
            # 重写 forward 方法，接受两个参数 x 和 y，返回一个元组 (bool, torch.Tensor)
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> Tuple[bool, torch.Tensor]:
                # 对 x 执行加法操作
                z = x + 1
                # 返回 x 是否和 y 相同的布尔值和计算后的 z
                return x is y, z

        # 创建输入数据 inp，包含两个随机生成的 Tensor
        inp = (torch.randn(10, 10), torch.rand(10, 10))
        # 调用 _check_equal_ts_ep_converter 方法，将 Module 实例和 inp 作为参数，并指定转换类型为 "script"
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    # 定义一个测试方法，用于测试 torch 模块中的 aten.__isnot__ 功能
    def test_aten___isnot__(self):
        # 定义一个继承自 torch.nn.Module 的内部类 Module
        class Module(torch.nn.Module):
            # 重写 forward 方法，接受两个参数 x 和 y，返回一个元组 (bool, torch.Tensor)
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> Tuple[bool, torch.Tensor]:
                # 对 x 执行加法操作
                z = x + 1
                # 返回 x 是否不等于 y 的布尔值和计算后的 z
                return x is not y, z

        # 创建输入数据 inp，包含两个随机生成的 Tensor
        inp = (torch.randn(10, 10), torch.rand(10, 10))
        # 调用 _check_equal_ts_ep_converter 方法，将 Module 实例和 inp 作为参数，并指定转换类型为 "script"
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    # 定义一个测试方法，用于测试 torch 模块中的 aten.__not__ 功能
    def test_aten___not__(self):
        # 定义一个继承自 torch.nn.Module 的内部类 Module
        class Module(torch.nn.Module):
            # 重写 forward 方法，接受两个参数 x 和 y，返回一个元组 (bool, torch.Tensor)
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> Tuple[bool, torch.Tensor]:
                # 对 x 执行加法操作
                z = x + 1
                # 返回 x 不是不等于 y 的布尔值（即 x 是否等于 y）和计算后的 z
                return not (x is not y), z

        # 创建输入数据 inp，包含两个随机生成的 Tensor
        inp = (torch.randn(10, 10), torch.rand(10, 10))
        # 调用 _check_equal_ts_ep_converter 方法，将 Module 实例和 inp 作为参数，并指定转换类型为 "script"
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    # 定义一个测试方法，用于测试 ts2ep_converter 方法的解包功能
    def test_ts2ep_converter_unpack(self):
        # 定义一个继承自 torch.nn.Module 的内部类 MUnpackList
        class MUnpackList(torch.nn.Module):
            # 重写 forward 方法，接受一个参数 x
            def forward(self, x):
                # 使用 torch.split 方法将 x 分割为两部分 x 和 y，并执行加法操作后返回
                x, y = torch.split(x, 2)
                return x + y

        # 定义一个继承自 torch.nn.Module 的内部类 MUnpackTuple
        class MUnpackTuple(torch.nn.Module):
            # 重写 forward 方法，接受一个元组参数 x_tuple，包含两个 Tensor
            def forward(self, x_tuple: Tuple[torch.Tensor, torch.Tensor]):
                # 将元组解包为两个 Tensor x 和 y，对 x 执行余弦函数操作后返回与 y 的加法结果
                x, y = x_tuple
                x = x.cos()
                return x + y

        # 创建输入数据 inp，包含一个长度为 4 的 Tensor
        inp = (torch.ones(4),)
        # 调用 _check_equal_ts_ep_converter 方法，将 MUnpackList 实例和 inp 作为参数
        self._check_equal_ts_ep_converter(MUnpackList(), inp)
        
        # 创建输入数据 inp，包含一个元组，元组内有两个形状为 (1, 4) 的 Tensor
        inp = ((torch.zeros(1, 4), torch.ones(1, 4)),)
        # 调用 _check_equal_ts_ep_converter 方法，将 MUnpackTuple 实例和 inp 作为参数
        self._check_equal_ts_ep_converter(MUnpackTuple(), inp)
    # 定义一个测试函数，用于测试带有嵌套参数的 PyTorch 神经网络模块转换
    def test_convert_nn_module_with_nested_param(self):
        # 定义一个简单的神经网络模块 M，包含一个线性层
        class M(torch.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)

            # 前向传播函数，将输入 x 通过线性层处理并返回结果
            def forward(self, x: torch.Tensor):
                return self.linear(x)

        # 定义一个嵌套的神经网络模块 NestedM，包含一个线性层和一个 M 类的实例
        class NestedM(torch.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)
                self.m = M(dim)

            # 前向传播函数，将输入 x 先通过嵌套的 M 模块处理，再通过线性层处理并返回结果
            def forward(self, x: torch.Tensor):
                return self.linear(self.m(x))

        # 定义一个超级嵌套的神经网络模块 SuperNestedM，包含一个线性层和一个 NestedM 类的实例
        class SuperNestedM(torch.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)
                self.m = NestedM(dim)

            # 前向传播函数，将输入 x 先通过超级嵌套的 NestedM 模块处理，再通过线性层处理并返回结果
            def forward(self, x: torch.Tensor):
                return self.linear(self.m(x))

        # 输入示例，创建包含三个元素的元组
        inp = (torch.ones(3),)
        # 创建一个 NestedM 模块的实例 orig_m，并调用自定义的转换器函数进行检查
        orig_m = NestedM(3)
        self._check_equal_ts_ep_converter(orig_m, inp)

        # 创建一个 SuperNestedM 模块的实例 orig_m，并调用自定义的转换器函数进行检查
        orig_m = SuperNestedM(3)
        self._check_equal_ts_ep_converter(orig_m, inp)

    # 定义一个测试函数，用于测试带有嵌套缓冲区的 PyTorch 神经网络模块转换
    def test_convert_nn_module_with_nested_buffer(self):
        # 定义一个包含缓冲区的神经网络模块 M
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 注册一个名为 "w" 的缓冲区，内容为随机生成的张量
                self.register_buffer("w", torch.randn(1))

            # 前向传播函数，返回输入 x 加上缓冲区 "w" 的结果
            def forward(self, x: torch.Tensor):
                return self.w + x

        # 定义一个嵌套的神经网络模块 NestedM，包含一个 M 类的实例和一个注册的缓冲区 "w"
        class NestedM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m = M()
                # 注册一个名为 "w" 的缓冲区，内容为随机生成的张量
                self.register_buffer("w", torch.randn(1))

            # 前向传播函数，返回输入 x 加上嵌套模块 M 处理后的结果和注册的缓冲区 "w" 的结果
            def forward(self, x: torch.Tensor):
                return self.w + self.m(x)

        # 定义一个超级嵌套的神经网络模块 SuperNestedM，包含一个 NestedM 类的实例和一个注册的缓冲区 "w"
        class SuperNestedM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m = NestedM()
                # 注册一个名为 "w" 的缓冲区，内容为随机生成的张量
                self.register_buffer("w", torch.randn(1))

            # 前向传播函数，返回输入 x 加上超级嵌套模块 NestedM 处理后的结果和注册的缓冲区 "w" 的结果
            def forward(self, x: torch.Tensor):
                return self.w + self.m(x)

        # 输入示例，创建包含一个元素的元组
        inp = (torch.ones(1),)
        # 创建一个 NestedM 模块的实例 orig_m，并调用自定义的转换器函数进行检查
        orig_m = NestedM()
        self._check_equal_ts_ep_converter(orig_m, inp)
        # 创建一个 SuperNestedM 模块的实例 orig_m，并调用自定义的转换器函数进行检查
        orig_m = SuperNestedM()
        self._check_equal_ts_ep_converter(orig_m, inp)
    def test_convert_nn_module_with_nested_if_and_buffer(self):
        # 定义一个测试函数，用于测试具有嵌套条件和缓冲区的神经网络模块转换

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 在模块中注册一个缓冲区 "w"，并用随机数初始化
                self.register_buffer("w", torch.randn(1))
                self.count = 1  # 设置一个计数器变量

            def forward(self, x: torch.Tensor):
                # 前向传播函数，返回输入 x、缓冲区 "w" 和计数器的和
                return self.w + x + self.count

        class NestedM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 实例化两个 M 类的对象作为嵌套模块的成员
                self.m1 = M()
                self.m2 = M()
                self.register_buffer("w", torch.randn(1))  # 注册一个名为 "w" 的缓冲区

            def forward(self, x: torch.Tensor):
                if torch.sum(x) > 1:
                    # 如果输入 x 的元素之和大于 1，则返回 "w" 加上 m1(x) 的结果
                    return self.w + self.m1(x)
                else:
                    # 否则返回 "w" 加上 m2(x) 的结果
                    return self.w + self.m2(x)

        # 超级嵌套模块，需要多次提升参数
        class SuperNestedM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 实例化两个 NestedM 类的对象作为超级嵌套模块的成员
                self.m1 = NestedM()
                self.m2 = NestedM()
                self.register_buffer("w", torch.randn(1))  # 注册一个名为 "w" 的缓冲区

            def forward(self, x: torch.Tensor):
                if torch.max(x) > 1:
                    # 如果输入 x 的最大值大于 1，则返回 "w" 加上 m1(x) 的结果
                    return self.w + self.m1(x)
                else:
                    # 否则返回 "w" 加上 m2(x) 的结果
                    return self.w + self.m2(x)

        # 超级嵌套模块测试
        inp = (torch.ones(1),)  # 输入为一个元素全为 1 的元组
        orig_m = SuperNestedM()  # 实例化一个超级嵌套模块对象
        # TODO: fix trace: state_dict is not equal.
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp, ["script"])

        t = inp[0]
        t -= 1
        for ep in ep_list:
            # 使用 torch.testing.assert_close 检查预期值和转换后的结果是否接近
            torch.testing.assert_close(
                ep.module()(*inp),
                orig_m(*inp),
            )

    def test_ts2ep_converter_contains(self):
        class MIn(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                # 返回 x 的数据类型是否在 [torch.float32, torch.float64] 中
                return x.dtype in [torch.float32, torch.float64]

        class MNotIn(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                # 返回 x 的数据类型是否在 [torch.int8] 中
                return x.dtype in [torch.int8]

        class MTensorIn(torch.nn.Module):
            def forward(self, x: torch.Tensor, x_dict: Dict[torch.Tensor, str]):
                # 返回 x 是否在 x_dict 的键中
                return x in x_dict

        # 跟踪函数必须返回具有张量的输出
        inp = (torch.tensor(4),)  # 输入为一个包含整数 4 的元组
        self._check_equal_ts_ep_converter(MIn(), inp, ["script"])  # 检查模型 MIn 的转换结果

        inp = (torch.tensor(4),)  # 输入为一个包含整数 4 的元组
        self._check_equal_ts_ep_converter(MNotIn(), inp, ["script"])  # 检查模型 MNotIn 的转换结果

        # TODO: update test to use reference for in.
        inp = (torch.tensor(4), {torch.tensor(4): "foo"})  # 输入为一个包含整数 4 和字典 {tensor: 'foo'} 的元组
        self._check_equal_ts_ep_converter(MTensorIn(), inp, ["script"])  # 检查模型 MTensorIn 的转换结果

        inp = (torch.tensor(1), {torch.tensor(4): "foo"})  # 输入为一个包含整数 1 和字典 {tensor: 'foo'} 的元组
        self._check_equal_ts_ep_converter(MTensorIn(), inp, ["script"])  # 再次检查模型 MTensorIn 的转换结果
    # 定义一个测试函数，用于测试自定义操作的转换器
    def test_ts2ep_converter_custom_op(self):
        # 使用 torch.library._scoped_library 函数进入名为 "mylib" 的库的 "FRAGMENT" 作用域
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            # 设置捕获标量输出配置为 True
            torch._dynamo.config.capture_scalar_outputs = True
            # 设置捕获动态输出形状操作配置为 True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

            # 定义名为 "mylib::foo" 的自定义操作，接受一个 Tensor x 并返回一个 Tensor
            torch.library.define(
                "mylib::foo",
                "(Tensor x) -> Tensor",
                lib=lib,
            )

            # 实现名为 "mylib::foo" 的自定义操作，使用 "CompositeExplicitAutograd" 方式
            @torch.library.impl(
                "mylib::foo",
                "CompositeExplicitAutograd",
                lib=lib,
            )
            def foo_impl(x):
                return x + x

            # 定义名为 "mylib::foo" 的自定义操作的元函数
            @torch.library.impl_abstract(
                "mylib::foo",
                lib=lib,
            )
            def foo_meta(x):
                return x + x

            # 定义一个继承自 torch.nn.Module 的类 M
            class M(torch.nn.Module):
                # 实现该类的前向传播方法
                def forward(self, x):
                    return torch.ops.mylib.foo(x)

            # 创建一个输入元组 inp，包含一个形状为 (3, 3) 的随机 Tensor
            inp = (torch.randn(3, 3),)
            # 实例化类 M
            m = M()
            # 调用测试方法 _check_equal_ts_ep_converter，验证模型 m 对输入 inp 的转换结果是否正确
            self._check_equal_ts_ep_converter(m, inp)

    # 定义一个测试函数，用于测试无参数的函数转换器
    def test_convert_func_without_param(self):
        # 定义一个接受两个参数 x 和 y 并返回它们的和的函数 func1
        def func1(x, y):
            return x + y

        # 定义一个接受两个参数 x 和 y 并根据 x 的和是否大于 0 返回不同结果的函数 func2
        def func2(x, y):
            if x.sum() > 0:
                return x + y
            else:
                return x - y

        # 创建一个输入元组 inp，包含两个 Tensor，值均为 1
        inp = (
            torch.tensor(1),
            torch.tensor(1),
        )
        # 调用测试方法 _check_equal_ts_ep_converter，验证函数 func1 对输入 inp 的转换结果是否正确
        self._check_equal_ts_ep_converter(func1, inp)

        # 调用测试方法 _check_equal_ts_ep_converter，验证函数 func2 对输入 inp 的转换结果是否正确，并返回结果列表 ep_list
        ep_list = self._check_equal_ts_ep_converter(func2, inp)

        # 取出输入元组中的第一个 Tensor，并将其值减去 1
        t = inp[0]
        t -= 1
        # 遍历 ep_list 中除第一个元素外的每个元素 ep，使用 torch.testing.assert_close 检查 ep 执行 (*inp) 的结果是否与 func2(*inp) 接近
        for ep in ep_list[1:]:
            torch.testing.assert_close(
                ep.module()(*inp),
                func2(*inp),
            )
    # 定义测试函数 test_implicit_constant_to_tensor_handling
    def test_implicit_constant_to_tensor_handling(self):
        # 定义函数 func1，对输入 x 加 2 后返回结果
        def func1(x):
            return x + 2

        # 定义函数 func2，对输入 x, y 执行复杂的数学运算并返回结果
        def func2(x, y):
            return x * y / (x - 2 * y) + y

        # 定义函数 func3，将输入 x 增加一个值为 3 的张量并返回结果
        def func3(x):
            return x + torch.tensor([3])

        # 定义函数 func4，创建一个值为正无穷大的张量，并用该值填充一个 10x10 的张量
        def func4():
            val = torch.tensor(float("inf"))
            return torch.full((10, 10), val)

        # 定义函数 func5，返回一个乘以 -1 的全 1 张量和一个全 0 张量
        def func5():
            x = -1
            return x * torch.ones(1, dtype=torch.float), torch.zeros(
                1, dtype=torch.float
            )

        # 定义函数 func6，返回多个张量的元素数量和尺寸，并确保下游操作仍然有效
        def func6(x1, x2, x3, x4):
            return (
                x1.numel(),
                x1.size(),
                x2.numel(),
                x2.size(),
                x3.numel(),
                x3.size(),
                x4.numel(),
                x4.size(),
                torch.ones(x1.numel()),  # 仅确保下游操作仍然有效
                torch.ones(x1.size()),  # 仅确保下游操作仍然有效
            )

        # 定义类 M1，继承自 torch.nn.Module，初始化时使用给定值创建张量 self.x
        class M1(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.x = torch.tensor(value)

            # 定义前向传播函数，返回克隆后的张量 self.x
            def forward(self):
                return self.x.clone()

        # 定义类 M2，继承自 torch.nn.Module，前向传播函数返回输入张量加上常量 4
        class M2(torch.nn.Module):
            def forward(self, x):
                return torch.tensor(4) + x

        # 准备输入 inp 为一个包含随机张量的元组
        inp = (torch.randn([2, 2]),)
        # 调用 _check_equal_ts_ep_converter 方法，比较 func1 在 inp 上的运行结果
        self._check_equal_ts_ep_converter(func1, inp)
        
        # 准备输入 inp 为包含两个随机张量的元组
        inp = (torch.randn([2, 2]), torch.randn([2, 2]))
        # 调用 _check_equal_ts_ep_converter 方法，比较 func2 在 inp 上的运行结果
        self._check_equal_ts_ep_converter(func2, inp)

        # 准备输入 inp 为一个包含随机张量的元组
        inp = (torch.randn([2, 2]),)
        # 调用 _check_equal_ts_ep_converter 方法，比较 func3 在 inp 上的运行结果
        self._check_equal_ts_ep_converter(func3, inp)

        # 调用 _check_equal_ts_ep_converter 方法，比较 func4 在空元组上的运行结果
        self._check_equal_ts_ep_converter(func4, ())

        # 调用 _check_equal_ts_ep_converter 方法，比较 M1 类在空元组上的运行结果
        self._check_equal_ts_ep_converter(M1(5), ())

        # 准备输入 inp 为一个随机张量的元组
        inp = (torch.randn(2),)
        # 调用 _check_equal_ts_ep_converter 方法，比较 M2 类在 inp 上的运行结果
        self._check_equal_ts_ep_converter(M2(), inp)

        # 调用 _check_equal_ts_ep_converter 方法，比较 func5 在空元组上的运行结果
        self._check_equal_ts_ep_converter(func5, ())

        # 准备输入 inp 为包含四个随机张量的元组，每个张量的数据类型不同
        inp = (
            torch.randn([2, 3, 4]).to(torch.int8),
            torch.randn([2, 3, 4]).to(torch.int32),
            torch.randn([2, 3, 4]).to(torch.float32),
            torch.randn([2, 3, 4]).to(torch.float64),
        )
        # 调用 _check_equal_ts_ep_converter 方法，比较 func6 在 inp 上的运行结果
        self._check_equal_ts_ep_converter(func6, inp)

    # 定义测试函数 test_prim_tolist
    def test_prim_tolist(self):
        # 定义类 Module，继承自 torch.nn.Module，前向传播函数接收一个张量 x，返回其转换为列表的结果
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> List[int]:
                return x.tolist()

        # 准备输入 inp 为一个包含整数张量的元组
        inp = (torch.tensor([1, 2, 3]),)
        # 调用 _check_equal_ts_ep_converter 方法，比较 Module 类在 inp 上的运行结果，并指定使用脚本模式
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        # 重新定义类 Module，继承自 torch.nn.Module，前向传播函数接收一个张量 x，返回其转换为嵌套列表的结果
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> List[List[int]]:
                return x.tolist()

        # 准备输入 inp 为一个包含嵌套整数张量的元组
        inp = (torch.tensor([[1, 2, 3], [4, 5, 6]]),)
        # 调用 _check_equal_ts_ep_converter 方法，比较 Module 类在 inp 上的运行结果，并指定使用脚本模式
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])
    def test_get_tensor_constants(self):
        # 定义一个测试方法，用于验证获取张量常量的情况

        # 定义一个名为 Foo 的神经网络模块类
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个形状为 (3, 2) 的随机张量，并赋给 self.data
                self.data = torch.randn(3, 2)

            # 前向传播方法，接受一个 torch.Tensor 类型的输入 x，并返回一个 torch.Tensor 类型的输出
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 返回输入 x 与 self.data 相加的结果
                return x + self.data

        # 定义一个名为 Goo 的神经网络模块类
        class Goo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个形状为 (3, 2) 的随机张量，并赋给 self.data
                self.data = torch.randn(3, 2)
                # 初始化一个 Foo 类的实例，并赋给 self.foo
                self.foo = Foo()

            # 前向传播方法，接受一个 torch.Tensor 类型的输入 x，并返回一个 torch.Tensor 类型的输出
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 返回输入 x、self.data、self.foo.data 和 self.foo(x) 四者相加的结果
                return x + self.data + self.foo.data + self.foo(x)

        # 构造一个输入元组 inp，包含一个形状为 (3, 2) 的随机张量
        inp = (torch.randn(3, 2),)
        # 创建一个 Goo 类的实例，并赋给变量 goo
        goo = Goo()
        # 调用 _check_equal_ts_ep_converter 方法进行 goo 实例与 inp 的相等性检查
        self._check_equal_ts_ep_converter(goo, inp)
# 如果当前脚本被直接执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数来运行测试用例
    run_tests()
```