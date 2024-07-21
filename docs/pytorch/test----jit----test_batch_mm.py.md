# `.\pytorch\test\jit\test_batch_mm.py`

```py
# Owner(s): ["oncall: jit"]

import torch  # 导入 PyTorch 库
from torch.testing import FileCheck  # 导入用于测试的文件检查工具
from torch.testing._internal.jit_utils import JitTestCase  # 导入用于 JIT 测试的基类

if __name__ == "__main__":
    # 如果此文件被直接运行，抛出运行时错误提示
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

class TestBatchMM(JitTestCase):
    @staticmethod
    def _get_test_tensors(n: int):
        # 根据给定的数量生成测试用的张量列表
        return [
            torch.tensor([[1 + x, 2 + x, 3 + x], [4 + x, 5 + x, 6 + x]])
            if x % 2 == 0
            else torch.tensor([[1 + x, 2 + x], [3 + x, 4 + x], [5 + x, 6 + x]])
            for x in range(n)
        ]

    def test_batch_mm_no_mutation(self):
        # 定义一个不允许变异的批量矩阵乘法测试函数
        def test_batch_mm(
            T1: torch.Tensor,
            T2: torch.Tensor,
            T3: torch.Tensor,
            T4: torch.Tensor,
            T5: torch.Tensor,
            T6: torch.Tensor,
            T7: torch.Tensor,
            T8: torch.Tensor,
        ):
            return (
                torch.mm(T1, T2)
                + torch.mm(T3, T4)
                + torch.mm(T5, T6)
                + torch.mm(T7, T8)
            )

        # 对测试函数进行脚本化
        test_batch_mm_scripted = torch.jit.script(test_batch_mm)

        # 获取测试用张量
        tensors = TestBatchMM._get_test_tensors(8)
        # 获取期望结果
        expected = test_batch_mm(*tensors)

        # 使用文件检查工具验证图中的 mm 操作数量是否为 4
        FileCheck().check_count("aten::mm", 4, exactly=True).run(
            test_batch_mm_scripted.graph
        )
        # 在脚本化的图上运行批量矩阵乘法优化 pass
        self.run_pass("batch_mm", test_batch_mm_scripted.graph)
        # 使用文件检查工具验证图中的 MMTreeReduce 操作数量是否为 1
        FileCheck().check_count("prim::MMTreeReduce", 1, exactly=True).run(
            test_batch_mm_scripted.graph
        )

        # 获取实际输出
        actual = test_batch_mm_scripted(*tensors)
        # 断言期望结果与实际输出是否相等，允许的数值误差为 1e-9
        self.assertEqual(expected, actual, atol=1e-9, rtol=1e-9)

    def test_batch_mm_permitted_mutation(self):
        # 定义一个允许变异的批量矩阵乘法测试函数
        def test_batch_mm(
            T1: torch.Tensor,
            T2: torch.Tensor,
            T3: torch.Tensor,
            T4: torch.Tensor,
            T5: torch.Tensor,
            T6: torch.Tensor,
            T7: torch.Tensor,
            T8: torch.Tensor,
        ):
            result = {}
            result["product"] = (
                torch.mm(T1, T2)
                + torch.mm(T3, T4)
                + torch.mm(T5, T6)
                + torch.mm(T7, T8)
            )
            result["constant"] = torch.tensor([42.0])
            return result

        # 对测试函数进行脚本化
        test_batch_mm_scripted = torch.jit.script(test_batch_mm)

        # 获取测试用张量
        tensors = TestBatchMM._get_test_tensors(8)
        # 获取期望结果
        expected = test_batch_mm(*tensors)

        # 使用文件检查工具验证图中的 mm 操作数量是否为 4
        FileCheck().check_count("aten::mm", 4, exactly=True).run(
            test_batch_mm_scripted.graph
        )
        # 在脚本化的图上运行批量矩阵乘法优化 pass
        self.run_pass("batch_mm", test_batch_mm_scripted.graph)
        # 使用文件检查工具验证图中的 MMTreeReduce 操作数量是否为 1
        FileCheck().check_count("prim::MMTreeReduce", 1, exactly=True).run(
            test_batch_mm_scripted.graph
        )

        # 获取实际输出
        actual = test_batch_mm_scripted(*tensors)
        # 断言期望结果与实际输出是否相等，允许的数值误差为 1e-9
        self.assertEqual(expected, actual, atol=1e-9, rtol=1e-9)
    def test_batch_mm_prohibited_mutation(self):
        # 定义一个通过 Torch 脚本声明的函数，用于测试批量矩阵乘法，禁止参数修改
        @torch.jit.script
        def test_batch_mm(n: int):
            # 创建多个零矩阵 T1 到 T8，每个矩阵大小为 n x n
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            # 对 T1 应用 ReLU 激活函数，原地修改
            torch.relu_(T1)
            # 计算结果，包括四个矩阵乘法的和
            result = (
                torch.mm(T1, T2)
                + torch.mm(T3, T4)
                + torch.mm(T5, T6)
                + torch.mm(T7, T8)
            )
            return result

        # 在生成的 Torch 图中检查矩阵乘法操作的数量是否为 4
        FileCheck().check_count("aten::mm", 4, exactly=True).run(test_batch_mm.graph)
        # 运行自定义的优化 pass，名称为 "batch_mm"，作用于 test_batch_mm 的图中
        self.run_pass("batch_mm", test_batch_mm.graph)
        # 检查生成的 Torch 图中是否有 4 个矩阵乘法操作，且没有使用 MMTreeReduce 操作
        FileCheck().check_count("aten::mm", 4, exactly=True).check_not(
            "prim::MMTreeReduce"
        ).run(test_batch_mm.graph)

    def test_batch_mm_prohibited_mutation_multiple_adds(self):
        # 定义另一个通过 Torch 脚本声明的函数，用于测试批量矩阵乘法，禁止参数修改，并包含多个加法操作
        @torch.jit.script
        def test_batch_mm(n: int):
            # 创建多个零矩阵 T1 到 T10，每个矩阵大小为 n x n
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            T9 = torch.zeros((n, n))
            T10 = torch.zeros((n, n))
            # 对 T1 应用 ReLU 激活函数，原地修改
            torch.relu_(T1)
            # 创建结果字典
            result = {}
            # 计算结果，分别为不可变参数的矩阵乘法和所有参数的矩阵乘法
            result["no_mutated_parameters"] = (
                torch.mm(T2, T3)
                + torch.mm(T4, T5)
                + torch.mm(T6, T7)
                + torch.mm(T8, T9)
            )
            result["all_parameters"] = (
                torch.mm(T1, T2)
                + torch.mm(T3, T4)
                + torch.mm(T5, T6)
                + torch.mm(T7, T8)
                + torch.mm(T9, T10)
            )
            return result

        # 运行自定义的优化 pass，名称为 "batch_mm"，作用于 test_batch_mm 的图中
        self.run_pass("batch_mm", test_batch_mm.graph)
        # 在生成的 Torch 图中检查是否有一个 MMTreeReduce 操作和五个矩阵乘法操作
        FileCheck().check_count("prim::MMTreeReduce", 1, exactly=True).check_count(
            "aten::mm", 5, exactly=True
        ).run(test_batch_mm.graph)
    # 定义一个测试函数，用于测试批量矩阵乘法禁止节点的变异性
    def test_batch_mm_prohibited_mutation_if_node(self):
        # 使用 Torch Script 标注的函数
        @torch.jit.script
        def test_batch_mm(n: int, use_t1: bool):
            # 初始化多个大小为 n x n 的零矩阵
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            T9 = torch.zeros((n, n))
            T10 = torch.zeros((n, n))
            # 如果 use_t1 为 True，则在 T1 上应用 relu_ 操作
            if use_t1:
                torch.relu_(T1)
                # 返回五个矩阵乘法结果的和
                return (
                    torch.mm(T1, T2)
                    + torch.mm(T3, T4)
                    + torch.mm(T5, T6)
                    + torch.mm(T7, T8)
                    + torch.mm(T9, T10)
                )
            else:
                # 返回四个矩阵乘法结果的和
                return (
                    torch.mm(T2, T3)
                    + torch.mm(T4, T5)
                    + torch.mm(T6, T7)
                    + torch.mm(T8, T9)
                )

        # 运行一个测试 pass 来验证 test_batch_mm 的图表达式
        self.run_pass("batch_mm", test_batch_mm.graph)
        # 使用 FileCheck 来验证图中 "aten::mm" 出现的次数为 5 次，且仅出现一次 "prim::MMTreeReduce"
        FileCheck().check_count("aten::mm", 5, exactly=True).check_count(
            "prim::MMTreeReduce", 1, exactly=True
        ).run(test_batch_mm.graph)

    # 定义一个测试函数，用于测试批量矩阵乘法允许边缘的变异性
    def test_batch_mm_side_permitted_mutation(self):
        # 使用 Torch Script 标注的函数
        @torch.jit.script
        def test_batch_mm(n: int):
            # 初始化一个空字典来存储结果
            result = {}
            # 初始化多个大小为 n x n 的零矩阵
            A = torch.zeros((n, n))
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            # 计算 A 与各 T 矩阵的乘积，将结果存入字典
            result["T1"] = torch.mm(A, T1)
            result["T2"] = torch.mm(A, T2)
            result["T3"] = torch.mm(A, T3)
            result["T4"] = torch.mm(A, T4)
            result["T5"] = torch.mm(A, T5)
            result["T6"] = torch.mm(A, T6)
            result["T7"] = torch.mm(A, T7)
            result["T8"] = torch.mm(A, T8)
            # 返回包含计算结果的字典
            return result

        # 使用 FileCheck 来验证图中 "aten::mm" 出现的次数为 8 次
        FileCheck().check_count("aten::mm", 8, exactly=True).run(test_batch_mm.graph)
        # 运行一个测试 pass 来验证 test_batch_mm 的图表达式，并检查 "prim::MMBatchSide" 出现一次，且没有 "aten::mm"
        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("prim::MMBatchSide", 1, exactly=True).check_not(
            "aten::mm"
        ).run(test_batch_mm.graph)
    def test_batch_mm_side_prohibited_mutation_uncommon_side(self):
        # 定义一个 Torch 脚本函数，用于测试批量矩阵乘法
        @torch.jit.script
        def test_batch_mm(n: int):
            # 创建 n x n 的全零张量 A
            A = torch.zeros((n, n))
            # 创建 T1 到 T10，都是 n x n 的全零张量
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            T9 = torch.zeros((n, n))
            T10 = torch.zeros((n, n))
            # 对 T1 应用 ReLU 激活函数（原地操作）
            torch.relu_(T1)
            # 初始化一个空字典 result
            result = {}
            # 执行 A 与 T1 到 T10 的矩阵乘法，并将结果存入 result 字典
            result["T1"] = torch.mm(A, T1)
            result["T2"] = torch.mm(A, T2)
            result["T3"] = torch.mm(A, T3)
            result["T4"] = torch.mm(A, T4)
            result["T5"] = torch.mm(A, T5)
            result["T6"] = torch.mm(A, T6)
            result["T7"] = torch.mm(A, T7)
            result["T8"] = torch.mm(A, T8)
            result["T9"] = torch.mm(A, T9)
            result["T10"] = torch.mm(A, T10)
            # 返回包含矩阵乘法结果的 result 字典
            return result

        # 检查 test_batch_mm 函数中 "aten::mm" 操作的确切调用次数为 10 次
        FileCheck().check_count("aten::mm", 10, exactly=True).run(test_batch_mm.graph)
        # 在 test_batch_mm 函数上运行自定义的批量矩阵乘法优化 pass
        self.run_pass("batch_mm", test_batch_mm.graph)

        # 再次检查 test_batch_mm 函数中 "aten::mm" 操作的确切调用次数为 1 次
        FileCheck().check_count("aten::mm", 1, exactly=True).run(test_batch_mm.graph)
        # 检查 test_batch_mm 函数中 "prim::MMBatchSide" 操作的确切调用次数为 1 次
        FileCheck().check_count("prim::MMBatchSide", 1, exactly=True).run(
            test_batch_mm.graph
        )

    def test_batch_mm_side_prohibited_mutation_common_side(self):
        # 定义另一个 Torch 脚本函数，用于测试批量矩阵乘法
        @torch.jit.script
        def test_batch_mm(n: int):
            # 创建 n x n 的全零张量 A
            A = torch.zeros((n, n))
            # 创建 T1 到 T10，都是 n x n 的全零张量
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            T9 = torch.zeros((n, n))
            T10 = torch.zeros((n, n))
            # 对 A 应用 ReLU 激活函数（原地操作）
            torch.relu_(A)
            # 初始化一个空字典 result
            result = {}
            # 执行 A 与 T1 到 T10 的矩阵乘法，并将结果存入 result 字典
            result["T1"] = torch.mm(A, T1)
            result["T2"] = torch.mm(A, T2)
            result["T3"] = torch.mm(A, T3)
            result["T4"] = torch.mm(A, T4)
            result["T5"] = torch.mm(A, T5)
            result["T6"] = torch.mm(A, T6)
            result["T7"] = torch.mm(A, T7)
            result["T8"] = torch.mm(A, T8)
            result["T9"] = torch.mm(A, T9)
            result["T10"] = torch.mm(A, T10)
            # 返回包含矩阵乘法结果的 result 字典
            return result

        # 检查 test_batch_mm 函数中 "aten::mm" 操作的确切调用次数为 10 次
        FileCheck().check_count("aten::mm", 10, exactly=True).run(test_batch_mm.graph)
        # 在 test_batch_mm 函数上运行自定义的批量矩阵乘法优化 pass
        self.run_pass("batch_mm", test_batch_mm.graph)
        # 再次检查 test_batch_mm 函数中 "aten::mm" 操作的确切调用次数为 10 次，并且不应有 "prim::MMBatchSide" 操作
        FileCheck().check_count("aten::mm", 10, exactly=True).check_not(
            "prim::MMBatchSide"
        ).run(test_batch_mm.graph)
```