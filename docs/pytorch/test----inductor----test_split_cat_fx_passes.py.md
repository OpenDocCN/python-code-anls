# `.\pytorch\test\inductor\test_split_cat_fx_passes.py`

```
# Owner(s): ["module: inductor"]

# 导入必要的库和模块
import torch
from torch._dynamo.utils import counters, optimus_scuba_log
from torch._inductor.fx_passes.misc_patterns import numpy_compat_normalization
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.triton_utils import requires_gpu

# 定义装饰器函数 patch，用于修改给定函数 f
def patch(f):
    # 使用 torch._inductor.config.patch 装饰器配置预梯度融合选项和后梯度融合选项
    f = torch._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},  # 正常化处理的融合选项
            "remove_split_with_size_one_pass": {},  # 移除大小为一的分割的融合选项
            "merge_getitem_cat_pass": {},  # 合并 getitem 和 cat 的融合选项
            "merge_stack_tahn_unbind_pass": {},  # 合并 stack、tanh 和 unbind 的融合选项
            "merge_splits_pass": {},  # 合并分割的融合选项
            "mutate_cat_pass": {},  # 变异 cat 的融合选项
            "split_cat_pass": {},  # 分割 cat 的融合选项
            "unbind_stack_pass": {},  # 解绑 stack 的融合选项
        },
        post_grad_fusion_options={},  # 后梯度融合选项为空字典
    )(f)
    return f

# 定义测试类 TestSplitCatFxPasses，继承自 TestCase 类
class TestSplitCatFxPasses(TestCase):
    @patch  # 使用 patch 装饰器修饰测试方法
    # 定义测试方法 test_split_normalization，用于测试张量分割的规范化处理
    def test_split_normalization(self):
        
        # 定义内部函数 arg_only，接受参数 x，对其在第二维度上以 2 的步长分割并应用 relu 函数
        def arg_only(x):
            return [torch.relu(s) for s in torch.split(x, 2, 1)]

        # 定义内部函数 arg_only_dim0，接受参数 x，对其在第一维度上以 2 的步长分割并应用 relu 函数
        def arg_only_dim0(x):
            return [torch.relu(s) for s in torch.split(x, 2, 0)]

        # 定义内部函数 kwarg1，接受参数 x，使用关键字参数 dim=1，在第二维度上以 2 的步长分割并应用 relu 函数
        def kwarg1(x):
            return [torch.relu(s) for s in torch.split(x, 2, dim=1)]

        # 定义内部函数 kwarg2，接受参数 x，使用关键字参数 split_size_or_sections=2 和 dim=1，在第二维度上分割并应用 relu 函数
        def kwarg2(x):
            return [
                torch.relu(s) for s in torch.split(x, split_size_or_sections=2, dim=1)
            ]

        # 定义内部函数 kwarg3，接受参数 x，使用关键字参数 tensor=x、split_size_or_sections=2 和 dim=-1，在最后一维度上分割并应用 relu 函数
        def kwarg3(x):
            return [
                torch.relu(s)
                for s in torch.split(tensor=x, split_size_or_sections=2, dim=-1)
            ]

        # 定义内部函数 list_replace，接受参数 x，对其在第二维度上按照指定长度分割并应用 relu 函数
        def list_replace(x):
            return [torch.relu(s) for s in torch.split(x, [16, 16], dim=1)]

        # 定义内部函数 multi_split，接受参数 x，对其在第二维度上分割成多个小块并返回分割后的张量列表
        def multi_split(x):
            return [torch.split(s, 2, 1) for s in torch.split(x, 2, 1)]

        # 定义内部函数 unequal_split，接受参数 x，对其在第二维度上以 3 的步长分割并应用 relu 函数
        def unequal_split(x):
            return [torch.relu(s) for s in torch.split(x, 3, 1)]

        # 定义内部函数 arg_only_cm，接受参数 x，使用张量对象的 split 方法，在第二维度上以 2 的步长分割并应用 relu 函数
        def arg_only_cm(x):
            return [torch.relu(s) for s in x.split(2, 1)]

        # 定义内部函数 kwarg1_cm，接受参数 x，使用张量对象的 split 方法，在第二维度上以 2 的步长分割并应用 relu 函数
        def kwarg1_cm(x):
            return [torch.relu(s) for s in x.split(2, dim=1)]

        # 定义内部函数 kwarg2_cm，接受参数 x，使用张量对象的 split 方法，在第二维度上以 2 的步长分割并应用 relu 函数
        def kwarg2_cm(x):
            return [torch.relu(s) for s in x.split(split_size=2, dim=1)]

        # 定义内部函数 multi_split_cm，接受参数 x，使用张量对象的 split 方法在第二维度上分割成多个小块并返回分割后的张量列表
        def multi_split_cm(x):
            return [s.split(2, 1) for s in x.split(2, 1)]

        # 定义内部函数 unequal_split_cm，接受参数 x，使用张量对象的 split 方法，在第二维度上以 3 的步长分割并应用 relu 函数
        def unequal_split_cm(x):
            return [torch.relu(s) for s in x.split(3, 1)]

        # 定义内部函数 cm_with_list，接受参数 x，使用张量对象的 split 方法，在最后一维度上按照指定长度分割并应用 relu 函数
        def cm_with_list(x):
            return [torch.relu(s) for s in x.split([16, 16], dim=-1)]

        # 初始化参数列表 args，包含一个形状为 (2, 32) 的随机张量
        args = [
            torch.randn(2, 32),
        ]
        
        # 遍历每个测试函数和预期的规范化处理次数的元组
        for fn, expected_split_norm_count in [
            (arg_only, 1),
            (arg_only_dim0, 1),
            (kwarg1, 1),
            (kwarg2, 1),
            (kwarg3, 1),
            (list_replace, 0),
            (multi_split, 17),
            (unequal_split, 1),
            (arg_only_cm, 1),
            (kwarg1_cm, 1),
            (kwarg2_cm, 1),
            (multi_split_cm, 17),
            (unequal_split_cm, 1),
            (cm_with_list, 1),
        ]:
            # 计算预期的输出
            expected = fn(*args)
            # 调用 torch 的编译函数，计算实际输出
            actual = torch.compile(fn)(*args)

            # 使用 torch 的测试函数断言实际输出和预期输出之间的近似程度
            torch.testing.assert_close(actual, expected)
            # 使用 unittest 框架的断言，检查计数器中规范化处理通过的次数是否符合预期
            self.assertEqual(
                counters["inductor"]["normalization_pass"],
                expected_split_norm_count,
                msg=f"for {fn}",
            )
            # 如果预期规范化处理次数大于 0，则检查日志中是否包含相应的预处理信息
            if expected_split_norm_count > 0:
                self.assertIn("normalization_pass_pre_grad", optimus_scuba_log)
            # 清空计数器以便下一次迭代使用
            counters.clear()

    # 使用 patch 装饰器修饰测试方法
    @patch
    @patch
    # 使用 torch 的配置修饰器设定前向梯度融合和后向梯度融合的选项为空字典
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={},
    )
    # 定义一个测试方法，用于验证配置标志是否被遵守
    def test_config_flag_is_respected(self):
        # 定义一个内部函数，用于在最后一个维度上对输入张量进行分割
        def split_with_cat(x):
            # 在最后一个维度上，将张量 x 拆分为长度分别为 4, 4, 24 的子张量列表
            fs = torch.split(x, [4, 4, 24], dim=-1)
            # 分别获取拆分后的子张量
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            # 创建最终的子张量列表，包含 item0 和 item1
            final_items = [item0, item1]
            # 将 item2 按长度分割成子张量，并加入到 final_items 中
            final_items.extend(item2.split((4, 4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        # 定义输入参数 args 为一个张量列表
        args = [
            torch.randn(2, 32),
        ]

        # 计算预期输出
        expected = split_with_cat(*args)
        # 使用 torch.compile 对 split_with_cat 进行编译，并计算实际输出
        actual = torch.compile(split_with_cat)(*args)

        # 断言实际输出与预期输出的近似性
        torch.testing.assert_close(actual, expected)
        # 断言计数器中 "inductor" 下的 "merge_splits_pass" 等于 0
        self.assertEqual(
            counters["inductor"]["merge_splits_pass"],
            0,
        )
        # 断言计数器中 "inductor" 下的 "normalization_pass" 等于 0
        self.assertEqual(
            counters["inductor"]["normalization_pass"],
            0,
        )

    # 使用 patch 修饰符定义一个测试方法，用于验证分割、连接、合并的变异
    @patch
    def test_split_cat_merge_mutation(self):
        # 定义输入参数 args 为一个张量列表
        args = [
            torch.randn(2, 32, 32, 16),
        ]

        # 定义一个内部函数，对输入张量进行分割、复制和连接操作
        def split_cat_mutation(x):
            # 在第一个维度上将张量 x 拆分为长度为 4 的子张量列表
            splits = torch.split(x, 4, dim=1)
            # 复制第一个子张量到第二个子张量
            splits[1].copy_(splits[0])
            # 将所有子张量连接在一起
            return torch.cat(splits, dim=1)

        # 计算预期输出
        expected = split_cat_mutation(*args)
        # 使用 torch.compile 对 split_cat_mutation 进行编译，并计算实际输出
        actual = torch.compile(split_cat_mutation)(*args)

        # 断言实际输出与预期输出的近似性
        torch.testing.assert_close(actual, expected)

        # 断言计数器中 "inductor" 下的 "scmerge_split_removed" 等于 0
        self.assertEqual(counters["inductor"]["scmerge_split_removed"], 0)
        # 断言计数器中 "inductor" 下的 "scmerge_cat_removed" 等于 0
        self.assertEqual(counters["inductor"]["scmerge_cat_removed"], 0)

    # 使用多个 patch 修饰符定义一个测试方法，用于验证堆叠、tanh、解绑和合并
    @patch
    @patch
    @patch
    @patch
    def test_stack_tahn_unbind_merge(self):
        # 定义一个内部函数，对输入张量进行堆叠、tanh、解绑和连接操作
        def stack_tahn_unbind(x):
            # 在第二个维度上将张量 x 拆分为长度分别为 20, 20, 20, 10, 10, 20, 20 的子张量列表
            l1_out = torch.split(x, [20, 20, 20, 10, 10, 20, 20], 1)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            # 在第一个维度上堆叠 item0, item1, item2
            stack = torch.stack(tensors=(item0, item1, item2), dim=0)
            # 在第二个维度上将 item3 和 item4 连接在一起
            cat_1 = torch.cat((item3, item4), 1)
            # 在第二个维度上将 item5 和 item6 连接在一起
            cat_2 = torch.cat((item5, item6), 1)
            # 对堆叠结果应用 tanh 函数
            tanh = torch.tanh(stack)
            # 对 tanh 结果进行解绑操作
            unbind = torch.unbind(tanh, 0)
            # 将解绑后的结果按第二个维度连接起来，并与 cat_1 和 cat_2 连接在一起
            return torch.cat((unbind[0], unbind[1], torch.cat((cat_1, cat_2), 1)), 1)

        # 定义输入参数 args 为一个张量列表
        args = [
            torch.randn(50, 120),
        ]

        # 对于函数 stack_tahn_unbind 和预期的 merge_stack_tahn_unbind_pass 的关系进行迭代
        for fn, expected_stack_tahn_unbind_merged in [
            (stack_tahn_unbind, 1),
        ]:
            # 计算预期输出
            expected = fn(*args)
            # 使用 torch.compile 对 stack_tahn_unbind 进行编译，并计算实际输出
            actual = torch.compile(fn)(*args)

            # 断言实际输出与预期输出的近似性
            torch.testing.assert_close(actual, expected)
            # 断言计数器中 "inductor" 下的 "merge_stack_tahn_unbind_pass" 等于 expected_stack_tahn_unbind_merged
            self.assertEqual(
                counters["inductor"]["merge_stack_tahn_unbind_pass"],
                expected_stack_tahn_unbind_merged,
            )
            # 断言优化器日志中包含 "merge_getitem_cat_pass_pre_grad"
            self.assertIn("merge_getitem_cat_pass_pre_grad", optimus_scuba_log)
            # 清空计数器
            counters.clear()
    # 定义一个测试函数，用于测试 numpy 兼容性的标准化处理
    def test_numpy_compat_normalization(self):
        # 定义内部函数 fn，接受输入参数 x 和 y
        def fn(x, y):
            # 将 x 和 y 沿着 axis=1 的方向堆叠成张量 a
            a = torch.stack([x, y], axis=1)
            # 计算 x 与 x2=y 的元素级乘积，存储在张量 b 中
            b = torch.mul(x, x2=y)
            # 计算 x 与 x2=y 的元素级乘积，存储在张量 c 中（重复的注释，可能是笔误）
            c = torch.mul(x, x2=y)
            # 计算 x 与 x2=y 的元素级乘积，存储在张量 d 中（重复的注释，可能是笔误）
            d = torch.mul(x, x2=y)
            # 沿着 dim=1 的方向找到 x 中的最大值，并保持维度为 True，结果存储在元组 e 中
            e = torch.max(x, dim=1, keepdims=True)
            # 对输入张量 x 执行 dropout 操作，丢弃概率为 0.5，train=True 表示训练模式
            f = torch.dropout(x=x, p=0.5, train=True)
            # 返回计算结果张量 a, b, c, d, e, f
            return a, b, c, d, e, f

        # 使用 torch.fx.symbolic_trace 对函数 fn 进行符号化跟踪，生成 fn_t
        fn_t = torch.fx.symbolic_trace(fn)
        # 调用 numpy_compat_normalization 函数处理 fn_t 的图形
        numpy_compat_normalization(fn_t.graph)

        # 遍历 fn_t 图形中的所有节点
        for n in fn_t.graph.nodes:
            # 遍历节点 n 的关键字参数 keys
            for k in n.kwargs.keys():
                # 断言关键字参数 k 不在 {"x", "x1", "x2", "a", "axis", "keepdims"} 中
                self.assertTrue(k not in {"x", "x1", "x2", "a", "axis", "keepdims"})

    # 使用 @patch 装饰器和 @requires_gpu 装饰器标记的测试函数
    def test_stack_normalization_axis_kwarg(self):
        # 定义内部函数 fn，接受输入参数 x 和 y
        def fn(x, y):
            # 使用 torch.stack 将 x 和 y 沿着 axis=1 的方向堆叠成张量
            return torch.stack([x, y], axis=1)

        # 使用 GPU_TYPE 设备生成随机张量 x 和 y，每个张量为 4x4 大小
        x, y = (torch.rand((4, 4), device=GPU_TYPE) for _ in range(2))
        # 期望的输出结果，调用 fn 函数计算 x 和 y 的堆叠结果
        expected = fn(x, y)
        # 使用 torch.compile 编译 fn 函数，并计算 x 和 y 的实际结果
        actual = torch.compile(fn)(x, y)

        # 断言实际结果与期望结果相等
        self.assertEqual(actual, expected)
# 如果脚本正在运行在 Linux 系统并且有 GPU 可用
if __name__ == "__main__":
    # 检查操作系统是否为 Linux，并且检查系统中是否存在 GPU
    if IS_LINUX and HAS_GPU:
        # 运行测试函数
        run_tests()
```