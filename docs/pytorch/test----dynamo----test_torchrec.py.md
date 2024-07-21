# `.\pytorch\test\dynamo\test_torchrec.py`

```py
# 导入系统、单元测试和类型提示相关模块
import sys
import unittest
from typing import Dict, List

# 导入 PyTorch 库
import torch

# 导入 Dynamo 配置和测试相关模块
import torch._dynamo.config
import torch._dynamo.test_case

# 导入 PyTorch 神经网络模块和测试工具类
from torch import nn
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import CompileCounter

# 导入 PyTorch 内部测试工具类
from torch.testing._internal.common_utils import NoTest

# 尝试导入 torchrec 相关模块
try:
    from torchrec.datasets.random import RandomRecDataset
    from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

    # 标记 torchrec 可用
    HAS_TORCHREC = True
except ImportError:
    # 如果导入失败，则标记 torchrec 不可用
    HAS_TORCHREC = False


# 应用装饰器，用于配置 patch
@torch._dynamo.config.patch(force_unspec_int_unbacked_size_like_on_torchrec_kjt=True)
# 定义类 BucketizeMod，继承自 torch.nn.Module
class BucketizeMod(torch.nn.Module):
    def __init__(self, feature_boundaries: Dict[str, List[float]]):
        super().__init__()
        
        # 初始化存储桶权重的参数字典
        self.bucket_w = torch.nn.ParameterDict()
        self.boundaries_dict = {}

        # 遍历特征边界的字典
        for key, boundaries in feature_boundaries.items():
            # 创建并注册存储桶权重的参数
            self.bucket_w[key] = torch.nn.Parameter(
                torch.empty([len(boundaries) + 1]).fill_(1.0),
                requires_grad=True,
            )

            # 创建边界的 tensor，并注册为缓冲区
            buf = torch.tensor(boundaries, requires_grad=False)
            self.register_buffer(
                f"{key}_boundaries",
                buf,
                persistent=False,
            )
            self.boundaries_dict[key] = buf

    # 前向传播方法，接收 KeyedJaggedTensor 类型的输入，返回相同类型的输出
    def forward(self, features: "KeyedJaggedTensor") -> "KeyedJaggedTensor":
        weights_list = []

        # 遍历边界字典，对每个特征进行处理
        for key, boundaries in self.boundaries_dict.items():
            jt = features[key]
            # 将 jt 的权重分桶化
            bucketized = torch.bucketize(jt.weights(), boundaries)
            # 使用哈希函数进行处理
            hashed = bucketized  # 实际上这里的哈希函数被注释掉了
            # 根据哈希结果从权重中进行收集
            weights = torch.gather(self.bucket_w[key], dim=0, index=hashed)
            weights_list.append(weights)

        # 返回处理后的 KeyedJaggedTensor 对象
        return KeyedJaggedTensor(
            keys=features.keys(),
            values=features.values(),
            weights=torch.cat(weights_list),
            lengths=features.lengths(),
            offsets=features.offsets(),
            stride=features.stride(),
            length_per_key=features.length_per_key(),
        )


# 如果没有 torchrec 模块，则输出警告信息并设置 TestCase 为 NoTest
if not HAS_TORCHREC:
    print("torchrec not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


# 根据 torchrec 是否可用，决定是否跳过测试
@unittest.skipIf(not HAS_TORCHREC, "these tests require torchrec")
class TorchRecTests(TestCase):
    def test_pooled(self):
        tables = [
            (nn.EmbeddingBag(2000, 8), ["a0", "b0"]),
            (nn.EmbeddingBag(2000, 8), ["a1", "b1"]),
            (nn.EmbeddingBag(2000, 8), ["b2"]),
        ]

        embedding_groups = {
            "a": ["a0", "a1"],
            "b": ["b0", "b1", "b2"],
        }

        counter = CompileCounter()

        @torch.compile(backend=counter, fullgraph=True, dynamic=True)
        def f(id_list_features: KeyedJaggedTensor):
            # 将 KeyedJaggedTensor 转换为字典形式的 JaggedTensor
            id_list_jt_dict: Dict[str, JaggedTensor] = id_list_features.to_dict()
            pooled_embeddings = {}

            # 对每个表（embedding table）执行以下操作
            for emb_module, feature_names in tables:
                features_dict = id_list_jt_dict
                # 遍历每个表中的特征名
                for feature_name in feature_names:
                    f = features_dict[feature_name]
                    # 计算嵌入向量并存储到 pooled_embeddings 中
                    pooled_embeddings[feature_name] = emb_module(
                        f.values(), f.offsets()
                    )

            pooled_embeddings_by_group = {}
            # 根据 embedding_groups 中的分组信息，汇总嵌入向量
            for group_name, group_embedding_names in embedding_groups.items():
                group_embeddings = [
                    pooled_embeddings[name] for name in group_embedding_names
                ]
                # 将同一组内的嵌入向量连接起来
                pooled_embeddings_by_group[group_name] = torch.cat(
                    group_embeddings, dim=1
                )

            return pooled_embeddings_by_group

        # 创建一个 RandomRecDataset 实例
        dataset = RandomRecDataset(
            keys=["a0", "a1", "b0", "b1", "b2"],
            batch_size=4,
            hash_size=2000,
            ids_per_feature=3,
            num_dense=0,
        )
        di = iter(dataset)

        # unsync 应该正常工作

        # 获取下一个 batch 的 unsync 特征数据
        d1 = next(di).sparse_features.unsync()
        d2 = next(di).sparse_features.unsync()
        d3 = next(di).sparse_features.unsync()

        # 使用 unsync 特征数据进行函数调用
        r1 = f(d1)
        r2 = f(d2)
        r3 = f(d3)

        # 断言 CompileCounter 计数为 1
        self.assertEqual(counter.frame_count, 1)
        counter.frame_count = 0

        # sync 也应该正常工作

        # 获取下一个 batch 的 sync 特征数据
        d1 = next(di).sparse_features.sync()
        d2 = next(di).sparse_features.sync()
        d3 = next(di).sparse_features.sync()

        # 使用 sync 特征数据进行函数调用
        r1 = f(d1)
        r2 = f(d2)
        r3 = f(d3)

        # 断言 CompileCounter 计数为 1
        self.assertEqual(counter.frame_count, 1)

        # export 只能在 unsync 模式下工作

        # 通过 unsync 特征数据导出函数图模块并打印可读形式
        gm = torch._dynamo.export(f)(next(di).sparse_features.unsync()).graph_module
        gm.print_readable()

        # 断言导出的函数模块在给定数据上的输出与之前的调用结果一致
        self.assertEqual(gm(d1), r1)
        self.assertEqual(gm(d2), r2)
        self.assertEqual(gm(d3), r3)
    # 定义一个单元测试方法，用于测试 BucketizeMod 类的功能
    def test_bucketize(self):
        # 创建一个 BucketizeMod 的实例 mod，传入一个字典作为参数
        mod = BucketizeMod({"f1": [0.0, 0.5, 1.0]})
        
        # 创建一个 KeyedJaggedTensor 的实例 features，从给定的数据创建
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1"],  # 设置键值为 "f1"
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),  # 设置张量的值
            lengths=torch.tensor([2, 0, 1, 1, 1, 3]),  # 设置长度张量
            weights=torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),  # 设置权重张量
        ).unsync()  # 将创建的对象解除同步状态

        # 定义一个内部函数 f，用于处理输入 x
        def f(x):
            # 这是一种技巧，用来填充计算缓存并告知 ShapeEnv 它们都是 sizey 类型
            x.to_dict()  # 将 x 转换为字典形式
            return mod(x)  # 使用 mod 处理 x

        # 导出函数 f，并输出 ATen 图形式的输出
        torch._dynamo.export(f, aten_graph=True)(features).graph_module.print_readable()

    # 声明一个预期失败的单元测试方法
    @unittest.expectedFailure
    def test_simple(self):
        # 创建一个 KeyedJaggedTensor 的实例 jag_tensor1，同步数据
        jag_tensor1 = KeyedJaggedTensor(
            values=torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),  # 设置张量的值
            keys=["index_0", "index_1"],  # 设置键值列表
            lengths=torch.tensor([0, 0, 1, 1, 1, 3]),  # 设置长度张量
        ).sync()  # 将创建的对象同步状态

        # 通常情况下，此处会触发一次专门化
        self.assertEqual(jag_tensor1.length_per_key(), [1, 5])  # 断言长度

        # 创建一个编译计数器对象
        counter = CompileCounter()

        # 定义一个优化函数 f，接受一个 jag_tensor 作为参数
        @torch._dynamo.optimize(counter, nopython=True)
        def f(jag_tensor):
            # 这里的索引需要更多的符号推理，目前不起作用
            return jag_tensor["index_0"].values().sum()  # 返回 "index_0" 键的值的总和

        f(jag_tensor1)  # 对 jag_tensor1 调用函数 f

        # 断言编译帧数为 1
        self.assertEqual(counter.frame_count, 1)

        # 创建另一个 KeyedJaggedTensor 的实例 jag_tensor2，同步数据
        jag_tensor2 = KeyedJaggedTensor(
            values=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),  # 设置张量的值
            keys=["index_0", "index_1"],  # 设置键值列表
            lengths=torch.tensor([2, 0, 1, 1, 1, 3]),  # 设置长度张量
        ).sync()  # 将创建的对象同步状态

        f(jag_tensor2)  # 对 jag_tensor2 调用函数 f

        # 断言编译帧数为 1
        self.assertEqual(counter.frame_count, 1)
# 如果当前脚本作为主程序运行（而非被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 导入 torch._dynamo.test_case 模块中的 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行测试函数 run_tests()
    run_tests()
```