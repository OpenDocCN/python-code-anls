# `.\pytorch\test\test_pruning_op.py`

```
# 导入所需的模块和函数
import hypothesis.strategies as st  # 导入Hypothesis库中的策略模块
from hypothesis import given  # 导入Hypothesis库中的给定装饰器
import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfTorchDynamo  # 导入PyTorch测试相关的工具和函数
import torch.testing._internal.hypothesis_utils as hu  # 导入PyTorch内部的Hypothesis测试工具
hu.assert_deadline_disabled()  # 禁用测试运行时限

# 定义一个测试类，继承自PyTorch的TestCase
class PruningOpTest(TestCase):

    # 生成基于指示器和阈值的逐行掩码向量
    # 指示器是一个包含每行权重值的向量，表示行的重要性
    # 如果行的指示器值小于阈值，则掩盖该行
    def _generate_rowwise_mask(self, embedding_rows):
        # 从随机数生成一个NumPy数组，转换为PyTorch张量作为指示器
        indicator = torch.from_numpy((np.random.random_sample(embedding_rows)).astype(np.float32))
        # 生成随机的阈值
        threshold = float(np.random.random_sample())
        # 创建布尔类型的张量作为掩码，根据指示器和阈值设置每行的掩码值
        mask = torch.BoolTensor([True if val >= threshold else False for val in indicator])
        return mask

    # 测试逐行修剪操作的方法
    def _test_rowwise_prune_op(self, embedding_rows, embedding_dims, indices_type, weights_dtype):
        # 初始化嵌入权重为None
        embedding_weights = None
        # 根据权重数据类型初始化嵌入权重张量
        if weights_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            embedding_weights = torch.randint(0, 100, (embedding_rows, embedding_dims), dtype=weights_dtype)
        else:
            embedding_weights = torch.rand((embedding_rows, embedding_dims), dtype=weights_dtype)
        # 生成逐行掩码
        mask = self._generate_rowwise_mask(embedding_rows)

        # 定义获取PyTorch实现结果的函数
        def get_pt_result(embedding_weights, mask, indices_type):
            return torch._rowwise_prune(embedding_weights, mask, indices_type)

        # 定义参考实现结果的函数
        def get_reference_result(embedding_weights, mask, indices_type):
            # 获取掩码张量的行数
            num_embeddings = mask.size()[0]
            # 创建压缩索引输出张量，初始化为零
            compressed_idx_out = torch.zeros(num_embeddings, dtype=indices_type)
            # 根据掩码提取修剪后的权重张量
            pruned_weights_out = embedding_weights[mask[:]]
            idx = 0
            # 遍历掩码，根据掩码值设置压缩索引输出
            for i in range(mask.size()[0]):
                if mask[i]:
                    compressed_idx_out[i] = idx
                    idx = idx + 1
                else:
                    compressed_idx_out[i] = -1
            return (pruned_weights_out, compressed_idx_out)

        # 调用PyTorch的行压缩方法获取结果
        pt_pruned_weights, pt_compressed_indices_map = get_pt_result(
            embedding_weights, mask, indices_type)
        # 调用参考实现方法获取结果
        ref_pruned_weights, ref_compressed_indices_map = get_reference_result(
            embedding_weights, mask, indices_type)

        # 使用PyTorch测试工具断言结果接近
        torch.testing.assert_close(pt_pruned_weights, ref_pruned_weights)
        # 使用PyTorch的断言方法断言压缩索引映射结果相等
        self.assertEqual(pt_compressed_indices_map, ref_compressed_indices_map)
        # 使用PyTorch的断言方法断言压缩索引映射的数据类型正确
        self.assertEqual(pt_compressed_indices_map.dtype, indices_type)

    # 使用装饰器标记，跳过在Torch Dynamo环境下的测试
    @skipIfTorchDynamo()
    # 使用Hypothesis给定装饰器，参数化测试
    @given(
        embedding_rows=st.integers(1, 100),  # 嵌入行数的整数策略，范围为1到100
        embedding_dims=st.integers(1, 100),  # 嵌入维度的整数策略，范围为1到100
        weights_dtype=st.sampled_from([torch.float64, torch.float32,  # 权重数据类型的策略，从给定的数据类型中采样
                                       torch.float16, torch.int8,
                                       torch.int16, torch.int32, torch.int64])
    )
    # 定义一个测试方法，用于测试基于行的稀疏操作，使用32位索引
    def test_rowwise_prune_op_32bit_indices(self, embedding_rows, embedding_dims, weights_dtype):
        # 调用内部方法 _test_rowwise_prune_op 进行测试，使用32位整数索引类型
        self._test_rowwise_prune_op(embedding_rows, embedding_dims, torch.int, weights_dtype)


    # 使用装饰器 @skipIfTorchDynamo() 标记该测试用例，在Torch Dynamo环境下跳过执行
    # 使用 @given 装饰器标记参数化测试，参数由以下范围生成：
    # embedding_rows: 1 到 100 之间的整数
    # embedding_dims: 1 到 100 之间的整数
    # weights_dtype: 从指定的数据类型列表中随机选择，包括 torch.float64, torch.float32,
    #               torch.float16, torch.int8, torch.int16, torch.int32, torch.int64
    @given(
        embedding_rows=st.integers(1, 100),
        embedding_dims=st.integers(1, 100),
        weights_dtype=st.sampled_from([torch.float64, torch.float32,
                                       torch.float16, torch.int8,
                                       torch.int16, torch.int32, torch.int64])
    )
    # 定义一个测试方法，用于测试基于行的稀疏操作，使用64位索引
    def test_rowwise_prune_op_64bit_indices(self, embedding_rows, embedding_dims, weights_dtype):
        # 调用内部方法 _test_rowwise_prune_op 进行测试，使用64位整数索引类型
        self._test_rowwise_prune_op(embedding_rows, embedding_dims, torch.int64, weights_dtype)
if __name__ == '__main__':
    # 如果这个模块是直接运行的主程序入口
    run_tests()
    # 调用运行测试函数
```