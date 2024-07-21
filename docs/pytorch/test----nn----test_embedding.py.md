# `.\pytorch\test\nn\test_embedding.py`

```
# 导入必要的库和模块
# 这些库包括了进行单元测试和深度学习模型测试所需的工具和函数
# Owner(s): ["module: nn"]
import itertools  # 导入 itertools 库，用于迭代操作
import random  # 导入 random 库，用于生成随机数
import unittest  # 导入 unittest 库，用于编写和运行单元测试
from itertools import product  # 从 itertools 中导入 product 函数，用于求笛卡尔积

import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
from torch.testing._internal.common_cuda import TEST_CUDA  # 导入 CUDA 相关测试工具
from torch.testing._internal.common_device_type import (  # 导入设备类型相关的测试工具
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    onlyCUDA,
    onlyNativeDeviceTypes,
    skipCUDAIf,
    skipMeta,
    TEST_WITH_ROCM,
)
from torch.testing._internal.common_nn import NNTestCase  # 导入神经网络测试基类
from torch.testing._internal.common_utils import (  # 导入常用测试工具函数
    _assertGradAndGradgradChecks,
    dtype2prec_DONTUSE,
    instantiate_parametrized_tests,
    IS_JETSON,
    parametrize_test,
    run_tests,
    set_default_dtype,
    skipIfTorchDynamo,
)

# 定义一个测试类 TestEmbeddingNN，继承自 NNTestCase
class TestEmbeddingNN(NNTestCase):
    _do_cuda_memory_leak_check = True  # 设置 CUDA 内存泄漏检查标志
    _do_cuda_non_default_stream = True  # 设置非默认 CUDA 流标志

    # 装饰器，如果 CUDA 不可用，则跳过测试
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_embedding_max_norm_unsorted_repeating_indices(self):
        # 内部函数，创建一个 Embedding 对象并返回
        def create_embedding(device):
            # 设置随机数种子以确保每次获得相同的 Embedding
            torch.manual_seed(0)
            return torch.nn.Embedding(
                num_embeddings=20, embedding_dim=64, max_norm=1.0
            ).to(device)

        ix = torch.arange(2, device="cpu", dtype=torch.long).repeat(2000)  # 在 CPU 上创建长整型张量 ix，并重复2000次
        out_cpu = create_embedding("cpu")(ix)  # 在 CPU 上创建 Embedding，并传入 ix，计算输出

        ix = ix.to("cuda")  # 将 ix 移动到 CUDA 设备
        out = create_embedding("cuda")(ix)  # 在 CUDA 设备上创建 Embedding，并传入 ix，计算输出
        self.assertEqual(out.cpu(), out_cpu)  # 断言 CUDA 输出与 CPU 输出相等

    # 测试稀疏 Embedding 的基本功能
    def test_embedding_sparse_basic(self):
        embedding = nn.Embedding(10, 20, sparse=True)  # 创建一个稀疏的 Embedding
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long)  # 创建一个长整型张量作为输入
        embedding(input).sum().backward()  # 对 Embedding 输出进行求和并反向传播梯度
        self.assertTrue(embedding.weight.grad.is_sparse)  # 断言权重梯度为稀疏张量
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)  # 断言权重梯度形状与权重形状相同

    # 测试稀疏 Embedding 处理空张量的情况
    def test_embedding_sparse_empty_tensor(self):
        embedding = nn.Embedding(0, 0, sparse=True)  # 创建一个稀疏的空 Embedding
        input = torch.tensor([], dtype=torch.int64)  # 创建一个空的长整型张量作为输入
        embedding(input).sum().backward()  # 对 Embedding 输出进行求和并反向传播梯度
        self.assertTrue(embedding.weight.grad.is_sparse)  # 断言权重梯度为稀疏张量
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)  # 断言权重梯度形状与权重形状相同

        embedding = nn.Embedding(10, 0, sparse=True)  # 创建一个稀疏的 Embedding，但是输出维度为0
        input = torch.LongTensor([[0, 2, 4, 5], [4, 3, 0, 9]])  # 创建一个二维长整型张量作为输入
        embedding(input).sum().backward()  # 对 Embedding 输出进行求和并反向传播梯度
        self.assertTrue(embedding.weight.grad.is_sparse)  # 断言权重梯度为稀疏张量
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)  # 断言权重梯度形状与权重形状相同
    # 测试稀疏半嵌入层的移动操作
    def test_move_sparse_half_embedding(self):
        # 创建一个稀疏的嵌入层，有10个嵌入向量，每个向量维度为3
        embedding = nn.Embedding(10, 3, sparse=True)
        # 断言嵌入层的权重张量位于CPU上
        self.assertEqual(embedding.weight.device.type, "cpu")
        # 断言嵌入层的权重张量数据类型与当前默认数据类型相同
        self.assertEqual(embedding.weight.dtype, torch.get_default_dtype())
        # 将嵌入层转换为float16数据类型
        embedding.to(torch.float16)
        # 断言嵌入层的权重张量数据类型已转换为float16
        self.assertEqual(embedding.weight.dtype, torch.float16)
        # 断言嵌入层的嵌入维度为3
        self.assertEqual(embedding.embedding_dim, 3)
        # 断言嵌入层的嵌入数量为10
        self.assertEqual(embedding.num_embeddings, 10)

        # 如果CUDA可用，则将嵌入层转移到CUDA设备
        if torch.cuda.is_available():
            embedding.to("cuda")
            # 断言嵌入层的权重张量已经移动到CUDA设备上
            self.assertEqual(embedding.weight.device.type, "cuda")
            # 将嵌入层转移回CPU
            embedding.to("cpu")
            # 断言嵌入层的权重张量已经移动回CPU
            self.assertEqual(embedding.weight.device.type, "cpu")

    # 测试嵌入层的最大范数限制
    def test_embedding_max_norm(self):
        # 创建一个带有最大范数限制为1.0的嵌入层，有22个嵌入向量，每个向量维度为5
        embedding = nn.Embedding(22, 5, max_norm=1.0)
        # 创建一个长整型张量作为输入
        input = torch.tensor([2, 8, 8, 6], dtype=torch.long)
        # 对输入进行嵌入操作
        output = embedding(input)
        # 断言输出的第1个和第2个元素相等
        self.assertEqual(output[1], output[2])
        # 断言输出的所有元素在L2范数下不大于1
        self.assertTrue(output.data.norm(p=2, dim=1).le(1).all())

    # 使用参数化测试进行嵌入层从预训练权重创建的测试
    @parametrize_test(
        "dtype",
        (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float,
            torch.double,
        ),
    )
    def test_embedding_from_pretrained(self, dtype):
        # 创建一个张量a，包含预训练的嵌入权重
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
        # 从预训练的权重创建一个嵌入层
        embedding = nn.Embedding.from_pretrained(a)
        # 断言嵌入层的权重数据与张量a相等
        self.assertEqual(a, embedding.weight.data)

        # 创建一个长整型张量作为输入
        input = torch.LongTensor([0, 1])
        # 对输入进行嵌入操作
        output = embedding(input)
        # 断言输出与张量a相等
        self.assertEqual(a, output)

    # 使用预训练权重创建嵌入袋的测试
    def test_embedding_bag_from_pretrained(self):
        # 创建一个张量a，包含预训练的嵌入权重
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # 从预训练的权重创建一个嵌入袋
        embedding = nn.EmbeddingBag.from_pretrained(a)
        # 断言嵌入袋的权重与张量a相等
        self.assertEqual(a, embedding.weight)

        # 创建一个长整型张量作为输入，创建一个标签张量
        input = torch.tensor([0, 1], dtype=torch.long)
        output = embedding(input, torch.arange(input.size(0)))
        # 断言输出与张量a相等
        self.assertEqual(a, output)

    # 使用带有填充索引的预训练权重创建嵌入层的测试
    def test_embedding_from_pretrained_padding_idx(self):
        padding_idx = 2
        padding_vec = torch.ones(3) * 7
        embeddings = torch.rand(4, 3, requires_grad=True)
        # 使用无梯度计算的方式将填充向量添加到预训练的权重中
        with torch.no_grad():
            embeddings[padding_idx] = padding_vec
        # 使用带有填充索引和填充向量创建一个嵌入层
        embedding_nn = nn.Embedding.from_pretrained(embeddings, padding_idx=padding_idx)
        # 断言嵌入层的权重中填充索引位置的向量与填充向量相等
        self.assertEqual(embedding_nn.weight[padding_idx], padding_vec)

    # 使用带有填充索引的预训练权重创建嵌入袋的测试
    def test_embedding_bag_from_pretrained_padding_idx(self):
        padding_idx = 2
        embeddings = torch.rand(4, 3, requires_grad=True)
        # 使用带有填充索引创建一个嵌入袋
        embedding_nn = nn.EmbeddingBag.from_pretrained(
            embeddings, padding_idx=padding_idx
        )
        # 断言嵌入袋的权重与张量embeddings相等
        self.assertEqual(embedding_nn.weight, embeddings)
    # 定义测试函数，用于测试从预训练选项创建嵌入层
    def test_embedding_from_pretrained_options(self):
        # 设置默认的张量数据类型为双精度浮点型
        with set_default_dtype(torch.double):
            # 创建一个张量 a，包含两个子列表，每个子列表包含三个浮点数
            a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            # 定义选项字典 opts
            opts = {
                "max_norm": 2.0,                    # 最大范数值
                "norm_type": 0.5,                   # 范数类型
                "scale_grad_by_freq": False,        # 是否按频率缩放梯度
                "sparse": True,                     # 是否使用稀疏模式
            }
            # 使用预训练选项创建嵌入层 embedding
            embedding = nn.Embedding.from_pretrained(a, **opts)
            # 创建输入张量 input，包含两个长整型元素
            input = torch.LongTensor([0, 1])
            # 将输入张量传入嵌入层，得到输出张量 output
            output = embedding(input)
            # 测试输出张量与原始张量 a 相等
            self.assertEqual(a, output)
            # 断言原始张量 a 不等于指定的张量
            self.assertTrue(a.ne(torch.arange(1, 7, dtype=a.dtype).view(2, 3)).all())
            # 断言输出张量的数据按照指定的范数类型和维度进行规范化后，均小于等于最大范数值
            self.assertTrue(
                output.data.norm(p=opts["norm_type"], dim=1).le(opts["max_norm"]).all()
            )

    # 定义测试函数，用于测试嵌入层的功能
    def test_embedding_functional(self):
        # 创建一个张量 a，包含两个子列表，每个子列表包含三个长整型元素
        a = torch.tensor([[1, 3, 2], [0, 2, 1]], dtype=torch.long)
        # 创建一个形状为 (4, 3) 的随机张量，需要计算梯度
        embeddings = torch.rand(4, 3, requires_grad=True)

        # 创建一个传统的嵌入层 embed_old
        embed_old = torch.nn.Embedding(4, 3)
        # 将嵌入层的权重数据设置为 embeddings 的数据
        embed_old.weight.data = embeddings.data
        # 断言嵌入层的权重数据与 embeddings 的数据相等
        self.assertEqual(embed_old.weight.data, embeddings.data)
        # 对输入张量 a 应用嵌入层 embed_old，得到结果 res_old
        res_old = embed_old(a)

        # 使用 functional 方式计算嵌入结果 res_F
        res_F = F.embedding(a, embeddings)
        # 断言传统方法的结果与 functional 方法的结果相等
        self.assertEqual(res_old, res_F)

        # 使用 from_pretrained 方法创建一个新的嵌入层 embed_old
        embed_old = torch.nn.Embedding(4, 3)
        embed_old = embed_old.from_pretrained(embeddings, padding_idx=2)
        # 对输入张量 a 应用新的嵌入层 embed_old，得到结果 res_old
        res_old = embed_old(a)
        # 使用 functional 方式计算带填充索引的嵌入结果 res_F
        res_F = F.embedding(a, embeddings, padding_idx=2)

        # 断言两种方式计算得到的结果相等
        self.assertEqual(res_old, res_F)

    # 定义测试函数，用于测试嵌入袋的 functional 方法
    def test_embedding_bag_functional(self):
        # 创建一个张量 a，包含两个子列表，每个子列表包含三个长整型元素
        a = torch.tensor([[1, 3, 2], [0, 2, 1]], dtype=torch.long)
        # 创建一个形状为 (4, 3) 的随机张量，需要计算梯度
        embeddings = torch.rand(4, 3, requires_grad=True)

        # 创建一个嵌入袋 embed_old
        embed_old = torch.nn.EmbeddingBag(4, 3)
        # 将嵌入袋的权重设置为 embeddings
        embed_old.weight = torch.nn.Parameter(embeddings)
        # 对输入张量 a 应用嵌入袋 embed_old，得到结果 res_old
        res_old = embed_old(a)

        # 使用 functional 方式计算嵌入袋结果 res_F
        res_F = F.embedding_bag(a, embeddings)
        # 断言传统方法的结果与 functional 方法的结果相等
        self.assertEqual(res_old, res_F)

        # 使用 from_pretrained 方法创建一个新的嵌入袋 embed_old
        embed_old = torch.nn.EmbeddingBag(4, 3)
        embed_old = embed_old.from_pretrained(embeddings, padding_idx=2)
        # 对输入张量 a 应用新的嵌入袋 embed_old，得到结果 res_old
        res_old = embed_old(a)
        # 使用 functional 方式计算带填充索引的嵌入袋结果 res_F
        res_F = F.embedding_bag(a, embeddings, padding_idx=2)

        # 断言两种方式计算得到的结果相等
        self.assertEqual(res_old, res_F)

    # 确保如果填充索引超出边界，会抛出错误
    def test_embedding_bag_padding_idx_error(self):
        # 创建一个示例张量 `a`，包含两个样本，每个样本有三个特征
        a = torch.tensor([[1, 3, 2], [0, 2, 1]], dtype=torch.long)
        num_embeddings = 4  # 设定嵌入矩阵的行数（嵌入数量）
        num_features = 3    # 设定嵌入矩阵的列数（每个嵌入的特征数）
        embeddings = torch.rand(num_embeddings, num_features, requires_grad=True)  # 创建一个随机初始化的嵌入矩阵，并设置为需要梯度计算

        functional_err_msg = r"padding_idx must be within the number of embeddings"  # 指定功能异常消息：padding_idx 必须在嵌入数量内
        module_err_msg = r"padding_idx must be within num_embeddings"  # 指定模块异常消息：padding_idx 必须在嵌入数量内

        # 遍历所有可能的 padding_idx 值，从 -(num_embeddings + 2) 到 (num_embeddings + 1)
        for padding_idx in range(-(num_embeddings + 2), (num_embeddings + 2)):
            # 如果 padding_idx 超出了有效范围
            if (padding_idx < -num_embeddings) or (padding_idx >= num_embeddings):
                # 断言调用 F.embedding_bag 时会触发 RuntimeError 异常，并匹配 functional_err_msg
                with self.assertRaisesRegex(RuntimeError, functional_err_msg):
                    F.embedding_bag(a, embeddings, padding_idx=padding_idx)
                # 断言调用 torch.nn.EmbeddingBag 时会触发 AssertionError 异常，并匹配 module_err_msg
                with self.assertRaisesRegex(AssertionError, module_err_msg):
                    torch.nn.EmbeddingBag(
                        num_embeddings, num_features, padding_idx=padding_idx
                    )
            else:
                # 在有效范围内调用 F.embedding_bag，不应触发异常
                F.embedding_bag(a, embeddings, padding_idx=padding_idx)
                # 在有效范围内调用 torch.nn.EmbeddingBag，不应触发异常
                torch.nn.EmbeddingBag(
                    num_embeddings, num_features, padding_idx=padding_idx
                )

    def test_embeddingbag_from_pretrained(self):
        # 创建一个示例张量 `a`，包含两个样本，每个样本有三个特征
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        embeddingbag = nn.EmbeddingBag.from_pretrained(a)  # 使用预训练的张量 `a` 创建一个嵌入层实例
        self.assertEqual(a, embeddingbag.weight.data)  # 断言预训练张量与嵌入层实例的权重数据相等

        input = torch.LongTensor([[0, 1]])  # 创建一个输入张量 `input`，包含两个索引
        output = embeddingbag(input)  # 使用嵌入层实例处理输入张量，得到输出
        self.assertEqual(a.mean(0, keepdim=True), output)  # 断言嵌入层的输出等于预训练张量按列求均值后的结果

    def test_embeddingbag_from_pretrained_options(self):
        # 使用默认数据类型设置为双精度浮点数
        with set_default_dtype(torch.double):
            # 创建一个示例张量 `a`，包含两个样本，每个样本有三个特征
            a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            # 定义初始化选项字典
            opts = {
                "max_norm": 2.0,
                "norm_type": 0.5,
                "scale_grad_by_freq": False,
                "mode": "max",
                "sparse": False,
            }
            embeddingbag = nn.EmbeddingBag.from_pretrained(a, **opts)  # 使用预训练的张量 `a` 和选项字典创建一个嵌入层实例

            input = torch.LongTensor([[0, 1]])  # 创建一个输入张量 `input`，包含两个索引
            output = embeddingbag(input)  # 使用嵌入层实例处理输入张量，得到输出
            self.assertEqual(a.max(0, keepdim=True)[0], output)  # 断言嵌入层的输出等于预训练张量按列取最大值后的结果
            self.assertTrue(a.ne(torch.arange(1, 7, dtype=a.dtype).view(2, 3)).all())  # 断言预训练张量与指定张量不相等的所有元素
            self.assertTrue(
                a.norm(p=opts["norm_type"], dim=1).le(opts["max_norm"]).all()
            )  # 断言预训练张量按给定范数类型计算的结果在最大范数限制内

    def test_embeddingbag_include_last_offset(self):
        # 从 https://github.com/pytorch/pytorch/issues/89677 中提取的测试用例
        embeddingbag = nn.EmbeddingBag(100, 3, include_last_offset=True, padding_idx=61)  # 创建一个包含最后偏移的嵌入层实例，指定 padding_idx
        input = torch.tensor([0, 1, 2, 3])  # 创建一个输入张量 `input`，包含四个索引
        out = embeddingbag(input, torch.tensor([0, 3, 3]))  # 使用嵌入层实例处理输入张量和偏移张量，得到输出
        out2 = embeddingbag(input, torch.tensor([0, 3, 4]))  # 使用嵌入层实例处理输入张量和不同偏移张量，得到输出

        weight = embeddingbag.weight  # 获取嵌入层实例的权重矩阵
        row0 = weight[0:3].mean(0)  # 计算权重矩阵前三行的均值作为参考行0
        row1 = weight[3]  # 获取权重矩阵的第四行作为参考行1
        ref_out = torch.stack([row0, row1])  # 将参考行0和参考行1堆叠成输出的参考结果

        self.assertEqual(ref_out, out)  # 断言嵌入层的输出与参考结果 `ref_out` 相等
        self.assertEqual(ref_out, out2)  # 断言另一个嵌入层的输出与参考结果 `ref_out` 相等
class TestEmbeddingNNDeviceType(NNTestCase):
    # 继承自NNTestCase的测试类，用于测试嵌入层的不同设备类型情况

    def test_embedding_dense_grad(self, device):
        # 测试嵌入层梯度计算
        with set_default_dtype(torch.double):
            # 设置默认数据类型为双精度
            embd = nn.Embedding(20, 20).to(device)
            # 创建一个嵌入层，大小为20x20，并移动到指定设备
            weight = embd.weight

            def fn_wrapper(device):
                # 包装函数，用于生成嵌入层的函数
                def fn(weight):
                    # 实际的嵌入层计算函数
                    inp = torch.tensor(
                        [[0, 1, 1, 2], [3, 5, 7, 11]], dtype=torch.long
                    ).to(device)
                    # 创建输入张量，类型为长整型，移动到指定设备
                    return torch.nn.functional.embedding(inp, weight)

                return fn

            fn = fn_wrapper(device)
            _assertGradAndGradgradChecks(self, fn, (weight,))
            # 使用自定义的梯度检查函数进行梯度和二阶梯度检查

    def test_embedding_scalar_weight_error(self, device):
        # 测试嵌入层使用标量权重时的错误处理
        indices = torch.rand(2, 2, device=device).long()
        # 创建随机索引张量，类型为长整型，移动到指定设备
        weights = [
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device).reshape(1, 1, 1),
        ]

        for weight in weights:
            # 遍历不同的权重
            with self.assertRaisesRegex(RuntimeError, "'weight' must be 2-D"):
                # 检查是否抛出正确的异常信息
                torch.nn.functional.embedding(indices, weight)

    @dtypesIfCUDA(torch.float16, torch.float64)
    @dtypes(torch.float64)
    def test_embedding_backward(self, device, dtype):
        # 测试嵌入层的反向传播
        embedding = nn.Embedding(10, 3, sparse=True)
        # 创建一个稀疏的嵌入层
        tensor = torch.tensor([[7, 1, 3]])
        ones = torch.tensor(1.0, dtype=dtype).expand(3, 3)
        tensorTwice = tensor.repeat(1, 2)
        onesTwice = torch.cat((ones, ones))

        embedding = embedding.to(dtype=dtype).to(device)
        tensor = tensor.to(device)
        ones = ones.to(device)
        tensorTwice = tensorTwice.to(device)
        onesTwice = onesTwice.to(device)

        embedding.zero_grad()
        # 将嵌入层的梯度清零
        embedding(tensor[0]).sum().backward()
        # 计算嵌入层输出的和，并进行反向传播
        self.assertEqual(embedding.weight.grad._indices(), tensor)
        self.assertEqual(embedding.weight.grad._values(), ones)
        # 断言梯度的索引和值

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        embedding(tensor[0]).sum().backward()
        # 再次进行反向传播
        self.assertEqual(embedding.weight.grad._indices(), tensorTwice)
        self.assertEqual(embedding.weight.grad._values(), onesTwice)
        # 断言更新后的梯度的索引和值

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        tensor[0, 0] = 8
        embedding(tensor[0]).sum().backward()
        tensorTwice[0, 3] = 8
        # 修改张量后再次进行反向传播
        self.assertEqual(embedding.weight.grad._indices(), tensorTwice)
        self.assertEqual(embedding.weight.grad._values(), onesTwice)
        # 断言更新后的梯度的索引和值

    @dtypesIfCUDA(
        *(
            (torch.float, torch.double, torch.bfloat16, torch.half)
            if TEST_WITH_ROCM
            else (torch.float, torch.double, torch.half)
        )
    )
    @dtypes(torch.float32)
    # 定义一个测试方法，用于验证嵌入操作中的最大范数约束的反向传播
    def test_embedding_max_norm_backward(self, device, dtype):
        # 由于原地归一化使得解析梯度与实际产生的梯度不同，因此无法使用 gradcheck 函数进行验证
        weight = torch.randn((4, 4), device=device, dtype=dtype) * 2
        weight.requires_grad_()
        # 创建输入索引列表
        inp_list = [0, 1, 2, 2]
        inp = torch.tensor(inp_list, device=device)
        # 计算嵌入向量并求和
        out = nn.functional.embedding(inp, weight, max_norm=1.0).sum()
        # 执行反向传播
        out.backward()

        # 预期的梯度值
        expected_grad = (
            torch.tensor([[1.0, 1.0, 2.0, 0.0]], device=device, dtype=dtype)
            .transpose(0, 1)
            .expand(4, 4)
        )
        # 断言权重的梯度与预期的梯度相等
        self.assertEqual(weight.grad, expected_grad)

    # 为了测试嵌入操作中最大范数约束的前向自动微分功能
    @dtypesIfCUDA(
        *(
            (torch.float, torch.double, torch.bfloat16, torch.half)
            if TEST_WITH_ROCM
            else (torch.float, torch.double, torch.half)
        )
    )
    @dtypes(torch.float32)
    def test_embedding_max_norm_fwd_AD(self, device, dtype):
        if torch.device(device).type == "xla":
            self.skipTest("forward AD doesn't work on xla")

        # 由于原地归一化使得解析梯度与实际产生的梯度不同，因此无法使用 gradcheck 函数进行验证
        weight = torch.randn((4, 4), device=device, dtype=dtype) * 2
        tangent = torch.ones((4, 4), device=device, dtype=dtype)
        inp = torch.tensor([[0, 1], [2, 2]], device=device)
        # 使用前向自动微分模式
        with torch.autograd.forward_ad.dual_level():
            dual_weight = torch.autograd.forward_ad.make_dual(weight, tangent)
            out = nn.functional.embedding(inp, dual_weight, max_norm=1.0)
            jvp = torch.autograd.forward_ad.unpack_dual(out).tangent

        # 预期的梯度值为全1张量
        expected_grad = torch.ones((2, 2, 4), device=device, dtype=dtype)
        # 断言输出的切向向量与预期的梯度值相等
        self.assertEqual(jvp, expected_grad)

    # 对于具有 padding_idx 的 torch.nn.functional.embedding_bag 前向和反向函数的正确性检查，
    # 给定一个使用偏移数组分隔成袋子的1D输入。与使用填充索引填充偏移数组指示的间隙的等效2D输入进行比较。
    @skipIfTorchDynamo("see https://github.com/pytorch/pytorch/pull/95621")
    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    @dtypesIfCUDA(torch.half, torch.bfloat16)
    # 对具有 padding_idx 的 torch.nn.functional.embedding_bag 前向和反向函数进行正确性检查，
    # 给定一个2D索引输入。与使用填充索引填充偏移数组指示的间隙的等效2D输入进行比较。
    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    @dtypesIfCUDA(torch.half, torch.bfloat16)
    @onlyCUDA
    @dtypes(
        *(
            (torch.float, torch.double, torch.bfloat16, torch.half)
            if TEST_WITH_ROCM
            else (torch.float, torch.double, torch.half)
        )
    )
    # 使用装饰器指定多种数据类型作为参数，根据 TEST_WITH_ROCM 决定是否包含额外的数据类型
    def test_embedding_max_norm_device(self, device, dtype):
        # 创建一个嵌入层，包含22个嵌入向量，每个向量长度为5，最大范数为1.0，并移动到指定设备和数据类型
        embedding = nn.Embedding(22, 5, max_norm=1.0).to(device, dtype=dtype)
        # 输入张量，包含多个索引，设备和数据类型与嵌入层相同
        input = torch.tensor([2, 8, 8, 6], device=device, dtype=torch.long)
        # 嵌入层的计算结果
        output = embedding(input)
        # 断言：输出的第1个和第2个索引对应的结果相等
        self.assertEqual(output[1], output[2])
        # 断言：所有样本的L2范数不超过1
        self.assertTrue(output.data.norm(p=2, dim=1).le(1).all())

    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long)))
    # 使用装饰器指定多种数据类型作为参数的笛卡尔积
    def test_embedding_bag_empty_input(self, device, dtypes):
        # 设定嵌入袋的大小为4x3，稀疏性由布尔变量sparse决定
        m = 4
        n = 3
        # 空输入张量，设备和数据类型与给定的dtypes匹配
        x = torch.tensor([], device=device, dtype=dtypes[0])
        # 针对稀疏性的两种情况进行迭代
        for sparse in [True, False]:
            # 创建一个嵌入袋，设定好设备
            Embed = torch.nn.EmbeddingBag(m, n, sparse=sparse)
            Embed.to(device)

            # 使用指定的输入和偏移量张量进行嵌入袋的计算
            output = Embed(
                input=x, offsets=torch.tensor([0], device=device, dtype=dtypes[1])
            )
            # 断言：输出应当与全零张量形状相同
            self.assertEqual(output, torch.zeros_like(output))

            # 使用指定的输入和偏移量张量进行嵌入袋的计算（偏移量为重复）
            output = Embed(
                input=x, offsets=torch.tensor([0, 0], device=device, dtype=dtypes[1])
            )
            # 断言：输出应当与全零张量形状相同
            self.assertEqual(output, torch.zeros_like(output))

    @skipCUDAIf(True, "no out-of-bounds check on CUDA for perf.")
    @dtypes(*itertools.product((torch.float, torch.double), (torch.int, torch.long)))
    @parametrize_test("padding_idx", [None, 0])
    @parametrize_test("mode", ["sum", "mean", "max"])
    # 使用装饰器指定多种数据类型和参数化测试参数
    def test_embedding_bag_out_of_bounds_idx(self, device, dtypes, padding_idx, mode):
        padding_idx = 0
        w_dtype, idx_dtype = dtypes
        # 负超出边界的索引
        idx1 = torch.tensor([[-1, 1]], device=device, dtype=idx_dtype)
        # 正超出边界的索引
        idx2 = torch.tensor([[11, 8]], device=device, dtype=idx_dtype)
        # 随机权重张量，设备和数据类型与给定的w_dtype匹配
        weight = torch.randn(10, 2, device=device, dtype=w_dtype)
        if mode == "sum":
            # 只有在`sum`模式下支持每个样本的权重
            per_sample_weights = (
                None,
                torch.randn_like(idx1, device=device, dtype=w_dtype),
            )
        else:
            per_sample_weights = (None,)

        # 针对每个样本的权重和超出边界索引进行迭代
        for p_s_weights, idx in itertools.product(per_sample_weights, (idx1, idx2)):
            msg = "Expected idx >= 0 && idx < num_embeddings"
            # 使用断言：预期会抛出包含指定消息的运行时错误
            with self.assertRaisesRegex(RuntimeError, msg):
                torch.nn.functional.embedding_bag(
                    idx,
                    weight,
                    per_sample_weights=p_s_weights,
                    padding_idx=padding_idx,
                    mode=mode,
                )
    # 定义一个测试函数，用于测试嵌入包函数的维度错误
    def test_embedding_bag_dimension_errors(self, device):
        # 定义包含多个函数的元组，每个函数执行不同的嵌入包操作
        funcs = (
            lambda x, y, z: torch.nn.functional.embedding_bag(y, x, z),
            torch.embedding_bag,
            torch._embedding_bag,
            torch._embedding_bag_forward_only,
        )
        # 遍历所有函数并测试
        for i, f in enumerate(funcs):
            # 根据函数索引选择相应的错误类型
            err_type = (ValueError, RuntimeError) if i == 0 else RuntimeError

            # 创建一个全零的权重张量，形状为 (2, 6)，数据类型为 float64，放置在指定设备上
            weight = torch.full(
                (
                    2,
                    6,
                ),
                0,
                dtype=torch.float64,
                device=device,
            )
            # 创建一个全 2 的索引张量，形状为 (2, 0, 0, 6, 6)，数据类型为 int64，放置在指定设备上
            indices = torch.full(
                (
                    2,
                    0,
                    0,
                    6,
                    6,
                ),
                2,
                dtype=torch.int64,
                device=device,
            )
            # 创建一个全零的偏移张量，形状为 (2, 0, 0, 6, 6)，数据类型为 int64，放置在指定设备上
            offsets = torch.full((2, 0, 0, 6, 6), 0, dtype=torch.int64, device=device)

            # 根据函数索引选择相应的错误消息
            if i == 0:
                error_msg = "input has to be 1D or 2D Tensor"
            else:
                error_msg = "input has to be a 1D or 2D Tensor"
            # 禁用特定错误类型的异常断言，测试函数调用时是否触发异常
            torch._dynamo.disable(self.assertRaisesRegex)(
                err_type, error_msg, lambda: f(weight, indices, offsets)
            )

            # 创建一个全零的权重张量，形状为 (2, 2)，数据类型为 float64，放置在指定设备上
            weight = torch.full((2, 2), 0, dtype=torch.float64, device=device)
            # 创建一个全 1 的索引张量，形状为 (2,)，数据类型为 int64，放置在指定设备上
            indices = torch.full((2,), 1, dtype=torch.int64, device=device)

            # 禁用特定错误类型的异常断言，测试函数调用时是否触发异常
            torch._dynamo.disable(self.assertRaisesRegex)(
                err_type,
                "offsets has to be a 1D Tensor",
                lambda: f(weight, indices, offsets),
            )

            # 创建一个全零的权重张量，形状为 (2, 2, 2)，数据类型为 float64，放置在指定设备上
            weight = torch.full((2, 2, 2), 0, dtype=torch.float64, device=device)
            # 创建一个全 2 的索引张量，形状为 (2,)，数据类型为 int64，放置在指定设备上
            indices = torch.full((2,), 2, dtype=torch.int64, device=device)
            # 创建一个全零的偏移张量，形状为 (2,)，数据类型为 int64，放置在指定设备上
            offsets = torch.full((2,), 0, dtype=torch.int64, device=device)

            # 禁用特定错误类型的异常断言，测试函数调用时是否触发异常
            torch._dynamo.disable(self.assertRaisesRegex)(
                err_type,
                "weight has to be a 2D Tensor",
                lambda: f(weight, indices, offsets),
            )
    # 测试 EmbeddingBag 函数在处理 per_sample_weights 失败时的情况
    def test_EmbeddingBag_per_sample_weights_failures(self, device, dtypes):
        # Failure 1: mismatched embeddings / per_sample_weights dtype
        # 创建一个 EmbeddingBag 实例 es，指定词汇表大小为 5，嵌入维度为 2，聚合模式为求和，并将其移至指定的设备和数据类型
        es = nn.EmbeddingBag(5, 2, mode="sum").to(dtype=torch.float, device=device)
        # 创建一个输入张量 input，包含索引 [3, 1, 1, 1, 4, 0]，数据类型从 dtypes[0] 中选择，并移至设备
        input = torch.tensor([3, 1, 1, 1, 4, 0], dtype=dtypes[0], device=device)
        # 创建一个偏移量张量 offsets，包含索引 [0, 0, 3, 3, 6]，数据类型从 dtypes[1] 中选择，并移至设备
        offsets = torch.tensor([0, 0, 3, 3, 6], dtype=dtypes[1], device=device)
        # 创建一个与 input 相同大小的 per_sample_weights 张量，数据类型为 torch.double，并移至设备
        per_sample_weights = torch.randn_like(input, dtype=torch.double, device=device)
        # 如果设备为 "cpu"，则预期会抛出 RuntimeError，其错误消息包含 "have the same type as"
        if device == "cpu":
            with self.assertRaisesRegex(RuntimeError, "have the same type as"):
                es(input, offsets, per_sample_weights)
        else:
            # 否则，在不支持的设备上预期抛出 RuntimeError，其错误消息包含 "expected scalar type"
            with self.assertRaisesRegex(RuntimeError, "expected scalar type"):
                es(input, offsets, per_sample_weights)

        # Failure 2.1: input/per_sample_weights have different sizes (1d input)
        # 重新定义 input 和 offsets 张量，此次 per_sample_weights 与 input 大小不匹配
        input = torch.tensor([3, 1, 1, 1, 4, 0], dtype=dtypes[0], device=device)
        offsets = torch.tensor([0, 0, 3, 3, 6], dtype=dtypes[1], device=device)
        # 创建一个大小不匹配的 per_sample_weights 张量，预期抛出 ValueError，错误消息包含 "same shape as the input"
        per_sample_weights = torch.randn(5, dtype=torch.float, device=device)
        with self.assertRaisesRegex(ValueError, "same shape as the input"):
            es(input, offsets, per_sample_weights)

        # Failure 2.2: input/per_sample_weights have different sizes (2d input)
        # 重新定义 input 为一个二维张量，offsets 为 None，此次 per_sample_weights 与 input 大小不匹配
        input = torch.randint(5, (7, 3), dtype=dtypes[0], device=device)
        offsets = None
        # 创建一个大小不匹配的 per_sample_weights 张量，预期抛出 ValueError，错误消息包含 "same shape as the input"
        per_sample_weights = torch.randn(7 * 3, dtype=torch.float, device=device)
        with self.assertRaisesRegex(ValueError, "same shape as the input"):
            es(input, offsets, per_sample_weights)

        # Failure 3: Unsupported per_sample_weights and mode=('max', 'mean')
        # 遍历不支持的聚合模式 ("max", "mean")
        for unsupported_mode in ("max", "mean"):
            # 创建一个 EmbeddingBag 实例 es，指定词汇表大小为 5，嵌入维度为 2，聚合模式为当前遍历的不支持模式，并将其移至指定的设备和数据类型
            es = nn.EmbeddingBag(5, 2, mode=unsupported_mode).to(
                dtype=torch.float, device=device
            )
            # 创建一个大小为 (7, 3) 的随机整数输入张量 input，数据类型从 dtypes[0] 中选择，并移至设备
            input = torch.randint(5, (7, 3), dtype=dtypes[0], device=device)
            offsets = None
            # 创建一个大小为 (7, 3) 的随机浮点数 per_sample_weights 张量，数据类型为 torch.float，并移至设备
            per_sample_weights = torch.randn(7, 3, dtype=torch.float, device=device)
            # 预期会抛出 NotImplementedError，错误消息包含 "only supported for mode='sum'"
            with self.assertRaisesRegex(
                NotImplementedError, "only supported for mode='sum'"
            ):
                es(input, offsets, per_sample_weights)

    # _embedding_bag_reference_impl 函数的参考实现
    def _embedding_bag_reference_impl(
        self,
        input,
        weight,
        offsets=None,
        mode="sum",
        per_sample_weights=None,
        include_last_offset=False,
    ):
        # 断言模式为 "sum" 或 per_sample_weights 为 None
        assert mode == "sum" or per_sample_weights is None
        # 断言 offsets 不为 None
        assert offsets is not None
        # 如果 per_sample_weights 为 None，则初始化为全1的张量，与 input 的数据类型和设备一致
        if per_sample_weights is None:
            per_sample_weights = torch.ones(input.size()).to(
                dtype=weight.dtype, device=weight.device
            )
        # 断言 input 和 per_sample_weights 的元素数量相同
        assert input.numel() == per_sample_weights.numel()

        bags = []
        # 将 input 转换为 long 类型张量
        long_input = input.to(torch.long)
        # 根据 long_input 从 weight 中选择对应的 embeddings，并乘以 per_sample_weights 的扩展形式
        embeddings = weight.index_select(0, long_input) * per_sample_weights.unsqueeze(
            1
        )
        # 如果 include_last_offset 为真，则遍历 offsets 除最后一个值外的所有偏移量
        if include_last_offset:
            for index in range(len(offsets) - 1):
                offset = offsets[index]
                next_offset = offsets[index + 1]
                length = next_offset - offset
                # 如果 length 为 0，则向 bags 中添加全零张量
                if length == 0:
                    bags.append(
                        torch.tensor([0] * weight.size(1)).to(
                            dtype=embeddings.dtype, device=embeddings.device
                        )
                    )
                else:
                    # 根据 mode 的不同执行不同的操作，将结果添加到 bags 中
                    if mode == "sum":
                        bags.append(embeddings.narrow(0, offset, length).sum(0))
                    elif mode == "mean":
                        bags.append(
                            embeddings.narrow(0, offset, length).sum(0).div(length)
                        )
                    else:
                        assert mode == "max"
                        bags.append(embeddings.narrow(0, offset, length).max(0)[0])
        # 如果 include_last_offset 为假，则遍历所有 offsets
        else:
            for index, offset in enumerate(offsets):
                if index + 1 < len(offsets):
                    next_offset = offsets[index + 1]
                else:
                    next_offset = len(long_input)
                length = next_offset - offset
                # 如果 length 为 0，则向 bags 中添加全零张量
                if length == 0:
                    bags.append(
                        torch.tensor([0] * weight.size(1)).to(
                            dtype=embeddings.dtype, device=embeddings.device
                        )
                    )
                else:
                    # 根据 mode 的不同执行不同的操作，将结果添加到 bags 中
                    if mode == "sum":
                        bags.append(embeddings.narrow(0, offset, length).sum(0))
                    elif mode == "mean":
                        bags.append(
                            embeddings.narrow(0, offset, length).sum(0).div(length)
                        )
                    else:
                        assert mode == "max"
                        bags.append(embeddings.narrow(0, offset, length).max(0)[0])
        # 将 bags 中的张量堆叠成一个张量返回
        return torch.stack(bags)

    @skipMeta
    @dtypes(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.half, torch.bfloat16, torch.float, torch.double),
        )
    )
    @dtypesIfCUDA(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half),
        )
    )
    # 定义测试函数，用于测试空输入、每个样本权重和反向传播。之前存在 CUDA 无效配置的 bug（更多背景在 #46572）
    def test_EmbeddingBag_empty_per_sample_weights_and_offsets(self, device, dtypes):
        # 测试每个样本权重的情况
        def test_per_sample_weights(mode, trainable_scale):
            # 在指定设备上创建 EmbeddingBag 实例，设置数据类型和设备
            es = nn.EmbeddingBag(5, 2, mode=mode).to(dtype=dtypes[2], device=device)
            # 将权重数据复制为从1到10的序列，并匹配数据类型
            es.weight.data.copy_(
                torch.arange(1, 11, device=device).view_as(es.weight).to(dtypes[2])
            )
            # 创建空的输入张量
            input = torch.tensor([], device=device, dtype=dtypes[0])
            # 定义偏移量张量，对于空输入使用固定的偏移量
            offsets = torch.tensor([0, 0, 0, 0, 0], device=device, dtype=dtypes[1])
            # 创建与输入张量形状相同的随机样本权重张量，并设置可训练标志
            per_sample_weights = torch.randn_like(
                input, dtype=dtypes[2]
            ).requires_grad_(trainable_scale)
            # 分离样本权重张量，同时设置可训练标志
            ref_per_sample_weights = per_sample_weights.detach().requires_grad_(
                trainable_scale
            )
            # 分离权重张量，同时设置可训练标志
            reference_weights = es.weight.detach().requires_grad_()

            # 调用参考实现计算期望结果
            expected = self._embedding_bag_reference_impl(
                input, reference_weights, offsets, mode, ref_per_sample_weights
            )
            # 调用 EmbeddingBag 实例计算结果
            result = es(input, offsets, per_sample_weights)
            # 断言结果与期望值相等，使用指定的数值精度容差
            self.assertEqual(
                result, expected, atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0
            )

            # 创建与期望结果形状相同的随机梯度张量
            grad = torch.randn_like(expected)
            # 计算反向传播
            result.backward(grad)
            # 对于空输入，参考实现不会有梯度函数；权重梯度应当是零张量
            ref_weights_grad = torch.zeros_like(es.weight)
            self.assertEqual(
                es.weight.grad,
                ref_weights_grad,
                atol=dtype2prec_DONTUSE[dtypes[2]],
                rtol=0,
            )
            # 如果设置了可训练标志，样本权重梯度应当是空张量
            if trainable_scale:
                ref_per_sample_weights_grad = torch.empty_like(per_sample_weights)
                self.assertEqual(
                    per_sample_weights.grad,
                    ref_per_sample_weights_grad,
                    atol=dtype2prec_DONTUSE[dtypes[2]],
                    rtol=0,
                )

        # 定义模式和可训练标志组合
        modes = ("sum",)
        trainable_scale = (True, False)
        # 遍历所有模式和可训练标志组合，执行测试函数
        for mode, trainable in itertools.product(modes, trainable_scale):
            test_per_sample_weights(mode, trainable)
    # 定义测试函数，用于测试 EmbeddingBag 类的 per_sample_weights 和 offsets 功能
    def test_EmbeddingBag_per_sample_weights_and_offsets(self, device, dtypes):
        # 定义内部函数 test_per_sample_weights，测试不同模式和可训练标度
        def test_per_sample_weights(mode, trainable_scale):
            # 创建一个 EmbeddingBag 实例，设置词汇表大小为 5，嵌入维度为 2，使用指定的 mode 和设备
            es = nn.EmbeddingBag(5, 2, mode=mode).to(dtype=dtypes[2], device=device)
            # 将权重数据设置为从 1 到 10 的序列，转换为指定类型的张量，并复制给 EmbeddingBag 实例的权重
            es.weight.data.copy_(
                torch.arange(1, 11, device=device).view_as(es.weight).to(dtypes[2])
            )
            # 创建输入张量，包含指定设备上指定类型的整数数据
            input = torch.tensor([3, 1, 1, 1, 4, 0], device=device, dtype=dtypes[0])
            # 创建偏移量张量，包含指定设备上指定类型的整数数据
            offsets = torch.tensor([0, 0, 3, 3, 6], device=device, dtype=dtypes[1])
            # 创建与输入张量形状相同的随机张量，作为每样本权重，允许梯度计算
            per_sample_weights = torch.randn_like(
                input, dtype=dtypes[2]
            ).requires_grad_(trainable_scale)
            # 分离 per_sample_weights 张量并设置允许梯度计算的属性，作为参考的每样本权重
            ref_per_sample_weights = per_sample_weights.detach().requires_grad_(
                trainable_scale
            )
            # 分离 es.weight 张量并设置允许梯度计算的属性，作为参考权重
            reference_weights = es.weight.detach().requires_grad_()

            # 使用自定义实现函数 _embedding_bag_reference_impl 计算期望输出
            expected = self._embedding_bag_reference_impl(
                input, reference_weights, offsets, mode, ref_per_sample_weights
            )
            # 调用 EmbeddingBag 实例计算结果
            result = es(input, offsets, per_sample_weights)
            # 断言实际结果与期望输出相等，使用指定的数值精度要求
            self.assertEqual(
                result, expected, atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0
            )

            # 创建与期望输出形状相同的随机梯度张量
            grad = torch.randn_like(expected).to(dtype=dtypes[2], device=device)
            # 对结果张量进行反向传播，计算梯度
            result.backward(grad)
            # 对期望输出张量进行反向传播，计算梯度
            expected.backward(grad)
            # 断言 EmbeddingBag 实例权重的梯度与参考权重的梯度相等，使用指定的数值精度要求
            self.assertEqual(
                es.weight.grad,
                reference_weights.grad,
                atol=dtype2prec_DONTUSE[dtypes[2]],
                rtol=0,
            )
            # 如果可训练标度为真，则断言每样本权重的梯度与参考每样本权重的梯度相等，使用指定的数值精度要求
            if trainable_scale:
                self.assertEqual(
                    per_sample_weights.grad,
                    ref_per_sample_weights.grad,
                    atol=dtype2prec_DONTUSE[dtypes[2]],
                    rtol=0,
                )

        # 定义模式集合为 "sum"
        modes = ("sum",)
        # 定义可训练标度集合为 True 和 False 的组合
        trainable_scale = (True, False)
        # 对模式和可训练标度进行组合测试
        for mode, trainable in itertools.product(modes, trainable_scale):
            test_per_sample_weights(mode, trainable)

    # 跳过元数据测试的装饰器
    @skipMeta
    # 指定数据类型的装饰器，用于不同类型的组合
    @dtypes(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half, torch.bfloat16),
        )
    )
    # 如果是 CUDA 设备，指定数据类型的装饰器，用于不同类型的组合
    @dtypesIfCUDA(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half),
        )
    )
    # 定义测试函数，测试EmbeddingBag类的使用，验证不同参数组合下的计算结果
    def test_EmbeddingBag_per_sample_weights_and_new_offsets(self, device, dtypes):
        # 定义内部测试函数，用于具体验证带有样本权重和新偏移量的情况
        def test_per_sample_weights_new_offsets(
            mode, trainable_scale, include_last_offset, has_weight=True
        ):
            # 创建EmbeddingBag对象，设置相关参数并移动到指定设备上
            es = nn.EmbeddingBag(
                5, 2, mode=mode, include_last_offset=include_last_offset
            ).to(dtype=dtypes[2], device=device)
            # 将权重数据初始化为从1到10的序列，并与EmbeddingBag对象的权重数据形状相匹配
            es.weight.data.copy_(
                torch.arange(1, 11, device=device).view_as(es.weight).to(dtypes[2])
            )
            # 创建输入张量，表示要嵌入的索引序列，并移动到指定设备上
            input = torch.tensor([3, 1, 1, 1, 4, 0], device=device, dtype=dtypes[0])
            # 创建偏移量张量，指定每个样本的起始偏移量，并移动到指定设备上
            offsets = torch.tensor([0, 0, 3, 3, 6], device=device, dtype=dtypes[1])

            # 如果设置了include_last_offset标志，则添加最后一个偏移量
            if include_last_offset:
                offsets = torch.cat(
                    (
                        offsets,
                        torch.tensor([input.size(0)], device=device, dtype=dtypes[1]),
                    ),
                    0,
                )

            # 如果有权重，则创建随机样本权重张量，并设置是否需要梯度计算
            if has_weight:
                per_sample_weights = torch.randn_like(
                    input, device=device, dtype=dtypes[2]
                ).requires_grad_(trainable_scale)
                # 创建参考的样本权重张量，并设置是否需要梯度计算
                ref_per_sample_weights = per_sample_weights.detach().requires_grad_(
                    trainable_scale
                )
            else:
                per_sample_weights = None
                ref_per_sample_weights = None

            # 创建参考的权重张量，并设置是否需要梯度计算
            reference_weights = es.weight.detach().requires_grad_()

            # 调用内部的参考实现函数，计算期望的嵌入结果
            expected = self._embedding_bag_reference_impl(
                input,
                reference_weights,
                offsets,
                mode,
                ref_per_sample_weights,
                include_last_offset,
            )
            # 调用EmbeddingBag对象，计算实际的嵌入结果
            result = es(input, offsets, per_sample_weights)
            # 断言实际计算结果与期望结果相等，设置绝对误差和相对误差的阈值
            self.assertEqual(
                result, expected, atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0
            )

            # 创建随机梯度张量
            grad = torch.randn_like(expected)
            # 计算嵌入结果的反向传播，并验证权重梯度是否正确
            result.backward(grad)
            expected.backward(grad)
            self.assertEqual(
                es.weight.grad,
                reference_weights.grad,
                atol=dtype2prec_DONTUSE[dtypes[2]],
                rtol=0,
            )
            # 如果有权重且需要梯度计算，则验证样本权重梯度是否正确
            if has_weight and trainable_scale:
                self.assertEqual(
                    per_sample_weights.grad,
                    ref_per_sample_weights.grad,
                    atol=dtype2prec_DONTUSE[dtypes[2]],
                    rtol=0,
                )

        # 定义训练尺度和是否包含最后偏移的列表
        trainable_scale = (True, False)
        include_last_offset_list = (True, False)
        # 定义嵌入模式和是否有权重的元组列表
        modes = (("sum", False), ("sum", True), ("max", False), ("mean", False))
        # 遍历所有参数组合，调用内部测试函数进行验证
        for (mode, has_weight), trainable, include_last_offset in itertools.product(
            modes, trainable_scale, include_last_offset_list
        ):
            test_per_sample_weights_new_offsets(
                mode, trainable, include_last_offset, has_weight
            )
    # 定义一个名为 _test_EmbeddingBag_vs_Embedding 的测试函数，用于比较 EmbeddingBag 和 Embedding 的性能差异
    def _test_EmbeddingBag_vs_Embedding(
        # 参数 N：表示样本数量
        self,
        # 参数 D：表示嵌入向量的维度
        N,
        # 参数 B：表示嵌入袋（EmbeddingBag）的模型数量
        D,
        # 参数 L：表示输入序列的最大长度
        B,
        # 参数 max_norm：表示嵌入向量的最大范数，可选，默认为 None
        max_norm=None,
        # 参数 mode：表示嵌入求和模式，可选，默认为 "mean"
        mode="mean",
        # 参数 device：表示设备类型，可选，默认为 "cpu"
        device="cpu",
        # 参数 wdtype：表示权重的数据类型，可选，默认为 torch.float
        wdtype=torch.float,
        # 参数 dtype：表示索引的数据类型，可选，默认为 torch.long
        dtype=torch.long,
        # 参数 test_per_sample_weights：表示是否测试每个样本的权重，可选，默认为 False
        test_per_sample_weights=False,
        # 参数 trainable_per_sample_weights：表示每个样本的权重是否可训练，可选，默认为 False
        trainable_per_sample_weights=False,
        # 参数 sparse：表示是否使用稀疏格式，可选，默认为 False
        sparse=False,
        # 参数 test_backward：表示是否测试反向传播，可选，默认为 True
        test_backward=True,
        # 参数 backward_prec：表示反向传播的精度，可选，默认为 None
        backward_prec=None,
    ):
        # 根据参数定义创建一个 EmbeddingBag 对象，并移动到指定的设备和数据类型上
        es = nn.EmbeddingBag(N, D, mode=mode, sparse=sparse, max_norm=max_norm).to(
            device, wdtype
        )
        # 根据参数定义创建一个 Embedding 对象，并移动到指定的设备和数据类型上
        e = nn.Embedding(N, D, max_norm=max_norm).to(device, wdtype)
        # 将 EmbeddingBag 的权重复制给 Embedding
        e.weight.data.copy_(es.weight)
        # 生成一个指定范围内的随机整数张量，用于输入
        input = torch.randint(N, (B, L), device=device, dtype=dtype)
        # 创建一个偏移量张量，用于指定每个“包”在 input 中的起始位置
        offsets = torch.arange(0, B, device=device, dtype=dtype).mul_(L)
        # 创建一个指定形状的随机张量，用作梯度输出
        grad_output = torch.rand(B, D, device=device, dtype=wdtype)

        if test_per_sample_weights:
            # 如果需要测试每个样本的权重，对权重进行 softmax 处理确保和为1
            per_sample_weights = torch.randn(B, L, device=device, dtype=wdtype).softmax(
                dim=-1
            )
            # 克隆并设置可训练的参考权重
            per_sample_weights_reference = per_sample_weights.clone().requires_grad_(
                trainable_per_sample_weights
            )
            # 设置当前权重为可训练状态
            per_sample_weights.requires_grad_(trainable_per_sample_weights)
            # 使用 EmbeddingBag 计算输出，根据需要展平输入和权重
            output = es(input.view(-1), offsets, per_sample_weights.view(-1))
        else:
            # 否则，直接使用 EmbeddingBag 计算输出，展平输入和偏移量
            output = es(input.view(-1), offsets)
            per_sample_weights = None
            per_sample_weights_reference = None

        if mode == "sum":
            if test_per_sample_weights:
                # 如果模式为 sum，并且需要测试每个样本的权重，计算参考输出
                ref_output = (
                    e(input) * per_sample_weights_reference.unsqueeze(-1)
                ).sum(1)
            else:
                # 否则，直接计算参考输出
                ref_output = e(input).sum(1)
        elif mode == "mean":
            # 如果模式为 mean，确保不需要测试每个样本的权重，计算参考输出
            assert not test_per_sample_weights
            ref_output = e(input).mean(1)
        elif mode == "max":
            # 如果模式为 max，确保不需要测试每个样本的权重，计算参考输出
            assert not test_per_sample_weights
            ref_output = e(input).max(1)[0]

        # 使用断言检查输出与参考输出的一致性，设置绝对误差和相对误差容忍度
        self.assertEqual(output, ref_output, atol=dtype2prec_DONTUSE[wdtype], rtol=0)

        # 如果不需要进行反向传播测试，则直接返回
        if not test_backward:
            return

        # 对输出进行反向传播，计算梯度
        output.backward(grad_output)
        ref_output.backward(grad_output)
        # 获取 EmbeddingBag 权重的梯度
        es_weight_grad = es.weight.grad
        # 如果使用了稀疏张量，则将稀疏梯度转换为密集梯度
        if sparse:
            es_weight_grad = es.weight.grad.to_dense()

        # 根据需要设置绝对误差和相对误差容忍度
        if backward_prec is None:
            needed_prec = dtype2prec_DONTUSE[wdtype] * 5
            rtol = 0.02 if wdtype == torch.half else 0
        else:
            needed_prec = backward_prec
            rtol = 0

        # 使用断言检查 EmbeddingBag 权重梯度与 Embedding 权重梯度的一致性
        self.assertEqual(es_weight_grad, e.weight.grad, atol=needed_prec, rtol=rtol)

        # 如果需要测试每个样本的权重，并且权重是可训练的，则检查权重梯度的一致性
        if test_per_sample_weights and trainable_per_sample_weights:
            self.assertEqual(
                per_sample_weights.grad,
                per_sample_weights_reference.grad,
                atol=dtype2prec_DONTUSE[wdtype],
                rtol=0,
            )

    # 用于定义不同 CUDA 设备上的数据类型组合的装饰器
    @dtypesIfCUDA(
        *itertools.product(
            (torch.int, torch.long), (torch.half, torch.float, torch.double)
        )
    )
    # 用于定义不同设备上的数据类型组合的装饰器
    @dtypes(*itertools.product((torch.int, torch.long), (torch.float, torch.double)))
    # 定义测试函数，测试带有不同参数组合的 EmbeddingBag 函数
    def test_EmbeddingBag_per_sample_weights_and_no_offsets(self, device, dtypes):
        # 内部函数，运行特定模式下的测试
        def run_tests(mode, sparse, trainable_per_sample_weights):
            # 设置测试参数
            kwargs = dict(
                test_per_sample_weights=True,  # 测试是否使用每个样本权重
                device=device,  # 设备类型，如 CPU 或 CUDA
                mode=mode,  # EmbeddingBag 的模式，如 'sum'
                wdtype=dtypes[1],  # 权重的数据类型
                dtype=dtypes[0],  # 输入的数据类型
                sparse=sparse,  # 是否稀疏输入
                trainable_per_sample_weights=trainable_per_sample_weights,  # 是否训练每个样本的权重
            )

            # 测试简单情况下的 EmbeddingBag 函数
            self._test_EmbeddingBag_vs_Embedding(2, 3, 5, 7, **kwargs)

            # 测试 B * L > 1000 的情况下的 EmbeddingBag 函数
            self._test_EmbeddingBag_vs_Embedding(2, 5, 53, 23, **kwargs)

            # 测试大 num_embedding 的情况下的 EmbeddingBag 函数
            self._test_EmbeddingBag_vs_Embedding(101, 5, 3, 7, **kwargs)

            # 测试大 embedding_dim 的情况下的 EmbeddingBag 函数
            self._test_EmbeddingBag_vs_Embedding(2, 101, 3, 7, **kwargs)

        # 定义测试模式和稀疏性
        modes = ("sum",)
        sparsity = (True, False)
        trainable_scale = (True, False)
        
        # 遍历模式、稀疏性和是否训练每个样本权重的所有组合，执行测试
        for mode, sparse, trainable_per_sample_weights in itertools.product(
            modes, sparsity, trainable_scale
        ):
            run_tests(mode, sparse, trainable_per_sample_weights)

        # 在 CUDA 环境下测试 Dense 的半精度
        if device == "cuda":
            modes = ("sum",)
            sparsity = (False,)
            trainable_scale = (True, False)
            for mode, sparse, trainable_per_sample_weights in itertools.product(
                modes, sparsity, trainable_scale
            ):
                run_tests(mode, sparse, trainable_per_sample_weights)

    # 定义测试 EmbeddingBag 函数的基本参数和选项
    def _test_EmbeddingBag(
        self,
        device,
        mode,
        sparse,
        wdtype=torch.double,  # 权重数据类型，默认为双精度
        dtype=torch.long,  # 输入数据类型，默认为长整型
        odtype=torch.long,  # 输出数据类型，默认为长整型
        test_backward=True,  # 是否测试反向传播，默认为 True
    ):
    # 测试嵌入包在指定设备和数据类型下的行为
    def test_embedding_bag_device(self, device, dtypes):
        # 如果是 Jetson 平台且数据类型包含 torch.bfloat16，并且设备是 CPU，则跳过测试
        if IS_JETSON and torch.bfloat16 in dtypes and device == "cpu":
            self.skipTest("bfloat16 not supported with Jetson cpu")
        
        # 设置默认数据类型为 torch.double
        with set_default_dtype(torch.double):
            # 测试 EmbeddingBag 在指定设备上进行求和操作的行为
            self._test_EmbeddingBag(
                device,
                "sum",
                False,
                wdtype=dtypes[2],  # 权重数据类型
                dtype=dtypes[0],   # 输入数据类型
                odtype=dtypes[1],  # 输出数据类型
            )
            # 测试 EmbeddingBag 在指定设备上进行均值操作的行为
            self._test_EmbeddingBag(
                device,
                "mean",
                False,
                wdtype=dtypes[2],  # 权重数据类型
                dtype=dtypes[0],   # 输入数据类型
                odtype=dtypes[1],  # 输出数据类型
            )
            # 测试 EmbeddingBag 在指定设备上进行最大值操作的行为
            self._test_EmbeddingBag(
                device,
                "max",
                False,
                wdtype=dtypes[2],  # 权重数据类型
                dtype=dtypes[0],   # 输入数据类型
                odtype=dtypes[1],  # 输出数据类型
            )

            # 初始化是否测试反向传播为 False
            test_backward = False
            if self.device_type == "cuda":
                # 如果设备类型是 CUDA，根据条件确定是否需要进行反向传播测试
                # 查看 'test_embedding_bag' 中的 'todo'
                test_backward = dtypes[2] is not torch.float16
            elif self.device_type == "cpu":
                # 如果设备类型是 CPU，TODO：找出为什么稀疏嵌入的精度与密集嵌入不同的原因
                test_backward = (
                    dtypes[2] is not torch.float and dtypes[2] is not torch.float16
                )

            # 测试 EmbeddingBag 在指定设备上进行求和操作的行为，包括反向传播测试
            self._test_EmbeddingBag(
                device,
                "sum",
                True,
                wdtype=dtypes[2],          # 权重数据类型
                dtype=dtypes[0],           # 输入数据类型
                odtype=dtypes[1],          # 输出数据类型
                test_backward=test_backward  # 是否进行反向传播测试
            )
            # 测试 EmbeddingBag 在指定设备上进行均值操作的行为，包括反向传播测试
            self._test_EmbeddingBag(
                device,
                "mean",
                True,
                wdtype=dtypes[2],          # 权重数据类型
                dtype=dtypes[0],           # 输入数据类型
                odtype=dtypes[1],          # 输出数据类型
                test_backward=test_backward  # 是否进行反向传播测试
            )

    # 标记为跳过元信息测试
    @skipMeta
    # 定义测试的数据类型组合，包括不同的整数类型和浮点数类型
    @dtypes(
        *itertools.product(
            (torch.int, torch.long),    # 输入数据类型
            (torch.int, torch.long),    # 输出数据类型
            (torch.float, torch.double, torch.half, torch.bfloat16),  # 权重数据类型
        )
    )
    # 如果是 CUDA 设备，定义测试的数据类型组合，包括不同的整数类型和浮点数类型
    @dtypesIfCUDA(
        *itertools.product(
            (torch.int, torch.long),    # 输入数据类型
            (torch.int, torch.long),    # 输出数据类型
            (torch.float, torch.double, torch.half),  # 权重数据类型
        )
    )
    # 测试非连续权重的嵌入包功能
    def test_embedding_bag_non_contiguous_weight(self, device, dtypes):
        # 创建一个指定设备和数据类型的随机张量作为权重
        weight_tensor = torch.randn(3, 4, dtype=dtypes[2], device=device)

        # 获取权重张量的非连续部分（非连续步幅）
        weight_tensor_non_contig = weight_tensor[
            :, :3
        ]  # 这是非连续的步幅。

        # 克隆并使权重张量连续化
        weight_tensor_contig = (
            weight_tensor_non_contig.clone().contiguous()
        )  # 连续的步幅。

        # 创建索引张量和偏移量张量，用于嵌入包函数的输入
        index = torch.tensor([0, 1, 2], dtype=dtypes[0], device=device)
        offsets = torch.tensor([0, 2], dtype=dtypes[1], device=device)

        # 对每种模式（"sum"、"mean"、"max"）分别执行嵌入包操作
        for mode in ["sum", "mean", "max"]:
            # 使用非连续步幅的权重张量进行嵌入包计算
            output_non_contig = F.embedding_bag(
                input=index,
                weight=weight_tensor_non_contig,
                offsets=offsets,
                mode=mode,
            )
            # 使用连续步幅的权重张量进行嵌入包计算
            output_contig = F.embedding_bag(
                input=index,
                weight=weight_tensor_contig,
                offsets=offsets,
                mode=mode,
            )

        # 断言非连续步幅和连续步幅计算的输出结果是否相等
        self.assertEqual(output_non_contig, output_contig)

    @onlyNativeDeviceTypes  # 目前在 XLA 上失败
    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long)))
    # 测试 bfloat16 类型的嵌入包功能
    def test_embedding_bag_bfloat16(self, device, dtypes):
        with set_default_dtype(torch.double):
            # 调用 _test_EmbeddingBag 方法测试嵌入包功能
            self._test_EmbeddingBag(
                device,
                "sum",
                True,
                wdtype=torch.bfloat16,
                dtype=dtypes[0],
                odtype=dtypes[1],
                test_backward=True,
            )
            # 调用 _test_EmbeddingBag 方法测试嵌入包功能
            self._test_EmbeddingBag(
                device,
                "mean",
                True,
                wdtype=torch.bfloat16,
                dtype=dtypes[0],
                odtype=dtypes[1],
                test_backward=True,
            )

    @onlyNativeDeviceTypes  # 目前在 XLA 上失败
    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long)))
    # 测试半精度浮点类型的嵌入包功能
    def test_embedding_bag_half(self, device, dtypes):
        # 调用 _test_EmbeddingBag 方法测试嵌入包功能
        self._test_EmbeddingBag(
            device,
            "sum",
            True,
            wdtype=torch.float16,
            dtype=dtypes[0],
            odtype=dtypes[1],
            test_backward=True,
        )
# 调用函数 instantiate_device_type_tests，初始化设备类型测试，传入测试类 TestEmbeddingNNDeviceType 和全局作用域
instantiate_device_type_tests(TestEmbeddingNNDeviceType, globals())

# 调用函数 instantiate_parametrized_tests，初始化参数化测试，传入测试类 TestEmbeddingNN
instantiate_parametrized_tests(TestEmbeddingNN)

# 检查当前脚本是否作为主程序运行，如果是则执行测试
if __name__ == "__main__":
    # 运行测试函数
    run_tests()
```