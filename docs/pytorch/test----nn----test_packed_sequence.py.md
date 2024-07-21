# `.\pytorch\test\nn\test_packed_sequence.py`

```py
# Owner(s): ["module: nn"]

import itertools  # 导入 itertools 库，用于迭代操作
import random  # 导入 random 库，用于生成随机数

import torch  # 导入 PyTorch 库
import torch.nn.utils.rnn as rnn_utils  # 导入 PyTorch 中的 rnn_utils 模块，用于处理序列的工具函数
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入测试相关的函数和类


class PackedSequenceTest(TestCase):
    _type_by_name = {
        "torch.DoubleTensor": (torch.DoubleTensor, "double"),  # 定义双精度张量的类型及其字符串表示
        "torch.FloatTensor": (torch.FloatTensor, "float"),  # 定义单精度张量的类型及其字符串表示
        # 'torch.HalfTensor': (torch.HalfTensor, 'half') 的定义被省略，因为 `pad_packed_sequence` 存在问题
        # > AttributeError: 'torch.HalfTensor' object has no attribute 'fill_'
        "torch.LongTensor": (torch.LongTensor, "long"),  # 定义长整型张量的类型及其字符串表示
        "torch.IntTensor": (torch.IntTensor, "int"),  # 定义整型张量的类型及其字符串表示
        "torch.ShortTensor": (torch.ShortTensor, "short"),  # 定义短整型张量的类型及其字符串表示
        "torch.CharTensor": (torch.CharTensor, "char"),  # 定义字符型张量的类型及其字符串表示
        "torch.ByteTensor": (torch.ByteTensor, "byte"),  # 定义字节型张量的类型及其字符串表示
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 5  # 初始化批量大小为 5
        self.max_length = 6  # 初始化最大序列长度为 6

    def _ordered_sequence(self, tensor_type):
        """Create ordered list of random sequences"""
        # 生成随机序列的有序列表
        seqs = [
            tensor_type(random.randint(1, self.max_length))  # 使用给定的张量类型创建随机长度的张量
            for _ in range(self.batch_size)  # 循环创建批量大小次数
        ]
        if tensor_type == torch.ByteTensor:
            seqs = [s.random_(0, 256) for s in seqs]  # 如果是字节张量，将其值随机设置在 [0, 256)
        else:
            seqs = [s.random_(-128, 128) for s in seqs]  # 否则将其值随机设置在 [-128, 128)
        ordered = sorted(seqs, key=len, reverse=True)  # 根据序列长度降序排序
        return ordered

    def _padded_sequence(self, tensor_type):
        """Create Tensor of random padded sequences"""
        ordered = self._ordered_sequence(tensor_type)  # 获取有序的随机序列
        lengths = [len(i) for i in ordered]  # 获取每个序列的长度
        padded_tensor = rnn_utils.pad_sequence(ordered)  # 对序列进行填充，生成填充后的张量
        return padded_tensor, lengths  # 返回填充后的张量和每个序列的长度列表

    def test_type_casts(self):
        """Test type casting of `PackedSequence` against type casting of tensor"""
        for input_type, _ in self._type_by_name.values():  # 遍历所有输入类型
            for expected_type_str, (_, cast_str) in self._type_by_name.items():  # 遍历所有期望的类型字符串
                for enforce_sorted in [True, False]:  # 对于每个是否强制排序的选项
                    padded, lengths = self._padded_sequence(input_type)  # 获取填充后的序列和长度
                    packed = rnn_utils.pack_padded_sequence(
                        padded, lengths, enforce_sorted=enforce_sorted  # 打包填充后的序列
                    )
                    # 对 `PackedSequence` 实例应用类型转换并解包
                    masked = getattr(packed, cast_str)()  # 调用相应的类型转换函数
                    unpacked, lengths_out = rnn_utils.pad_packed_sequence(masked)  # 解包成填充后的张量
                    self.assertEqual(unpacked.type(), expected_type_str)  # 断言解包后的张量类型与期望的类型字符串相等

    def test_wrong_order(self):
        a = torch.ones(25, 300)  # 创建形状为 (25, 300) 的全 1 张量 a
        b = torch.ones(22, 300)  # 创建形状为 (22, 300) 的全 1 张量 b
        b_a = rnn_utils.pad_sequence([b, a])  # 对张量 b 和 a 进行填充
        self.assertRaises(
            RuntimeError,
            lambda: rnn_utils.pack_padded_sequence(b_a, [22, 25], enforce_sorted=True),  # 尝试打包填充后的序列，期望引发 RuntimeError
        )
    # 定义测试函数，用于测试使用张量序列作为输入的 pad_sequence 函数
    def test_pad_sequence_with_tensor_sequences(self):
        # 创建一个包含两个张量的元组，并进行填充序列操作
        seq_tuple_input = torch.nn.utils.rnn.pad_sequence(
            (torch.tensor([[7, 6]]), torch.tensor([[-7, -1]]))
        )
        # 创建一个包含两个张量的列表，并进行填充序列操作
        seq_tensor_input = torch.nn.utils.rnn.pad_sequence(
            torch.tensor([[[7, 6]], [[-7, -1]]])
        )
        # 断言两种不同输入方式得到的结果相等
        self.assertEqual(seq_tuple_input, seq_tensor_input)
        # 断言填充后的张量形状为 [1, 2, 2]
        self.assertEqual(seq_tuple_input.shape, torch.Size([1, 2, 2]))

    # 定义测试函数，用于测试 pad_sequence 函数对非可迭代序列的处理
    def test_pad_sequence_with_non_iterable_sequences(self):
        # 期望捕获 RuntimeError，并验证错误消息格式是否符合预期
        msg = r"Expected iterable for input sequences, but got arg of type"
        with self.assertRaisesRegex(RuntimeError, msg):
            # 调用 pad_sequence 函数，传入一个非可迭代对象（整数）
            torch.nn.utils.rnn.pad_sequence(5)

    # 定义测试函数，用于测试总长度参数（total_length）对 pad_packed_sequence 函数的影响
    def test_total_length(self):
        # 调用内部方法 _padded_sequence 返回填充后的序列和长度列表
        padded, lengths = self._padded_sequence(torch.FloatTensor)
        # 获取长度列表中的最大长度
        max_length = max(lengths)
        # 使用 pack_padded_sequence 函数打包填充后的序列
        packed = rnn_utils.pack_padded_sequence(padded, lengths)

        # 测试当 total_length < max_length 时，是否会抛出 ValueError 异常
        for total_length in (-1, 0, max_length - 1):
            for batch_first in (True, False):
                # 定义内部函数 err_fn，用于捕获 ValueError 异常
                def err_fn():
                    rnn_utils.pad_packed_sequence(
                        packed, batch_first=batch_first, total_length=total_length
                    )

                # 断言是否捕获到预期的 ValueError 异常，并验证错误消息格式
                self.assertRaisesRegex(
                    ValueError,
                    r"Expected total_length to be at least the "
                    r"length of the longest sequence in input",
                    err_fn,
                )

        # 测试 pad_packed_sequence 函数返回的解包结果是否具有正确的长度
        for batch_first in (True, False):
            # 调用 pad_packed_sequence 函数解包填充后的序列
            no_extra_pad, _ = rnn_utils.pad_packed_sequence(
                packed, batch_first=batch_first
            )
            # 遍历不同的 total_length_delta 值
            for total_length_delta in (0, 1, 8):
                # 计算当前的 total_length 值
                total_length = max_length + total_length_delta
                # 调用 pad_packed_sequence 函数解包填充后的序列
                unpacked, lengths_out = rnn_utils.pad_packed_sequence(
                    packed, batch_first=batch_first, total_length=total_length
                )
                # 断言解包后的长度列表与原始长度列表相等
                self.assertEqual(lengths, lengths_out)
                # 断言解包后的张量维度与指定的 total_length 相符
                self.assertEqual(unpacked.size(1 if batch_first else 0), total_length)
                # 根据 total_length_delta 的不同情况，生成参考输出 ref_output
                if total_length_delta == 0:
                    ref_output = no_extra_pad
                elif batch_first:
                    extra_pad = no_extra_pad.new_zeros(
                        self.batch_size, total_length_delta
                    )
                    ref_output = torch.cat([no_extra_pad, extra_pad], 1)
                else:
                    extra_pad = no_extra_pad.new_zeros(
                        total_length_delta, self.batch_size
                    )
                    ref_output = torch.cat([no_extra_pad, extra_pad], 0)
                # 断言解包后的张量与参考输出 ref_output 相等
                self.assertEqual(unpacked, ref_output)
    # 定义一个测试函数，测试数据转换操作
    def test_to(self):
        # 遍历 enforce_sorted 参数的两种取值 True 和 False
        for enforce_sorted in (True, False):
            # 调用 _padded_sequence 方法获取填充后的序列和长度
            padded, lengths = self._padded_sequence(torch.IntTensor)
            # 使用 rnn_utils.pack_padded_sequence 进行序列打包，并转移到 CPU
            a = rnn_utils.pack_padded_sequence(
                padded, lengths, enforce_sorted=enforce_sorted
            ).cpu()

            # 断言 a 和 a.to("cpu") 返回的对象是同一个对象
            self.assertIs(a, a.to("cpu"))
            # 断言 a 和 a.cpu() 返回的对象是同一个对象
            self.assertIs(a, a.cpu())
            # 断言 a 和 a.to("cpu", dtype=torch.int32) 返回的对象是同一个对象
            self.assertIs(a, a.to("cpu", dtype=torch.int32))
            # 断言 a.long() 和 a.to(torch.int64) 返回的对象相等
            self.assertEqual(a.long(), a.to(torch.int64))

            # 如果 CUDA 可用，进入 CUDA 相关的测试
            if torch.cuda.is_available():
                # 遍历不同的 CUDA 设备
                for cuda in [
                    "cuda",
                    "cuda:0" if torch.cuda.device_count() == 1 else "cuda:1",
                ]:
                    # 将 a 转移到指定的 CUDA 设备，使用指定的设备名 cuda
                    b = a.cuda(device=cuda)
                    # 断言 b 和 b.to(cuda) 返回的对象是同一个对象
                    self.assertIs(b, b.to(cuda))
                    # 断言 b 和 b.cuda() 返回的对象是同一个对象
                    self.assertIs(b, b.cuda())
                    # 断言 a 和 b.to("cpu") 返回的对象相等
                    self.assertEqual(a, b.to("cpu"))
                    # 断言 b 和 a.to(cuda) 返回的对象相等
                    self.assertEqual(b, a.to(cuda))
                    # 断言 a 和 b.to("cpu", dtype=torch.int32) 返回的对象相等
                    self.assertEqual(a, b.to("cpu", dtype=torch.int32))
                    # 断言 b 和 b.to(dtype=torch.int32) 返回的对象是同一个对象
                    self.assertIs(b, b.to(dtype=torch.int32))
                    # 断言 b.long() 和 b.to(dtype=torch.int64) 返回的对象相等
                    self.assertEqual(b.long(), b.to(dtype=torch.int64))

    # 定义一个测试函数，测试内存格式设置操作
    def test_to_memory_format(self):
        # 创建一个 Conv2d 模块，设置输入通道数、输出通道数、卷积核大小和是否带偏置
        m = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, bias=True)
        # 将模块 m 转移到通道优先的内存格式
        m = m.to(memory_format=torch.channels_last)
        # 遍历模块 m 的参数
        for param in m.parameters():
            # 如果参数的维度是 4
            if param.dim() == 4:
                # 断言参数在通道优先的内存格式下是连续的
                self.assertTrue(param.is_contiguous(memory_format=torch.channels_last))
    # 定义一个测试函数，用于测试序列填充的功能
    def test_pad_sequence(self):
        # 定义一个内部函数pad，用于将张量填充到指定长度
        def pad(tensor, length):
            return torch.cat(
                [
                    tensor.data,
                    # 创建一个与原张量相同类型和形状的全零张量，用于填充
                    tensor.data.new(length - tensor.size(0), *tensor.size()[1:]).zero_(),
                ]
            )

        # 单维度张量的测试数据
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        c = torch.tensor([6])

        # 当batch_first = True时的预期结果
        expected = torch.tensor([[4, 5, 0], [1, 2, 3], [6, 0, 0]])
        # 调用pad_sequence进行填充，并断言结果与预期一致
        padded = rnn_utils.pad_sequence([b, a, c], True)
        self.assertEqual(padded, expected)

        # 当batch_first = False时的填充测试
        padded = rnn_utils.pad_sequence([b, a, c])
        # 断言结果与预期的转置一致
        self.assertEqual(padded, expected.transpose(0, 1))

        # 使用非零值进行填充的测试
        expected = torch.tensor([[4, 5, 1], [1, 2, 3], [6, 1, 1]])
        padded = rnn_utils.pad_sequence([b, a, c], True, 1)
        self.assertEqual(padded, expected)

        # 测试按排序后的序列进行填充
        expected = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]])
        padded = rnn_utils.pad_sequence([a, b, c], True)
        self.assertEqual(padded, expected)

        # 多维度张量的测试
        maxlen = 9
        for num_dim in (0, 1, 2, 3):
            sequences = []
            trailing_dims = [4] * num_dim
            # 生成不同长度的随机张量序列
            for i in range(1, maxlen + 1):
                seq_len = i * i
                sequences.append(torch.rand(seq_len, 5, *trailing_dims))
            # 随机打乱序列顺序
            random.shuffle(sequences)
            expected = []
            # 对每个序列进行填充操作
            for seq in sequences:
                expected.append(pad(seq, maxlen * maxlen))
            # 当batch_first = True时的测试
            expected = torch.stack(expected)
            padded = rnn_utils.pad_sequence(sequences, True)
            self.assertEqual(padded, expected)

            # 当batch_first = False时的测试
            padded = rnn_utils.pad_sequence(sequences)
            self.assertEqual(padded, expected.transpose(0, 1))
    # 定义一个测试函数，用于测试解除填充（unpad）序列的功能
    def test_unpad_sequence(self):
        # 创建三个单维张量作为测试数据
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        c = torch.tensor([6])
        sequences = [a, b, c]

        # 计算每个序列的长度并转换为张量
        lengths = torch.as_tensor([v.size(0) for v in sequences])
        
        # 遍历两种不同的 batch_first 参数值进行测试
        for batch_first in [True, False]:
            # 对序列进行填充操作，根据 batch_first 参数决定填充方式
            padded_sequences = rnn_utils.pad_sequence(
                sequences, batch_first=batch_first
            )
            # 解除填充操作，根据长度信息和 batch_first 参数恢复原始序列
            unpadded_sequences = rnn_utils.unpad_sequence(
                padded_sequences, lengths, batch_first=batch_first
            )
            # 断言解除填充后的序列与原始序列相等
            self.assertEqual(sequences, unpadded_sequences)

        # 测试更高维度的情况
        maxlen = 9
        for num_dim in (0, 1, 2, 3):
            sequences = []
            trailing_dims = [4] * num_dim
            # 创建具有不同长度和维度的随机张量序列并随机排序
            for i in range(1, maxlen + 1):
                seq_len = i * i
                sequences.append(torch.rand(seq_len, 5, *trailing_dims))
            random.shuffle(sequences)

            # 计算每个序列的长度并转换为张量
            lengths = torch.as_tensor([v.size(0) for v in sequences])
            
            # 对序列进行填充操作，根据 batch_first 参数决定填充方式
            padded_sequences = rnn_utils.pad_sequence(
                sequences, batch_first=batch_first
            )
            # 解除填充操作，根据长度信息和 batch_first 参数恢复原始序列
            unpadded_sequences = rnn_utils.unpad_sequence(
                padded_sequences, lengths, batch_first=batch_first
            )
            # 断言解除填充后的序列与原始序列相等
            self.assertEqual(sequences, unpadded_sequences)

    # 定义一个测试函数，用于测试打包（pack）序列的功能
    def test_unpack_sequence(self):
        # 创建三个单维张量作为测试数据
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        c = torch.tensor([6])
        sequences = [a, b, c]

        # 打包序列，不强制排序
        packed_sequences = rnn_utils.pack_sequence(sequences, enforce_sorted=False)
        # 解包序列
        unpacked_sequences = rnn_utils.unpack_sequence(packed_sequences)
        # 断言解包后的序列与原始序列相等
        self.assertEqual(sequences, unpacked_sequences)

        # 测试更高维度的情况
        maxlen = 9
        for num_dim in (0, 1, 2, 3):
            sequences = []
            trailing_dims = [4] * num_dim
            # 创建具有不同长度和维度的随机张量序列并随机排序
            for i in range(1, maxlen + 1):
                seq_len = i * i
                sequences.append(torch.rand(seq_len, 5, *trailing_dims))
            random.shuffle(sequences)

            # 打包序列，不强制排序
            packed_sequences = rnn_utils.pack_sequence(sequences, enforce_sorted=False)
            # 解包序列
            unpacked_sequences = rnn_utils.unpack_sequence(packed_sequences)
            # 断言解包后的序列与原始序列相等
            self.assertEqual(sequences, unpacked_sequences)
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```