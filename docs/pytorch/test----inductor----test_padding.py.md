# `.\pytorch\test\inductor\test_padding.py`

```py
# Owner(s): ["module: inductor"]

# 导入必要的模块和库
import copy  # 导入 copy 模块用于深拷贝
import functools  # 导入 functools 模块用于高阶函数
import os  # 导入 os 模块提供操作系统相关的功能
import unittest  # 导入 unittest 模块用于编写和运行测试

import torch  # 导入 PyTorch 深度学习库
from torch import nn, Tensor  # 导入神经网络模块及 Tensor 数据类型
from torch._dynamo.convert_frame import maybe_cprofile  # 导入 Dynamo 模块中的性能分析工具
from torch._dynamo.test_case import run_tests, TestCase  # 导入 Dynamo 模块中的测试运行器和测试用例基类
from torch._dynamo.testing import rand_strided, reduce_to_scalar_loss  # 导入 Dynamo 模块中的测试工具
from torch._inductor import config, ir, metrics  # 导入 Inductor 模块的配置、IR、度量等
from torch._inductor.fx_passes import pad_mm as pad_mm_pass  # 导入 Inductor 模块中的 FX passes
from torch._inductor.runtime.runtime_utils import do_bench  # 导入 Inductor 运行时工具函数
from torch._inductor.utils import run_and_get_code  # 导入 Inductor 工具函数
from torch.testing._internal.common_utils import serialTest  # 导入内部测试工具中的 serialTest
from torch.testing._internal.inductor_utils import HAS_CUDA  # 导入 Inductor 工具中的 CUDA 判断

# 设置性能测试和精度测试的环境变量标志
DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"
DO_ACC_TEST = os.environ.get("DO_ACC_TEST", "1") == "1"
WITH_STACK = os.environ.get("WITH_STACK") == "1"
USE_CUDA_GRAPHS = os.environ.get("USE_CUDA_GRAPHS", "1") == "1"

# 尝试导入 transformers 模块，标记是否成功
try:
    import transformers  # noqa: F401  # 尝试导入 transformers 模块，忽略未使用警告
    HAS_TRANSFORMER = True  # 设置标志表明成功导入 transformers 模块
except ImportError:
    HAS_TRANSFORMER = False  # 设置标志表明未成功导入 transformers 模块


def get_optim(m):
    # 返回一个 Adam 优化器对象，用于模型 m 的参数优化
    return torch.optim.Adam(m.parameters(), lr=0.01, capturable=True, foreach=True)


def gen_transformer_inputs(vocab_size, bs, seq_length):
    def geninp():
        # 生成一个指定大小和类型的随机整数张量
        return torch.randint(
            0, vocab_size, (bs, seq_length), dtype=torch.int64, requires_grad=False
        )

    input_dict = {"input_ids": geninp(), "labels": geninp()}  # 创建输入字典
    return input_dict  # 返回输入字典


class LinearAndSoftmax(nn.Module):
    """
    It's very common that a transformer model will do a matmul and then
    softmax/log_softmax in the end.

    Creating this toy model to capture the pattern and make sure we do
    proper padding.
    """

    def __init__(self, vocab_size=30523, bias=True):
        """
        The default vocab size for BertForMaskedLM is 30522.
        We run a few test cases with good or bad vocab_size around Bert's
        default value.
        """
        super().__init__()
        self.vocab_size = vocab_size  # 初始化词汇表大小
        self.linear = nn.Linear(768, vocab_size, bias=bias)  # 初始化线性层
        self.ce = nn.CrossEntropyLoss()  # 初始化交叉熵损失函数

    def forward(self, x, label):
        x = self.linear(x)  # 执行线性变换
        return self.ce(x.view(-1, self.vocab_size), label.view(-1))  # 计算交叉熵损失

    def get_example_inputs(self, batch_size=16):
        # 生成示例输入张量
        return torch.randn(batch_size, 512, 768), torch.randint(
            0, self.vocab_size, (batch_size, 512)
        )


def forward_and_backward_pass(m, inputs):
    # 执行模型前向传播和反向传播
    m(*inputs).sum().backward()


@config.patch(
    {
        "benchmark_kernel": True,  # 开启内核性能基准测试
        "triton.unique_kernel_names": True,  # 使用唯一的内核名称
        "triton.cudagraphs": USE_CUDA_GRAPHS,  # 根据环境变量确定是否启用 CUDA 图形
    }
)
class TestCaseBase(TestCase):
    # 定义一个方法用于检查两个数值的接近程度，若类型为 LongformerMaskedLMOutput，则比较 loss 属性；若类型为 SequenceClassifierOutput，则比较 logits 属性；若是字典且包含 "loss" 键，则比较其 "loss" 值
    def check_close(self, ref, act, tol=1e-3):
        if type(ref).__name__ == "LongformerMaskedLMOutput":
            ref = ref.loss
            act = act.loss
        if type(ref).__name__ == "SequenceClassifierOutput":
            ref = ref.logits
            act = act.logits
        if isinstance(ref, dict) and "loss" in ref:
            ref = ref["loss"]
            act = act["loss"]
        # 使用 assertTrue 方法断言两个张量在指定的公差范围内接近
        self.assertTrue(
            torch.allclose(ref, act, atol=tol, rtol=tol), f"ref:\n{ref}\nact:\n{act}"
        )

    # 定义一个通用的数值检查方法，调用给定函数 f 两次并比较结果的接近程度
    def common_numeric_check(self, f, *args, tol=1e-3, **kwargs):
        # 调用函数 f 获取参考值 ref
        ref = f(*args, **kwargs)
        # 使用 torch.compile 对函数 f 进行优化
        opt_f = torch.compile(f)
        # 调用优化后的函数获取实际值 act
        act = opt_f(*args, **kwargs)
        # 调用 check_close 方法进行结果比较
        self.check_close(ref, act, tol)

    # 定义一个性能分析方法，分别对左右两个函数进行多次迭代调用，并记录性能数据
    def do_profiling(
        self,
        f_lhs,
        f_rhs,
        tag_lhs="With padding",
        tag_rhs="Without padding",
        args=(),
        kwargs=None,
    ):
        if kwargs is None:
            kwargs = {}
        # 同步 CUDA 设备以确保之前的操作完成
        torch.cuda.synchronize()
        # 使用 torch.profiler.profile 进行性能分析，并记录函数调用栈信息（如果 WITH_STACK 为 True）
        with torch.profiler.profile(with_stack=WITH_STACK) as p:
            niter = 3  # 设置迭代次数
            # 进行 niter 次迭代
            for _ in range(niter):
                # 使用 torch.profiler.record_function 记录 tag_lhs 对应函数的调用
                with torch.profiler.record_function(tag_lhs):
                    f_lhs(*args, **kwargs)

                # 使用 torch.profiler.record_function 记录 tag_rhs 对应函数的调用
                with torch.profiler.record_function(tag_rhs):
                    f_rhs(*args, **kwargs)
            # 再次同步 CUDA 设备，确保所有操作均已完成
            torch.cuda.synchronize()

        # 指定 Chrome trace 的输出路径
        profile_path = "/tmp/chrome.json"
        # 将性能分析结果导出到 Chrome trace 文件中
        p.export_chrome_trace(profile_path)
        # 打印提示信息，显示 Chrome trace 文件的输出路径
        print(f"Chrome trace is written to {profile_path}")
# 基于 TestCaseBase 的性能测试类，比较使用不同形状的性能
class PerfTestBetweenGoodAndBadShape(TestCaseBase):

    # 如果未启用性能测试，则跳过该测试
    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    # 测试不带偏置的 LinearAndSoftmax 的性能
    def test_nobias_LinearAndSoftmax_both_shapes(self):
        self.test_LinearAndSoftmax_both_shapes(bias=False)

    # 如果未启用性能测试，则跳过该测试
    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    # 测试带有偏置（默认为 True）的 LinearAndSoftmax 的性能
    def test_LinearAndSoftmax_both_shapes(self, bias=True):
        """
        Compare the perf with good and bad shape.
        """
        # 创建具有 30523 词汇量和给定偏置的 LinearAndSoftmax 模型
        m_bad_shape = LinearAndSoftmax(vocab_size=30523, bias=bias)
        # 获取坏形状模型的示例输入
        inptus_bad_shape = m_bad_shape.get_example_inputs()
        
        # 创建具有 30528 词汇量和给定偏置的 LinearAndSoftmax 模型
        m_good_shape = LinearAndSoftmax(vocab_size=30528, bias=bias)
        # 获取好形状模型的示例输入
        inputs_good_shape = m_good_shape.get_example_inputs()

        # 使用 Torch 编译优化坏形状模型
        m_bad_shape_opt = torch.compile(m_bad_shape)
        # 使用 Torch 编译优化好形状模型
        m_good_shape_opt = torch.compile(m_good_shape)

        # 测量使用好形状模型进行前向和反向传播的延迟
        latency_good_shape = do_bench(
            lambda: forward_and_backward_pass(m_good_shape_opt, inputs_good_shape)
        )
        # 测量使用坏形状模型进行前向和反向传播的延迟
        latency_bad_shape = do_bench(
            lambda: forward_and_backward_pass(m_bad_shape_opt, inptus_bad_shape)
        )
        # 打印好形状和坏形状的延迟比较结果
        print(
            f"Latency for good shape v.s. bad shape: {latency_good_shape:.3f}ms v.s. {latency_bad_shape:.3f}ms"
        )

    # 如果未启用性能测试或未安装 transformers 库，则跳过该测试
    @unittest.skipIf(not DO_PERF_TEST or not HAS_TRANSFORMER, "Perf test not enabled")
    # 测试 BertForMaskedLM 模型的性能
    def test_BertForMaskedLM(self, num_layers=1):
        """
        Compare the perf between doing padding and good shape.
        """
        # 导入 BertForMaskedLM 模型
        from transformers import BertForMaskedLM

        # 获取 BertForMaskedLM 的配置类
        config_cls = BertForMaskedLM.config_class
        bs = 16  # 批量大小
        seq_length = 512  # 序列长度

        # 创建模型的函数，接受词汇量作为参数
        def create_model(vocab_size):
            # 使用配置类创建配置对象
            config = config_cls()
            config.num_hidden_layers = num_layers
            config.vocab_size = vocab_size
            # 生成变压器输入并获取模型的示例输入
            inputs = gen_transformer_inputs(config.vocab_size, bs, seq_length)
            model = BertForMaskedLM(config)

            # 获取模型的优化器
            optim = get_optim(model)

            # 定义前向和反向传播函数
            def f(**inputs):
                optim.zero_grad(True)
                with torch.cuda.amp.autocast():
                    pred = model(**inputs)
                    loss = pred[0]
                loss.backward()
                optim.step()

            return torch.compile(f), inputs

        # 创建好形状和坏形状的 BertForMaskedLM 模型及其示例输入
        f_good_shape, inputs_good_shape = create_model(30528)
        f_bad_shape, inputs_bad_shape = create_model(30522)

        # 打印好形状模型的基准测试信息
        print("benchmark for good shape")
        # 测量使用好形状模型的延迟
        latency_good_shape = do_bench(lambda: f_good_shape(**inputs_good_shape))
        # 打印坏形状模型的基准测试信息
        print("benchmark for bad shape")
        # 测量使用坏形状模型的延迟
        latency_bad_shape = do_bench(lambda: f_bad_shape(**inputs_bad_shape))
        # 打印好形状和坏形状模型的延迟比较结果
        print(
            f"Latency with good and bad shape: {latency_good_shape:.3f} v.s. {latency_bad_shape:.3f}"
        )

        # 执行性能分析
        self.do_profiling(
            lambda: f_good_shape(**inputs_good_shape),
            lambda: f_bad_shape(**inputs_bad_shape),
            tag_lhs="With good shape",
            tag_rhs="With bad shape",
        )


# 继承自 TestCaseBase 的性能测试类，用于测试带和不带填充的性能
class PerfTestWithAndWithoutPadding(TestCaseBase):
    @maybe_cprofile
    def run_acc_and_perf_test(self, model, inputs, perf_inputs=None, tol=1e-3):
        """
        Run accuracy test.

        Also compare the perf with and without the comprehensive padding if
        DO_PERF_TEST is true.
        """
        # 如果没有提供性能输入，则使用与输入相同的输入
        if perf_inputs is None:
            perf_inputs = inputs

        def _process_inputs(x):
            """
            return args and kwargs
            """
            # 如果输入是字典，则返回空列表和输入字典
            if isinstance(x, dict):
                return [], x

            # 如果输入不是元组或列表，则将其包装为列表
            if not isinstance(inputs, (tuple, list)):
                x = [x]

            return x, {}

        # 处理输入和性能输入
        args, kwargs = _process_inputs(inputs)
        perf_args, perf_kwargs = _process_inputs(perf_inputs)

        # 如果需要进行准确性测试
        if DO_ACC_TEST:
            # 将模型设置为评估模式
            model.eval()
            # 进行数值检查以验证模型准确性
            self.common_numeric_check(model, *args, **kwargs, tol=tol)
        else:
            # 如果不需要准确性测试，则打印跳过的消息
            print("Accuracy test skipped")

        # 将模型设置为训练模式
        model.train()

        # 如果需要进行性能测试
        if DO_PERF_TEST:
            print("Do performance test")

            # 定义获取优化函数的函数
            def get_f(m, optim):
                def f(*args, **kwargs):
                    # 在每次调用之前将优化器梯度归零
                    optim.zero_grad(True)
                    # 使用自动混合精度上下文
                    with torch.cuda.amp.autocast():
                        # 调用模型生成预测
                        pred = m(*args, **kwargs)
                        # 计算损失并转换为标量
                        loss = reduce_to_scalar_loss(pred)
                    # 反向传播损失
                    loss.backward()
                    # 执行优化步骤
                    optim.step()

                return f

            latency_with_padding = None
            print("Benchmark with padding")
            # 使用综合填充功能进行性能评估
            with config.patch(comprehensive_padding=True):
                # 深度复制模型
                m_copy_with_padding = copy.deepcopy(model)
                # 获取带填充的优化器
                optim_with_padding = get_optim(m_copy_with_padding)
                # 编译优化函数
                opt_f_with_padding = torch.compile(
                    get_f(m_copy_with_padding, optim_with_padding)
                )
                # 执行基准测试
                latency_with_padding = do_bench(
                    lambda: opt_f_with_padding(*perf_args, **perf_kwargs)
                )
            latency_without_padding = None
            print("Benchmark without padding")
            # 不使用综合填充功能进行性能评估
            with config.patch(comprehensive_padding=False):
                # 深度复制模型
                m_copy_without_padding = copy.deepcopy(model)
                # 获取不带填充的优化器
                optim_without_padding = get_optim(m_copy_without_padding)
                # 编译优化函数
                opt_f_without_padding = torch.compile(
                    get_f(m_copy_without_padding, optim_without_padding)
                )
                # 执行基准测试
                latency_without_padding = do_bench(
                    lambda: opt_f_without_padding(*perf_args, **perf_kwargs)
                )
            # 打印带和不带填充的延迟结果
            print(
                f"Latency with and without padding: {latency_with_padding:.3f} v.s. {latency_without_padding:.3f}"
            )

            # 进行性能分析
            self.do_profiling(
                opt_f_with_padding,
                opt_f_without_padding,
                args=perf_args,
                kwargs=perf_kwargs,
            )
    # 定义一个测试函数，用于测试 NVIDIA DeepRecommender 模型的性能
    def test_nvidia_deeprecommender(self):
        """
        Compared the perf with and without comprehensive padding.
        """
        # 定义模型每一层的大小
        layer_sizes = [197951, 512, 512, 1024, 512, 512, 197951]
        # 生成一个随机张量作为输入数据
        x = torch.randn(4, layer_sizes[0])

        # 定义一个神经网络模型类
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                mod_list = []
                # 遍历每一层的大小，构建模型层次结构
                for i in range(len(layer_sizes) - 1):
                    mod_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                    mod_list.append(nn.SELU())

                    # 在特定的层（索引为2的层）添加 Dropout 层
                    if i == 2:
                        mod_list.append(nn.Dropout(0.8))
                # 使用 nn.Sequential 封装模型层次结构
                self.seq = nn.Sequential(*mod_list)

            def forward(self, x):
                return self.seq(x)

        # 创建一个模型实例
        m = Model()
        # 生成一个性能测试用的输入数据张量
        perf_inputs = torch.randn(256, layer_sizes[0])
        # 调用方法运行精度和性能测试
        self.run_acc_and_perf_test(m, x, perf_inputs)

    # 如果未启用性能测试或者缺少 transformer 库，则跳过测试
    @unittest.skipIf(not DO_PERF_TEST or not HAS_TRANSFORMER, "Perf test not enabled")
    def test_longformer(self, bs=4):
        # 从 transformers 库导入配置和模型
        from transformers import AutoConfig, AutoModelForMaskedLM

        # 从预训练模型加载配置
        config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
        # 根据配置创建模型
        model = AutoModelForMaskedLM.from_config(config)

        # 获取模型的词汇表大小和序列长度
        vocab_size = model.config.vocab_size
        seq_length = 1024
        # 生成 Transformer 模型的输入字典
        input_dict = gen_transformer_inputs(vocab_size, bs, seq_length)

        # 调用方法运行精度和性能测试
        self.run_acc_and_perf_test(model, input_dict)

    # 如果未启用性能测试或者缺少 transformer 库，则跳过测试
    @unittest.skipIf(not DO_PERF_TEST or not HAS_TRANSFORMER, "Perf test not enabled")
    def test_longformer_small_bs(self):
        """
        The model exists in both HF and TB. In TB it uses a samller batch size.
        """
        # 调用 test_longformer 方法，指定较小的批量大小（bs=2）
        self.test_longformer(bs=2)
class PaddingTest(TestCaseBase):
    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_mm_padding_perf(self):
        def naive_mm(a, b):
            return a @ b  # 执行普通的矩阵乘法运算

        def _compute_padding(s, align):
            return (s + align - 1) // align * align - s  # 计算对齐填充量

        @torch.compile
        def pad_mm(a, b, align=16):
            """
            NOTE: this function only pad a single dimension which is good
            enough for testing.
            """
            m_padding = _compute_padding(a.size(0), align)  # 计算矩阵 a 的第一维度填充量
            k_padding = _compute_padding(a.size(1), align)  # 计算矩阵 a 的第二维度填充量
            n_padding = _compute_padding(b.size(1), align)  # 计算矩阵 b 的第二维度填充量
            return pad_mm_pass.pad_mm(a, b, m_padding, k_padding, n_padding)  # 调用实际的填充矩阵乘法函数

        for M, K, N, f in (
            (8192, 768, 30523, naive_mm),  # 使用普通矩阵乘法的性能测试参数
            (8192, 768, 30523, pad_mm),    # 使用填充矩阵乘法的性能测试参数
            (8192, 768, 30528, naive_mm),  # 使用普通矩阵乘法的性能测试参数
            (30523, 8192, 768, naive_mm),  # 使用普通矩阵乘法的性能测试参数
            (30528, 8192, 768, naive_mm),  # 使用普通矩阵乘法的性能测试参数
        ):
            a = torch.randn(M, K)  # 创建随机矩阵 a
            b = torch.randn(K, N)  # 创建随机矩阵 b
            ms = do_bench(lambda: f(a, b))  # 测试函数执行时间
            print(f"MxKxN {M}x{K}x{N} {f.__name__}: {ms:.3f}ms")  # 打印性能测试结果

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_padmm(self):
        """
        Latency between origional matmul and padded matmul: 2.717 v.s. 2.356
        """
        mat1_pad = torch.randn(8192, 30522, dtype=torch.float16)  # 创建指定大小的随机矩阵 mat1_pad
        mat2_pad = torch.randn(30522, 768, dtype=torch.float16)   # 创建指定大小的随机矩阵 mat2_pad

        def f():
            return mat1_pad @ mat2_pad  # 执行矩阵乘法运算

        def pad_dim(x: Tensor, padded_length: int, dim: int) -> Tensor:
            pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1 :])  # 创建填充张量
            return torch.cat([x, pad], dim=dim)  # 在指定维度上拼接张量

        @torch.compile(fullgraph=True, options={"triton.cudagraphs": False})
        def g():
            mat1 = mat1_pad
            mat2 = mat2_pad
            mat1 = pad_dim(mat1, 6, 1)  # 对矩阵 mat1 在第一维度进行填充
            mat2 = pad_dim(mat2, 6, 0)  # 对矩阵 mat2 在第零维度进行填充
            return torch.ops.aten.mm(mat1, mat2)  # 使用 mm 操作执行矩阵乘法

        ori_time = do_bench(f)  # 测试原始矩阵乘法的性能
        pad_time = do_bench(g)  # 测试填充矩阵乘法的性能

        print(
            f"Latency between origional matmul and padded matmul: {ori_time:.3f} v.s. {pad_time:.3f}"
        )  # 打印性能比较结果
        self.do_profiling(f, g, "No MM Padding", "With mm padding")  # 执行性能分析函数
    def test_matmul(self):
        """
        Latency with good and bad shapes: 1.705 v.s. 2.625
        """
        # 创建具有良好形状的张量
        x_good_shape = torch.randn(8192, 30528, dtype=torch.float16)
        weight_good_shape = torch.randn(30528, 768, dtype=torch.float16)
        out_good_shape = torch.randn(8192, 768, dtype=torch.float16)

        # 使用不良形状的步长 (30522, 1)，在此处不会产生任何影响。
        x_bad_shape = rand_strided(
            (8192, 30522), (30528, 1), device="cuda", dtype=torch.float16
        )
        weight_bad_shape = torch.randn(30522, 768, dtype=torch.float16)
        out_bad_shape = torch.randn(8192, 768, dtype=torch.float16)

        # 定义一个函数 f，对输入进行矩阵乘法并将结果存储在指定的输出张量中
        def f(x, weight, out):
            torch.mm(x, weight, out=out)
            return out

        # 编译函数 f 的部分应用，使用良好形状的输入和输出张量
        f1 = torch.compile(
            functools.partial(f, x_good_shape, weight_good_shape, out_good_shape)
        )
        # 编译函数 f 的部分应用，使用不良形状的输入和输出张量
        f2 = torch.compile(
            functools.partial(f, x_bad_shape, weight_bad_shape, out_bad_shape)
        )
        # 测量使用良好形状的延迟
        latency_good_shape = do_bench(f1)
        # 测量使用不良形状的延迟
        latency_bad_shape = do_bench(f2)
        # 打印延迟比较结果
        print(
            f"Latency with good and bad shapes: {latency_good_shape:.3f} v.s. {latency_bad_shape:.3f}"
        )
        # 对 f1 和 f2 进行性能分析
        self.do_profiling(f1, f2)

    @serialTest()
    def test_nobias_LinearAndSoftmax_codegen(self):
        # 调用 test_LinearAndSoftmax_codegen 方法，设置 bias=False 进行测试
        self.test_LinearAndSoftmax_codegen(bias=False)

    def test_LinearAndSoftmax_codegen(self, bias=True):
        # 创建具有不良形状的 LinearAndSoftmax 模型
        m_bad_shape = LinearAndSoftmax(vocab_size=30523, bias=bias)
        # 获取模型的示例输入
        inputs_bad_shape = m_bad_shape.get_example_inputs()
        # 使用深拷贝编译 m_bad_shape 模型
        m_bad_shape_opt = torch.compile(copy.deepcopy(m_bad_shape))

        # 运行并获取 forward_and_backward_pass 函数的代码和包装器代码
        _, wrapper_codes = run_and_get_code(
            forward_and_backward_pass, m_bad_shape_opt, inputs_bad_shape
        )
        # 对 m_bad_shape 进行 forward_and_backward_pass 操作
        forward_and_backward_pass(m_bad_shape, inputs_bad_shape)
        # 断言优化后的梯度与原始模型的梯度是否接近
        self.assertTrue(
            torch.allclose(
                m_bad_shape.linear.weight.grad, m_bad_shape_opt.linear.weight.grad
            )
        )
        # 断言 wrapper_codes 的长度为 2，一个用于前向传播，一个用于反向传播
        self.assertTrue(len(wrapper_codes) == 2)  # one for forward and oen for backward
        forward_wrapper = wrapper_codes[0]

        # 确保 softmax 的加载在 forward_wrapper 中对齐
        self.assertTrue(
            "tl.load(in_ptr0 + (r1 + (30528*x0))" in forward_wrapper,
            f"forward_wrapper: {forward_wrapper}",
        )

        # 如果进行性能测试，则测量前向和反向传播的延迟
        if DO_PERF_TEST:
            latency = do_bench(
                lambda: forward_and_backward_pass(m_bad_shape_opt, inputs_bad_shape)
            )
            print(f"latency: {latency:.3f}ms")

    @config.patch(pattern_matcher=False)
    def test_attention(self):
        # 定义测试中使用的批大小、序列长度、注意力头数和隐藏层大小
        batch_size, seq_len, num_heads, hidden_size = 1, 4, 1, 16
        # 计算缩放因子，用于注意力权重计算
        inv_scale = (num_heads / hidden_size) ** 0.5

        # 定义注意力模型
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义查询、键、值的线性变换
                self.query = nn.Linear(hidden_size, hidden_size)
                self.key = nn.Linear(hidden_size, hidden_size)
                self.value = nn.Linear(hidden_size, hidden_size)

            @staticmethod
            def reshape(x):
                # 将输入张量重塑成指定形状，用于多头注意力计算
                return x.view(batch_size, seq_len, num_heads, -1).permute(0, 2, 1, 3)

            @staticmethod
            def cancel_reshape(x):
                # 取消重塑操作，返回原始形状的张量
                return x.permute(0, 2, 1, 3).view(batch_size, seq_len, hidden_size)

            def forward(self, x):
                # 前向传播函数，计算注意力权重并应用到值上
                query, key, value = self.query(x), self.key(x), self.value(x)
                weights = (
                    torch.matmul(
                        self.reshape(query), self.reshape(key).permute(0, 1, 3, 2)
                    )
                    * inv_scale
                ).softmax(dim=-1)
                return self.cancel_reshape(torch.matmul(weights, self.reshape(value)))

        # 创建Attention类的实例
        attn = Attention()
        # 生成随机输入张量
        x = torch.randn(batch_size, seq_len, hidden_size)

        # 调用公共的数值检查方法，验证Attention模型的输出
        self.common_numeric_check(attn, x)

    def test_view(self):
        def f(x):
            # 定义一个函数，将输入张量重塑为3x3x3的形状
            return x.view(3, 3, 3)

        # 生成随机输入张量
        x = torch.randn(3, 9)
        # 调用公共的数值检查方法，验证重塑函数的输出
        self.common_numeric_check(f, x)

    def test_pad_strides(self):
        """
        注意：即使dim0的步幅已经是16的倍数，仍然对其进行了填充。原因是我们对dim1的步幅进行了填充。
        我们必须相应地增加dim0的步幅。
        """
        # 定义输入大小和步幅
        sizes = [2, 16, 2047]
        in_strides = [2047 * 16, 2047, 1]
        # 调用布局对象的步幅填充方法，返回填充后的输出步幅
        out_strides = list(ir.Layout._pad_strides(in_strides, sizes, torch.float32))
        expected_strides = [2048 * 16, 2048, 1]
        # 断言填充后的输出步幅与预期结果相等
        self.assertEqual(
            expected_strides, out_strides, f"{expected_strides} v.s. {out_strides}"
        )

    def test_pad_strides_skip(self):
        """
        为了避免内存开销过大，跳过了填充操作。
        """
        # 定义输入大小和步幅
        sizes = [2, 32, 127]
        in_strides = [4064, 127, 1]
        # 调用布局对象的步幅填充方法，返回填充后的输出步幅
        out_strides = list(ir.Layout._pad_strides(in_strides, sizes, torch.float32))
        expected_strides = [4064, 127, 1]
        # 断言填充后的输出步幅与预期结果相等
        self.assertEqual(
            expected_strides, out_strides, f"{expected_strides} v.s. {out_strides}"
        )
    def test_pad_3d_tensor(self):
        """
        构建这个测试用例的指导思想是不对占位符或用户可见输出的步幅进行填充。

        在开头和结尾添加一个矩阵乘法，以便我们可以为中间张量填充步幅。
        """

        def f(x, y):
            x = torch.matmul(x, y)  # 执行矩阵乘法运算
            x = x + 1  # 对结果进行加法操作
            return torch.matmul(x, y)  # 再次执行矩阵乘法运算

        x = torch.randn(2, 16, 2047)  # 生成一个指定形状的随机张量 x
        y = torch.randn(2047, 2047)  # 生成一个指定形状的随机张量 y
        self.common_numeric_check(f, x, y, tol=1e-2)  # 调用公共数值检查函数，检查 f 的数值结果
        self.assertTrue(metrics.num_comprehensive_padding > 0)  # 检查综合填充数量是否大于 0

    def test_conv(self):
        """
        对卷积输入进行填充可能导致额外的复制内核被调用。
        查看这个示例跟踪：https://gist.github.com/shunting314/ce45398f7d51a63ce05fc8d411faddb3
        """
        x_shape = (1, 128, 640, 959)  # 定义输入张量 x1 的形状
        x1 = torch.randn(*x_shape)  # 生成一个指定形状的随机张量 x1

        padded_stride = ir.Layout._pad_strides(x1.stride(), x1.shape, torch.float32)  # 对 x1 的步幅进行填充
        x2 = rand_strided(x_shape, padded_stride, device="cuda")  # 根据填充后的步幅生成一个张量 x2，并移动到 CUDA 设备上
        x2.copy_(x1)  # 将 x1 的内容复制到 x2

        weight = torch.randn(64, 128, 3, 3)  # 生成一个指定形状的随机权重张量

        def fun(x, weight):
            return torch.convolution(
                x,
                weight,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
                bias=None,
            )

        ref = fun(x1, weight)  # 对 x1 进行卷积操作，作为参考结果
        act = fun(x2, weight)  # 对 x2 进行卷积操作，作为实际结果
        self.check_close(ref, act)  # 检查参考结果和实际结果是否接近
        if DO_PERF_TEST:  # 如果进行性能测试
            latency_with_padding = do_bench(lambda: fun(x2, weight))  # 测试填充后的卷积运行时间
            latency_without_padding = do_bench(lambda: fun(x1, weight))  # 测试不填充的卷积运行时间
            print(
                f"Latency with and without padding: {latency_with_padding:.3f} v.s. {latency_without_padding:.3f}"
            )

            self.do_profiling(lambda: fun(x2, weight), lambda: fun(x1, weight))  # 进行性能分析比较

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_cat(self):
        """
        比较 aten cat 和编译后 cat 的性能。

        Eager 和编译版本之间的延迟：1.596 v.s. 0.601

        Eager 版本可能比导线内核慢 2.66 倍。
        """
        x = torch.randn(8192, 30522, dtype=torch.float16)  # 生成一个指定形状和数据类型的随机张量 x

        def f(x):
            pad = x.new_zeros(x.size(0), 6)  # 生成一个与 x 形状相同的零张量作为填充
            return torch.cat([x, pad], dim=1)  # 在维度 1 上对 x 和 pad 进行拼接

        # 禁用 cudagraphs，因为 cudagraphs 需要复制输入，这会大大扭曲延迟！（这里编译版本的延迟是非编译版本的两倍）
        with config.patch("triton.cudagraphs", False):
            opt_f = torch.compile(f)  # 编译函数 f
            opt_f(x)  # 调用编译后的函数 opt_f 进行运算
        eager_time = do_bench(lambda: f(x))  # 测试非编译版本的运行时间
        opt_time = do_bench(lambda: opt_f(x))  # 测试编译版本的运行时间
        print(
            f"Latency between eager and compiled: {eager_time:.3f} v.s. {opt_time:.3f}"
        )
        self.do_profiling(lambda: f(x), lambda: opt_f(x), "Eager Cat", "Compiled Cat")  # 进行性能分析比较
    # 定义一个测试方法，用于测试通道位于最后的张量的填充功能
    def test_pad_channels_last(self):
        # 创建一个大小为 (2, 3, 5, 1025) 的随机张量
        t = torch.randn(2, 3, 5, 1025)
        # 获取张量的步幅信息
        in_strides = t.stride()
        # 调用 Layout 类的 _pad_strides 方法，对步幅进行填充以匹配指定的张量形状和数据类型
        out_strides = ir.Layout._pad_strides(in_strides, t.shape, torch.float32)
        # 断言原始步幅和填充后的步幅不相等，验证填充操作有效
        self.assertTrue(in_strides != out_strides)

        # 将张量转换为通道最后的内存格式
        t = t.to(memory_format=torch.channels_last)
        # 再次获取张量的步幅信息
        in_strides = t.stride()
        # 调用 Layout 类的 _pad_strides 方法，对步幅进行填充以匹配指定的张量形状和数据类型
        out_strides = ir.Layout._pad_strides(in_strides, t.shape, torch.float32)
        # 断言原始步幅和填充后的步幅相等，验证填充操作对通道最后格式的张量不改变其步幅
        self.assertTrue(in_strides == out_strides)
# 如果当前脚本被直接执行（而非被导入其他模块），则执行以下代码块
if __name__ == "__main__":
    # 如果系统中有 CUDA 支持可用
    if HAS_CUDA:
        # 设置使用高精度浮点数进行矩阵乘法计算
        torch.set_float32_matmul_precision("high")
        # 设置默认使用 CUDA 设备加速计算
        torch.set_default_device("cuda")
        # 运行测试函数或代码块
        run_tests()
```