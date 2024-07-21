# `.\pytorch\test\functorch\test_dims.py`

```py
# 导入 Python 标准库中的垃圾回收模块
import gc

# 从 unittest 模块中导入跳过装饰器和条件跳过装饰器
from unittest import skip, skipIf

# 从 attn_ft 模块中导入 BertSelfAttentionA 类和 Linear 类
from attn_ft import BertSelfAttention as BertSelfAttentionA, Linear

# 从 attn_positional 模块中导入 BertSelfAttentionB 类
from attn_positional import BertSelfAttention as BertSelfAttentionB

# 导入 PyTorch 库
import torch

# 从 functorch._C 模块中导入 dim 函数
from functorch._C import dim as _C

# 从 functorch.dim 模块中导入多个类和函数
from functorch.dim import (
    Dim,
    DimensionBindError,
    DimList,
    dimlists,
    dims,
    stack,
    Tensor,
)

# 从 torch.testing._internal.common_utils 模块中导入多个函数和类
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    TEST_CUDA,
    TestCase,
)

# 尝试导入 torchvision.models 模块中的 resnet18 模型，如果导入失败则将 resnet18 设置为 None
try:
    from torchvision.models import resnet18
except ImportError:
    resnet18 = None

# 从 functorch._C 模块中导入 _test_c, _parse_test, _set_pointwise_optimize 函数
_test_c, _parse_test, _set_pointwise_optimize = (
    _C._test_c,
    _C._parse_test,
    _C._set_pointwise_optimize,
)

# 从 contextlib 模块中导入上下文管理器 contextmanager
from contextlib import contextmanager

# 从 time 模块中导入 perf_counter 函数
from time import perf_counter

# 设置性能测量标志为 False
measure_perf = False

# 如果 measure_perf 为 True，则从 torchdim.magic_trace 模块中导入 magic_trace 函数
if measure_perf:
    from torchdim.magic_trace import magic_trace
# 否则定义一个名为 magic_trace 的上下文管理器函数
else:

    @contextmanager
    def magic_trace(*args, **kwargs):
        yield

# 定义一个名为 measure 的上下文管理器函数，用于测量执行时间
@contextmanager
def measure(what):
    b = perf_counter()  # 记录开始时间
    yield
    e = perf_counter()  # 记录结束时间
    print(f"{what}: {e - b:.20f} seconds")  # 输出执行时间差

# 定义一个名为 triu 的函数，计算输入张量 A 的上三角部分
def triu(A):
    i, j = dims()  # 从 dims 函数中获取维度 i 和 j
    a = A[i, j]  # 选择 A 张量的第 i 行第 j 列元素
    zero = torch.tensor(0, dtype=torch.float)  # 创建一个值为 0 的浮点型张量
    return torch.where(i <= j, a, zero).order(i, j)  # 返回满足条件的张量元素，并按照 i, j 维度排序

# 定义一个名为 gpu_time 的函数，用于测量 GPU 执行时间
def gpu_time(lmb, name, r=100):
    b = torch.cuda.Event(enable_timing=True)  # 创建 CUDA 事件，用于测量时间
    e = torch.cuda.Event(enable_timing=True)  # 创建 CUDA 事件，用于测量时间
    for _ in range(r):
        lmb()  # 执行指定的 lambda 函数
    b.record()  # 记录开始时间
    for _ in range(r):
        lmb()  # 再次执行指定的 lambda 函数
    e.record()  # 记录结束时间
    e.synchronize()  # 等待 GPU 操作完成
    elapsed = b.elapsed_time(e)  # 计算执行时间
    print(name, elapsed / r)  # 输出平均每次执行的时间
    return elapsed / r  # 返回平均每次执行的时间

# 跳过 TorchDynamo 测试的装饰器类
@skipIfTorchDynamo("Bad interaction")
class TestMin(TestCase):
    def setUp(self):
        super().setUp()  # 调用父类的 setUp 方法
        gc.disable()  # 禁用垃圾回收器
        gc.collect()  # 执行垃圾回收
        self.interesting = set()  # 创建一个空集合，用于存储感兴趣的对象
        for o in gc.get_objects():  # 遍历所有 Python 对象
            if isinstance(o, (torch.Tensor, Dim, Tensor, DimList)):  # 如果是 Tensor 或者相关对象
                self.interesting.add(id(o))  # 将其 id 添加到感兴趣集合中
        if "cuda" in self._testMethodName:  # 如果测试方法名中包含 'cuda'
            self.mem_allocated = torch.cuda.memory_allocated()  # 记录当前 CUDA 内存分配量
    # 在测试结束时执行清理操作
    def tearDown(self):
        # 初始化一个空列表，用于存储感兴趣的对象
        interesting = []
        # 遍历所有 Python 对象
        for o in gc.get_objects():
            # 判断对象是否为 torch.Tensor, Dim, Tensor, DimList 类型，并且不在感兴趣列表中
            if (
                isinstance(o, (torch.Tensor, Dim, Tensor, DimList))
                and id(o) not in self.interesting
            ):
                # 将满足条件的对象添加到感兴趣列表中
                interesting.append(o)

        # 初始化额外内存变量
        extra_memory = 0
        # 如果测试方法名中包含 "cuda"
        if "cuda" in self._testMethodName:
            # 计算额外内存
            extra_memory += torch.cuda.memory_allocated() - self.mem_allocated

        # 如果存在额外内存或者感兴趣列表不为空
        if extra_memory != 0 or len(interesting) != 0:
            # 导入 refcycle 模块
            import refcycle
            # 导出垃圾图像到 "garbage.pdf"
            refcycle.garbage().export_image("garbage.pdf")
        # 执行垃圾回收
        gc.collect()
        # 断言额外内存为0
        assert extra_memory == 0, f"extra cuda memory left allocated: {extra_memory}"
        # 断言感兴趣列表为空
        assert len(interesting) == 0, (
            f"extra torch.Tensor, Dim, or Tensor left allocated: {len(interesting)} objects of types:"
            f" { [type(t) for t in interesting] }"
        )

    # 测试手动操作
    def test_manual_stuff(self):
        # 创建随机张量 A_ 和 B_
        A_ = torch.rand(3, 4)
        B_ = torch.rand(4, 5)
        # 获取维度 i, j, k
        i, j, k = dims()
        # 对张量 A_ 和 B_ 进行操作
        A = A_[i, k]
        B = B_[k, j]
        C = (A.expand(j) * B.expand(i)).sum(k)
        # 断言 C 的顺序
        self.assertTrue(torch.allclose(C.order(i, j), torch.mm(A_, B_)))
        self.assertTrue(torch.allclose(torch.triu(A_, 0), triu(A_)))

        # 创建随机整数张量 D_
        D_ = torch.randint(0, 3, (6,))
        # 获取维度 d
        d = dims()
        # 对张量 D_ 进行操作
        D = D_[d]

        # 对张量 A 进行索引和排序
        A.index([i], [D]).order(k, d)

    # 定义注意力函数
    def attn(
        self,
        batch_size=1,
        sequence_length=4,
        hidden_size=6,
        num_attention_heads=3,
        linear=Linear,
        device=None,
        time=False,
    def test_attn(self):
        # 调用注意力函数
        self.attn()

    # 测试原地操作
    def test_inplace(self):
        # 创建一个嵌入表 embeddings
        embeddings = torch.zeros(10, 3)

        # 对嵌入表进行稀疏更新
        indices = torch.arange(2) + 1
        values = torch.rand(2, 3)

        # 获取维度 i, n, f
        i, n, f = dims()

        # 对 embeddings 进行原地操作
        embeddings[indices[i], f] += values[i, f]

    # 测试适应性
    def test_adapt(self):
        # 定义函数 f
        def f():
            ci, co = dims()

        # Python 3.11 在一定迭代次数后会调整字节码
        # 检查我们是否仍然正确匹配名称
        for i in range(10):
            f()

    # 测试 CUDA 上的注意力函数
    @skipIf(not TEST_CUDA, "no CUDA")
    def test_attn_cuda(self):
        # 调用 CUDA 上的注意力函数
        self.attn(
            batch_size=256,
            hidden_size=768,
            sequence_length=128,
            num_attention_heads=12,
            device="cuda",
            time=measure_perf,
            linear=torch.nn.Linear,
        )

    # 测试堆叠操作
    def test_stack(self):
        # 获取维度 i, j, d
        i, j, d = dims()
        # 创建随机张量 A
        A = torch.rand(4, 5)
        # 对 A 进行堆叠操作
        r = stack([A[i, j]], d, j)
        # 解绑 r，并进行断言
        # a, b = r.unbind(d)
        # self.assertTrue(torch.allclose(a.order(i, j), i.expand(j).order(i, j)))
        # self.assertTrue(torch.allclose(b.order(i, j), j.expand(i).order(i, j)))
    # 定义一个测试函数，用于测试 max 函数
    def test_max(self):
        # 创建一个形状为 (2, 3, 2) 的随机张量
        ap = torch.rand(2, 3, 2)
        # 调用 dims 函数返回四个索引 i, j, k
        i, j, k = dims()
        # 从 ap 中根据索引 i, j, k 获取子张量 a
        a = ap[i, j, k]
        # 对张量 a 沿着维度 k 进行最大值计算，返回结果和索引 i0
        r, i0 = a.max(dim=k)
        # 使用 assertTrue 方法验证 r 的排序后的顺序与 ap 在维度 2 上的最大值的一致性
        self.assertTrue(torch.allclose(r.order(i, j), ap.max(2)[0]))

    # 定义一个测试函数，测试 mm 操作
    def test_mm(self):
        # 调用 dims 函数返回四个索引 i, j, k, q
        i, j, k, q = dims()
        # 创建形状为 (3, 4) 的随机张量 a 和形状为 (4, 5) 的随机张量 b
        a = torch.rand(3, 4)
        b = torch.rand(4, 5)
        # 从张量 a 中根据索引 i, k 获取子张量 a_
        a_ = a[i, k]
        # 从张量 b 中根据索引 k, j 获取子张量 b_
        b_ = b[k, j]
        # 设置 q 的大小为 1
        q.size = 1
        # 计算两个子张量的扩展乘积，然后沿着维度 k 求和，最后按照顺序 q, i, j 排序结果为 r
        r = (a_.expand(j, q) * b_.expand(i, q)).sum(k).order(q, i, j)
        # 注释掉的代码段为等效的 mm 操作的实现方式
        # r = (a_*b_).sum(k).order(q, i, j)
        # print(r)
        # print(a @ b)

    # 定义一个测试函数，测试带有维度拆分的操作
    def test_with_dims_split(self):
        # 创建一个形状为 (3, 12) 的张量 a
        a = torch.arange(3 * 12).view(3, 12)
        # 调用 dims 函数返回三个索引 i, j, k
        i, j, k = dims()
        # 设置 k 的大小为 4
        k.size = 4
        # 从张量 a 中根据索引 i, [j, k] 获取子张量 r
        r = a[i, [j, k]]
        # 按照顺序 i, [j, k] 对 r 排序得到 x
        x = r.order(i, [j, k])
        # 使用 assertTrue 方法验证张量 a 和 x 的近似相等性
        self.assertTrue(torch.allclose(a, x))

    # 定义一个简单的测试函数，测试基本操作
    def test_simple(self):
        # 调用 dims 函数返回三个索引 i, j, k
        i, j, k = dims()
        # 创建一个形状为 (3, 4) 的随机张量 x
        x = torch.rand(3, 4)
        # 从张量 x 中根据索引 i, j 获取子张量 z
        z = x[i, j]
        # 对 z 进行四次加法操作
        (z + z + z + z)
        # 按照顺序 i, j 对 z 排序
        (z.order(i, j))

    # 定义一个测试函数，测试 mm 融合操作
    def test_mm_fuse(self):
        # 调用 dims 函数返回三个索引 i, j, k
        i, j, k = dims()
        # 创建形状为 (3, 4) 和 (4, 5) 的随机张量 A 和 B
        A = torch.rand(3, 4)
        B = torch.rand(4, 5)
        # 计算张量 A 和 B 的 mm 融合结果 C，并按照顺序 i, j 对结果进行排序
        C = (A[i, k] * B[k, j]).sum(k).order(i, j)
        # 使用 assert 语句验证张量 C 与矩阵乘法 A @ B 的近似相等性
        assert torch.allclose(C, A @ B)

    # 定义一个测试函数，测试 mm 时间性能
    def test_time_mm_fuse(self):
        # 调用 dims 函数返回三个索引 i, j, k
        i, j, k = dims()
        # 创建形状为 (3, 4) 和 (4, 5) 的随机张量 A 和 B
        A = torch.rand(3, 4)
        B = torch.rand(4, 5)

        # 循环 10 次执行 A @ B 的矩阵乘法
        for _ in range(10):
            r0 = A @ B

        # 循环 10 次执行 mm 操作的矩阵乘法
        for _ in range(10):
            a = A[i, k]
            b = B[k, j]
            r1 = (a * b).sum(k)

        # 使用 measure 函数测量时间，执行 10000 次 A @ B 的矩阵乘法
        with measure("pp"):
            for _ in range(10000):
                A @ B
        # magic_trace_stop_indicator()

        # 使用 measure 函数测量时间，执行 10000 次 mm 操作的矩阵乘法，并按照顺序 i, j 排序
        with measure("fc"):
            for _ in range(10000):
                (A[i, k] * B[k, j]).sum(k).order(i, j)

        # 使用 magic_trace 函数测量时间，执行 10000 次 mm 操作的矩阵乘法，并记录到 "f.fxt" 中
        with magic_trace("f.fxt"):
            for _ in range(10000):
                (A[i, k] * B[k, j]).sum(k).order(i, j)

        # 使用 magic_trace 函数测量时间，执行 10000 次 A @ B 的矩阵乘法，并记录到 "p.fxt" 中
        with magic_trace("p.fxt"):
            for _ in range(10000):
                A @ B

        # 使用 assert 语句验证张量 r1 按照顺序 i, j 排序后与 r0 的近似相等性
        assert torch.allclose(r1.order(i, j), r0)

    # 定义一个测试函数，测试维度比较操作
    def test_compare_dims(self):
        # 调用 dims 函数返回两个索引 i, j
        i, j = dims()
        # 设置 i 的大小为 3，j 的大小为 4
        i.size = 3
        j.size = 4
        # 执行 i 与 j 的比较操作，禁用 B015 警告
        (i < j)  # noqa: B015

    # 定义一个测试函数，测试 C 函数的调用
    def test_c(self):
        # 调用 _test_c 函数，测试 C 函数的功能
        _test_c()

    # 定义一个测试函数，测试分段操作
    def test_seg(self):
        # 创建形状为 (3, 4) 的随机张量 A
        A = torch.rand(3, 4)
        # 调用 dims 函数返回两个索引 i, k
        i, k = dims()
        # 设置 i 的大小为 4，k 的大小为 3
        i.size = 4
        k.size = 3
        # 计算 i 和 k 的加法减一
        r = i + k - 1

    # 定义一个测试函数，测试张量扩展操作
    def test_expand(self):
        # 创建形状为 (3, 4) 的随机张量 A
        A = torch.rand(3, 4)
        # 调用 dims 函数返回一个索引 i
        i =
    def test_parse(self):
        # 测试_parse_test函数的不同参数组合
        self.assertEqual(("x", None, None, None), _parse_test(1, 0, "x"))
        self.assertEqual(("x", None, "y", None), _parse_test(1, 0, "x", c="y"))
        self.assertEqual(("x", None, "y", "z"), _parse_test(1, 0, "x", d="z", c="y"))

        # 测试_parse_test函数的另外两个参数组合
        self.assertEqual(("x", "4", None, None), _parse_test(2, 0, "x", b="4"))
        self.assertEqual(("x", "y", "z", "q"), _parse_test(2, 0, "x", "y", "z", "q"))

        # 测试_parse_test函数抛出TypeError异常的情况
        with self.assertRaises(TypeError):
            _parse_test(2, 0, "x", "y", "z", "q", "5")
        with self.assertRaises(TypeError):
            _parse_test(2, 0, "x", "y", b="y")

        with self.assertRaises(TypeError):
            _parse_test(2, 0, "x", c="y")
        with self.assertRaises(TypeError):
            _parse_test(2, 0, "x")

    def test_network(self):
        # 如果resnet18为None，则跳过该测试
        if resnet18 is None:
            self.skipTest("no torchvision")

        # 使用resnet18创建网络模型rn，并设置norm_layer为BatchNorm2d
        rn = resnet18(
            norm_layer=lambda x: torch.nn.BatchNorm2d(x, track_running_stats=False)
        )
        rn.train()

        # 创建随机张量img，将其视图变换为imgf
        img = torch.rand(1, 1, 2, 3, 224, 224)
        imgf = img.view(2, 3, 224, 224)

        # 调用dims函数获取i和j
        i, j = dims()

        # 对rn模型应用img[i, j]的索引，得到r
        r = rn(img[i, j])

        # 将r按照顺序i, j进行排序，然后再视图变换为(2, 1000)
        r = r.order(i, j).view(2, 1000)

        # 对imgf应用rn模型，得到r2
        r2 = rn(imgf)

        # 断言r2与r在给定的绝对误差下相似
        assert torch.allclose(r2, r, atol=1e-06)

    def test_dim_args(self):
        # 测试dims和dimlists函数返回类型是否正确
        a = dimlists()
        assert isinstance(a, DimList)
        a = dims()
        b = dimlists()
        assert isinstance(a, Dim)
        assert isinstance(b, DimList)

        # 测试dims函数的其他用例
        assert str(a) == "a"
        a, b = dims(sizes=[3, 4])
        assert a.size == 3
        assert b.size == 4
        a = dims(sizes=[3])
        b = dimlists(sizes=[4])
        assert len(b) == 4
        a = dims()
        b = dimlists(sizes=[[4, 5]])
        assert b[0].size == 4
        assert b[1].size == 5

    def test_diag(self):
        # 调用dims函数获取i
        i = dims()

        # 创建4x4的随机张量A，对其应用索引A[i, i]
        A = torch.rand(4, 4)
        (A[i, i])

    def test_softmax_split(self):
        # 创建长度为16的随机张量a
        a = torch.rand(16)

        # 调用dims函数获取g和i
        g, i = dims(sizes=[2, None])

        # 对a应用索引a[[i, g],]，得到a2
        a2 = a[[i, g],]

        # 计算a2在维度i上的最大值和索引
        m_b, _ = a2.max(i)

        # 计算指数函数exp(a2 - m_b)，得到f_b
        f_b = torch.exp(a2 - m_b)

        # 计算f_b在维度i上的和，得到l_b
        l_b = f_b.sum(i)

        # 计算m_b在维度g上的最大值和索引
        m, _ = m_b.max(g)

        # 计算指数函数exp(m_b - m)，得到c
        c = torch.exp(m_b - m)

        # 计算(c * f_b)并按照(i, g)顺序排序，得到f
        f = (c * f_b).order((i, g))

        # 计算(c * l_b)在维度g上的和，得到l
        l = (c * l_b).sum(g)

        # 断言f / l与torch.nn.functional.softmax(a, dim=0)在给定的绝对误差下相似
        assert torch.allclose(f / l, torch.nn.functional.softmax(a, dim=0))

    def test_index(self):
        # 创建3x4的随机张量A和4x5的随机张量B
        A = torch.rand(3, 4)
        B = torch.rand(4, 5)

        # 调用dims函数获取i, j, k
        i, j, k = dims()

        # 调用dims函数获取o, l，并设置o.size = 2
        o, l = dims()
        o.size = 2

        # 对A[i, k]应用索引函数index(k, [o, l])，得到r
        r = A[i, k].index(k, [o, l])

        # 断言r按照顺序(i, o, l)排序后与A.view(-1, 2, 2)在给定的绝对误差下相似
        assert torch.allclose(r.order(i, o, l), A.view(-1, 2, 2))

        # 对r应用索引函数index([o, l], k)，得到rr
        rr = r.index([o, l], k)

        # 断言rr按照顺序(i, k)排序后与A在给定的绝对误差下相似
        assert torch.allclose(A, rr.order(i, k))

        # 调用dims函数获取z
        z = dims()

        # 创建长度为2的张量C
        C = torch.arange(2)

        # 对A[i, k]应用索引函数index(k, C[z])，并按照顺序(i, z)排序，得到x
        x = A[i, k].index(k, C[z]).order(i, z)

        # 断言x的列选择与A的[:, 0:2]在给定的绝对误差下相似
        assert torch.allclose(A[:, 0:2], x)

        # 创建3x4x5的随机张量C
        C = torch.rand(3, 4, 5)

        # 调用dims函数获取ik
        ik = dims()

        # 对C应用索引函数index((0, 2), ik)，并按照顺序(ik)排序，得到新的张量
        assert torch.allclose(
            C.index((0, 2), ik).order(ik), C.permute(0, 2, 1).reshape(15, 4)
        )
    # 对一些操作符进行 monkey patching 后出现的失败情况...
    def test_monkey(self):
        # 创建一个 3x4 的随机张量 A
        A = torch.rand(3, 4)
        # 修改 A 的第一个元素为 5
        A[0, 0] = 5
        # 创建一个形状为 3x4x4x4x3 的随机张量 x
        x = torch.randn(3, 4, 4, 4, 3)
        # 克隆张量 x 得到 x_clone1
        x_clone1 = x.clone()
        # 创建一个张量 ia，包含 [0, 2, 1]
        ia = torch.tensor([0, 2, 1])
        # 创建一个张量 ib，包含 [0, 2, 1]
        ib = torch.tensor([0, 2, 1])
        # 计算 x[:, ia, None, ib, 0] 的形状，并赋值给 first_shape
        first_shape = x[:, ia, None, ib, 0].shape
        # 使用随机生成的数据替换 x_clone1 的 x[:, ia, None, ib, 0] 的值
        x_clone1[:, ia, None, ib, 0] = torch.randn(first_shape).to(x_clone1)
        # 创建一个空的 torch.autograd.Variable 张量 x
        x = torch.autograd.Variable(torch.tensor([]))
        # 创建一个包含 [1, 2, 3] 的 torch.autograd.Variable 张量 z
        z = torch.autograd.Variable(torch.IntTensor([1, 2, 3]))
        # 创建一个列表 a，包含 z 的第三个元素和 z 的第一个元素加 3
        a = [z[2], z[0] + 3]
        # 使用 x.new(a) 创建一个新的张量
        x.new(a)
        # 断言 x.new([z[2], z[0] + 3]).tolist() 的结果等于 [3, 4]
        # self.assertEqual(x.new([z[2], z[0] + 3]).tolist(), [3, 4])

    # 测试索引位置
    def test_index_placement(self):
        # 创建一个 1x2x3x4 的随机张量 A
        A = torch.rand(1, 2, 3, 4)
        # 调用 dims 函数获取 i, j
        i, j = dims(sizes=[2, 4])
        # 根据 i, j 索引张量 A 的子集 a
        a = A[:, i + 0, :, j + 0]
        # 对 a 进行排序，并赋值给 r
        r = a.order(i, j)
        # 断言 A.permute(1, 3, 0, 2) 和 r 的所有元素相近
        assert torch.allclose(A.permute(1, 3, 0, 2), r)

    # 测试顺序
    def test_order(self):
        # 调用 dims 函数获取 i, j
        i, j = dims()
        # 创建一个 3x4x5 的随机张量 A
        A = torch.rand(3, 4, 5)
        # 断言 A[i].order(1, i) 和 A.permute(2, 0, 1) 的所有元素相近
        assert torch.allclose(A[i].order(1, i), A.permute(2, 0, 1))

    # 测试掩码
    def test_mask(self):
        # 创建一个包含 5 个随机数的张量 a
        a = torch.rand(5)
        # 调用 dims 函数获取 i, j
        i, j = dims(sizes=[a.size(0), a.size(0)])
        # 计算 ((i >= j) * a[i]).sum(j).order(i)
        ((i >= j) * a[i]).sum(j).order(i)

    # 测试相等
    def test_eq(self):
        # 调用 dims 函数获取 i, j
        i, j = dims(sizes=[3, 3])
        # 断言 (i == j).sum((i, j)) 等于 3
        assert (i == j).sum((i, j)) == 3

    # 测试带有大小的维度
    def test_dims_with_size(self):
        # 调用 dims 函数获取 x
        x = dims(3)
        # 断言 x 的长度等于 3 并且 x[0] 是 Dim 类的实例
        assert len(x) == 3 and isinstance(x[0], Dim)
        # 创建一个类 Foo
        class Foo:
            pass
        # 创建对象 y 为 Foo 类的实例
        y = Foo()
        # 调用 dims 函数获取 z, y.x, q
        z, y.x, q = dims(3)
        # 断言 str(z) 等于 "z"
        assert str(z) == "z"
        # 断言 str(y.x) 等于 "d1"
        assert str(y.x) == "d1"
        # 断言 str(q) 等于 "d2"
        assert str(q) == "d2"

    # 测试 dir 函数
    def test_dir(self):
        # 调用 dims 函数获取 i, j
        i, j = dims(sizes=[3, 3])
        # 调用 dir 函数，参数为 i <= j
        dir(i <= j)

    # 测试文档字符串
    def test_doc(self):
        # 断言 Tensor.clamp.__doc__ 等于 torch.Tensor.clamp.__doc__
        assert Tensor.clamp.__doc__ == torch.Tensor.clamp.__doc__

    # 测试嵌入
    def test_embed(self):
        # 创建一个 8x32 的随机张量 embeddings
        embeddings = torch.rand(8, 32)
        # 创建一个张量 ids，包含 [1, 0, 3, 4]
        ids = torch.tensor([1, 0, 3, 4])
        # 使用慢速但 Pythonic 的方法填充 values_
        values_ = torch.empty(4, 32)
        for batch in range(ids.size(0)):
            for feature in range(embeddings.size(1)):
                values_[batch, feature] = embeddings[ids[batch], feature]
        # 使用 torchdim 进行单索引核操作，填充 values
        batch, feature = dims(2)
        values = embeddings[ids[batch], feature].order(batch, feature)
        # 断言 values 和 values_ 的所有元素相近
        assert torch.allclose(values, values_)

    # 测试 functorch
    def test_functorch(self):
        # 创建三个随机张量 A, B, C
        A = torch.rand(3, 4, 5)
        B = torch.rand(3, 4, 5)
        C = torch.rand(5, 2)
        # 调用 dims 函数获取 i, j
        i, j = dims()
        # 计算 AA 和 BB
        AA = torch.mm(A[i], C)  # 3, 4, 2
        BB = torch.mm(B[j], C)  # 3, 4, 2
        # 断言 torch.mm(AA.T, BB).order(i, j) 的形状为 [3, 3, 2, 2]
        assert list(torch.mm(AA.T, BB).order(i, j).shape) == [3, 3, 2, 2]

    # 测试 permute_orig 函数
    def test_permute_orig(self):
        # 调用 dims 函数获取 d
        d = dims(1)
        # 创建一个 1x2x3x4 的随机张量 t_fc
        t_fc = torch.rand(1, 2, 3, 4)[d]
        # 断言 t_fc.permute(dims=(1, 0, 2)).shape 和 t_fc.permute(1, 0, 2).shape 相等
        assert t_fc.permute(dims=(1, 0, 2)).shape == t_fc.permute(1, 0, 2).shape

    # 测试 order_keyword 函数
    def test_order_keyword(self):
        # 调用 dims 函数获取 d
        d = dims(1)
        # 创建一个包含 3 个随机数的张量 t
        t = torch.rand(3)[d]
        # 断言调用 t.order(wrong=3) 会引发 TypeError 异常
        self.assertRaises(TypeError, lambda: t.order(wrong=3))
    # 定义一个测试方法，用于测试大数据集的分割操作
    def test_big_split(self):
        # 初始化总数变量为 0
        total = 0
        # 初始化空列表
        l = []
        # 当总数小于 6400 时执行以下循环
        while total < 6400:
            # 在列表 l 中添加一个在范围 [2, 10) 内随机整数，并取其单元素值
            l.append(torch.randint(2, 10, (1,)).item())
            # 将列表 l 的最后一个元素累加到总数中
            total += l[-1]
        # 生成一个形状为 (total, 1) 的随机张量 x
        x = torch.randn(total, 1)
        # 使用列表 l 中的元素作为分割长度，沿着维度 0 分割张量 x
        x.split(l, 0)
# 定义要跳过的测试函数名称列表，这些函数不会被执行
skip_functorch_only = ["test_time_mm_fuse", "test_attn_cuda"]

# 定义一个测试类 TestMinFunctorchOnly，继承自 TestMin 类
class TestMinFunctorchOnly(TestMin):
    # 在每个测试方法执行前调用，设置点对点优化为 False
    def setUp(self):
        super().setUp()  # 调用父类的 setUp 方法
        _set_pointwise_optimize(False)

    # 在每个测试方法执行后调用，设置点对点优化为 True
    def tearDown(self):
        _set_pointwise_optimize(True)
        super().tearDown()  # 调用父类的 tearDown 方法

# 遍历 skip_functorch_only 列表中的每个函数名称 n
for n in skip_functorch_only:
    # 给 TestMinFunctorchOnly 类动态设置属性，属性名为 n，值为装饰器 skip("skip_functorch_only")(lambda self: None)
    setattr(TestMinFunctorchOnly, n, skip("skip_functorch_only")(lambda self: None))

# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 运行测试
    run_tests()
```