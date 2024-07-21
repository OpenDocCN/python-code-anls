# `.\pytorch\test\inductor\test_pad_mm.py`

```py
# 导入unittest模块，用于编写和运行测试用例
# 导入torch模块，PyTorch深度学习框架的核心库
# 导入inductor_config模块，包含有关编译器自动调优的配置
# 导入rand_strided函数，用于生成随机张量
# 导入pad_mm模块中的相关函数：get_alignment_size, get_pad_cache, get_padded_length, should_pad_common
# 导入TestCase类，用于编写测试用例的基类
# 导入fresh_inductor_cache和run_and_get_code函数，用于刷新编译器缓存和执行并获取代码
# 导入FileCheck类，用于验证生成的代码是否符合预期的格式
# 导入HAS_CUDA，检查系统是否支持CUDA

class PadMMTest(TestCase):
    # 使用装饰器设置inductor_config的patch，配置编译器的最大自动调优和GEMM后端为TRITON
    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    # 定义测试方法test_pad_mm_dyn_m，测试动态矩阵乘法
    def test_pad_mm_dyn_m(self):
        M = 40
        K1 = 581
        K2 = 49
        N = 30

        # 定义模型类Model，继承自torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模型权重self.w，调用rand_strided生成随机初始化的张量
                self.w = rand_strided(
                    (K2, N), (1, K2), device="cuda", dtype=torch.float32
                )

            # 定义前向传播函数forward，接受输入a，返回torch.mm(a1, self.w)的结果
            def forward(self, a):
                # 使用torch.narrow从a中选择部分张量a1
                a1 = torch.narrow(a, 1, 0, K2)
                return torch.mm(a1, self.w)

        # 创建Model类的实例fn，并将其部署到CUDA设备上
        fn = Model().cuda()
        # 生成随机张量a，使用rand_strided生成，部署到CUDA设备上
        a = rand_strided((M, K1), (K1, 1), device="cuda", dtype=torch.float32)
        # 计算对齐后的K值，使用get_padded_length和get_alignment_size函数
        aligned_k = get_padded_length(K2, get_alignment_size(a)) + K2
        # 标记张量a为动态张量
        torch._dynamo.mark_dynamic(a, 0)
        # 使用unittest.mock.patch装饰器修改pad_mm模块的_skip_do_bench_times为True
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            # 调用模型fn，传入a作为参数，计算结果保存到res1
            res1 = fn(a)
            # 编译模型fn
            compiled_fn = torch.compile(fn)
            # 执行编译后的模型fn，传入a作为参数，获取结果res2和生成的代码code
            res2, (code,) = run_and_get_code(compiled_fn, a)
            # 使用FileCheck检查生成的code中是否包含指定格式的字符串
            FileCheck().check(f"K = {aligned_k}").run(code)
        # 断言res1和res2相等
        self.assertEqual(res1, res2)

    # 使用装饰器设置inductor_config的patch，配置编译器的最大自动调优和GEMM后端为TRITON
    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    # 定义测试方法test_cat_pad_mm_dyn_m，测试动态矩阵拼接和乘法
    def test_cat_pad_mm_dyn_m(self):
        M1 = 128
        M2 = 40
        K1 = 129
        K2 = 111
        N = 100

        # 定义模型类Model，继承自torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模型权重self.w，调用rand_strided生成随机初始化的张量
                self.w = rand_strided(
                    (K2, N), (1, K2), device="cuda", dtype=torch.float32
                )

            # 定义前向传播函数forward，接受输入a和b，返回torch.mm(a1, self.w)的结果
            def forward(self, a, b):
                # 使用torch.cat对a和b在维度0上进行拼接，得到张量c
                c = torch.cat([a, b], dim=0)
                # 使用torch.narrow从c中选择部分张量a1
                a1 = torch.narrow(c, 1, 0, K2)
                return torch.mm(a1, self.w)

        # 创建Model类的实例fn，并将其部署到CUDA设备上
        fn = Model().cuda()
        # 生成随机张量a和b，使用rand_strided生成，部署到CUDA设备上
        a = rand_strided((M1, K1), (K1, 1), device="cuda", dtype=torch.float32)
        b = rand_strided((M2, K1), (K1, 1), device="cuda", dtype=torch.float32)
        # 标记张量a和b为动态张量
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        # 计算对齐后的K值，使用get_padded_length和get_alignment_size函数
        aligned_k = get_padded_length(K2, get_alignment_size(a)) + K2
        # 使用unittest.mock.patch装饰器修改pad_mm模块的_skip_do_bench_times为True
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            # 调用模型fn，传入a和b作为参数，计算结果保存到res1
            res1 = fn(a, b)
            # 编译模型fn
            compiled_fn = torch.compile(fn)
            # 执行编译后的模型fn，传入a和b作为参数，获取结果res2和生成的代码code
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            # 使用FileCheck检查生成的code中是否包含指定格式的字符串
            FileCheck().check(f"K = {aligned_k}").run(code)
        # 断言res1和res2相等
        self.assertEqual(res1, res2)
    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    # 应用装饰器配置最大自动调优和使用TRITON作为后端的自动调优
    def test_pad_mm_dyn_n(self):
        # 设置矩阵维度
        M = 20
        K = 81
        N = 30

        # 定义模型类，继承自torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 前向传播方法，计算矩阵相乘
            def forward(self, a, b):
                return torch.mm(a, b)

        # 创建Model类的实例，并将其部署到CUDA设备上
        fn = Model().cuda()

        # 生成随机的CUDA张量a和b，按指定的步幅分布
        a = rand_strided((M, K), (K, 1), device="cuda", dtype=torch.float32)
        b = rand_strided((K, N), (1, K), device="cuda", dtype=torch.float32)

        # 获取对齐后的K值，并将动态特性标记为1
        aligned_k = get_padded_length(K, get_alignment_size(a)) + K
        torch._dynamo.mark_dynamic(b, 1)

        # 使用unittest.mock.patch装饰器，跳过性能评估时间的计算
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            # 执行前向传播计算
            res1 = fn(a, b)
            # 编译模型并运行获取生成的代码及结果
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            # 检查生成的代码中是否包含对齐后的K值
            FileCheck().check(f"K = {aligned_k}").run(code)

        # 断言两种计算方式得到的结果是否一致
        self.assertEqual(res1, res2)

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    # 应用装饰器配置最大自动调优和使用TRITON作为后端的自动调优
    def test_pad_mm_dyn_k(self):
        # 设置矩阵维度
        M = 21
        K = 80
        N = 30

        # 定义模型类，继承自torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 前向传播方法，计算矩阵相乘
            def forward(self, a, b):
                return torch.mm(a, b)

        # 创建Model类的实例，并将其部署到CUDA设备上
        fn = Model().cuda()

        # 生成随机的CUDA张量a和b，按指定的步幅分布
        a = rand_strided((M, K), (K, 1), device="cuda", dtype=torch.float32)
        b = rand_strided((K, N), (1, K), device="cuda", dtype=torch.float32)

        # TODO: 获取正确的对齐需要运行在新添加节点上的模式匹配器
        # 获取对齐后的M值，并将动态特性标记为1或0
        aligned_m = get_padded_length(M, get_alignment_size(a)) + M
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)

        # 使用unittest.mock.patch装饰器，跳过性能评估时间的计算
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            # 执行前向传播计算
            res1 = fn(a, b)
            # 编译模型并运行获取生成的代码及结果
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            # 检查生成的代码中是否包含对齐后的M值
            FileCheck().check(f"M = {aligned_m}").run(code)

        # 断言两种计算方式得到的结果是否一致
        self.assertEqual(res1, res2)
    def test_pad_mm_dyn_mnk(self):
        M = 20  # 定义矩阵 A 的行数
        K = 81  # 定义矩阵 A 的列数，也是矩阵 B 的行数
        N = 30  # 定义矩阵 B 的列数

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                # 实现矩阵乘法运算
                return torch.mm(a, b)

        fn = Model().cuda()  # 实例化模型，并移动到 GPU 上
        a = rand_strided((M, K), (K, 1), device="cuda", dtype=torch.float32)  # 生成随机张量 a
        b = rand_strided((K, N), (1, K), device="cuda", dtype=torch.float32)  # 生成随机张量 b
        torch._dynamo.mark_dynamic(a, 0)  # 标记张量 a 的动态维度为 0
        torch._dynamo.mark_dynamic(a, 1)  # 标记张量 a 的动态维度为 1
        torch._dynamo.mark_dynamic(b, 0)  # 标记张量 b 的动态维度为 0
        torch._dynamo.mark_dynamic(b, 1)  # 标记张量 b 的动态维度为 1
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)  # 使用模型 fn 计算 a 和 b 的乘积
            compiled_fn = torch.compile(fn)  # 编译模型 fn
            res2, (code,) = run_and_get_code(compiled_fn, a, b)  # 运行编译后的模型，并获取运行结果和代码
        self.assertEqual(res1, res2)  # 断言两种计算方法的结果应相等

    @inductor_config.patch(force_shape_pad=True)
    def test_zero_dim(self):
        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        x = torch.randn(100).cuda()  # 生成一个大小为 100 的随机张量 x，并移动到 GPU 上
        a = torch.randn(0, 10).cuda()  # 生成一个大小为 0x10 的随机张量 a，并移动到 GPU 上
        b = torch.randn(10, 100).cuda()  # 生成一个大小为 10x100 的随机张量 b，并移动到 GPU 上
        self.assertEqual(torch.compile(addmm)(x, a, b), addmm(x, a, b))  # 断言编译后的 addmm 函数与原始函数结果相等

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    def test_pad_bmm_dyn_b(self):
        B = 10  # 定义批次大小
        M = 128  # 定义矩阵 A 的行数
        K = 33   # 定义矩阵 A 的列数，也是矩阵 B 的行数
        N = 40   # 定义矩阵 B 的列数

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                # 实现批次矩阵乘法运算
                return torch.bmm(a, b)

        fn = Model().cuda()  # 实例化模型，并移动到 GPU 上
        a = torch.randn(B, M, K, device="cuda", dtype=torch.float32)  # 生成随机张量 a
        b = torch.randn(B, K, N, device="cuda", dtype=torch.float32)  # 生成随机张量 b
        aligned_k = get_padded_length(K, get_alignment_size(a)) + K  # 计算对齐后的 K 值
        torch._dynamo.mark_dynamic(a, 0)  # 标记张量 a 的动态维度为 0
        torch._dynamo.mark_dynamic(b, 0)  # 标记张量 b 的动态维度为 0
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)  # 使用模型 fn 计算 a 和 b 的批次乘积
            compiled_fn = torch.compile(fn)  # 编译模型 fn
            res2, (code,) = run_and_get_code(compiled_fn, a, b)  # 运行编译后的模型，并获取运行结果和代码
            FileCheck().check(f"K = {aligned_k}").run(code)  # 检查编译后的代码中是否包含对齐后的 K 值
        self.assertEqual(res1, res2)  # 断言两种计算方法的结果应相等
    def test_pad_bmm_dyn_k(self):
        B = 10  # 定义批量大小 B
        M = 128  # 定义矩阵 A 的维度 M
        K = 40   # 定义矩阵 A 和矩阵 B 公共维度 K
        N = 41   # 定义矩阵 B 的维度 N

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                return torch.bmm(a, b)  # 执行批量矩阵乘法操作

        fn = Model().cuda()  # 创建一个 CUDA 加速的模型实例
        a = torch.randn(B, M, K, device="cuda", dtype=torch.float32)  # 生成随机的输入张量 a
        b = torch.randn(B, K, N, device="cuda", dtype=torch.float32)  # 生成随机的输入张量 b
        aligned_n = get_padded_length(N, get_alignment_size(b)) + N  # 计算对齐后的长度
        torch._dynamo.mark_dynamic(a, 2)  # 将张量 a 的第 2 维标记为动态维度
        torch._dynamo.mark_dynamic(b, 1)  # 将张量 b 的第 1 维标记为动态维度
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)  # 执行模型的前向传播
            compiled_fn = torch.compile(fn)  # 编译模型
            res2, (code,) = run_and_get_code(compiled_fn, a, b)  # 运行并获取编译后的代码和结果
            FileCheck().check(f"N = {aligned_n}").run(code)  # 检查编译后的代码中是否包含对齐后长度的检查
        self.assertEqual(res1, res2)  # 断言两种执行方式的结果一致

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    def test_pad_bmm_dyn_bm(self):
        B = 10  # 定义批量大小 B
        M = 128  # 定义矩阵 A 的维度 M
        K = 40   # 定义矩阵 A 和矩阵 B 公共维度 K
        N = 41   # 定义矩阵 B 的维度 N

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                return torch.bmm(a, b)  # 执行批量矩阵乘法操作

        fn = Model().cuda()  # 创建一个 CUDA 加速的模型实例
        a = torch.randn(B, M, K, device="cuda", dtype=torch.float32)  # 生成随机的输入张量 a
        b = torch.randn(B, K, N, device="cuda", dtype=torch.float32)  # 生成随机的输入张量 b
        aligned_n = get_padded_length(N, get_alignment_size(b)) + N  # 计算对齐后的长度
        torch._dynamo.mark_dynamic(a, 0)  # 将张量 a 的第 0 维标记为动态维度
        torch._dynamo.mark_dynamic(a, 1)  # 将张量 a 的第 1 维标记为动态维度
        torch._dynamo.mark_dynamic(b, 0)  # 将张量 b 的第 0 维标记为动态维度
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)  # 执行模型的前向传播
            compiled_fn = torch.compile(fn)  # 编译模型
            res2, (code,) = run_and_get_code(compiled_fn, a, b)  # 运行并获取编译后的代码和结果
            FileCheck().check(f"N = {aligned_n}").run(code)  # 检查编译后的代码中是否包含对齐后长度的检查
        self.assertEqual(res1, res2)  # 断言两种执行方式的结果一致

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    def test_pad_addmm_dyn_m(self):
        M = 128
        K = 33
        N = 40

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                # 使用 torch.addmm 执行矩阵相乘和相加操作
                return torch.addmm(a, b, c)

        fn = Model().cuda()
        # 在 GPU 上生成随机张量 a, b, c
        a = torch.randn(M, N, device="cuda", dtype=torch.float32)
        b = torch.randn(M, K, device="cuda", dtype=torch.float32)
        c = torch.randn(K, N, device="cuda", dtype=torch.float32)
        # 标记张量 a, b 为动态张量
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            # 在模型 fn 上执行前向传播
            res1 = fn(a, b, c)
            # 编译模型 fn
            compiled_fn = torch.compile(fn)
            # 运行编译后的模型 fn，并获取结果
            res2, (code,) = run_and_get_code(compiled_fn, a, b, c)
            # 检查代码中是否有包含 aligned_k 的部分
            FileCheck().check(f"K = {aligned_k}").run(code)
        # 断言两次执行的结果是否一致
        self.assertEqual(res1, res2)

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    def test_pad_addmm_dyn_mn(self):
        M = 128
        K = 33
        N = 40

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                # 使用 torch.addmm 执行矩阵相乘和相加操作
                return torch.addmm(a, b, c)

        fn = Model().cuda()
        # 在 GPU 上生成随机张量 a, b, c
        a = torch.randn(M, N, device="cuda", dtype=torch.float32)
        b = torch.randn(M, K, device="cuda", dtype=torch.float32)
        c = torch.randn(K, N, device="cuda", dtype=torch.float32)
        # 标记张量 a, b, c 为动态张量
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)
        torch._dynamo.mark_dynamic(c, 1)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            # 在模型 fn 上执行前向传播
            res1 = fn(a, b, c)
            # 编译模型 fn
            compiled_fn = torch.compile(fn)
            # 运行编译后的模型 fn，并获取结果
            # 检查代码中是否没有 padding 操作
            FileCheck().check(f"K = {K}").run(code)
        # 断言两次执行的结果是否一致
        self.assertEqual(res1, res2)

    @inductor_config.patch(force_shape_pad=True)
    def test_pad_single_cat(self):
        @torch.compile()
        def foo(x, y):
            # 执行矩阵乘法运算
            return x @ y

        # 生成两个随机张量
        inps = [torch.rand([5, 5], device="cuda") for _ in range(2)]
        # 调用 foo 函数执行矩阵乘法
        out = foo(*inps)
        # 断言输出结果与预期结果是否一致
        self.assertEqual(out, inps[0] @ inps[1])

    @inductor_config.patch(force_shape_pad=True)
    @fresh_inductor_cache()
    # 定义一个测试方法，用于测试带偏置的二维加法和矩阵乘法操作
    def test_pad_addmm_2d_bias(self):
        # 使用 torch 的编译装饰器，编译以下函数
        @torch.compile()
        def foo(input, x, y):
            # 调用 torch 操作的 addmm 函数，执行矩阵乘法并加上偏置
            return torch.ops.aten.addmm(input, x, y)

        # 第一组循环，测试不同维度的输入矩阵组合
        for a in [1, 4]:
            for b in [1, 6]:
                # 创建随机数填充的 CUDA 设备上的张量输入
                inps = (
                    torch.rand([a, b], device="cuda"),
                    torch.rand([4, 5], device="cuda"),
                    torch.rand([5, 6], device="cuda"),
                )
                # 调用 foo 函数计算输出
                out = foo(*inps)
                # 直接调用 torch 的 addmm 操作计算期望输出
                out_eager = torch.ops.aten.addmm(*inps)
                # 使用断言比较 foo 函数和直接调用的输出是否一致
                self.assertEqual(out, out_eager)

        # 第二组循环，测试不同维度的输入矩阵组合
        for a in [1, 6]:
            # 创建随机数填充的 CUDA 设备上的张量输入
            inps = (
                torch.rand([a], device="cuda"),
                torch.rand([4, 5], device="cuda"),
                torch.rand([5, 6], device="cuda"),
            )
            # 调用 foo 函数计算输出
            out = foo(*inps)
            # 直接调用 torch 的 addmm 操作计算期望输出
            out_eager = torch.ops.aten.addmm(*inps)
            # 使用断言比较 foo 函数和直接调用的输出是否一致
            self.assertEqual(out, out_eager)

    # 使用特定的配置参数修饰测试方法，强制进行形状的填充
    @inductor_config.patch(force_shape_pad=True)
    def test_pad_batch(self):
        # 定义矩阵的维度参数
        m = 6
        n = 9
        k = 11
        # 定义批处理大小
        batch_size = 3
        # 创建全为 1 的 float16 类型的张量 mat1 和 mat2，存储在 CUDA 设备上
        mat1 = torch.ones((batch_size, m, k), device="cuda", dtype=torch.float16)
        mat2 = torch.ones((batch_size, k, n), device="cuda", dtype=torch.float16)
        # 获取 mat1 的对齐大小
        expected_alignment = get_alignment_size(mat1)

        # 断言对齐大小为 8，适用于 float16 类型
        assert expected_alignment == 8, "Alignment for float16 should be 8"
        # 断言 mat1 和 mat2 应通过常见的填充标准
        assert should_pad_common(
            mat1, mat2
        ), "This should pass the common padding criteria"

        # 使用 torch 的编译装饰器，编译以下函数
        @torch.compile()
        def bmm(mat1, mat2):
            # 调用 torch 的批量矩阵乘法函数 bmm
            return torch.bmm(mat1, mat2)

        # 运行并获取 bmm 函数的执行结果和生成的代码
        res2, (code,) = run_and_get_code(bmm, mat1, mat2)
        # 计算使用 bmm 函数得到的预期结果
        bmm_expected_result = torch.bmm(mat1, mat2)
        # 在调用代码中，期望看到每个输入都有一个单独的填充，然后应看到输出的填充分配
        FileCheck().check("del async_compile").check_count(
            ".run(", 2, exactly=True
        ).check("empty_strided_cuda((3, 8, 16)").run(code)

        # 使用 allclose 函数检查 res2 和 bmm_expected_result 是否近似相等
        assert torch.allclose(
            res2, bmm_expected_result
        ), "BMM results are not identical"

    # 使用新的感应器缓存配置修饰测试方法
    @fresh_inductor_cache()
    # 定义一个测试方法，用于测试排除填充（padding）情况
    def test_exclude_padding(self):
        # 定义一个矩阵乘法的编译版本函数
        @torch.compile()
        def mm(a, b):
            return a @ b

        # 调用矩阵乘法函数，传入两个随机生成的 CUDA 设备上的张量
        mm(torch.rand([25, 25], device="cuda"), torch.rand([25, 25], device="cuda"))
        
        # 获取填充缓存的本地缓存
        local_cache = get_pad_cache().get_local_cache()
        
        # 断言本地缓存的长度为2
        self.assertTrue(len(local_cache) == 2)
        
        # 使用文件检查器检查本地缓存中是否确实包含两个 "exclude_pad:False" 字符串
        FileCheck().check_count("exclude_pad:False", 2, exactly=True).run(
            repr(local_cache)
        )

        # 重新定义矩阵乘法的编译版本函数，这次在第一个张量上加1再进行乘法操作
        @torch.compile()
        def mm(a, b):
            return (a + 1) @ b

        # 再次调用修改后的矩阵乘法函数，传入两个随机生成的 CUDA 设备上的张量
        mm(torch.rand([25, 25], device="cuda"), torch.rand([25, 25], device="cuda"))
        
        # 获取填充缓存的本地缓存
        local_cache = get_pad_cache().get_local_cache()
        
        # 断言本地缓存的长度为3，因为已经执行了第二次矩阵乘法操作
        self.assertTrue(len(local_cache) == 3)

        # 使用文件检查器检查本地缓存中是否确实包含三个 "exclude_pad:False" 字符串
        FileCheck().check_count("exclude_pad:False", 3, exactly=True).run(
            repr(local_cache)
        )
        
        # 使用文件检查器检查本地缓存中是否确实包含一个 "exclude_pad:True" 字符串
        FileCheck().check_count("exclude_pad:True", 1, exactly=True).run(
            repr(local_cache)
        )
# 如果当前脚本被直接执行（而不是被导入到其他模块中执行），则执行以下代码
if __name__ == "__main__":
    # 如果系统配置中有 CUDA 加速设备可用
    if HAS_CUDA:
        # 运行测试函数
        run_tests()
```