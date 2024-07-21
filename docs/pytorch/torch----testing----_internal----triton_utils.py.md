# `.\pytorch\torch\testing\_internal\triton_utils.py`

```
# mypy: ignore-errors

# 导入 unittest 模块，用于单元测试
import unittest

# 从 torch.testing._internal.inductor_utils 导入 HAS_CUDA 和 HAS_GPU
from torch.testing._internal.inductor_utils import HAS_CUDA, HAS_GPU

# 从 torch.utils._triton 导入 has_triton 函数
from torch.utils._triton import has_triton

# 如果系统支持 Triton，执行以下代码块
if has_triton():
    # 导入 triton 库及其 language 模块
    import triton
    from triton import language as tl

    # 定义 add_kernel 函数，使用 triton.jit 进行 JIT 编译
    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # 获取程序的 ID，axis=0 表示一维计算
        pid = tl.program_id(axis=0)
        # 计算当前块的起始位置
        block_start = pid * BLOCK_SIZE
        # 生成偏移量数组，表示当前块内的索引
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # 创建掩码，用于检测偏移量是否超出元素总数
        mask = offsets < n_elements
        # 从输入指针加载数据到 x
        x = tl.load(in_ptr0 + offsets, mask=mask)
        # 从输入指针加载数据到 y，根据 ARGS_PASSED 可选参数决定是否加载
        y = tl.load(in_ptr1 + offsets, mask=mask) if ARGS_PASSED == "two" else 0
        # 执行加法操作
        output = x + y
        # 将结果存储到输出指针对应位置
        tl.store(out_ptr + offsets, output, mask=mask)

    # 定义带可选参数的 add_kernel 函数，使用 triton.jit 进行 JIT 编译
    @triton.jit
    def add_kernel_with_optional_param(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        ARGS_PASSED: "tl.constexpr",
        BLOCK_SIZE: "tl.constexpr",
    ):
        # 获取程序的 ID，axis=0 表示一维计算
        pid = tl.program_id(axis=0)
        # 计算当前块的起始位置
        block_start = pid * BLOCK_SIZE
        # 生成偏移量数组，表示当前块内的索引
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # 创建掩码，用于检测偏移量是否超出元素总数
        mask = offsets < n_elements
        # 从输入指针加载数据到 x
        x = tl.load(in_ptr0 + offsets, mask=mask)
        # 根据 ARGS_PASSED 的值决定是否加载第二个输入指针的数据到 y
        if ARGS_PASSED == "two":
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
        else:
            output = x
        # 将结果存储到输出指针对应位置
        tl.store(out_ptr + offsets, output, mask=mask)

    # 定义带自动调优功能的 add_kernel_autotuned 函数，使用 triton.jit 进行 JIT 编译
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 128}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
        ],
        key=[],
    )
    @triton.jit
    def add_kernel_autotuned(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # 获取程序的 ID，axis=0 表示一维计算
        pid = tl.program_id(axis=0)
        # 计算当前块的起始位置
        block_start = pid * BLOCK_SIZE
        # 生成偏移量数组，表示当前块内的索引
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # 创建掩码，用于检测偏移量是否超出元素总数
        mask = offsets < n_elements
        # 从输入指针加载数据到 x 和 y
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        # 执行加法操作
        output = x + y
        # 将结果存储到输出指针对应位置
        tl.store(out_ptr + offsets, output, mask=mask)

    # 定义带自动调优功能和多参数的 add_kernel_autotuned 函数，使用 triton.jit 进行 JIT 编译
    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_Y": 128}, num_stages=3, num_warps=8
            ),
            triton.Config(
                {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_Y": 128}, num_stages=4, num_warps=4
            ),
            triton.Config(
                {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 64}, num_stages=3, num_warps=8
            ),
            triton.Config(
                {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 64}, num_stages=4, num_warps=4
            ),
        ],
        key=[],
    )
    @triton.jit
    def add_kernel_autotuned_with_multiple_params(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE_X: "tl.constexpr",
        BLOCK_SIZE_Y: "tl.constexpr",
    ):
        # 获取程序的 ID，axis=0 表示一维计算
        pid = tl.program_id(axis=0)
        # 计算当前块的起始位置
        block_start = pid * BLOCK_SIZE_X * BLOCK_SIZE_Y
        # 生成偏移量数组，表示当前块内的索引
        offsets_x = block_start + tl.arange(0, BLOCK_SIZE_X)
        offsets_y = block_start + tl.arange(0, BLOCK_SIZE_Y)
        # 创建掩码，用于检测偏移量是否超出元素总数
        mask_x = offsets_x < n_elements
        mask_y = offsets_y < n_elements
        # 从输入指针加载数据到 x 和 y
        x = tl.load(in_ptr0 + offsets_x, mask=mask_x)
        y = tl.load(in_ptr1 + offsets_y, mask=mask_y)
        # 执行加法操作
        output = x + y
        # 将结果存储到输出指针对应位置
        tl.store(out_ptr + offsets_x, output, mask=mask_x)
        tl.store(out_ptr + offsets_y, output, mask=mask_y)
    )
    @triton.jit
    def add_kernel_2d_autotuned(
        in_ptr0,
        in_ptr1,
        out_ptr,
        x_elements,
        y_elements,
        BLOCK_SIZE_X: "tl.constexpr",
        BLOCK_SIZE_Y: "tl.constexpr",
    ):
        # 计算 x 方向的偏移量
        xoffset = tl.program_id(0) * BLOCK_SIZE_X
        # 生成 x 方向的索引
        xindex = xoffset + tl.arange(0, BLOCK_SIZE_X)[:, None]
        # 创建 x 方向的掩码
        xmask = xindex < x_elements
        # 计算 y 方向的偏移量
        yoffset = tl.program_id(1) * BLOCK_SIZE_Y
        # 生成 y 方向的索引
        yindex = yoffset + tl.arange(0, BLOCK_SIZE_Y)[None, :]
        # 创建 y 方向的掩码
        ymask = yindex < y_elements
        # 计算第一个输入数据的索引
        x1 = xindex
        # 计算第二个输入数据的索引
        y0 = yindex
        # 从第一个输入指针加载数据到 tmp0
        tmp0 = tl.load(in_ptr0 + (x1 + (x_elements * y0)), xmask & ymask)
        # 从第二个输入指针加载数据到 tmp1
        tmp1 = tl.load(in_ptr0 + (y0 + (y_elements * x1)), xmask & ymask)
        # 计算结果并存储到 tmp2
        tmp2 = tmp0 + tmp1
        # 将结果存储到输出指针
        tl.store(out_ptr + (x1 + (x_elements * y0)), tmp2, xmask & ymask)

    @triton.jit
    def add_kernel_with_scaling(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        scaling_factor,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # 获取程序 ID
        pid = tl.program_id(axis=0)
        # 计算块的起始位置
        block_start = pid * BLOCK_SIZE
        # 生成偏移量数组
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # 创建掩码
        mask = offsets < n_elements
        # 从第一个输入指针加载数据到 x
        x = tl.load(in_ptr0 + offsets, mask=mask)
        # 从第二个输入指针加载数据到 y
        y = tl.load(in_ptr1 + offsets, mask=mask)
        # 计算加权和并乘以缩放因子
        output = (x + y) * scaling_factor
        # 将结果存储到输出指针
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def mul2_kernel(
        in_ptr0,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # 获取程序 ID
        pid = tl.program_id(axis=0)
        # 计算块的起始位置
        block_start = pid * BLOCK_SIZE
        # 生成偏移量数组
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # 创建掩码
        mask = offsets < n_elements
        # 从输入指针加载数据到 x
        x = tl.load(in_ptr0 + offsets, mask=mask)
        # 计算乘以2的结果
        output = 2 * x
        # 将结果存储到输出指针
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def mul2_inplace_kernel(
        ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # 获取程序 ID
        pid = tl.program_id(axis=0)
        # 计算块的起始位置
        block_start = pid * BLOCK_SIZE
        # 生成偏移量数组
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # 创建掩码
        mask = offsets < n_elements
        # 从输入指针加载数据到 x
        x = tl.load(ptr + offsets, mask=mask)
        # 计算乘以2的结果
        output = 2 * x
        # 将结果存储回输入指针
        tl.store(ptr + offsets, output, mask=mask)

    @triton.jit
    def zero_negs(x):
        # 使用 Triton 的 where 函数实现将负数置零
        return tl.where(x >= 0, x, 0)

    @triton.jit
    def indirection_kernel(
        in_ptr0,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
        ACTIVATION: "tl.constexpr",
    ):
        # 获取程序 ID
        pid = tl.program_id(axis=0)
        # 计算块的起始位置
        block_start = pid * BLOCK_SIZE
        # 生成偏移量数组
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # 创建掩码
        mask = offsets < n_elements
        # 根据 ACTIVATION 参数选择不同的内核函数进行调用
        if ACTIVATION == "mul2_inplace_kernel":
            mul2_inplace_kernel(in_ptr0, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        elif ACTIVATION == "add_kernel":
            add_kernel(in_ptr0, in_ptr0, out_ptr, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        # 从输入指针加载数据到 x
        x = tl.load(in_ptr0 + offsets, mask=mask)
        # 将结果存储到输出指针
        tl.store(out_ptr + offsets, x, mask=mask)
    # 定义一个函数，用于执行双重步进的计算核函数
    def double_strided_kernel(
        in_ptr,
        out_ptr,
        in_y_stride,
        out_y_stride,
        X_BLOCK_SIZE: "tl.constexpr",
        Y_BLOCK_SIZE: "tl.constexpr",
    ):
        # 获取当前线程在 x 轴和 y 轴上的索引
        xid = tl.program_id(axis=0)
        yid = tl.program_id(axis=1)
        # 计算当前线程在二维数据块中的起始位置
        x_start = xid * X_BLOCK_SIZE
        y_start = yid * Y_BLOCK_SIZE
        # 计算 x 和 y 方向上的偏移量
        x_offsets = x_start + tl.arange(0, X_BLOCK_SIZE)
        y_offsets = y_start + tl.arange(0, Y_BLOCK_SIZE)
        # 计算输入数据的起始地址偏移量
        src_offsets = y_offsets[:, None] * in_y_stride + x_offsets[None, :]
        # 计算输出数据的起始地址偏移量
        dst_offsets = y_offsets[:, None] * out_y_stride + x_offsets[None, :]
        # 从输入地址加载数据
        src = tl.load(in_ptr + src_offsets)
        # 将加载的数据乘以2后存储到输出地址
        tl.store(out_ptr + dst_offsets, src * 2.0)
    
    # 定义一个使用内联汇编的计算核函数
    @triton.jit
    def inline_asm_kernel(X, Y, Z, n: "tl.constexpr", BLOCK: "tl.constexpr"):
        # 从地址X处加载长度为BLOCK的数据
        x = tl.load(X + tl.arange(0, BLOCK))
        # 从地址Y处加载长度为BLOCK的数据
        y = tl.load(Y + tl.arange(0, BLOCK))
        # 创建一个长度为BLOCK的常量数组
        s = tl.full([BLOCK], n, tl.int32)
        # 执行内联汇编操作，并将结果存储到z中
        z = tl.inline_asm_elementwise(
            "shf.l.wrap.b32 $0, $1, $2, $3;",
            "=r,r, r, r",
            [x, y, s],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        # 将结果z存储到地址Z处
        tl.store(Z + tl.arange(0, BLOCK), z)
    
    # 定义一个带有块指针的加法计算核函数
    @triton.jit
    def add_kernel_with_block_ptr(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # 获取当前线程在 x 轴上的索引
        pid = tl.program_id(axis=0)
        # 计算当前块的起始位置
        block_start = pid * BLOCK_SIZE
        # 从 x_ptr 地址的块指针中加载数据块
        x = tl.load(
            tl.make_block_ptr(
                base=x_ptr,
                shape=[n_elements],
                strides=[1],
                offsets=[block_start],
                block_shape=[BLOCK_SIZE],
                order=[0],
            ),
            boundary_check=[0],
        )
        # 从 y_ptr 地址的块指针中加载数据块
        y = tl.load(
            tl.make_block_ptr(
                base=y_ptr,
                shape=[n_elements],
                strides=[1],
                offsets=[block_start],
                block_shape=[BLOCK_SIZE],
                order=[0],
            ),
            boundary_check=[0],
        )
        # 计算输出数据
        output = x + y
        # 将结果存储到 output_ptr 地址的块指针中
        tl.store(
            tl.make_block_ptr(
                base=output_ptr,
                shape=[n_elements],
                strides=[1],
                offsets=[block_start],
                block_shape=[BLOCK_SIZE],
                order=[0],
            ),
            output,
            boundary_check=[0],
        )
    
    # 定义一个二维数据块指针的计算核函数
    @triton.jit
    def kernel_with_block_ptr_2d(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # 这部分代码未完整，需要根据实际情况添加注释
    ):
        # 获取当前线程的程序 ID，用于确定块的起始位置
        pid = tl.program_id(axis=0)
        # 计算当前块在数据中的起始位置
        block_start = pid * BLOCK_SIZE
        # 加载输入数据的部分块，只加载当前块的数据
        x = tl.load(
            tl.make_block_ptr(
                base=x_ptr,
                shape=[n_elements, 1],
                strides=[1, 1],
                offsets=[block_start, 0],
                block_shape=[BLOCK_SIZE, 1],
                order=[1, 0],
            ),
            boundary_check=[0],  # 设置边界检查参数为 0
        )
        # 将加载的数据输出
        output = x
        # 将数据存储回指定位置，只存储当前块的数据
        tl.store(
            tl.make_block_ptr(
                base=output_ptr,
                shape=[n_elements, 1],
                strides=[1, 1],
                offsets=[block_start, 0],
                block_shape=[BLOCK_SIZE, 1],
                order=[1, 0],
            ),
            output,
            boundary_check=[0],  # 设置边界检查参数为 0
        )

    from triton.language import load, store

    @triton.jit
    def add_kernel_with_import(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # 获取当前线程的程序 ID，用于确定块的起始位置
        pid = tl.program_id(axis=0)
        # 计算当前块在数据中的起始位置
        block_start = pid * BLOCK_SIZE
        # 计算当前块内的偏移量范围
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # 创建一个布尔掩码，标记有效数据的位置
        mask = offsets < n_elements
        # 加载输入数据的部分块，只加载当前块内的有效数据
        x = load(in_ptr0 + offsets, mask=mask)
        y = load(in_ptr1 + offsets, mask=mask)
        # 对加载的数据进行加法操作
        output = x + y
        # 将结果存储回指定位置，只存储当前块内的有效数据
        store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def cond_op_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # 获取当前线程的程序 ID，用于确定块的起始位置
        pid = tl.program_id(axis=0)
        # 计算当前块在数据中的起始位置
        block_start = pid * BLOCK_SIZE
        # 计算当前块内的偏移量范围
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # 创建一个布尔掩码，标记有效数据的位置
        mask = offsets < n_elements
        # 加载输入数据的部分块，只加载当前块内的有效数据
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        # 根据程序 ID 判断执行加法还是乘法操作
        if tl.program_id(0) == 0:
            output = x + y
        else:
            output = x * y
        # 将结果存储回指定位置，只存储当前块内的有效数据
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def atomic_add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # 获取当前线程的程序 ID，用于确定块的起始位置
        pid = tl.program_id(axis=0)
        # 计算当前块在数据中的起始位置
        block_start = pid * BLOCK_SIZE
        # 计算当前块内的偏移量范围
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # 创建一个布尔掩码，标记有效数据的位置
        mask = offsets < n_elements
        # 加载输入数据的部分块，只加载当前块内的有效数据
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        # 对加载的数据进行加法操作，并使用原子加操作将结果存储回指定位置
        output = x + y
        tl.atomic_add(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def add_4_times_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # 此函数尚未完成，暂无注释
    # 定义一个使用 Triton 库编译的 JIT 函数，实现并行计算的加法内核
    @triton.jit
    def add_kernel_out_of_order_fn1(
        in_ptr0,
        in_ptr1,
        n_elements,
        out_ptr,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # 获取当前程序在指定轴上的唯一标识符
        pid = tl.program_id(axis=0)
        # 计算当前块的起始位置
        block_start = pid * BLOCK_SIZE
        # 生成当前块内的偏移量数组
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # 创建一个掩码，标识有效的偏移量范围
        mask = offsets < n_elements
        # 从输入指针0和指针1加载数据到变量x和y，仅对有效偏移应用掩码
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        
        # 进行两次循环，计算并存储结果到输出指针
        for i in range(2):
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)
        
        # 另一种计算并存储结果的方式，采用递减循环两次
        i = 2
        while i > 0:
            i -= 1
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)
    
    # 定义另一个使用 Triton 库编译的 JIT 函数，实现并行计算的加法内核（顺序无关）
    @triton.jit
    def add_kernel_out_of_order_fn2(
        in_ptr0,
        in_ptr1,
        n_elements,
        out_ptr,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # 获取当前程序在指定轴上的唯一标识符
        pid = tl.program_id(axis=0)
        # 计算当前块的起始位置
        block_start = pid * BLOCK_SIZE
        # 生成当前块内的偏移量数组
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # 创建一个掩码，标识有效的偏移量范围
        mask = offsets < n_elements
        # 从输入指针0和指针1加载数据到变量x和y，仅对有效偏移应用掩码
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        
        # 计算并存储结果到输出指针，顺序不影响计算的正确性
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)
```