# `.\pytorch\test\inductor\test_cudacodecache.py`

```
# Owner(s): ["module: inductor"]

import ctypes  # 引入 ctypes 库，用于与 C 语言兼容的数据类型转换
import unittest  # 引入 unittest 框架，用于编写和运行单元测试

import torch  # 引入 PyTorch 深度学习库

from torch._inductor import config  # 引入模块 config，用于配置相关
from torch._inductor.async_compile import AsyncCompile  # 引入异步编译器 AsyncCompile
from torch._inductor.codecache import CUDACodeCache  # 引入 CUDA 代码缓存模块 CUDACodeCache
from torch._inductor.codegen.cuda.cuda_env import nvcc_exist  # 引入 nvcc_exist 函数，检查是否存在 nvcc 编译器
from torch._inductor.exc import CUDACompileError  # 引入 CUDA 编译错误异常类
from torch._inductor.test_case import TestCase as InductorTestCase  # 引入 InductorTestCase 作为 TestCase 的别名

_SOURCE_CODE = r"""

#include <stdio.h>

__global__
void saxpy_device(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

extern "C" {

__attribute__((__visibility__("default")))
int saxpy(int n, float a, float *x, float *y) {
  // Perform SAXPY
  saxpy_device<<<(n+255)/256, 256>>>(n, a, x, y);
  return 0;
}

}
"""

# 在特定条件下跳过测试，这里检查是否在 fbcode 环境下，fbcode 需要不同的 CUDA_HOME 设置
@unittest.skipIf(config.is_fbcode(), "fbcode requires different CUDA_HOME setup")
class TestCUDACodeCache(InductorTestCase):
    def test_cuda_load(self):
        # 测试 .o 和 .so 编译的两种方式
        object_file_path, object_hash_key, source_code_path0 = CUDACodeCache.compile(
            _SOURCE_CODE, "o"
        )
        dll_wrapper, so_hash_key, source_code_path1 = CUDACodeCache.load(
            _SOURCE_CODE, "so"
        )
        self.assertNotEqual(source_code_path0, source_code_path1)  # 断言编译时的源代码路径不同
        self.assertNotEqual(object_hash_key, so_hash_key)  # 断言编译生成的对象哈希键值不同

        # 测试加载并调用 .so 中的函数
        x = torch.rand(10).float().cuda()  # 生成一个在 GPU 上的随机张量 x
        y = torch.rand(10).float().cuda()  # 生成一个在 GPU 上的随机张量 y
        a = 5.0  # 设定一个浮点数常量 a
        expected_y = a * x + y  # 期望的计算结果
        res = dll_wrapper.saxpy(
            ctypes.c_int(10),  # 将整数 10 转换为 ctypes.c_int 类型
            ctypes.c_float(a),  # 将浮点数 a 转换为 ctypes.c_float 类型
            ctypes.c_void_p(x.data_ptr()),  # 获取张量 x 的数据指针并转换为 void* 类型
            ctypes.c_void_p(y.data_ptr()),  # 获取张量 y 的数据指针并转换为 void* 类型
        )
        torch.testing.assert_close(y, expected_y)  # 断言实际结果与期望结果相近

    def test_compilation_error(self):
        error_source_code = _SOURCE_CODE.replace("saxpy_device", "saxpy_wrong", 1)  # 将源代码中的一个函数名替换为另一个
        with self.assertRaises(CUDACompileError):  # 断言捕获到 CUDA 编译错误异常
            CUDACodeCache.compile(error_source_code, "o")  # 编译修改后的错误源代码为 .o 文件

    def test_async_compile(self):
        async_compile = AsyncCompile()  # 创建异步编译器对象
        compiled_res = async_compile.cuda(_SOURCE_CODE, "so")  # 异步编译源代码为 .so 文件
        async_compile.wait(globals())  # 等待异步编译完成并使得全局作用域可见编译结果

        # 测试加载并调用 .so 中的函数
        x = torch.rand(5).float().cuda()  # 生成一个在 GPU 上的随机张量 x
        y = torch.rand(5).float().cuda()  # 生成一个在 GPU 上的随机张量 y
        a = 2.0  # 设定一个浮点数常量 a
        expected_y = a * x + y  # 期望的计算结果
        res = compiled_res.result().saxpy(
            ctypes.c_int(5),  # 将整数 5 转换为 ctypes.c_int 类型
            ctypes.c_float(a),  # 将浮点数 a 转换为 ctypes.c_float 类型
            ctypes.c_void_p(x.data_ptr()),  # 获取张量 x 的数据指针并转换为 void* 类型
            ctypes.c_void_p(y.data_ptr()),  # 获取张量 y 的数据指针并转换为 void* 类型
        )
        torch.testing.assert_close(y, expected_y)  # 断言实际结果与期望结果相近

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if nvcc_exist():  # 检查是否存在 nvcc 编译器
        run_tests("cuda")  # 运行 CUDA 相关的测试
```