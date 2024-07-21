# `.\pytorch\test\test_kernel_launch_checks.py`

```py
# 导入所需模块和函数
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.check_kernel_launches import (
    check_cuda_kernel_launches, check_code_for_cuda_kernel_launches
)

# 定义测试类，继承自TestCase类
class AlwaysCheckCudaLaunchTest(TestCase):
    
    # 定义测试方法，验证CUDA内核启动检查的正则表达式对多种情况是否有效
    def test_check_code(self):
        
        """Verifies that the regex works for a few different situations"""
        
        # 测试CUDA内核启动检查函数对多种不同间距情况的验证
        self.assertEqual(2, check_code_for_cuda_kernel_launches("""
some_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);
C10_CUDA_KERNEL_LAUNCH_CHECK();
some_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);

some_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);
C10_CUDA_KERNEL_LAUNCH_CHECK();
some_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);
some_other_stuff;
some_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);
C10_CUDA_KERNEL_LAUNCH_CHECK();
some_function_call<TemplateArg><<<1,2,0,stream>>> (arg1,arg2,arg3);
C10_CUDA_KERNEL_LAUNCH_CHECK();
some_function_call<TemplateArg><<<1,2,0,stream>>> ( arg1 , arg2 , arg3 ) ;

    C10_CUDA_KERNEL_LAUNCH_CHECK();
        """))

        # 验证CUDA内核启动检查函数对宏的有效性
        self.assertEqual(0, check_code_for_cuda_kernel_launches(r"""
#define SOME_MACRO(x) some_function_call<<<1,2>>> ( x ) ;  \
    C10_CUDA_KERNEL_LAUNCH_CHECK();

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)  \
  indexAddSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM> \
        <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                \
          selfInfo, sourceInfo, indexInfo,                                               \
          selfAddDim, sourceAddDim, sliceSize, selfAddDimSize);                          \
      C10_CUDA_KERNEL_LAUNCH_CHECK();

# 启动 CUDA 核函数，使用指定的线程格 `smallIndexGrid` 和线程块 `smallIndexBlock`，无动态共享内存，使用 `stream` 流
# 传递给核函数的参数依次是：selfInfo, sourceInfo, indexInfo, selfAddDim, sourceAddDim, sliceSize, selfAddDimSize
# 检查 CUDA 核函数启动是否成功，如果失败则抛出异常

        # Does it work for lambdas?
        self.assertEqual(1, check_code_for_cuda_kernel_launches(r"""
            rrelu_with_noise_cuda_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
                    numel,
                    rng_engine_inputs,
                    output_data,
                    input_data,
                    noise_data,
                    lower,
                    upper,
                    [] __device__ (curandStatePhilox4_32_10_t* state) {
                    return curand_uniform2_double(state);
                    });
                    C10_CUDA_KERNEL_LAUNCH_CHECK();

            rrelu_with_noise_cuda_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
                    numel,
                    rng_engine_inputs,
                    output_data,
                    input_data,
                    noise_data,
                    lower,
                    upper,
                    [] __device__ (curandStatePhilox4_32_10_t* state) {
                    return curand_uniform2_double(state);
                    });
                    uh oh;
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
        """))

# 检查 CUDA 核函数启动的辅助函数，期望返回值为 1
# 在 CUDA 代码块中，启动 `rrelu_with_noise_cuda_kernel` 核函数，使用指定的线程格 `grid` 和线程块 `block`，无动态共享内存，使用 `stream` 流
# 传递给核函数的参数依次是：numel, rng_engine_inputs, output_data, input_data, noise_data, lower, upper
# 定义了一个设备端的 Lambda 函数，返回值为使用 `curand_uniform2_double` 生成的随机数，接受一个 `curandStatePhilox4_32_10_t` 类型的指针作为参数
# 检查 CUDA 核函数启动是否成功，如果失败则抛出异常

    def test_check_cuda_launches(self):
        unsafeLaunchesCount = check_cuda_kernel_launches()
        self.assertTrue(unsafeLaunchesCount == 0)

# 测试函数：检查 CUDA 核函数启动的数量
# 调用 `check_cuda_kernel_launches` 函数，返回未安全启动的核函数数量
# 断言未安全启动的核函数数量为 0，如果不是则测试失败
# 如果当前脚本作为主程序运行（而不是被导入到其他脚本中），则执行以下代码块
if __name__ == '__main__':
    # 调用名为 run_tests() 的函数，用于执行测试或主要的程序逻辑
    run_tests()
```