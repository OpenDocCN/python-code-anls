# `.\pytorch\torch\_inductor\codegen\codegen_device_driver.py`

```py
# 导入 torch 库，用于与 CUDA 相关的操作
import torch

# 提供 aoti 模块启动 hip/cuda 驱动程序。该文件还用于单元测试目的

def cuda_kernel_driver() -> str:
    # CUDA_DRIVER_CHECK 宏定义，用于检查 CUDA 驱动程序执行结果
    source_codes = """
            #define CUDA_DRIVER_CHECK(EXPR)                    \\
            do {                                               \\
                CUresult code = EXPR;                          \\
                const char *msg;                               \\
                cuGetErrorString(code, &msg);                  \\
                if (code != CUDA_SUCCESS) {                    \\
                    throw std::runtime_error(                  \\
                        std::string("CUDA driver error: ") +   \\
                        std::string(msg));                     \\
                }                                              \\
            } while (0);

            namespace {

            // 定义 Grid 结构体，表示 CUDA 格子配置
            struct Grid {
                Grid(uint32_t x, uint32_t y, uint32_t z)
                  : grid_x(x), grid_y(y), grid_z(z) {}
                uint32_t grid_x;
                uint32_t grid_y;
                uint32_t grid_z;

                // 检查格子是否有效
                bool is_non_zero() {
                    return grid_x > 0 && grid_y > 0 && grid_z > 0;
                }
            };

            }  // 匿名命名空间

            // 加载 CUDA kernel 函数
            static inline CUfunction loadKernel(
                    std::string filePath,
                    const std::string &funcName,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string> &cubinDir = std::nullopt) {
                // 如果提供了 cubinDir，根据路径设置文件路径
                if (cubinDir) {
                    std::filesystem::path p1{*cubinDir};
                    std::filesystem::path p2{filePath};
                    filePath = (p1 / p2.filename()).string();
                }

                CUmodule mod;
                CUfunction func;
                
                // 加载 CUDA 模块
                CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
                // 获取 CUDA 函数
                CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
                
                // 如果需要设置共享内存大小，则设置相应属性
                if (sharedMemBytes > 0) {
                    CUDA_DRIVER_CHECK(cuFuncSetAttribute(
                        func,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        sharedMemBytes
                    ))
                }
                return func;
            }

            // 启动 CUDA kernel 函数
            static inline void launchKernel(
                    CUfunction func,
                    uint32_t gridX,
                    uint32_t gridY,
                    uint32_t gridZ,
                    uint32_t numWarps,
                    uint32_t sharedMemBytes,
                    void* args[],
                    cudaStream_t stream) {
                CUDA_DRIVER_CHECK(cuLaunchKernel(
                    func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
                ));
            }
    """
    # 检查当前是否在使用 PyTorch 的 HIP 版本（AMD GPU 上的端口），如果是则进行以下操作
    if torch.version.hip is not None:
        # 在源代码中，将所有出现的 "32*numWarps" 替换为 "64*numWarps"
        # 这是因为在 NV GPU 上，线程束（warp）大小为 32，在 AMD GPU 上，波前（wavefront）大小为 64
        source_codes = source_codes.replace("32*numWarps", "64*numWarps")
    # 返回替换后的源代码
    return source_codes
# 定义一个函数，返回包含 CUDA 内核头文件的字符串
def cuda_kernel_header() -> str:
    # 定义包含 CUDA 相关头文件的多行字符串
    source_codes = """
        #include <c10/cuda/CUDAGuard.h>
        #include <c10/cuda/CUDAStream.h>
        #include <ATen/cuda/EmptyTensor.h>
    """
    # 返回包含 CUDA 头文件的字符串
    return source_codes
```