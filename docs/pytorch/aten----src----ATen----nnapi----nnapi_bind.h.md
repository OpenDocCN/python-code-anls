# `.\pytorch\aten\src\ATen\nnapi\nnapi_bind.h`

```
// 如果 NNAPI_BIND_H_ 未定义，则定义 NNAPI_BIND_H_，以避免重复包含
#ifndef NNAPI_BIND_H_
#define NNAPI_BIND_H_

// 包含必要的头文件
#include <vector>  // 包含向量容器的头文件

#include <ATen/ATen.h>  // 包含 PyTorch 的 ATen 库
#include <torch/custom_class.h>  // 包含 PyTorch 的自定义类支持

#include <ATen/nnapi/nnapi_wrapper.h>  // 包含 NNAPI 的封装头文件

namespace torch {  // PyTorch 命名空间
namespace nnapi {  // NNAPI 命名空间
namespace bind {  // 绑定相关的命名空间

// 声明 nnapi 和 check_nnapi 为外部链接，使用 NNAPI 封装的全局变量
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TORCH_API extern nnapi_wrapper* nnapi;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TORCH_API extern nnapi_wrapper* check_nnapi;

// 定义一个宏，用于创建智能指针类型
#define MAKE_SMART_PTR(type) \
  struct type ## Freer {  // 定义释放器结构体
    void operator()(ANeuralNetworks ## type * obj) {  // 定义释放操作符
      if (!nnapi) { /* obj must be null. */ return; }  // 如果 nnapi 未初始化，则直接返回
      nnapi-> type ## _free(obj);  // 调用 NNAPI 封装的释放函数
    } \
  }; \
  typedef std::unique_ptr<ANeuralNetworks ## type, type ## Freer> type ## Ptr;  // 使用 unique_ptr 定义智能指针类型

// 使用宏创建 ModelPtr、CompilationPtr 和 ExecutionPtr 智能指针类型
MAKE_SMART_PTR(Model)
MAKE_SMART_PTR(Compilation)
MAKE_SMART_PTR(Execution)

#undef MAKE_SMART_PTR  // 取消宏定义

// 定义 NnapiCompilation 类，继承自 torch::jit::CustomClassHolder
struct NnapiCompilation : torch::jit::CustomClassHolder {
    NnapiCompilation() = default;  // 默认构造函数
    ~NnapiCompilation() override = default;  // 虚析构函数，使用默认实现

    // 初始化函数，用于旧模型调用 init() 的兼容性处理
    TORCH_API void init(
      at::Tensor serialized_model_tensor,  // 序列化的模型张量
      std::vector<at::Tensor> parameter_buffers  // 参数缓冲区向量
    );

    // 初始化函数，支持更多参数，用于 init2() 的实现
    TORCH_API void init2(
      at::Tensor serialized_model_tensor,  // 序列化的模型张量
      const std::vector<at::Tensor>& parameter_buffers,  // 参数缓冲区向量（常量引用）
      int64_t compilation_preference,  // 编译偏好设置
      bool relax_f32_to_f16  // 是否将 float32 放宽到 float16
    );

    // 运行函数，执行编译后的模型
    TORCH_API void run(std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs);

    // 静态函数，获取张量的操作数类型
    static void get_operand_type(const at::Tensor& t, ANeuralNetworksOperandType* operand, std::vector<uint32_t>* dims);

    ModelPtr model_;  // 智能指针，指向编译后的模型
    CompilationPtr compilation_;  // 智能指针，指向编译对象
    int32_t num_inputs_ {};  // 输入数量
    int32_t num_outputs_ {};  // 输出数量
};

} // namespace bind
} // namespace nnapi
} // namespace torch

#endif // 结束 NNAPI_BIND_H_ 的条件编译
```