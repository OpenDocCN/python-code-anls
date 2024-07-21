# `.\pytorch\aten\src\ATen\native\mps\operations\MultiTensorApply.h`

```
#pragma once
// 包含 ATen 库的 Tensor 头文件
#include <ATen/core/Tensor.h>
// 包含 ATen 库的 MPSProfiler 头文件
#include <ATen/mps/MPSProfiler.h>
// 包含 ATen 库的 MPS 操作的头文件
#include <ATen/native/mps/operations/FusedOptimizerOps.h>

// 定义命名空间 at::native::mps
namespace at::native::mps {

// 声明常量 kChunkSize，表示块的大小为 65536
static constexpr int64_t kChunkSize = 65536;
// 声明常量 kmaxThreadGroups，表示最大线程组数为 32
static constexpr int64_t kmaxThreadGroups = 32;
// 声明常量 kmaxTensors，表示最大张量数为 32
static constexpr int64_t kmaxTensors = 32;

// 定义 MetadataArguments 结构体，用于存储元数据参数
struct MetadataArguments { // the size of this struct must be less than 4 bytes
  // 存储每个张量的元素数量的数组
  uint numels[kmaxTensors];
  // 将线程组映射到张量的索引的数组
  uint threadgroup_to_tensor[kmaxThreadGroups];
  // 将线程组映射到块的索引的数组
  uint threadgroup_to_chunk[kmaxThreadGroups];
};

// 定义 FusedAdamEncodingFunctor 结构体
struct FusedAdamEncodingFunctor {
  // 定义函数调用操作符，用于编码融合 Adam 优化器的参数
  void operator()(
      id<MTLComputeCommandEncoder>& computeEncoder,
      id<MTLBuffer>& tensorArgumentBuffer,
      const MetadataArguments& metadata_arguments,
      const double lr,
      const double beta1,
      const double beta2,
      const double weight_decay,
      const double eps,
      const bool maximize
    ) const {

    // 将 lr、beta1、beta2、weight_decay、eps 和 maximize 转换为适当的数据类型
    float lr_lv = lr;
    float beta1_lv = beta1;
    float beta2_lv = beta2;
    float weight_decay_lv = weight_decay;
    float eps_lv = eps;
    uint8_t maximize_lv = maximize;

    // 设置计算编码器的第一个缓冲区参数
    [computeEncoder setBuffer:tensorArgumentBuffer
                                  offset:0
                                  atIndex:0];
    // 设置计算编码器的第二个缓冲区参数，传递元数据参数结构体
    [computeEncoder setBytes:&metadata_arguments
                                  length:sizeof(MetadataArguments)
                                  atIndex:1];
    // 设置计算编码器的第三个缓冲区参数，传递 lr_lv（学习率）
    [computeEncoder setBytes:&lr_lv length:sizeof(float) atIndex:2];
    // 设置计算编码器的第四个缓冲区参数，传递 beta1_lv（第一动量系数）
    [computeEncoder setBytes:&beta1_lv length:sizeof(float) atIndex:3];
    // 设置计算编码器的第五个缓冲区参数，传递 beta2_lv（第二动量系数）
    [computeEncoder setBytes:&beta2_lv length:sizeof(float) atIndex:4];
    // 设置计算编码器的第六个缓冲区参数，传递 weight_decay_lv（权重衰减）
    [computeEncoder setBytes:&weight_decay_lv length:sizeof(float) atIndex:5];
    // 设置计算编码器的第七个缓冲区参数，传递 eps_lv（数值稳定性参数）
    [computeEncoder setBytes:&eps_lv length:sizeof(float) atIndex:6];
    // 设置计算编码器的第八个缓冲区参数，传递 maximize_lv（是否最大化）
    [computeEncoder setBytes:&maximize_lv length:sizeof(uint8_t) atIndex:7];
  }
};

// 定义 FusedSgdEncodingFunctor 结构体
struct FusedSgdEncodingFunctor {
  // 定义函数调用操作符，用于编码融合 SGD 优化器的参数
  void operator()(
    id<MTLComputeCommandEncoder>& computeEncoder,
      id<MTLBuffer>& tensorArgumentBuffer,
      const MetadataArguments& metadata_arguments,
      const double weight_decay,
      const double momentum,
      const double lr,
      const double dampening,
      const bool nesterov,
      const bool maximize,
      const bool is_first_step
    ) const {
      // 将传入的参数保存到局部变量中，以便在 Metal compute 函数中使用
      float weight_decay_lv = weight_decay;
      float momentum_lv = momentum;
      float lr_lv = lr;
      float dampening_lv = dampening;
      uint8_t nesterov_lv = nesterov;
      uint8_t maximize_lv = maximize;
      uint8_t is_first_step_lv = is_first_step;

      // 设置 Metal compute encoder 的参数：
      // 设置第 0 个索引处的缓冲区为 tensorArgumentBuffer，偏移量为 0
      [computeEncoder setBuffer:tensorArgumentBuffer
                          offset:0
                         atIndex:0];
      // 设置第 1 个索引处的字节数据为 metadata_arguments，长度为 MetadataArguments 结构体的大小
      [computeEncoder setBytes:&metadata_arguments
                          length:sizeof(MetadataArguments)
                         atIndex:1];
      // 设置第 2 个索引处的字节数据为 weight_decay_lv，长度为 float 的大小
      [computeEncoder setBytes:&weight_decay_lv length:sizeof(float) atIndex:2];
      // 设置第 3 个索引处的字节数据为 momentum_lv，长度为 float 的大小
      [computeEncoder setBytes:&momentum_lv length:sizeof(float) atIndex:3];
      // 设置第 4 个索引处的缓冲区为 lr 对应的 Metal 缓冲区，偏移量为 lr 的存储偏移量乘以元素大小
      [computeEncoder setBuffer:getMTLBufferStorage(lr)
                          offset:lr.storage_offset() * lr.element_size()
                         atIndex:4];
      // 设置第 5 个索引处的字节数据为 dampening_lv，长度为 float 的大小
      [computeEncoder setBytes:&dampening_lv length:sizeof(float) atIndex:5];
      // 设置第 6 个索引处的字节数据为 nesterov_lv，长度为 uint8_t 的大小
      [computeEncoder setBytes:&nesterov_lv length:sizeof(uint8_t) atIndex:6];
      // 设置第 7 个索引处的字节数据为 maximize_lv，长度为 uint8_t 的大小
      [computeEncoder setBytes:&maximize_lv length:sizeof(uint8_t) atIndex:7];
      // 设置第 8 个索引处的字节数据为 is_first_step_lv，长度为 uint8_t 的大小
      [computeEncoder setBytes:&is_first_step_lv length:sizeof(uint8_t) atIndex:8];
  }

  void operator()(
    id<MTLComputeCommandEncoder>& computeEncoder,
      id<MTLBuffer>& tensorArgumentBuffer,
      const MetadataArguments& metadata_arguments,
      const double weight_decay,
      const double momentum,
      const at::Tensor& lr,
      const double dampening,
      const bool nesterov,
      const bool maximize,
      const bool is_first_step
    ) const {
      // 将传入的参数保存到局部变量中，以便在 Metal compute 函数中使用
      float weight_decay_lv = weight_decay;
      float momentum_lv = momentum;
      float dampening_lv = dampening;
      uint8_t nesterov_lv = nesterov;
      uint8_t maximize_lv = maximize;
      uint8_t is_first_step_lv = is_first_step;

      // 设置 Metal compute encoder 的参数：
      // 设置第 0 个索引处的缓冲区为 tensorArgumentBuffer，偏移量为 0
      [computeEncoder setBuffer:tensorArgumentBuffer
                          offset:0
                         atIndex:0];
      // 设置第 1 个索引处的字节数据为 metadata_arguments，长度为 MetadataArguments 结构体的大小
      [computeEncoder setBytes:&metadata_arguments
                          length:sizeof(MetadataArguments)
                         atIndex:1];
      // 设置第 2 个索引处的字节数据为 weight_decay_lv，长度为 float 的大小
      [computeEncoder setBytes:&weight_decay_lv length:sizeof(float) atIndex:2];
      // 设置第 3 个索引处的字节数据为 momentum_lv，长度为 float 的大小
      [computeEncoder setBytes:&momentum_lv length:sizeof(float) atIndex:3];
      // 设置第 4 个索引处的缓冲区为 lr 对应的 Metal 缓冲区，偏移量为 lr 的存储偏移量乘以元素大小
      [computeEncoder setBuffer:getMTLBufferStorage(lr)
                          offset:lr.storage_offset() * lr.element_size()
                         atIndex:4];
      // 设置第 5 个索引处的字节数据为 dampening_lv，长度为 float 的大小
      [computeEncoder setBytes:&dampening_lv length:sizeof(float) atIndex:5];
      // 设置第 6 个索引处的字节数据为 nesterov_lv，长度为 uint8_t 的大小
      [computeEncoder setBytes:&nesterov_lv length:sizeof(uint8_t) atIndex:6];
      // 设置第 7 个索引处的字节数据为 maximize_lv，长度为 uint8_t 的大小
      [computeEncoder setBytes:&maximize_lv length:sizeof(uint8_t) atIndex:7];
      // 设置第 8 个索引处的字节数据为 is_first_step_lv，长度为 uint8_t 的大小
      [computeEncoder setBytes:&is_first_step_lv length:sizeof(uint8_t) atIndex:8];
  }
// 定义一个模板函数，用于在融合优化器中多线程应用于张量操作
template <int depth, uint32_t kThreadGroupSize, typename encoder_func_t, typename... ArgTypes>
static void multi_tensor_apply_for_fused_optimizer(
    const std::string& kernel_name,  // 核函数名称
    std::vector<std::vector<at::Tensor>>& tensor_lists,  // 包含张量列表的二维向量
    at::TensorList state_steps,  // 状态步骤的张量列表
    encoder_func_t encode,  // 编码器函数对象
    ArgTypes... args  // 可变参数列表
    ) {
  const auto num_tensors = tensor_lists[0].size();  // 获取第一个张量列表的张量数量

  if (num_tensors == 0) {
    return;  // 如果张量数量为零，直接返回
  }

  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth");  // 检查张量列表的深度是否与指定的深度匹配
  for (const auto& d : c10::irange(depth)) {
    TORCH_CHECK(
      tensor_lists[d][0].scalar_type() == at::ScalarType::Float || tensor_lists[d][0].scalar_type() == at::ScalarType::Half, "Only float and half are supported");  // 检查每个张量列表中的第一个张量是否为浮点数或半精度数
  }

  id<MTLDevice> device = MPSDevice::getInstance()->device();  // 获取 Metal 设备
  MPSStream* mpsStream = getCurrentMPSStream();  // 获取当前的 MPS 流

  // 移除用于调试的注释
  /*
  mpsStream->addCompletedHandler(^(id<MTLCommandBuffer> cb) {
    [cb.logs enumerateObjectsUsingBlock:^(NSString* log, NSUInteger idx, BOOL* stop) {
      NSLog(@"MPSStream: %@", log);  // 输出 MPS 流的日志信息
      }
    ];
  });
  */

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    // 在 MPS 流的队列上同步执行的块
    }
  });
}

} // namespace mps
} // namespace at::native
```