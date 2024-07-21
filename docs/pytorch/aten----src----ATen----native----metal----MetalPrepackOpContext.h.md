# `.\pytorch\aten\src\ATen\native\metal\MetalPrepackOpContext.h`

```
#pragma once


// 预处理指令，确保此头文件只被编译一次


#include <ATen/Tensor.h>
#include <torch/custom_class.h>


// 包含必要的头文件，用于张量操作和自定义类支持


namespace at::native::metal {


// 进入命名空间 at::native::metal


using SerializationTypeConv2dPrePack = std::tuple<
    Tensor,
    std::optional<Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    std::optional<Scalar>,
    std::optional<Scalar>>;


// 定义别名 SerializationTypeConv2dPrePack，表示 Conv2d 操作的预打包序列化类型


class Conv2dOpContext : public torch::jit::CustomClassHolder {
 public:


// 定义类 Conv2dOpContext，继承自 torch::jit::CustomClassHolder


  SerializationTypeConv2dPrePack pack() {
    return std::make_tuple(
        weight_,
        bias_,
        stride_,
        padding_,
        dilation_,
        groups_,
        output_min_,
        output_max_);
  }


// pack 方法：返回预打包的序列化数据，包括权重、偏置、步长、填充、膨胀、分组、输出最小值和最大值


  Conv2dOpContext() = delete;


// 默认构造函数被删除，禁止默认构造 Conv2dOpContext 对象


  Conv2dOpContext(
      at::Tensor&& weight,
      std::optional<at::Tensor>&& bias,
      std::vector<int64_t> stride,
      std::vector<int64_t> padding,
      std::vector<int64_t> dilation,
      int64_t groups,
      std::optional<Scalar> output_min,
      std::optional<Scalar> output_max)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        output_min_(std::move(output_min)),
        output_max_(std::move(output_max)) {}


// 构造函数：初始化 Conv2dOpContext 对象的成员变量，包括权重、偏置、步长、填充、膨胀、分组、输出最小值和最大值


  ~Conv2dOpContext() override {
    if (releaseCallback_) {
      releaseCallback_(conv2dOp_);
    }
  }


// 虚析构函数：释放资源，调用 releaseCallback_ 函数释放 conv2dOp_ 指向的资源


  void release_resources() override {
    if (releaseCallback_) {
      releaseCallback_(conv2dOp_);
    }
  }


// 重写虚函数 release_resources，释放资源，调用 releaseCallback_ 函数释放 conv2dOp_ 指向的资源


  const Tensor& get_weight() const {
    return weight_;
  }


// 返回权重张量的引用


  const std::optional<Tensor>& get_bias() const {
    return bias_;
  }


// 返回偏置张量的可选引用


  const std::vector<int64_t>& get_stride() const {
    return stride_;
  }


// 返回步长向量的引用


  const std::vector<int64_t>& get_padding() const {
    return padding_;
  }


// 返回填充向量的引用


  const std::vector<int64_t>& get_dilation() const {
    return dilation_;
  }


// 返回膨胀向量的引用


  int64_t get_groups() const {
    return groups_;
  }


// 返回分组数


  const std::optional<Scalar>& get_output_min() const {
    return output_min_;
  }


// 返回输出最小值的可选引用


  const std::optional<Scalar>& get_output_max() const {
    return output_max_;
  }


// 返回输出最大值的可选引用


  void set_conv2dOpPtr(void* ptr) {
      conv2dOp_ = ptr;
  }


// 设置 conv2dOp_ 指针的值


  void* get_conv2dOpPtr() const {
    return conv2dOp_;
  }


// 返回 conv2dOp_ 指针的值


  void set_releaseCallback(const std::function<void(void*)>& func) {
    releaseCallback_ = func;
  }


// 设置 releaseCallback_ 函数


  std::function<void(void*)>& get_releaseCallback() {
     return releaseCallback_;
  }


// 返回 releaseCallback_ 函数的引用


  private:
    Tensor weight_;
    std::optional<Tensor> bias_;
    std::vector<int64_t> stride_;
    std::vector<int64_t> padding_;
    std::vector<int64_t> dilation_;
    int64_t groups_;
    std::optional<Scalar> output_min_;
    std::optional<Scalar> output_max_;
    std::function<void(void*)> releaseCallback_ = nullptr;
    void* conv2dOp_ = nullptr; // reserved to hold MPSCNNConv2dOp objects
};


// 类 Conv2dOpContext 的私有成员变量声明，包括权重、偏置、步长、填充、膨胀、分组、输出最小值和最大值，以及资源释放回调函数和指向 MPSCNNConv2dOp 对象的指针


using SerializationTypeLinearPrePack = std::tuple<
    Tensor,
    std::optional<Tensor>,
    std::optional<Scalar>,
    std::optional<Scalar>>;


// 定义别名 SerializationTypeLinearPrePack，表示 Linear 操作的预打包序列化类型


} // namespace at::native::metal


// 结束命名空间 at::native::metal
class LinearOpContext : public torch::jit::CustomClassHolder {
 public:
  // 定义序列化函数 pack，返回包含权重、偏置、输出最小值和输出最大值的元组
  SerializationTypeLinearPrePack pack() {
    return std::make_tuple(weight_, bias_, output_min_, output_max_);
  }
  
  // 禁用默认构造函数
  LinearOpContext() = delete;
  
  // 自定义构造函数，接受权重、可选的偏置、可选的输出最小值和可选的输出最大值
  LinearOpContext(
      at::Tensor&& weight,
      std::optional<at::Tensor>&& bias,
      std::optional<Scalar> output_min,
      std::optional<Scalar> output_max)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        output_min_(std::move(output_min)),
        output_max_(std::move(output_max)) {}

  // 虚析构函数，释放资源时若存在 releaseCallback_，调用 releaseCallback_ 释放 opaqueOpPtr_
  ~LinearOpContext() override {
    if (releaseCallback_) {
      releaseCallback_(opaqueOpPtr_);
    }
  }

  // 重写 release_resources 函数，释放资源时若存在 releaseCallback_，调用 releaseCallback_ 释放 opaqueOpPtr_
  void release_resources() override {
    if (releaseCallback_) {
      releaseCallback_(opaqueOpPtr_);
    }
  }

  // 返回权重 Tensor 的引用
  const Tensor& get_weight() const {
    return weight_;
  }

  // 返回偏置的可选值的引用
  const std::optional<Tensor>& get_bias() const {
    return bias_;
  }

  // 返回输出最小值的可选值的引用
  const std::optional<Scalar>& get_output_min() const {
    return output_min_;
  }

  // 返回输出最大值的可选值的引用
  const std::optional<Scalar>& get_output_max() const {
    return output_max_;
  }

  // 设置 opaqueOpPtr_ 的值
  void set_opaqueOpPtr(void* ptr) {
    opaqueOpPtr_ = ptr;
  }

  // 返回 opaqueOpPtr_ 的值
  void* get_opaqueOpPtr() const {
    return opaqueOpPtr_;
  }

  // 设置 releaseCallback_ 函数
  void set_releaseCallback(const std::function<void(void*)>& func) {
    releaseCallback_ = func;
  }

  // 返回 releaseCallback_ 函数的引用
  std::function<void(void*)>& get_releaseCallback() {
    return releaseCallback_;
  }

 private:
  Tensor weight_; // 权重 Tensor
  std::optional<Tensor> bias_; // 可选的偏置 Tensor
  std::optional<Scalar> output_min_; // 可选的输出最小值 Scalar
  std::optional<Scalar> output_max_; // 可选的输出最大值 Scalar
  void* opaqueOpPtr_ = nullptr; // 保留，用于持有 MPSCNNFullyConnected 对象
  std::function<void(void*)> releaseCallback_ = nullptr; // 释放资源的回调函数
};

} // namespace at::native::metal
```