# `.\pytorch\torch\csrc\jit\codegen\onednn\LlgaTensorImpl.h`

```py
// 预处理指令，表示这个头文件只会被包含一次
#pragma once

// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含 ATen 库的配置头文件
#include <ATen/Config.h>

// 包含 oneDNN Graph 的头文件
#include <oneapi/dnnl/dnnl_graph.hpp>
// 包含 PyTorch JIT 的 IR 头文件
#include <torch/csrc/jit/ir/ir.h>

// 声明 torch 命名空间下的 jit 命名空间、fuser 命名空间、onednn 命名空间
namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// Engine 表示一个设备及其上下文。根据设备类型，Engine 知道如何为目标设备生成代码，
// 以及预期的设备对象类型。设备 ID 确保为每个设备创建唯一的 Engine。从 PyTorch 传入的
// 设备句柄允许 oneDNN 图实现在 PyTorch 指定的设备上工作，目前只支持 CPU，因此只有一个 Engine。
// Ref: https://spec.oneapi.io/onednn-graph/latest/programming_model.html#engine
struct Engine {
  // 获取 CPU 引擎的单例实例
  static dnnl::engine& getEngine();
  // 禁止拷贝构造函数
  Engine(const Engine&) = delete;
  // 禁止赋值运算符重载
  void operator=(const Engine&) = delete;
};

// Stream 是执行单元的逻辑抽象。它建立在 oneDNN 图引擎之上。一个编译好的 oneDNN 图分区
// 会被提交到一个 Stream 中进行执行。
struct Stream {
  // 获取 CPU 流的单例实例
  static dnnl::stream& getStream();
  // 禁止拷贝构造函数
  Stream(const Stream&) = delete;
  // 禁止赋值运算符重载
  void operator=(const Stream&) = delete;
};

// LlgaTensorDesc 用于描述逻辑张量。使用 oneDNN 图的 logical_tensor 类作为基础。
struct LlgaTensorDesc {
  // 使用逻辑张量的别名
  using desc = dnnl::graph::logical_tensor;

  // 构造函数，初始化 LlgaTensorDesc 对象
  LlgaTensorDesc(
      size_t tid,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides,
      desc::data_type dtype,
      desc::property_type property_type)
      : tid_(tid),
        sizes_(sizes),
        strides_(strides),
        dtype_(dtype),
        property_type_(property_type),
        layout_type_(desc::layout_type::strided),
        layout_id_(-1) {}

  // 从已有的 logical_tensor 构造 LlgaTensorDesc 对象
  LlgaTensorDesc(const desc& t)
      : tid_(t.get_id()),
        sizes_(t.get_dims()),
        strides_({-1}),
        dtype_(t.get_data_type()),
        property_type_(t.get_property_type()),
        layout_type_(t.get_layout_type()),
        layout_id_(-1) {
    // 如果是不透明布局，获取布局 ID
    if (is_opaque()) {
      layout_id_ = t.get_layout_id();
    }
    // 如果是分块布局，获取步长信息
    if (is_strided()) {
      strides_ = t.get_strides();
    }
  }

  // 从 PyTorch JIT 的值构造 LlgaTensorDesc 对象
  LlgaTensorDesc(const torch::jit::Value* v)
      : LlgaTensorDesc(
            v->unique(),
            {},
            {},
            desc::data_type::f32,
            get_property_type(v)) {
    if (v->type()->isSubtypeOf(TensorType::get())) {
      auto tt = v->type()->cast<TensorType>();

      // 获取张量的数据类型
      if (tt->scalarType()) {
        dtype_ = getLlgaDataType(tt->scalarType().value());
      }

      // 获取张量的尺寸信息
      auto sizes = tt->sizes();
      if (sizes.sizes()) {
        for (auto d : *sizes.sizes()) {
          sizes_.push_back(d.value_or(DNNL_GRAPH_UNKNOWN_DIM));
        }
      }

      // 获取张量的步长信息
      auto strides = tt->strides();
      if (strides.sizes()) {
        for (auto d : *strides.sizes()) {
          strides_.push_back(d.value_or(DNNL_GRAPH_UNKNOWN_DIM));
        }
      }
    }
  }
  }

  LlgaTensorDesc supplementTensorInfo(const at::Tensor& t) const;
  # 声明一个函数 supplementTensorInfo，接受一个常量引用参数 t，返回类型为 LlgaTensorDesc

  desc::data_type getLlgaDataType(at::ScalarType dt) const;
  # 声明一个函数 getLlgaDataType，接受一个 at::ScalarType 类型参数 dt，返回类型为 desc::data_type

  at::ScalarType aten_scalar_type() const;
  # 声明一个函数 aten_scalar_type，返回类型为 at::ScalarType

  const std::vector<int64_t>& sizes() const {
    return sizes_;
  }
  # 定义一个成员函数 sizes，返回 sizes_ 成员变量的常量引用

  const std::vector<int64_t>& strides() const {
    TORCH_CHECK(!is_opaque(), "Cannot get strides on opaque layout");
    return strides_;
  }
  # 定义一个成员函数 strides，返回 strides_ 成员变量的常量引用，并在非透明布局时执行 TORCH_CHECK 断言

  size_t tid() const {
    return tid_;
  }
  # 定义一个成员函数 tid，返回 tid_ 成员变量的值

  LlgaTensorDesc tid(uint64_t new_id) const {
    auto ret = *this;
    ret.tid_ = new_id;
    return ret;
  }
  # 定义一个成员函数 tid，接受一个 uint64_t 类型的新 id 参数，返回一个修改了 tid_ 成员变量后的新 LlgaTensorDesc 对象

  desc::data_type dtype() const {
    return dtype_;
  }
  # 定义一个成员函数 dtype，返回 dtype_ 成员变量的值

  LlgaTensorDesc dtype(desc::data_type new_dtype) const {
    return LlgaTensorDesc(tid_, sizes_, strides_, new_dtype, property_type_);
  }
  # 定义一个成员函数 dtype，接受一个新的 data_type 参数，返回一个新的 LlgaTensorDesc 对象，使用新的 dtype

  desc::layout_type layout_type() const {
    return layout_type_;
  }
  # 定义一个成员函数 layout_type，返回 layout_type_ 成员变量的值

  LlgaTensorDesc layout_type(desc::layout_type new_layout_type) {
    auto ret = *this;
    ret.layout_type_ = new_layout_type;
    return ret;
  }
  # 定义一个成员函数 layout_type，接受一个新的 layout_type 参数，返回一个修改了 layout_type_ 成员变量后的新 LlgaTensorDesc 对象

  desc::property_type get_property_type(const torch::jit::Value* v) {
    switch (v->node()->kind()) {
      case prim::Constant:
        return desc::property_type::constant;
      default:
        return desc::property_type::variable;
    }
  }
  # 定义一个成员函数 get_property_type，接受一个 torch::jit::Value* 类型的参数 v，返回对应的 desc::property_type 值

  LlgaTensorDesc any() {
    return layout_type(desc::layout_type::any);
  }
  # 定义一个成员函数 any，返回一个新的 LlgaTensorDesc 对象，使用 layout_type 设置为 any

  size_t storage_size() const {
    return logical_tensor().get_mem_size();
  }
  # 定义一个成员函数 storage_size，返回 logical_tensor() 的内存大小

  desc logical_tensor() const {
    if (is_dimensionality_unknown()) {
      return desc(
          tid_, dtype_, DNNL_GRAPH_UNKNOWN_NDIMS, layout_type_, property_type_);
    } else if (is_opaque()) {
      return desc(tid_, dtype_, sizes_, layout_id_, property_type_);
    } else if (is_any()) {
      return desc(tid_, dtype_, sizes_, layout_type_, property_type_);
    } else {
      return desc(tid_, dtype_, sizes_, strides_, property_type_);
    }
  }
  # 定义一个成员函数 logical_tensor，根据对象的不同状态返回不同的 desc 对象

  bool is_strided() const {
    return layout_type_ == desc::layout_type::strided;
  }
  # 定义一个成员函数 is_strided，判断对象的 layout_type_ 是否为 strided

  bool is_any() const {
    return layout_type_ == desc::layout_type::any;
  }
  # 定义一个成员函数 is_any，判断对象的 layout_type_ 是否为 any

  bool is_opaque() const {
    return layout_type_ == desc::layout_type::opaque;
  }
  # 定义一个成员函数 is_opaque，判断对象的 layout_type_ 是否为 opaque

  bool operator==(const LlgaTensorDesc& desc) const {
    return tid_ == desc.tid_ && sizes_ == desc.sizes_ &&
        dtype_ == desc.dtype_ && layout_type_ == desc.layout_type_ &&
        ((is_opaque() && layout_id_ == desc.layout_id_) ||
         strides_ == desc.strides_);
  }
  # 定义一个成员函数 operator==，比较对象的各个成员变量是否相等，考虑不同的布局类型

  bool operator!=(const LlgaTensorDesc& desc) const {
    return (tid_ != desc.tid_) || (sizes_ != desc.sizes_) ||
        (dtype_ != desc.dtype_) || (layout_type_ != desc.layout_type_) ||
        !((is_opaque() && (layout_id_ == desc.layout_id_)) ||
          (strides_ == desc.strides_));
  }
  # 定义一个成员函数 operator!=，比较对象的各个成员变量是否不相等，考虑不同的布局类型

  static size_t hash(const LlgaTensorDesc& desc) {
    return c10::get_hash(
        desc.tid_,
        desc.sizes_,
        desc.dtype_,
        desc.layout_type_,
        desc.layout_id_);
  }
  # 定义一个静态成员函数 hash，计算给定 LlgaTensorDesc 对象的哈希值

  void set_compute_inplace() {
    compute_inplace_ = true;
  }
  # 定义一个成员函数 set_compute_inplace，将成员变量 compute_inplace_ 设置为 true

  void set_input_tensor_index(size_t index) {
  // 设置输入张量的索引为给定的索引值
  input_tensor_index_ = index;
}

bool reuses_input_tensor() {
  // 返回是否允许就地计算（重用输入张量）
  return compute_inplace_;
}

size_t get_input_tensor_index() {
  // 返回输入张量的索引
  return input_tensor_index_;
}

private:
bool is_dimensionality_unknown() const {
  // 判断张量的维度是否未知
  return sizes_.size() == 0;
}

size_t tid_;
std::vector<int64_t> sizes_;
std::vector<int64_t> strides_;
desc::data_type dtype_;
desc::property_type property_type_;
desc::layout_type layout_type_;
size_t layout_id_;
// 如果这是一个输出张量，并且查询编译分区确定该张量将重用其输入张量，
// 则 compute_inplace_ 为 true，input_tensor_index_ 为 LlgaKernel 对象的 inputSpecs_ 中相应输入张量的索引。
bool compute_inplace_ = false;
size_t input_tensor_index_;
};

// 结构体定义：LlgaTensorImpl，继承自c10::TensorImpl，用于支持LLGA张量的实现
// TORCH_API宏指示符合Torch API的结构体
// 这个结构体用于处理LLGA张量的实现，绕过了守卫检查，即使在oneDNN图形中使用了分块布局，现在已经改用分块之间的分步张量。
struct TORCH_API LlgaTensorImpl : public c10::TensorImpl {
  // 构造函数：接受存储、数据类型和LLGA张量描述作为参数
  LlgaTensorImpl(
      at::Storage&& storage,
      const caffe2::TypeMeta& data_type,
      const LlgaTensorDesc& desc);

  // 返回LLGA张量描述
  const LlgaTensorDesc& desc() const {
    return desc_;
  }

  // 静态方法：将LLGA张量实现转换为ATen张量
  static at::Tensor llga_to_aten_tensor(LlgaTensorImpl* llgaImpl);

 private:
  LlgaTensorDesc desc_;  // 成员变量：LLGA张量描述
};

// 函数声明：创建一个空的LLGA张量
at::Tensor empty_llga(
    const LlgaTensorDesc& desc,
    const c10::TensorOptions& options);

// 函数声明：从ATen张量创建LLGA张量
dnnl::graph::tensor llga_from_aten_tensor(const at::Tensor& tensor);

// 命名空间结束：onednn
} // namespace onednn

// 命名空间结束：fuser
} // namespace fuser

// 命名空间结束：jit
} // namespace jit

// 命名空间结束：torch
} // namespace torch
```