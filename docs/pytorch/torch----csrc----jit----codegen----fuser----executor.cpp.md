# `.\pytorch\torch\csrc\jit\codegen\fuser\executor.cpp`

```py
// 引入Torch相关的头文件，用于执行器功能的实现
#include <torch/csrc/jit/codegen/fuser/executor.h>

// 引入ATen相关的头文件，提供张量操作和扩展工具
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/functional.h>
#include <ATen/core/stack.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>

// 引入Torch的代码生成相关头文件，包括编译器、接口、内核缓存、内核规范和张量信息
#include <torch/csrc/jit/codegen/fuser/compiler.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>
#include <torch/csrc/jit/codegen/fuser/kernel_spec.h>
#include <torch/csrc/jit/codegen/fuser/tensor_info.h>
#include <torch/csrc/jit/passes/graph_fuser.h>

// 引入标准库头文件，包括算法、异常处理、元组和向量容器等
#include <algorithm>
#include <iostream> // TODO: remove, debugging only
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

// Torch命名空间下jit模块下fuser子模块内的实现
namespace torch {
namespace jit {
namespace fuser {

// 返回当前运行的“映射大小”，即所有中间张量的常见大小
static std::optional<std::vector<int64_t>> getMapSize(
    const KernelSpec& spec,               // 内核规范，描述了编译过程中的规范
    at::TensorList args,                  // 张量列表，表示传递给内核的参数
    at::IntArrayRef arg_subset) {         // 参数子集的整数数组引用
  // TODO: 每次迭代都重新分配map_size，但我们知道需要的存储空间大小，可以在每个步骤中直接修复。
  // 注意: 留空以便在每个步骤中广播到任何形状
  std::vector<int64_t> map_size;          // 存储映射大小的向量
  map_size.reserve(8);                    // 预留8个元素的空间
  for (const auto arg_idx : arg_subset) { // 对于参数子集中的每个索引
    auto& arg = args.at(arg_idx);         // 获取对应索引处的参数
    auto& chunk_desc = spec.inputChunks().at(arg_idx); // 获取输入块描述
    if (chunk_desc.nSubTensors() == 1) {  // 如果子张量数为1
      try {
        map_size = at::infer_size(map_size, arg.sizes()); // 推断并更新map_size的大小
      } catch (...) {
        return c10::nullopt;               // 捕获异常，返回空optional
      }
    } else {
      auto tensor_sizes = arg.sizes().vec();  // 获取参数的大小向量
      const auto num_chunks = chunk_desc.nSubTensors(); // 子张量的数量
      const auto dim = at::maybe_wrap_dim(chunk_desc.dim(), tensor_sizes.size()); // 确定维度
      if (tensor_sizes[dim] % num_chunks != 0) { // 如果维度不能被子张量数整除
        return c10::nullopt;               // 返回空optional
      }
      tensor_sizes[dim] /= num_chunks;     // 调整维度大小
      try {
        map_size = at::infer_size(map_size, tensor_sizes); // 推断并更新map_size的大小
      } catch (...) {
        return c10::nullopt;               // 捕获异常，返回空optional
      }
    }
  }

  return {map_size};                      // 返回计算得到的映射大小
}

// 尝试为实例化的内核确定一个映射大小（见上文）
static std::optional<std::vector<int64_t>> canRunKernel(
    const KernelSpec& spec,               // 内核规范，描述了编译过程中的规范
    at::TensorList args) {                // 张量列表，表示传递给内核的参数
  // 如果参数数量与规范中的输入块描述数不匹配，则抛出异常
  TORCH_CHECK(
      args.size() == spec.inputChunks().size(),
      "Expected ",
      spec.inputChunks().size(),
      " arguments, but got ",
      args.size());

  std::optional<std::vector<int64_t>> map_size; // 存储映射大小的可选向量
  for (const auto& broadcast_group : spec.inputBroadcastGroups()) { // 遍历输入广播组
    if (!map_size) {
      map_size = getMapSize(spec, args, broadcast_group); // 获取映射大小
      if (!map_size)
        return c10::nullopt;               // 如果映射大小为空，则返回空optional
      // TODO: 如果映射大小不为空，可能还有其他操作
    } else {
      // 获取使用指定参数和广播组计算得到的映射大小
      const auto group_map_size = getMapSize(spec, args, broadcast_group);
      // 注意：此处检查 group_map_size 是否已定义且等于 map_size
      if (map_size != group_map_size)
        // 如果不相等，则返回空的optional对象
        return c10::nullopt;
    }
  }

  // 返回计算得到的映射大小
  return map_size;
}

// Arguments are expanded to a common shape, referred to as the "map size,"
// (see above).
// Note: Arguments are mutated by this call, although map_size is restored
// to its original value.
static bool expandArgs(
    const KernelSpec& spec,                             // 确定内核规范的参数
    std::vector<at::Tensor>& args,                      // 输入张量列表
    std::vector<int64_t>& map_size,                     // 映射大小列表
    bool dry_run) {                                     // 是否为试运行模式的标志位
  bool has_broadcast = false;                           // 是否进行了广播操作的标志位
  for (size_t i = 0; i < args.size(); ++i) {            // 迭代处理每一个输入张量
    auto& arg = args[i];                                // 获取当前输入张量的引用
    const auto& pdesc = spec.inputChunks()[i];          // 获取当前输入的分块描述信息
    if (pdesc.nSubTensors() == 1) {                     // 如果分块数为1
      if (arg.sizes().equals(map_size))                 // 如果当前张量的大小与映射大小相等
        continue;                                       // 继续下一轮迭代
      if (!dry_run) {                                   // 如果不是试运行模式
        arg = arg.expand(map_size);                     // 对当前张量进行扩展到映射大小
        has_broadcast = true;                           // 设置广播操作标志位为真
      } else {
        return true;                                    // 返回真，表示需要扩展参数
      }
    } else {                                            // 如果分块数不为1
      map_size.at(pdesc.dim()) *= pdesc.nSubTensors();  // 更新映射大小中对应维度的大小
      if (!arg.sizes().equals(map_size)) {              // 如果当前张量的大小与更新后的映射大小不相等
        if (!dry_run) {                                 // 如果不是试运行模式
          arg = arg.expand(map_size);                   // 对当前张量进行扩展到映射大小
          has_broadcast = true;                         // 设置广播操作标志位为真
        } else {
          return true;                                  // 返回真，表示需要扩展参数
        }
      }
      map_size.at(pdesc.dim()) /= pdesc.nSubTensors();  // 恢复映射大小中对应维度的大小
    }
  }
  return has_broadcast;                                 // 返回是否进行了广播操作的标志位
}

static bool shouldExpandArgs(
    const KernelSpec& spec,                             // 内核规范参数
    std::vector<at::Tensor>& args,                      // 输入张量列表
    std::vector<int64_t>& map_size) {                   // 映射大小列表
  return expandArgs(spec, args, map_size, /*dry_run=*/true);  // 调用参数扩展函数进行试运行
}

// Note: assumes that inputs are 32-bit addressable
static uint32_t computeNumel(const at::ArrayRef<int64_t> sizes) {  // 计算张量大小的函数
  uint32_t result = 1;                                  // 初始化结果为1

  for (const auto& size : sizes)                        // 迭代处理每一个维度大小
    result *= size;                                     // 计算所有维度大小的乘积

  return result;                                        // 返回计算结果
}

// Note: Assumes that after at::chunk, all inputs are the same size
static std::vector<int64_t> computeMapSize(
    const at::Tensor& tensor,                           // 输入张量
    const PartitionDesc& chunkDesc) {                   // 分块描述信息
  std::vector<int64_t> sizes(tensor.sizes().begin(), tensor.sizes().end());  // 初始化大小为张量的维度大小
  AT_ASSERT(sizes[chunkDesc.dim()] % chunkDesc.nSubTensors() == 0);  // 断言分块后的维度大小能整除分块数
  sizes[chunkDesc.dim()] /= chunkDesc.nSubTensors();    // 更新分块后的维度大小
  return sizes;                                         // 返回更新后的大小列表
}

// Tries to compress sizes and strides according to cont. Emits the result t
// c_sizes, c_strides and throws an error on failure (if can't compress)
static void compressContiguous(
    const at::IntArrayRef& sizes,                       // 维度大小列表
    const at::IntArrayRef& strides,                     // 步长列表
    const std::vector<bool>& cont,                      // 是否连续的标志位列表
    uint32_t* c_sizes,                                  // 压缩后的维度大小
    uint32_t* c_strides) {                              // 压缩后的步长
  size_t compressed_dims = 0;                           // 初始化压缩维度数量为0
  size_t cur = 0;                                       // 当前处理的维度索引
  size_t ndim = sizes.size();                           // 获取总维度数量
  while (cur < ndim) {                                  // 迭代处理每一个维度
    size_t total_size = sizes[cur];                     // 初始化总大小为当前维度大小
    cur++;                                              // 增加当前维度索引
    while (cont[cur - 1] && cur < ndim) {               // 如果当前维度是连续的并且不是最后一个维度
      AT_ASSERT(strides[cur - 1] == sizes[cur] * strides[cur]);  // 断言连续维度的步长符合要求
      total_size *= sizes[cur];                         // 更新总大小
      cur++;                                            // 增加当前维度索引
    }
    c_sizes[compressed_dims] = total_size;              // 将压缩后的维度大小存入数组
    c_strides[compressed_dims] = strides[cur - 1];      // 将压缩后的步长存入数组
    compressed_dims++;                                  // 增加压缩维度数量
  }

  if (ndim > 0)                                         // 如果总维度数量大于0
    AT_ASSERT(!cont.back() || strides.back() == 1);     // 断言最后一个维度是连续的或步长为1
}

// Launches the requested fusion on the given device with the given inputs.
// Output pointers are stored in outputs (to be put on the stack later).
static void launchFusion(
    const FusedKernel& fusion,                          // 融合内核对象
    const at::Device device,                            // 设备对象
    const at::ArrayRef<at::Tensor>& inputs,             // 输入张量列表
    const at::ArrayRef<IValue>& all_inputs,
    std::vector<at::Tensor>& outputs) {
  // 如果融合和给定的输入不一致，则失败
  AT_ASSERT(inputs.size() == fusion.inputDesc().size());

  // 计算展平后的输入和输出数量
  size_t flat_inputs_size = 0;
  size_t flat_outputs_size = 0;
  for (const auto& c : fusion.chunkDesc())
    flat_inputs_size += c.nSubTensors();
  for (const auto& c : fusion.concatDesc())
    flat_outputs_size += c.nSubTensors();

  // 如果第一个张量的元素不能表示为32位整数，则失败
  // 注意：此代码假设输入是32位可寻址的
  // 注意：此代码假设所有输入大小相同
  AT_ASSERT(inputs[0].numel() <= std::numeric_limits<uint32_t>::max());

  // 根据第一个输入计算 map_size 和 numel
  at::IntArrayRef map_size;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint32_t numel;
  std::vector<int64_t> keep_alive_size;
  if (fusion.chunkDesc()[0].isNoop()) {
    map_size = inputs[0].sizes();
    numel = inputs[0].numel();
  } else {
    keep_alive_size = computeMapSize(inputs[0], fusion.chunkDesc()[0]);
    map_size = keep_alive_size;
    numel = computeNumel(map_size);
  }

  // 计算标量输入的数量并转换为浮点数
  std::vector<double> scalar_inputs;
  scalar_inputs.reserve(all_inputs.size());
  for (auto const& input : all_inputs) {
    if (input.isDouble())
      scalar_inputs.push_back(input.to<float>());
  }

  // 计算存储空间以存储输入和输出的 TensorInfo 结构所需的空间
  size_t uncompressedDim = fusion.inputDesc().at(0).contiguity.size();
  size_t maxPossibleTensorInfoSize =
      sizeof(TensorInfo) + 2 * sizeof(uint32_t) * uncompressedDim;
  size_t maxPossibleBufferSize =
      maxPossibleTensorInfoSize * (flat_inputs_size + flat_outputs_size);
  std::vector<char> buffer(maxPossibleBufferSize);
  char* buffer_next = buffer.data();

  // 内核参数的向量 (numel, *input_desc_s, *output_desc_s)
  std::vector<void*> arguments;
  arguments.reserve(
      3 + scalar_inputs.size() + flat_inputs_size + flat_outputs_size);
  arguments.push_back(&numel);

  auto addTensorInfoRaw = [&](const TensorDesc& desc,
                              void* data_ptr,
                              at::IntArrayRef sizes,
                              at::IntArrayRef strides) {
    const auto nDim = desc.nDim(); // 注意：这是压缩维度
    AT_ASSERT(nDim <= uncompressedDim); // 否则会溢出空间
    auto ti = reinterpret_cast<TensorInfo*>(buffer_next);
    ti->data = data_ptr;
    compressContiguous(
        sizes, strides, desc.contiguity, ti->sizes(nDim), ti->strides(nDim));
    buffer_next += maxPossibleTensorInfoSize;
    arguments.push_back(ti);

# 将 `ti` 添加到 `arguments` 向量的末尾。

  };

  // 断言张量 `t` 的维度可以像 `desc` 中描述的那样进行压缩（这是内核假设的），并将其附加到 arguments 向量中。
  auto addTensorInfo = [&](const TensorDesc& desc, const at::Tensor& t) {
    // 调用 addTensorInfoRaw 函数，将张量 `t` 的相关信息添加到 arguments 向量中
    addTensorInfoRaw(desc, t.data_ptr(), t.sizes(), t.strides());
  };

  // 添加（展开的）输入参数
  for (size_t i = 0; i < fusion.inputDesc().size(); ++i) {
    const auto& chunk = fusion.chunkDesc()[i];
    const at::Tensor& tensor = inputs[i];
    if (chunk.isNoop()) {
      // 如果 chunk 是 Noop（无操作），则将 fusion.inputDesc()[i] 描述的张量信息添加到 arguments 向量中
      addTensorInfo(fusion.inputDesc()[i], tensor);
    } else {
      size_t chunk_offset = map_size[chunk.dim()] * tensor.stride(chunk.dim()) *
          elementSize(tensor.scalar_type());
      char* data_ptr = reinterpret_cast<char*>(tensor.data_ptr());
      for (size_t chunks = 0; chunks < chunk.nSubTensors(); ++chunks) {
        // 对于每个 chunk，将其子张量的信息添加到 arguments 向量中
        addTensorInfoRaw(
            *chunk.subTensorDesc(), data_ptr, map_size, tensor.strides());
        data_ptr += chunk_offset;
      }
    }
  }
  // 添加标量参数
  for (double& s : scalar_inputs) {
    // 将每个标量 s 添加到 arguments 向量中
    arguments.push_back(&s);
  }

  // 添加（展开的）输出参数
  outputs.reserve(fusion.outputDesc().size());
  const auto& ref_options = inputs[0].options();
  for (size_t i = 0; i < fusion.outputDesc().size(); ++i) {
    const auto& c = fusion.concatDesc()[i];
    if (c.isNoop()) {
      // 如果 c 是 Noop（无操作），则创建一个空的输出张量，并将其描述信息添加到 arguments 向量中
      outputs.push_back(at::empty(
          map_size, ref_options.dtype(fusion.outputDesc()[i].scalar_type)));
      addTensorInfo(fusion.outputDesc()[i], outputs[i]);
    } else {
      size_t small_size = map_size[c.dim()];
      std::vector<int64_t> concat_size(map_size.begin(), map_size.end());
      concat_size[c.dim()] = small_size * c.nSubTensors();
      // 创建一个拼接后的输出张量，并将其子张量的信息添加到 arguments 向量中
      outputs.push_back(at::empty(concat_size, ref_options));
      const auto& o = outputs[i];
      size_t offset = 0;
      for (size_t j = 0; j < c.nSubTensors(); ++j) {
        // 对于每个子张量，创建一个视图并将其信息添加到 arguments 向量中
        const auto view = o.narrow(c.dim(), offset, small_size);
        addTensorInfo(*c.subTensorDesc(), view);
        offset += small_size;
      }
    }
  }
  // 如果元素个数大于 0，则调用 fusion.launch_raw 函数启动内核计算，并传递 arguments 向量作为参数。
  if (numel > 0) {
    fusion.launch_raw(numel, arguments);
  }
} // 结束命名空间 torch
} // 结束命名空间 jit
} // 结束命名空间 fuser

// 函数: 在给定的 key 下运行融合操作
bool runFusion(const int64_t key, Stack& stack, std::string* code_out) {
    // 如果 CPU 旧版或 GPU 上不能进行融合操作，则直接返回 false
    if (!canFuseOnCPULegacy() && !canFuseOnGPU())
        return false;

    // 获取与指定 key 相关联的融合规范
    auto maybe_spec = retrieve(key);
    AT_ASSERT(maybe_spec);
    auto& spec = *(*maybe_spec);

    // 从堆栈中获取输入数据
    auto all_inputs = last(stack, spec.nInputs());
    std::vector<at::Tensor> inputs;
    inputs.reserve(spec.nTensorInputs());

    // 将张量输入装入 inputs 向量
    // 我们知道张量输入在前面
    for (const auto i : c10::irange(spec.nTensorInputs())) {
        inputs.emplace_back(all_inputs[i].toTensor());
    }

    // 如果第一个输入未定义，则返回 false
    if (!inputs.at(0).defined()) {
        return false;
    }

    // 确定要分派到的设备
    at::Device device = inputs.at(0).device();

    // 如果输入中存在设备不匹配或者有稀疏张量，则返回 false
    for (const auto& t : at::TensorList(inputs).slice(1)) {
        // 稀疏张量不支持 CUDA 融合，因此我们退出
        if (t.device() != device || t.is_sparse()) {
            return false;
        }
    }

    // 如果设备是 CUDA 并且 GPU 融合被禁用，则返回 false
    if (device.is_cuda() && !canFuseOnGPU())
        return false;

    // 如果设备是 CPU 并且 CPU 旧版融合被禁用，则返回 false
    if (device.is_cpu() && !canFuseOnCPULegacy())
        return false;

    // 如果设备是 XPU，则返回 false
    if (device.is_xpu())
        return false;

    // 验证尺寸并根据需要扩展输入
    auto maybe_map_size = canRunKernel(spec, inputs);

    // 如果无法计算映射大小，则尝试运行回退操作
    if (!maybe_map_size)
        return false;

    // 如果规范中包含随机性，则检查是否需要扩展参数
    if (spec.hasRandom()) {
        bool hasBroadcast = shouldExpandArgs(spec, inputs, *maybe_map_size);
        if (hasBroadcast)
            return false;
    }

    // 根据需要扩展参数
    expandArgs(spec, inputs, *maybe_map_size, /*dry_run=*/false);

    // 检索内核，必要时编译并缓存
    ArgSpec arg_spec{inputs, device.index()};
    auto maybe_kernel = spec.findKernel(arg_spec);
    if (!maybe_kernel) {
        const auto kernel = compileKernel(spec, arg_spec, *maybe_map_size, device);
        spec.cacheKernel(arg_spec, kernel);
    }
    maybe_kernel = spec.findKernel(arg_spec);
    AT_ASSERT(maybe_kernel);

    // 如果提供了 code_out 指针，则将内核代码复制到指定位置
    if (code_out) {
        *code_out = maybe_kernel.value()->code();
    }

    // 启动融合操作
    std::vector<at::Tensor> outputs;
    launchFusion(*(*maybe_kernel), device, inputs, all_inputs, outputs);

    // 更新堆栈
    drop(stack, spec.nInputs());
    stack.insert(
        stack.end(),
        std::make_move_iterator(outputs.begin()),
        std::make_move_iterator(outputs.end()));

    return true;
}
```