# `.\pytorch\torch\csrc\cuda\comm.cpp`

```py
// 引入 Torch CUDA 通信头文件
#include <torch/csrc/cuda/comm.h>

// 引入 Torch CUDA 设备集头文件
#include <torch/csrc/cuda/device_set.h>
// 引入 Torch 张量展平工具头文件
#include <torch/csrc/utils/tensor_flatten.h>

// 如果定义了 USE_NCCL，则引入 Torch NCCL 头文件
#ifdef USE_NCCL
#include <torch/csrc/cuda/nccl.h>
#endif

// 引入 ATen 头文件
#include <ATen/ATen.h>
// 引入 ATen 张量维度包装工具头文件
#include <ATen/WrapDimUtils.h>
// 引入 ATen CUDA 上下文头文件
#include <ATen/cuda/CUDAContext.h>
// 引入 C10 CUDA 守卫头文件
#include <c10/cuda/CUDAGuard.h>
// 引入 C10 可选类型头文件
#include <c10/util/Optional.h>
// 引入 C10 整数范围工具头文件
#include <c10/util/irange.h>
// 引入 Torch 自动求导变量头文件
#include <torch/csrc/autograd/variable.h>

// 引入标准库头文件
#include <cstddef>
#include <vector>

// 定义 Torch CUDA 命名空间
namespace torch::cuda {
// 使用 ATen 和 Torch 自动求导命名空间
using namespace at;
using namespace torch::autograd;

// Some operations can be performed more efficiently if we're handling tensors
// of a single type only. Adding this logic directly in the loop makes it a bit
// ugly, so here's a helper for it.
// 定义用于检查唯一类型的辅助结构体
struct unique_type_checker {
  // 显示给定类型 ID
  void show(size_t type_id) {
    // 如果不是唯一类型，则直接返回
    if (!unique) {
      return;
    }
    // 如果类型 ID 未定义，则将其定义为当前类型 ID
    if (!type_id_) {
      type_id_ = type_id;
    }
    // 判断是否是唯一类型
    unique = type_id_.value() == type_id;
  }

  std::optional<size_t> type_id_; // 可选的类型 ID
  bool unique = true;             // 是否唯一的标志
};

// ***************** Broadcast *******************
//
// Broadcast a source tensor (CPU or CUDA) to a list of CUDA devices, or CUDA
// tensors on one or more devices.

// 不进行检查的情况下实现广播
static inline std::vector<Tensor>& _broadcast_out_impl(
    const Tensor& tensor,
    std::vector<Tensor>& out_tensors) {
#ifdef USE_NCCL
  // 准备用于 NCCL 的张量列表，预留空间
  std::vector<Tensor> nccl_list;
  nccl_list.reserve(out_tensors.size() + 1);
  // 将源张量添加到 NCCL 列表中
  nccl_list.emplace_back(tensor);
  // 将每个输出张量添加到 NCCL 列表中
  for (auto& out_tensor : out_tensors) {
    nccl_list.emplace_back(out_tensor);
  }
  // 如果 NCCL 可用，则使用 NCCL 进行广播
  if (nccl::is_available(nccl_list)) {
    nccl::broadcast(nccl_list);
  } else {
#else
  {
#endif
    // 否则，逐个复制源张量到每个输出张量（非阻塞方式）
    for (auto& out_tensor : out_tensors) {
      out_tensor.copy_(tensor, /*non_blocking=*/true);
    }
  }
  return out_tensors; // 返回输出张量列表
}

// 对外部接口，实现广播操作
std::vector<Tensor>& broadcast_out(
    const Tensor& tensor,
    std::vector<Tensor>& out_tensors) {
  // 遍历输出张量列表
  for (const auto i : c10::irange(out_tensors.size())) {
    // 检查每个输出张量是否为 CUDA 张量
    TORCH_CHECK(
        out_tensors[i].is_cuda(),
        "Expected all output tensors to be CUDA tensors, but output tensor at index ",
        i,
        " has device '",
        out_tensors[i].device(),
        "'");
    // 检查每个输出张量的形状是否与源张量相同
    TORCH_CHECK(
        out_tensors[i].sizes() == tensor.sizes(),
        "Expected all output tensors to have same shape as the source tensor ",
        tensor.sizes(),
        ", but output tensor at index ",
        i,
        " has shape ",
        out_tensors[i].sizes());
  }
  return _broadcast_out_impl(tensor, out_tensors); // 调用实现函数进行广播
}

// 对外部接口，实现按给定设备列表进行广播操作
std::vector<Tensor> broadcast(const Tensor& tensor, IntArrayRef devices) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 准备不同设备上的目标张量列表，预留空间
  std::vector<Tensor> diff_device_dst_tensors;
  diff_device_dst_tensors.reserve(devices.size());
  // 遍历设备列表
  for (auto device : devices) {
    // 检查设备索引是否为非负数
    TORCH_CHECK(
        device >= 0, "Expected non-negative device index, but got ", device);
    // 检查张量是否不在指定的设备上
    if (device != tensor.get_device()) {
      // 如果不在指定设备上，创建一个空张量，与原张量的大小和设备相匹配，并添加到不同设备目标张量列表中
      diff_device_dst_tensors.emplace_back(at::empty(
          tensor.sizes(),
          tensor.options().device(at::Device(
              DeviceType::CUDA,
              static_cast<DeviceIndex>(device))))); // 保留内存格式
    }
  }
  // 将原始张量广播到不同设备目标张量列表中的每个张量
  _broadcast_out_impl(tensor, diff_device_dst_tensors);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 准备目标张量的向量，并预留设备数量的空间
  std::vector<Tensor> dst_tensors;
  dst_tensors.reserve(devices.size());
  // 开始遍历不同设备目标张量列表和设备列表
  auto it = diff_device_dst_tensors.begin();
  for (auto device : devices) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    // 如果当前设备不是原始张量的设备，则从不同设备目标张量列表中取出对应的张量，添加到目标张量的向量中
    if (device != tensor.get_device()) {
      dst_tensors.emplace_back(*it++);
    } else {
      // 如果当前设备是原始张量的设备，则直接添加原始张量到目标张量的向量中
      dst_tensors.emplace_back(tensor);
    }
  }
  // 断言所有不同设备目标张量都已经遍历完
  TORCH_INTERNAL_ASSERT(it == diff_device_dst_tensors.end());
  // 返回填充好的目标张量的向量
  return dst_tensors;
```cpp`
// NOTE [ Version Counter in comm.*_coalesced ]
//
// broadcast_coalesced
// ~~~~~~~~~~~~~~~~~~~
//
// In broadcast_coalesced, multiple variables may be coalesced into a single
// large one, broadcast to other devices, and then split according to the
// original shapes.
//
// When splitting, the view operations ensure all Variables broadcast together
// to share a single version counter, although they do not share storage after
// the initial large Variable is discarded.
//
// For instance, in `DataParallel`, if two buffers are broadcast together and
// one is modified in-place during `forward` while the other is required in
// backward, the autograd engine will raise an error.
//
// To resolve this, Variables are re-wrapped after broadcasting (similar to .data
// in Python), each receiving individual version counters.
//
// NB: Simply calling detach() on variables is inadequate.
//
// NB: For `device[0]` in broadcast_coalesced, input Variables are always returned
//     as-is and should **not** be re-wrapped.
//
// reduce_add_coalesced
// ~~~~~~~~~~~~~~~~~~~~
//
// Similarly in reduce_add_coalesced, when the outputs are newly created Variables.
tensor_list2d broadcast_coalesced(
    TensorList tensors,
    IntArrayRef devices,
    size_t buffer_size) {
  TORCH_CHECK(
      std::all_of(
          tensors.begin(),
          tensors.end(),
          [&](const at::Tensor& t) { return t.get_device() == devices[0]; }),
      "All tensors must be on devices[0]: ",
      devices[0]);
#ifdef USE_NCCL
  buffer_size = std::min(torch::cuda::nccl::get_max_count(), buffer_size);
#endif

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  tensor_list2d outputs(devices.size());
  outputs[0] = tensors.vec();
  for (auto& o : outputs)
    o.reserve(tensors.size());

  unique_type_checker type_checker;
  at::cuda::CUDAGuard device_guard(static_cast<DeviceIndex>(devices[0]));
  for (auto& chunk : torch::utils::take_tensors(tensors, buffer_size)) {
    auto type_id = chunk.type_id();
    type_checker.show(type_id);
    std::vector<at::Tensor> results;
    // 如果块的选项是稀疏的
    if (chunk.options().is_sparse()) {
      // 展开稀疏张量并进行广播
      auto flat_tuple = torch::utils::flatten_sparse_tensors(chunk.tensors);
      auto broadcast_indices = broadcast(flat_tuple.first, devices);
      auto broadcast_values = broadcast(flat_tuple.second, devices);
      // 预留空间以容纳每个设备的结果
      results.reserve(devices.size());
      // 对于每个设备，从第二个开始（第一个设备已经处理过）
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        // 设置当前设备的索引
        device_guard.set_index(static_cast<DeviceIndex>(devices[i]));
        // 获取当前设备的输出引用
        auto& device_outputs = outputs[i];
        // 获取当前设备的广播后的索引和值
        auto& inds = broadcast_indices[i];
        auto& vals = broadcast_values[i];
        // 根据广播后的索引和值，解析稀疏张量
        for (const auto& var : torch::utils::unflatten_sparse_tensors(
                 inds, vals, chunk.tensors)) {
          // 查看注释中的“NOTE [ Version Counter in comm.*_coalesced ]”
          // 将处理后的变量加入设备的输出中
          device_outputs.emplace_back(make_variable(var.tensor_data(), false));
        }
      }
    } else {
      // 如果块的选项不是稀疏的
      // 展开稠密张量并进行广播
      auto results = broadcast(
          torch::utils::flatten_dense_tensors(chunk.tensors), devices);
      // 对于每个设备，从第二个开始（第一个设备已经处理过）
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        // 设置当前设备的索引
        device_guard.set_index(static_cast<DeviceIndex>(devices[i]));
        // 获取当前设备的输出引用
        auto& device_outputs = outputs[i];
        // 根据广播后的结果和原始张量，解析稠密张量
        for (auto& var :
             torch::utils::unflatten_dense_tensors(results[i], chunk.tensors)) {
          // 查看注释中的“NOTE [ Version Counter in comm.*_coalesced ]”
          // 将处理后的变量加入设备的输出中
          device_outputs.emplace_back(make_variable(var.tensor_data(), false));
        }
      }
    }
  }

  // 如果仅看到了单一类型的张量，则可以跳过昂贵的重新排序操作
  if (!type_checker.unique) {
    // 对每个输出列表进行与张量相似的重新排序
    for (auto& o : outputs)
      torch::utils::reorder_tensors_like(o, tensors);
  }
  // 返回处理后的输出结果
  return outputs;
// ***************** Scatter *******************
//
// Scatter a source tensor (CPU or CUDA) to a list of CUDA tensors on one or
// more devices.

// 将源张量（CPU 或 CUDA）分散到一个或多个设备上的 CUDA 张量列表中。

std::vector<at::Tensor>& scatter_out(
    const at::Tensor& tensor, // 源张量，可以是 CPU 或 CUDA
    std::vector<at::Tensor>& out_tensors, // 目标 CUDA 张量列表
    int64_t dim, // 分散的维度
    const std::optional<std::vector<std::optional<at::cuda::CUDAStream>>>&
        streams) { // 可选的 CUDA 流列表
  TORCH_CHECK(
      !out_tensors.empty(),
      "Expected at least one output tensor to scatter to"); // 检查目标张量列表不能为空
  dim = at::maybe_wrap_dim(dim, tensor); // 将维度 dim 转换成合适的张量维度
  int64_t total_size = 0; // 总大小初始化为 0
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<int64_t> chunk_sizes; // 各部分大小的向量
  chunk_sizes.reserve(out_tensors.size()); // 预留空间以容纳目标张量列表大小
  for (const auto i : c10::irange(out_tensors.size())) { // 遍历目标张量列表
    TORCH_CHECK(
        out_tensors[i].is_cuda(), // 检查当前目标张量是否为 CUDA 张量
        "Expected all output tensors to be CUDA tensors, but output tensor at index ",
        i,
        " has device '",
        out_tensors[i].device(),
        "'");
    auto out_sizes = out_tensors[i].sizes().vec(); // 获取目标张量的尺寸
    bool same_ndim = out_sizes.size() == static_cast<size_t>(tensor.dim()); // 检查目标张量与源张量维度是否相同
    if (same_ndim) {
      total_size += out_sizes[dim]; // 累加总大小
      chunk_sizes.emplace_back(out_sizes[dim]); // 将当前维度大小添加到部分大小的向量中
      out_sizes[dim] = tensor.size(dim); // 更新目标张量在 dim 维度上的大小
    }
    TORCH_CHECK(
        same_ndim && out_sizes == tensor.sizes(),
        "Output tensor at index ",
        i,
        " has incorrect shape: ",
        out_tensors[i].sizes(),
        ". Expected same "
        "shape except for scatter dim ",
        dim,
        " as the source tensor: ",
        at::IntArrayRef(tensor.sizes()));
  }
  TORCH_CHECK(
      total_size == tensor.size(dim), // 检查总大小是否与源张量在 dim 维度上的大小相等
      "Total size for output tensors along scatter dim ",
      dim,
      " does not match "
      "the source tensor size at dim ",
      dim,
      ". Expected ",
      tensor.size(dim),
      ", but got total size ",
      total_size);

  auto chunks =
      tensor.split_with_sizes(/*split_sizes=*/chunk_sizes, /*dim=*/dim); // 将源张量按照给定的部分大小在 dim 维度上进行分割
  at::cuda::OptionalCUDAStreamGuard cuda_guard; // CUDA 流的可选保护器
  for (const auto i : c10::irange(chunks.size())) { // 遍历分割后的部分
    if (i < (streams ? streams->size() : 0U) && (*streams)[i]) { // 如果存在 CUDA 流并且流是可选的
      const auto device_index =
          static_cast<int16_t>(out_tensors[i].get_device()); // 获取目标张量的设备索引
      TORCH_CHECK(
          (*streams)[i]->device_index() == device_index, // 检查流的设备索引与目标张量的设备索引是否匹配
          "Expected the device associated with the stream at index ",
          i,
          " (was ",
          (*streams)[i]->device_index(),
          ") ",
          "to match the device supplied at that index ",
          "(expected ",
          device_index,
          ")");
      cuda_guard.reset_stream(*(*streams)[i]); // 重置 CUDA 流
    }
    // NB: We don't detect the case where `out_tensor` is already the correct
    //     view of `tensor` since that would be nontrivial and involve checking
    //     ptr, offset, and strides. So `scatter_out(src, src.chunk(...))` does
    //     more copying than `scatter(src)`.
    // 注意：我们没有检测 `out_tensor` 已经是 `tensor` 的正确视图的情况，因为这将是非平凡的，
    //      需要检查指针、偏移和步幅。因此 `scatter_out(src, src.chunk(...))` 比 `scatter(src)` 执行更多的复制。
    out_tensors[i].copy_(chunks[i], /*non_blocking=*/true); // 将分割后的部分复制到目标张量中
  }
  return out_tensors; // 返回更新后的目标 CUDA 张量列表
}
// Scatter函数：将一个张量分散到多个设备上，并返回一个包含分散后张量的向量。
std::vector<at::Tensor> scatter(
    const at::Tensor& tensor, // 输入张量，需要分散到多个设备上
    at::IntArrayRef devices, // 目标设备列表
    const std::optional<std::vector<int64_t>>& chunk_sizes, // 可选的每块大小列表
    int64_t dim, // 分割维度
    const std::optional<std::vector<std::optional<at::cuda::CUDAStream>>>& streams) { // 可选的CUDA流列表
  TORCH_CHECK(!devices.empty(), "Expected at least one device to scatter to"); // 检查设备列表不为空

  if (chunk_sizes.has_value()) { // 如果有指定块大小
    TORCH_CHECK(
        chunk_sizes->size() == devices.size(), // 检查设备列表与块大小列表长度相同
        "Expected devices and chunk_sizes to be of same length, but got "
        "len(devices) = ",
        devices.size(),
        " and len(chunk_sizes) = ",
        chunk_sizes->size());
  }

  dim = at::maybe_wrap_dim(dim, tensor); // 确保维度dim有效

  // 根据块大小分割或者分块输入张量
  std::vector<at::Tensor> chunks = chunk_sizes
      ? tensor.split_with_sizes(/*split_sizes=*/*chunk_sizes, /*dim=*/dim)
      : tensor.chunk(
            /*chunks=*/static_cast<int64_t>(devices.size()), /*dim=*/dim);

  at::cuda::OptionalCUDAStreamGuard cuda_guard; // CUDA流的可选保护

  // 遍历分割后的张量块
  for (const auto i : c10::irange(chunks.size())) {
    const auto device_index = static_cast<int16_t>(devices[i]); // 获取设备索引

    // 如果当前块的设备索引不是输入张量的设备索引
    if (device_index != tensor.get_device()) {
      // 如果提供了流列表且当前索引处有有效流
      if (i < (streams ? streams->size() : 0U) && (*streams)[i]) {
        // 检查流的设备索引与预期的设备索引是否匹配
        TORCH_CHECK(
            (*streams)[i]->device_index() == device_index,
            "Expected the device associated with the stream at index ",
            i,
            " (was ",
            (*streams)[i]->device_index(),
            ") ",
            "to match the device supplied at that index ",
            "(expected ",
            device_index,
            ")");
        cuda_guard.reset_stream(*(*streams)[i]); // 重置CUDA流
      }

      TORCH_CHECK(
          device_index >= 0, // 确保设备索引非负
          "Expected non-negative device index, but got ",
          device_index);

      // 将当前块移到目标设备上
      chunks[i] = chunks[i].to(
          {DeviceType::CUDA, device_index}, // 目标设备类型和索引
          /*non_blocking=*/true, // 非阻塞操作
          /*copy=*/false, // 不复制数据
          /*memory_format=*/at::MemoryFormat::Preserve); // 保持内存格式不变
    }
  }

  return chunks; // 返回分散后的张量块向量
}

// ***************** Gather *******************
//
// Gather函数：将一个或多个CUDA张量收集到目标张量或设备上，可以是CPU或CUDA设备。

// 没有额外的检查
static inline at::Tensor& _gather_out_impl(
    at::TensorList tensors, // 输入张量列表
    at::Tensor& out_tensor, // 输出目标张量
    int64_t dim) { // 收集的维度
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<int64_t> chunk_sizes; // 块大小列表
  chunk_sizes.reserve(tensors.size()); // 预留空间以容纳输入张量数量的块大小

  // 遍历输入张量列表，收集每个张量在指定维度上的大小
  for (auto& tensor : tensors) {
    chunk_sizes.emplace_back(tensor.size(dim));
  }

  // 根据块大小分割目标张量
  auto chunks =
      out_tensor.split_with_sizes(/*split_sizes=*/chunk_sizes, /*dim=*/dim);

  // 遍历输入张量列表，将每个张量复制到相应的块中
  for (const auto i : c10::irange(tensors.size())) {
    chunks[i].copy_(tensors[i], /*non_blocking=*/out_tensor.is_cuda());
  }

  return out_tensor; // 返回输出目标张量的引用
}

// Gather_out函数：将输入张量列表中的张量收集到目标张量中。
at::Tensor& gather_out(
    at::TensorList tensors, // 输入张量列表
    at::Tensor& out_tensor, // 输出目标张量
    int64_t dim) { // 收集的维度
    // 检查张量列表不为空，至少应包含一个张量用于聚合操作
    TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to gather from");
    // 初始化总大小为0
    int64_t total_size = 0;
    // 获取第一个张量，并记录其大小
    auto& first = tensors.front();
    const auto first_size = first.sizes();
    // 调整维度参数dim，确保其在有效范围内
    dim = at::maybe_wrap_dim(dim, first);
    // 创建一个期望大小与第一个张量相同的向量
    std::vector<int64_t> expected_size(first_size.begin(), first_size.end());
    // 遍历所有张量，进行形状和设备的检查
    for (const auto i : c10::irange(tensors.size())) {
      const auto& tensor = tensors[i];
      // 检查所有输入张量都是CUDA张量
      TORCH_CHECK(
          tensor.is_cuda(),
          "Expected all input tensors to be CUDA tensors, but "
          "tensor at index ",
          i,
          " has device '",
          tensor.device(),
          "'");
      // 检查所有输入张量的维度数量与第一个张量相同
      TORCH_CHECK(
          tensor.ndimension() == static_cast<int64_t>(expected_size.size()),
          "Expected all input tensors to have the same number of dimensions, but ",
          "tensor at index ",
          i,
          " has ",
          tensor.ndimension(),
          " dimensions, (expected ",
          expected_size.size(),
          ")");
      // 更新期望大小中的dim维度，确保每个张量的该维度大小相同
      expected_size[dim] = tensor.size(dim);
      // 检查张量的每个维度与期望大小是否匹配
      for (const auto dimension : c10::irange(expected_size.size())) {
        TORCH_CHECK(
            expected_size[dimension] == tensor.size(dimension),
            "Input tensor at index ",
            i,
            " has invalid shape ",
            tensor.sizes(),
            ", but expected ",
            at::IntArrayRef(expected_size));
      }
      // 累加当前维度上的总大小
      total_size += tensor.size(dim);
    }
    // 更新期望大小中的dim维度，使其等于总大小
    expected_size[dim] = total_size;
    // 检查输出张量的形状是否符合期望大小
    TORCH_CHECK(
        out_tensor.sizes() == expected_size,
        "Expected out tensor to have shape ",
        at::IntArrayRef(expected_size),
        ", but got ",
        out_tensor.sizes())
    
    // 调用实际的聚合函数实现，返回聚合结果张量
    return _gather_out_impl(tensors, out_tensor, dim);
// 结束 namespace torch::cuda

} // namespace torch::cuda

// 定义函数 gather，用于从给定的张量列表中收集数据到一个张量中
at::Tensor gather(
    at::TensorList tensors,                   // 输入参数：包含要收集数据的张量列表
    int64_t dim,                              // 输入参数：指定在哪个维度上收集数据
    std::optional<int32_t> destination_index) // 输入参数（可选）：指定结果张量的目标设备索引
{
  TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to gather from"); // 检查输入张量列表不能为空
  int64_t total_size = 0; // 初始化总大小为 0
  auto& first = tensors.front(); // 获取列表中的第一个张量的引用
  const auto first_size = first.sizes(); // 获取第一个张量的尺寸
  dim = at::maybe_wrap_dim(dim, first); // 对维度进行包装，确保在有效范围内
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<int64_t> expected_size(first_size.begin(), first_size.end()); // 使用第一个张量的尺寸初始化预期尺寸向量
  auto memory_format = first.suggest_memory_format(); // 获取第一个张量推荐的内存格式
  // 遍历张量列表
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i]; // 获取当前张量的引用
    TORCH_CHECK(
        tensor.is_cuda(), // 检查当前张量是否在 CUDA 上
        "Expected all input tensors to be CUDA tensors, but "
        "tensor at index ",
        i,
        " has device ",
        tensor.device()); // 报错信息：期望所有输入张量都是 CUDA 张量
    TORCH_CHECK(
        tensor.ndimension() == static_cast<int64_t>(expected_size.size()), // 检查当前张量的维度是否与预期一致
        "Expected all input tensors to have the same number of dimensions, but ",
        "tensor at index ",
        i,
        "has ",
        tensor.ndimension(),
        " dimensions, (expected ",
        expected_size.size(),
        ")");
    expected_size[dim] = tensor.size(dim); // 更新预期尺寸中指定维度的大小
    // 遍历所有维度，检查当前张量的形状是否符合预期
    for (const auto dimension : c10::irange(expected_size.size())) {
      TORCH_CHECK(
          expected_size[dimension] == tensor.size(dimension),
          "Input tensor at index ",
          i,
          " has invalid shape ",
          tensor.sizes(),
          ", but expected ",
          at::IntArrayRef(expected_size));
    }
    total_size += tensor.size(dim); // 累加当前张量在指定维度上的大小到总大小中
    // 如果当前张量推荐的内存格式不是与第一个张量一致，将内存格式设为 Contiguous
    if (memory_format != MemoryFormat::Contiguous &&
        tensor.suggest_memory_format() != memory_format) {
      memory_format = MemoryFormat::Contiguous;
    }
  }
  expected_size[dim] = total_size; // 更新预期尺寸中指定维度的大小为总大小
  at::Device device(DeviceType::CPU); // 创建一个 CPU 设备对象
  // 如果没有提供目标设备索引或者索引值为 -1，则设备为 CUDA 设备，否则根据提供的索引值设定设备
  if (!destination_index || *destination_index != -1) {
    device = at::Device(
        DeviceType::CUDA,
        destination_index ? static_cast<DeviceIndex>(*destination_index)
                          : DeviceIndex(-1));
  }
  // 创建一个空的张量 result，其尺寸为 expected_size，设备和内存格式由第一个张量决定
  at::Tensor result =
      at::empty(expected_size, first.options().device(device), memory_format);
  // 调用内部实现函数 _gather_out_impl，将收集的结果写入 result 张量中，并返回 result
  return _gather_out_impl(tensors, result, dim);
}
```