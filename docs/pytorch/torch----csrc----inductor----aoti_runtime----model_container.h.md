# `.\pytorch\torch\csrc\inductor\aoti_runtime\model_container.h`

```
#pragma once

#include <algorithm>
#include <deque>
#include <future>
#include <mutex>
#include <shared_mutex>

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/model.h>

namespace torch {
namespace aot_inductor {

class AOTInductorModelContainer {
 public:
  // 构造函数，初始化 AOTInductorModelContainer 实例
  AOTInductorModelContainer(
      size_t num_models, // 模型数量
      const std::string& device_str, // 设备字符串
      std::optional<std::string> cubin_dir = std::nullopt) { // 可选的 cubin 目录
    // 创建常量映射和常量数组的共享指针
    constants_map_ = std::make_shared<ConstantMap>();
    constants_array_ = std::make_shared<std::vector<ConstantHandle>>();
    // 设置是否使用辅助模型和是否进行常量折叠的初始状态
    use_secondary_ = false;
    constant_folded_ = false;
    // 预留空间以容纳模型
    models_.reserve(num_models);
    available_models_.reserve(num_models);
    // 循环创建指定数量的 AOTInductorModel 实例并添加到 models_ 和 available_models_
    for (size_t i = 0; i < num_models; ++i) {
      models_.push_back(AOTInductorModel::Create(
          constants_map_, constants_array_, device_str, cubin_dir));
      available_models_.push_back(models_.back().get());
    }

    // 使用可用模型的第一个模型初始化输入名称列表
    auto* model = available_models_[0];
    size_t num_inputs = model->num_inputs();
    input_names_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
      input_names_.push_back(model->input_name(i));
    }

    // 使用可用模型的第一个模型初始化输出名称列表
    size_t num_outputs = model->num_outputs();
    output_names_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
      output_names_.push_back(model->output_name(i));
    }

    // 载入常量数据到模型
    model->load_constants();
#ifdef USE_CUDA
    // 如果使用 CUDA，则释放常量数据的内存，并计算 CUDA 相关的常量数据
    constant_blob_ = model->release_constant_blob();
    constants_internal_offset_.resize(model->num_constants());
    model->compute_cuda_constant_blob(blob_size_, constants_internal_offset_);
#endif

    // 更新所有模型的常量映射
    for (auto& model : models_) {
      model->update_constants_map(constants_map_);
    }

    // 获取输入和输出规范
    in_spec_ = model->get_in_spec();
    out_spec_ = model->get_out_spec();
  }

  // 运行模型推断
  void run(
      AtenTensorHandle*
          input_handles, // 输入 AtenTensorHandle 数组；句柄被窃取，数组本身被借用
      AtenTensorHandle*
          output_handles, // 用于写入输出 AtenTensorHandle 的数组；句柄将被调用者窃取，数组本身被借用
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor) {
    // 使用 shared_lock 对 model_exec_mutex_ 进行加锁，以保证多个线程可以同时读取 model_exec_mutex_
    std::shared_lock model_lk(model_exec_mutex_);
    // 获取一个可用的模型指针
    auto* model = get_available_model();

    // 如果常量还未折叠
    if (!constant_folded_) {
      // 在这一点上，常量还没有准备好。我们需要在执行模型之前调用常量折叠。
      // 在这一点上获取一个独占锁以确保所有的常量都准备好了。
      model_lk.unlock();
      // 获取一个独占锁，确保常量只折叠一次。
      std::unique_lock constants_folding_lk(model_exec_mutex_);
      // 再次检查常量是否折叠
      if (!constant_folded_) {
        // 运行常量折叠操作，并初始化常量映射
        auto folded_const_map = model->run_const_fold(
            stream, proxy_executor, /* initialization = */ true);
        // 更新常量缓冲区
        update_constant_buffer(
            folded_const_map,
            /* use_inactive = */ false,
            /* validate_full_update = */ false);
        // 标记常量已折叠
        constant_folded_ = true;
      }
      constants_folding_lk.unlock();
      // 重新获取 shared_lock
      model_lk.lock();
    }

    try {
      // 运行模型
      model->run(input_handles, output_handles, stream, proxy_executor);
    } catch (...) {
      // 捕获任何异常后，将模型重新加入可用模型列表并抛出异常
      std::lock_guard lk(models_mutex_);
      available_models_.push_back(model);
      throw;
    }

    {
      // 将模型添加到待处理模型列表中
      std::lock_guard lk(models_mutex_);
      pending_models_.push_back(model);
    }
    // 通知等待的线程有新的待处理模型可用
    pending_models_available_.notify_one();
  }

  // 获取常量的数量
  size_t num_constants() const {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
    // 返回第一个模型的常量数量
    return models_[0]->num_constants();
  }

  // 获取 constants_info_ 中第 idx 个常量的名称
  const char* constant_name(size_t idx) const {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
    // 返回第一个模型 constants_info_ 中第 idx 个常量的名称
    return models_[0]->constant_name(idx);
  }

  // 获取 constants_info_ 中第 idx 个常量的原始完全限定名（FQN）
  const char* constant_original_fqn(size_t idx) const {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
    // 返回第一个模型 constants_info_ 中第 idx 个常量的原始 FQN
    return models_[0]->constant_original_fqn(idx);
  }

  // 获取 constants_info_ 中第 idx 个常量是否来自折叠的信息
  bool constant_from_folded(size_t idx) const {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
    // 返回第一个模型 constants_info_ 中第 idx 个常量是否来自折叠的信息
    return models_[0]->constant_from_folded(idx);
  }

  // 获取 constants_info_ 中第 idx 个常量的数据类型（dtype）
  int32_t constant_dtype(size_t idx) const {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
    // 返回第一个模型 constants_info_ 中第 idx 个常量的数据类型（dtype）
    return models_[0]->constant_dtype(idx);
  }

  // 运行常量折叠
  void run_const_fold(
      bool inactive_buffer,
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor) {
    // 使用 shared_lock 对 model_exec_mutex_ 进行加锁
    std::shared_lock model_lk(model_exec_mutex_);
    // 获取一个可用的模型指针
    auto* model = get_available_model();
    if (!inactive_buffer) {
      // 如果不是使用非活跃缓冲区，则执行以下操作：
      // 需要获取独占锁以在活跃缓冲区上运行常量折叠。
      model_lk.unlock();  // 解锁模型锁，以便其他线程可以访问模型
      std::unique_lock constants_folding_lk(model_exec_mutex_);  // 获取常量折叠所需的互斥锁
      try {
        auto folded_const_map = model->run_const_fold(stream, proxy_executor);  // 运行模型的常量折叠操作
        update_constant_buffer(
            folded_const_map,
            /* use_inactive = */ false,
            /* validate_full_update = */ false);  // 更新常量缓冲区，不使用非活跃缓冲区，不进行完整更新验证
      } catch (...) {
        std::lock_guard lk(models_mutex_);  // 获取模型容器的互斥锁
        available_models_.push_back(model);  // 将模型放回可用模型列表中
        throw;  // 抛出异常，向上层传递异常
      }
      constants_folding_lk.unlock();  // 解锁常量折叠的互斥锁
      model_lk.lock();  // 重新锁定模型锁，以便后续操作可以继续保持模型的一致性
    } else {
      // 如果使用非活跃缓冲区，则执行以下操作：
      // 将常量映射交换到模型的非活跃缓冲区，以运行常量运行。
      auto constants_map = get_constants_map(/* get_inactive= */ true);  // 获取非活跃缓冲区的常量映射
      auto constants_array = get_constants_array(/* get_inactive= */ true);  // 获取非活跃缓冲区的常量数组

      try {
        model->update_constants_map(
            constants_map, /* remap_constants_array= */ false);  // 更新模型的常量映射，不重新映射常量数组
        model->update_constants_array(constants_array);  // 更新模型的常量数组

        auto folded_const_map = model->run_const_fold(stream, proxy_executor);  // 运行模型的常量折叠操作
        update_constant_buffer(
            folded_const_map,
            /* use_inactive = */ true,
            /* validate_full_update = */ false);  // 更新常量缓冲区，使用非活跃缓冲区，不进行完整更新验证

        // 将模型的常量映射切换回活跃缓冲区
        constants_map = get_constants_map(/* get_inactive= */ false);  // 获取活跃缓冲区的常量映射
        constants_array = get_constants_array(/* get_inactive= */ false);  // 获取活跃缓冲区的常量数组
        model->update_constants_map(
            constants_map, /* remap_constants_array= */ false);  // 更新模型的常量映射，不重新映射常量数组
        model->update_constants_array(constants_array);  // 更新模型的常量数组
      } catch (...) {
        std::lock_guard lk(models_mutex_);  // 获取模型容器的互斥锁
        available_models_.push_back(model);  // 将模型放回可用模型列表中
        throw;  // 抛出异常，向上层传递异常
      }
    }

    {
      std::lock_guard lk(models_mutex_);  // 获取模型容器的互斥锁
      pending_models_.push_back(model);  // 将模型添加到待处理模型列表中
    }
    pending_models_available_.notify_one();  // 通知等待的线程有待处理模型可用
  }

  bool _is_tensor_constant(const std::string& constant_name) const {
    return constant_name.rfind("_tensor_constant", 0) == 0;  // 检查常量名是否以 "_tensor_constant" 开头，返回布尔值
  }
  // This function updates the buffer for storing constants.
  // It will update the buffer, the mapping and the array mapping.
  void update_constant_buffer(
      const std::unordered_map<std::string, AtenTensorHandle>& constants_map,
      bool use_inactive,
      bool validate_full_update) {
    if (this->num_models() == 0) {
      throw std::runtime_error("No model available in container!");  // 如果模型容器中没有可用的模型，则抛出运行时异常
    }
    auto num_constants = models_[0]->num_constants();  // 获取模型容器中第一个模型的常量数量
    // 如果需要进行完全更新验证
    if (validate_full_update) {
      // 遍历所有常量
      for (size_t idx = 0; idx < num_constants; idx++) {
        // 如果该常量是从折叠状态中得到的，跳过此次循环
        if (models_[0]->constant_from_folded(idx)) {
          continue;
        }

        // 获取常量的名称
        auto constant_name = std::string(models_[0]->constant_name(idx));
        // 在常量映射表中查找该常量名
        auto it = constants_map.find(constant_name);
        // 如果在常量映射表中找不到该常量名
        if (it == constants_map.end()) {
          // 如果该常量是张量常量并且在使用非活动状态，则跳过此次循环并输出警告信息
          if (_is_tensor_constant(constant_name)) {
            std::cerr << "[WARNING] Found constant " << constant_name
                      << " in model, but not provided by user!\n";
            continue;
          }
          // 抛出运行时错误，指出在常量映射表中找不到该常量名
          throw std::runtime_error(
              std::string("Cannot find constants ") + constant_name +
              std::string(" in constants_map!"));
        }
      }
    }

    // 获取原始常量映射表，如果不使用非活动状态则取反
    auto original_constants_map = get_constants_map(!use_inactive);
    // 获取要更新的常量映射表，根据使用非活动状态的布尔值决定
    auto constants_map_to_update = get_constants_map(use_inactive);

    // 再次遍历所有常量
    for (size_t idx = 0; idx < num_constants; idx++) {
      // 获取常量的名称
      auto constant_name = std::string(models_[0]->constant_name(idx));
      // 在常量映射表中查找该常量名
      auto it = constants_map.find(constant_name);
      // 如果在常量映射表中找不到该常量名，并且该常量不是张量常量且使用非活动状态
      if (it == constants_map.end() &&
          !(_is_tensor_constant(constant_name) && use_inactive)) {
        // 跳过此次循环
        continue;
      }
#ifdef USE_CUDA
      // 如果定义了 USE_CUDA 宏，则使用 CUDA 加速
      AtenTensorHandle tensor;
      // 检查常量是否为常量张量，并且使用非活动状态
      if (_is_tensor_constant(constant_name) && use_inactive) {
        // 如果是常量张量且使用非活动状态，则从原始常量映射中获取张量
        tensor = original_constants_map->find(constant_name)->second.get();
      } else {
        // 否则，从当前常量映射中获取张量
        tensor = it->second;
      }
      auto* constants_blob_ptr =
          static_cast<uint8_t*>(get_constant_blob_ptr(use_inactive));

      // 将数据移动到容器处理的数据块中
      uint8_t* internal_constants_ptr =
          constants_blob_ptr + constants_internal_offset_[idx];
      void* user_constant_ptr;
      int64_t constant_size;
      // 获取张量的数据指针和存储大小
      aoti_torch_get_data_ptr(tensor, &user_constant_ptr);
      aoti_torch_get_storage_size(tensor, &constant_size);

      // 使用 CUDA 函数进行内存拷贝
      AOTI_RUNTIME_DEVICE_CHECK(cudaMemcpy(
          internal_constants_ptr,
          user_constant_ptr,
          constant_size,
          cudaMemcpyDefault));

      // 从容器处理的数据块中生成张量
      // 从提供的张量中提取步长和偏移量，因为不保证张量是连续的
      AtenTensorHandle tensor_handle;
      int64_t* stride;
      int64_t offset;
      int device_idx = models_[0]->get_device_idx();
      // 获取张量的步长和存储偏移
      AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(tensor, &stride));
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_get_storage_offset(tensor, &offset));
      // 从数据块中创建张量
      AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob(
          internal_constants_ptr,
          models_[0]->constant_ndim(idx),
          models_[0]->constant_shape(idx),
          stride,
          offset,
          models_[0]->constant_dtype(idx),
          aoti_torch_device_type_cuda(),
          device_idx,
          &tensor_handle));
#else // USE_CUDA
      // 如果未定义 USE_CUDA 宏，则直接使用当前的张量句柄
      AtenTensorHandle tensor_handle = it->second;
#endif // USE_CUDA

      // 将张量句柄放入常量映射中，此时将接管张量的所有权
      constants_map_to_update->emplace(constant_name, tensor_handle);
    }
    // 更新非活动状态的常量数组
    update_array_from_map(
        get_constants_array(use_inactive), constants_map_to_update);
  }

  // 从映射中更新常量数组
  void update_array_from_map(
      std::shared_ptr<std::vector<ConstantHandle>> constants_array,
      std::shared_ptr<ConstantMap> constants_map) {
    auto num_constants = models_[0]->num_constants();
    for (size_t idx = 0; idx < num_constants; idx++) {
      // 如果常量在映射中存在，则更新常量数组中的对应项
      if (constants_map->find(models_[0]->constant_name(idx)) !=
          constants_map->end()) {
        constants_array->at(idx) = ConstantHandle(
            constants_map->find(models_[0]->constant_name(idx))->second);
      }
    }
  }

  // 交换常量缓冲区
  void swap_constant_buffer() {
    // 使用互斥锁保护操作的原子性
    std::lock_guard unique_lk(model_exec_mutex_);

    // 获取非活动状态的常量映射和常量数组
    auto constants_map = get_constants_map(/* get_inactive= */ true);
    auto constants_array = get_constants_array(/* get_inactive= */ true);
    for (auto& model : models_) {
      // 对于每个模型，更新其常量映射，不重新映射常量数组
      model->update_constants_map(
          constants_map, /* remap_constants_array = */ false);
      // 更新模型的常量数组
      model->update_constants_array(constants_array);
    }

    // 切换使用辅助模型的标志
    use_secondary_ = !use_secondary_;
  }

  // 返回输入名称的数量
  size_t num_inputs() const {
    return input_names_.size();
  }

  // 返回输出名称的数量
  size_t num_outputs() const {
    return output_names_.size();
  }

  // 返回指定索引的输入名称的 C 字符串表示
  const char* input_name(size_t idx) const {
    return input_names_.at(idx).c_str();
  }

  // 返回指定索引的输出名称的 C 字符串表示
  const char* output_name(size_t idx) const {
    return output_names_.at(idx).c_str();
  }

  // 返回模型的数量
  size_t num_models() const {
    return models_.size();
  }

  // 返回输入规范的 C 字符串表示
  const char* get_in_spec() const {
    return in_spec_;
  }

  // 返回输出规范的 C 字符串表示
  const char* get_out_spec() const {
    return out_spec_;
  }

 private:
  // 输入名称的字符串向量
  std::vector<std::string> input_names_;
  // 输出名称的字符串向量
  std::vector<std::string> output_names_;
  // 输入规范的 C 字符串指针
  const char* in_spec_;
  // 输出规范的 C 字符串指针
  const char* out_spec_;
#ifdef USE_CUDA
  // 用于存储 CUDA 下的常量 at::Tensor 的内存块
  CUDAPtr constant_blob_;
  // 用于存储 CUDA 下的次要常量 at::Tensor 的内存块
  CUDAPtr constant_blob_secondary_;

  // 在完全支持 CPU 案例更新之前，将其放置在 USE_CUDA 内部
  size_t blob_size_;
  // 常量内部偏移的大小向量
  std::vector<size_t> constants_internal_offset_;
#endif // USE_CUDA

  // 确定模型正在使用的常量
  bool use_secondary_;

  // 确定是否已经进行了常量折叠
  bool constant_folded_;

  // 持有常量到 at::Tensor 映射的指针
  // at::Tensor 的底层数据存储在 constant_blob_（用于 CUDA）或 _binary_constants_bin_start（用于 CPU）中
  std::shared_ptr<ConstantMap> constants_map_;
  // 持有常量到 at::Tensor 映射的指针（次要常量）
  std::shared_ptr<ConstantMap> constants_map_secondary_;

  // 持有常量索引数组，用于在运行时加快查找速度
  std::shared_ptr<std::vector<ConstantHandle>> constants_array_;
  // 持有常量索引数组的指针（次要常量）
  std::shared_ptr<std::vector<ConstantHandle>> constants_array_secondary_;

  // 持有该容器拥有的所有 AOTInductorModel 实例的唯一指针数组
  std::vector<std::unique_ptr<AOTInductorModel>> models_;

  // 持有可用于推理的 AOTInductorModel 实例数组
  std::vector<AOTInductorModel*> available_models_;

  // 持有已经开始运行推理且可以放置到 available_models_ 中的 AOTInductorModel 实例队列
  std::deque<AOTInductorModel*> pending_models_;

  // 保护 available_models_ 和 pending_models_ 的互斥量
  std::mutex models_mutex_;

  // 每当将模型放置到 pending_models_ 上时发出通知
  std::condition_variable pending_models_available_;

  // 获取可用的模型实例
  AOTInductorModel* get_available_model() {
    std::unique_lock lk(models_mutex_);
    if (available_models_.empty()) {
      reclaim_finished_models(lk);
    }
    auto* result = available_models_.back();
    available_models_.pop_back();
    return result;
  }

  // 用于保护模型执行的互斥量
  // 如果允许并发执行，则以共享模式获取互斥量
  // 当需要独占访问模型时，以独占模式获取互斥量，例如权重交换时确保没有其他人正在执行模型
  std::shared_mutex model_exec_mutex_;

#ifdef USE_CUDA
  // 获取常量内存块的指针，根据是否需要获取非活动常量决定使用 primary 或 secondary
  void* get_constant_blob_ptr(bool get_inactive) {
    if ((get_inactive && use_secondary_) ||
        (!get_inactive && !use_secondary_)) {
      return constant_blob_.get();
    } else {
      if (!constant_blob_secondary_) {
        constant_blob_secondary_ = RAII_cudaMalloc(blob_size_);
      }
      return constant_blob_secondary_.get();
    }
  }
#endif // USE_CUDA

  // 获取常量映射的指针，根据是否需要获取非活动常量决定使用 primary 或 secondary
  std::shared_ptr<ConstantMap> get_constants_map(bool get_inactive) {
    if ((get_inactive && use_secondary_) ||
        (!get_inactive && !use_secondary_)) {
      return constants_map_;
    } else {
      return constants_map_secondary_;
    }
  }
  } else {
    // 如果没有使用常量映射的辅助副本，则创建一个新的常量映射对象
    if (!constants_map_secondary_) {
      constants_map_secondary_ = std::make_shared<ConstantMap>();
    }
    // 返回常量映射的辅助副本
    return constants_map_secondary_;
  }
}

std::shared_ptr<std::vector<ConstantHandle>> get_constants_array(
    bool get_inactive) {
  // 根据参数决定返回活跃或非活跃状态的常量数组
  if ((get_inactive && use_secondary_) ||
      (!get_inactive && !use_secondary_)) {
    return constants_array_;
  } else {
    // 如果没有使用常量数组的辅助副本，则创建一个新的常量数组
    if (!constants_array_secondary_) {
      constants_array_secondary_ =
          std::make_shared<std::vector<ConstantHandle>>(
              models_[0]->num_constants());
    }
    // 返回常量数组的辅助副本
    return constants_array_secondary_;
  }
}

void reclaim_finished_models(std::unique_lock<std::mutex>& lk) {
  // 将已完成的模型实例推入 pending_models_ 的末尾
  auto it = std::stable_partition(
      pending_models_.begin(),
      pending_models_.end(),
      [](AOTInductorModel* m) { return !m->is_finished(); });

  if (it != pending_models_.end()) {
    // 将已完成的模型实例推入 available_models_ 的末尾，以避免在等待 pending_models_available_ 条件时阻塞
    available_models_.insert(
        available_models_.end(), it, pending_models_.end());
    pending_models_.erase(it, pending_models_.end());
    return;
  }

  // 等待 pending_models_ 非空，以便获取已完成的模型实例
  pending_models_available_.wait(
      lk, [this]() { return !pending_models_.empty(); });

  // 简化调度策略：总是等待第一个 pending_models_ 中的模型完成
  auto* model = pending_models_.front();
  pending_models_.pop_front();
  lk.unlock();

  // 等待模型完成
  try {
    model->wait_for_completion();
  } catch (...) {
    lk.lock();
    available_models_.push_back(model);
    throw;
  }

  lk.lock();
  available_models_.push_back(model);
}
};

} // namespace aot_inductor
} // namespace torch
```