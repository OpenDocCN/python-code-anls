# `.\pytorch\c10\cuda\CUDAAllocatorConfig.cpp`

```
// 包含 CUDAAllocatorConfig.h 头文件，用于 CUDA 分配器配置
#include <c10/cuda/CUDAAllocatorConfig.h>
// 包含 CUDACachingAllocator.h 头文件，用于 CUDA 缓存分配器
#include <c10/cuda/CUDACachingAllocator.h>
// 包含 llvmMathExtras.h 头文件，提供 LLVM 的数学辅助函数
#include <c10/util/llvmMathExtras.h>

// 如果不使用 ROCm 并且支持 PYTORCH_C10_DRIVER_API_SUPPORTED 宏定义，则包含 driver_api.h 头文件
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#endif

// 定义命名空间 c10::cuda::CUDACachingAllocator
namespace c10::cuda::CUDACachingAllocator {

// 定义常量 kRoundUpPowerOfTwoIntervals，表示向上取整的幂次数间隔数目为 16
constexpr size_t kRoundUpPowerOfTwoIntervals = 16;

// CUDAAllocatorConfig 构造函数的实现
CUDAAllocatorConfig::CUDAAllocatorConfig()
    : m_max_split_size(std::numeric_limits<size_t>::max()), // 最大分割大小设置为 size_t 类型的最大值
      m_garbage_collection_threshold(0), // 垃圾回收阈值初始化为 0
      m_pinned_num_register_threads(1), // 固定内存注册线程数初始化为 1
      m_expandable_segments(false), // 可扩展段初始化为 false
      m_release_lock_on_cudamalloc(false), // 在 cudamalloc 上释放锁初始化为 false
      m_pinned_use_cuda_host_register(false), // 使用 CUDA 主机注册的固定内存初始化为 false
      m_last_allocator_settings("") { // 最后一次分配器设置为空字符串
  // 初始化向上取整的幂次数分割数组，大小为 kRoundUpPowerOfTwoIntervals，每个元素初始化为 0
  m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);
}

// 定义静态成员函数 roundup_power2_divisions，实现向上取整的幂次数分割
size_t CUDAAllocatorConfig::roundup_power2_divisions(size_t size) {
  // 计算 size 的对数值
  size_t log_size = (63 - llvm::countLeadingZeros(size));

  // 我们的间隔从1MB到64GB
  const size_t interval_start =
      63 - llvm::countLeadingZeros(static_cast<size_t>(1048576));
  const size_t interval_end =
      63 - llvm::countLeadingZeros(static_cast<size_t>(68719476736));
  // 检查间隔数是否与 kRoundUpPowerOfTwoIntervals 匹配
  TORCH_CHECK(
      (interval_end - interval_start == kRoundUpPowerOfTwoIntervals),
      "kRoundUpPowerOfTwoIntervals mismatch");

  // 计算索引值
  int index = static_cast<int>(log_size) - static_cast<int>(interval_start);

  // 确保 index 在有效范围内
  index = std::max(0, index);
  index = std::min(index, static_cast<int>(kRoundUpPowerOfTwoIntervals) - 1);
  // 返回预先计算好的向上取整的幂次数分割值
  return instance().m_roundup_power2_divisions[index];
}

// 定义 lexArgs 函数，用于解析环境变量字符串并存储到配置向量中
void CUDAAllocatorConfig::lexArgs(
    const char* env,
    std::vector<std::string>& config) {
  std::vector<char> buf;

  size_t env_length = strlen(env);
  for (size_t i = 0; i < env_length; i++) {
    // 根据特定字符进行分割，将解析出的字符串存储到 config 向量中
    if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
      if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
        buf.clear();
      }
      config.emplace_back(1, env[i]);
    } else if (env[i] != ' ') {
      buf.emplace_back(static_cast<char>(env[i]));
    }
  }
  // 处理剩余的字符，将最后一个字符串存储到 config 向量中
  if (!buf.empty()) {
    config.emplace_back(buf.begin(), buf.end());
  }
}

// 定义 consumeToken 函数，用于检查并消耗指定位置的字符 token
void CUDAAllocatorConfig::consumeToken(
    const std::vector<std::string>& config,
    size_t i,
    const char c) {
  // 检查 config 中第 i 个位置的字符是否与 c 匹配，如果不匹配则抛出错误信息
  TORCH_CHECK(
      i < config.size() && config[i] == std::string(1, c),
      "Error parsing CachingAllocator settings, expected ",
      c,
      "");
}

// 定义 parseMaxSplitSize 函数，用于解析最大分割大小的配置
size_t CUDAAllocatorConfig::parseMaxSplitSize(
    const std::vector<std::string>& config,
    size_t i) {
  // 检查并消耗 ':' 符号后的值，将其解析为 size_t 类型的最大分割大小
  consumeToken(config, ++i, ':');
  constexpr int mb = 1024 * 1024;
  if (++i < config.size()) {
    size_t val1 = stoi(config[i]);
    // 检查解析的值是否大于预定义的 kLargeBuffer 大小
    TORCH_CHECK(
        val1 > kLargeBuffer / mb,
        "CachingAllocator option max_split_size_mb too small, must be > ",
        kLargeBuffer / mb,
        "");
    // 将解析的值限制在合理范围内，并赋值给 m_max_split_size
    val1 = std::max(val1, kLargeBuffer / mb);
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / mb));
    m_max_split_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_split_size_mb value", "");


    // 使用 TORCH_CHECK 宏来检查条件，若条件为 false，则抛出错误
    // 第一个参数是条件 false，表示此处必须为 false 才会触发错误
    // 第二个参数是错误消息字符串，指示出错的原因
    // 第三个参数是额外的说明信息，本例中为空字符串
    TORCH_CHECK(false, "Error, expecting max_split_size_mb value", "");



  }
  return i;


  // 返回变量 i 的值
  return i;
}



// 解析垃圾回收阈值配置参数
size_t CUDAAllocatorConfig::parseGarbageCollectionThreshold(
    const std::vector<std::string>& config,
    size_t i) {
  // 消耗掉冒号前的token
  consumeToken(config, ++i, ':');
  // 检查是否有下一个元素
  if (++i < config.size()) {
    // 将字符串转换为双精度浮点数
    double val1 = stod(config[i]);
    // 检查阈值是否大于0
    TORCH_CHECK(
        val1 > 0, "garbage_collect_threshold too small, set it 0.0~1.0", "");
    // 检查阈值是否小于1.0
    TORCH_CHECK(
        val1 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0", "");
    // 将阈值存储到成员变量中
    m_garbage_collection_threshold = val1;
  } else {
    // 报错，期望找到垃圾回收阈值的值
    TORCH_CHECK(
        false, "Error, expecting garbage_collection_threshold value", "");
  }
  // 返回当前处理的配置索引
  return i;
}

// 解析2的幂次方向上取整分区配置参数
size_t CUDAAllocatorConfig::parseRoundUpPower2Divisions(
    const std::vector<std::string>& config,
    size_t i) {
  // 消耗掉冒号前的token
  consumeToken(config, ++i, ':');
  // 标记第一个值为true
  bool first_value = true;

  // 检查是否有下一个元素
  if (++i < config.size()) {
    // 如果下一个元素是"["
    if (std::string_view(config[i]) == "[") {
      // 上一个索引位置初始化为0
      size_t last_index = 0;
      // 循环直到找到"]"为止
      while (++i < config.size() && std::string_view(config[i]) != "]") {
        // 获取当前值
        const std::string& val1 = config[i];
        size_t val2 = 0;

        // 消耗掉冒号前的token
        consumeToken(config, ++i, ':');
        // 检查是否有下一个元素
        if (++i < config.size()) {
          // 将字符串转换为整数
          val2 = stoi(config[i]);
        } else {
          // 报错，解析roundup_power2_divisions值出错
          TORCH_CHECK(
              false, "Error parsing roundup_power2_divisions value", "");
        }
        // 检查值是否为0或者是2的幂次方
        TORCH_CHECK(
            val2 == 0 || llvm::isPowerOf2_64(val2),
            "For roundups, the divisons has to be power of 2 or 0 to disable roundup ",
            "");

        // 如果val1是大于号">"
        if (std::string_view(val1) == ">") {
          // 使用val2填充从last_index开始的m_roundup_power2_divisions
          std::fill(
              std::next(
                  m_roundup_power2_divisions.begin(),
                  static_cast<std::vector<unsigned long>::difference_type>(
                      last_index)),
              m_roundup_power2_divisions.end(),
              val2);
        } else {
          // 将val1转换为长整型
          size_t val1_long = stoul(val1);
          // 检查val1_long是否是2的幂次方
          TORCH_CHECK(
              llvm::isPowerOf2_64(val1_long),
              "For roundups, the intervals have to be power of 2 ",
              "");

          // 计算索引值
          size_t index = 63 - llvm::countLeadingZeros(val1_long);
          index = std::max((size_t)0, index);
          index = std::min(index, m_roundup_power2_divisions.size() - 1);

          // 如果是第一个值
          if (first_value) {
            // 使用val2填充从开始到index位置的m_roundup_power2_divisions
            std::fill(
                m_roundup_power2_divisions.begin(),
                std::next(
                    m_roundup_power2_divisions.begin(),
                    static_cast<std::vector<unsigned long>::difference_type>(
                        index)),
                val2);
            first_value = false;
          }
          // 如果index小于m_roundup_power2_divisions的大小
          if (index < m_roundup_power2_divisions.size()) {
            // 将val2存储到m_roundup_power2_divisions的index位置
            m_roundup_power2_divisions[index] = val2;
          }
          // 更新last_index为index
          last_index = index;
        }

        // 如果下一个元素不是"]"
        if (std::string_view(config[i + 1]) != "]") {
          // 消耗掉逗号前的token
          consumeToken(config, ++i, ',');
        }
      }
    } else { // 对于向后兼容性保留此代码
      // 将 config[i] 转换为 size_t 类型
      size_t val1 = stoi(config[i]);
      // 使用 TORCH_CHECK 进行断言，确保 val1 是 2 的幂次方
      TORCH_CHECK(
          llvm::isPowerOf2_64(val1),
          "For roundups, the divisons has to be power of 2 ",
          "");
      // 使用 val1 填充 m_roundup_power2_divisions 的整个范围
      std::fill(
          m_roundup_power2_divisions.begin(),
          m_roundup_power2_divisions.end(),
          val1);
    }
  } else {
    // 如果不满足条件，抛出错误信息
    TORCH_CHECK(false, "Error, expecting roundup_power2_divisions value", "");
  }
  // 返回循环变量 i 的值
  return i;
size_t CUDAAllocatorConfig::parseAllocatorConfig(
    const std::vector<std::string>& config,
    size_t i,
    bool& used_cudaMallocAsync) {
  // 跳过配置项后的冒号
  consumeToken(config, ++i, ':');
  // 如果还有下一个配置项
  if (++i < config.size()) {
    // 检查分配器后端是否为"native"或"cudaMallocAsync"
    TORCH_CHECK(
        ((config[i] == "native") || (config[i] == "cudaMallocAsync")),
        "Unknown allocator backend, "
        "options are native and cudaMallocAsync");
    // 设置是否使用cudaMallocAsync
    used_cudaMallocAsync = (config[i] == "cudaMallocAsync");
    // 如果不是使用ROCm
#ifndef USE_ROCM
    // HIP支持hipMallocAsync，不需要检查版本
    if (used_cudaMallocAsync) {
      // 如果CUDA版本大于或等于11.4，则继续
#if CUDA_VERSION >= 11040
      int version = 0;
      // 获取CUDA驱动程序版本
      C10_CUDA_CHECK(cudaDriverGetVersion(&version));
      // 检查CUDA运行时版本是否大于或等于11.4
      TORCH_CHECK(
          version >= 11040,
          "backend:cudaMallocAsync requires CUDA runtime "
          "11.4 or newer, but cudaDriverGetVersion returned ",
          version);
#else
      // 如果CUDA版本不支持cudaMallocAsync，抛出错误
      TORCH_CHECK(
          false,
          "backend:cudaMallocAsync requires PyTorch to be built with "
          "CUDA 11.4 or newer, but CUDA_VERSION is ",
          CUDA_VERSION);
#endif
    }
#endif
    // 检查分配器后端在运行时是否与加载时一致
    TORCH_INTERNAL_ASSERT(
        config[i] == get()->name(),
        "Allocator backend parsed at runtime != "
        "allocator backend parsed at load time");
  } else {
    // 如果没有后续配置项，抛出错误
    TORCH_CHECK(false, "Error parsing backend value", "");
  }
  // 返回当前处理的配置项索引
  return i;
}

void CUDAAllocatorConfig::parseArgs(const char* env) {
  // 如果环境变量为空，设置默认值并返回
  m_max_split_size = std::numeric_limits<size_t>::max();
  m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);
  m_garbage_collection_threshold = 0;
  bool used_cudaMallocAsync = false;
  bool used_native_specific_option = false;

  if (env == nullptr) {
    return;
  }
  {
    // 加锁保护最后的分配器设置
    std::lock_guard<std::mutex> lock(m_last_allocator_settings_mutex);
    m_last_allocator_settings = env;
  }

  // 将环境变量解析为配置项列表
  std::vector<std::string> config;
  lexArgs(env, config);

  // 遍历配置项列表
  for (size_t i = 0; i < config.size(); i++) {
    std::string_view config_item_view(config[i]);
    // 解析max_split_size_mb选项
    if (config_item_view == "max_split_size_mb") {
      i = parseMaxSplitSize(config, i);
      used_native_specific_option = true;
    // 解析garbage_collection_threshold选项
    } else if (config_item_view == "garbage_collection_threshold") {
      i = parseGarbageCollectionThreshold(config, i);
      used_native_specific_option = true;
    // 解析roundup_power2_divisions选项
    } else if (config_item_view == "roundup_power2_divisions") {
      i = parseRoundUpPower2Divisions(config, i);
      used_native_specific_option = true;
    // 解析backend选项
    } else if (config_item_view == "backend") {
      i = parseAllocatorConfig(config, i, used_cudaMallocAsync);
    } else if (config_item_view == "expandable_segments") {
      // 标记使用了本地特定选项
      used_native_specific_option = true;
      // 消耗 ':' 标记
      consumeToken(config, ++i, ':');
      // 增加索引
      ++i;
      // 检查下一个配置项是否是 "True" 或 "False"
      TORCH_CHECK(
          i < config.size() &&
              (std::string_view(config[i]) == "True" ||
               std::string_view(config[i]) == "False"),
          "Expected a single True/False argument for expandable_segments");
      // 将配置项赋值给 config_item_view
      config_item_view = config[i];
      // 设置 m_expandable_segments 标志
      m_expandable_segments = (config_item_view == "True");
    } else if (
        // ROCm 构建中的 hipify 步骤会将 "cuda" 替换为 "hip"，为了方便起见，接受两者。我们必须在此处分隔字符串以防止 hipify。
        config_item_view == "release_lock_on_hipmalloc" ||
        config_item_view ==
            "release_lock_on_c"
            "udamalloc") {
      // 标记使用了本地特定选项
      used_native_specific_option = true;
      // 消耗 ':' 标记
      consumeToken(config, ++i, ':');
      // 增加索引
      ++i;
      // 检查下一个配置项是否是 "True" 或 "False"
      TORCH_CHECK(
          i < config.size() &&
              (std::string_view(config[i]) == "True" ||
               std::string_view(config[i]) == "False"),
          "Expected a single True/False argument for release_lock_on_cudamalloc");
      // 将配置项赋值给 config_item_view
      config_item_view = config[i];
      // 设置 m_release_lock_on_cudamalloc 标志
      m_release_lock_on_cudamalloc = (config_item_view == "True");
    } else if (
        // ROCm 构建中的 hipify 步骤会将 "cuda" 替换为 "hip"，为了方便起见，接受两者。我们必须在此处分隔字符串以防止 hipify。
        config_item_view == "pinned_use_hip_host_register" ||
        config_item_view ==
            "pinned_use_c"
            "uda_host_register") {
      // 解析 pinned_use_cudahostregister 配置项
      i = parsePinnedUseCudaHostRegister(config, i);
      // 标记使用了本地特定选项
      used_native_specific_option = true;
    } else if (config_item_view == "pinned_num_register_threads") {
      // 解析 pinned_num_register_threads 配置项
      i = parsePinnedNumRegisterThreads(config, i);
      // 标记使用了本地特定选项
      used_native_specific_option = true;
    } else {
      // 若未识别到有效的 CachingAllocator 选项，则报错
      TORCH_CHECK(
          false, "Unrecognized CachingAllocator option: ", config_item_view);
    }

    // 如果还有更多配置项，消耗逗号标记
    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }

  // 如果使用了 cudaMallocAsync 且使用了本地特定选项，则发出警告
  if (used_cudaMallocAsync && used_native_specific_option) {
    TORCH_WARN(
        "backend:cudaMallocAsync ignores max_split_size_mb,"
        "roundup_power2_divisions, and garbage_collect_threshold.");
  }
}

size_t CUDAAllocatorConfig::parsePinnedUseCudaHostRegister(
    const std::vector<std::string>& config,
    size_t i) {
  // 跳过冒号标记，进入下一个配置参数
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    // 检查下一个参数是否为 "True" 或 "False"
    TORCH_CHECK(
        (config[i] == "True" || config[i] == "False"),
        "Expected a single True/False argument for pinned_use_cuda_host_register");
    // 根据参数设置 m_pinned_use_cuda_host_register 的值
    m_pinned_use_cuda_host_register = (config[i] == "True");
  } else {
    // 如果参数不足，抛出错误信息
    TORCH_CHECK(
        false, "Error, expecting pinned_use_cuda_host_register value", "");
  }
  // 返回当前处理的参数索引
  return i;
}

size_t CUDAAllocatorConfig::parsePinnedNumRegisterThreads(
    const std::vector<std::string>& config,
    size_t i) {
  // 跳过冒号标记，进入下一个配置参数
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    // 将字符串转换为整数
    size_t val2 = stoi(config[i]);
    // 检查是否为 2 的幂次方
    TORCH_CHECK(
        llvm::isPowerOf2_64(val2),
        "Number of register threads has to be power of 2 ",
        "");
    // 获取最大线程数并进行验证
    auto maxThreads = CUDAAllocatorConfig::pinned_max_register_threads();
    TORCH_CHECK(
        val2 <= maxThreads,
        "Number of register threads should be less than or equal to " +
            std::to_string(maxThreads),
        "");
    // 设置 m_pinned_num_register_threads 的值
    m_pinned_num_register_threads = val2;
  } else {
    // 如果参数不足，抛出错误信息
    TORCH_CHECK(
        false, "Error, expecting pinned_num_register_threads value", "");
  }
  // 返回当前处理的参数索引
  return i;
}

// 设置分配器的配置参数
void setAllocatorSettings(const std::string& env) {
  // 解析环境变量并应用配置
  CUDACachingAllocator::CUDAAllocatorConfig::instance().parseArgs(env.c_str());
}

} // namespace c10::cuda::CUDACachingAllocator
```