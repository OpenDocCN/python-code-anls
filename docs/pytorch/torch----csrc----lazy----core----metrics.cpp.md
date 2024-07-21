# `.\pytorch\torch\csrc\lazy\core\metrics.cpp`

```py
// 引入Torch Lazy模块中的度量相关头文件
#include <torch/csrc/lazy/core/metrics.h>

// 引入C10库中的日志和范围工具
#include <c10/util/Logging.h>
#include <c10/util/irange.h>

// 引入Torch Lazy模块中的后端接口和配置、辅助工具
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/util.h>

// 引入STL标准库中的算法、时间、字符串流等组件
#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

// Torch Lazy命名空间
namespace torch {
namespace lazy {
namespace {

// 从环境变量中读取百分位数列表
const std::vector<double>* ReadEnvPercentiles() {
  // 将环境变量中的百分位数字符串按':'分割成列表
  std::vector<std::string> percentiles_list =
      StrSplit(FLAGS_torch_lazy_metrics_percentiles, ':');
  // 创建存储百分位数的unique_ptr，使用make_unique分配内存
  std::unique_ptr<std::vector<double>> metrics_percentiles =
      std::make_unique<std::vector<double>>();
  // 遍历百分位数列表
  for (auto& pct_str : percentiles_list) {
    // 将字符串转换为double类型的百分位数
    double pct = std::stod(pct_str);
    // 检查百分位数是否在有效范围内 (0, 1)
    TORCH_CHECK(pct > 0.0 && pct < 1.0, "Invalid percentile: ", pct);
    // 将百分位数添加到百分位数列表中
    metrics_percentiles->push_back(pct);
  }
  // 对百分位数列表进行排序
  std::sort(metrics_percentiles->begin(), metrics_percentiles->end());
  // 释放unique_ptr的所有权，返回指向百分位数列表的指针
  return metrics_percentiles.release();
}

// 获取存储环境变量百分位数的静态常量引用
const std::vector<double>& GetPercentiles() {
  // 仅在首次调用时执行，从环境变量中读取百分位数并保存静态常量中
  static const std::vector<double>* metrics_percentiles = ReadEnvPercentiles();
  return *metrics_percentiles;
}

// 发送度量信息到输出流，包括名称、数据以及相关统计信息
void EmitMetricInfo(
    const std::string& name,
    MetricData* data,
    std::stringstream* ss) {
  // 累加器和总样本数初始化
  double accumulator = 0.0;
  size_t total_samples = 0;
  // 获取度量数据中的样本集合
  std::vector<Sample> samples = data->Samples(&accumulator, &total_samples);
  // 输出度量信息的名称
  (*ss) << "Metric: " << name << std::endl;
  // 输出总样本数和累加器的表示
  (*ss) << "  TotalSamples: " << total_samples << std::endl;
  (*ss) << "  Accumulator: " << data->Repr(accumulator) << std::endl;
  // 如果样本集合非空，计算样本值的速率
  if (!samples.empty()) {
    double total = 0.0;
    for (auto& sample : samples) {
      total += sample.value;
    }
    // 计算时间间隔以及样本值的速率
    int64_t delta_time =
        samples.back().timestamp_ns - samples.front().timestamp_ns;
    if (delta_time > 0) {
      double value_sec = 1e6 * (total / (delta_time / 1000.0));
      (*ss) << "  ValueRate: " << data->Repr(value_sec) << " / second"
            << std::endl;
      double count_sec =
          1e6 * (static_cast<double>(samples.size()) / (delta_time / 1000.0));
      (*ss) << "  Rate: " << count_sec << " / second" << std::endl;
    }
  }

  // 获取百分位数列表的引用，并按样本值排序
  const std::vector<double>& metrics_percentiles = GetPercentiles();
  std::sort(
      samples.begin(), samples.end(), [](const Sample& s1, const Sample& s2) {
        return s1.value < s2.value;
      });
  (*ss) << "  Percentiles: ";
  // 计算并输出各百分位数的样本值
  for (const auto i : c10::irange(metrics_percentiles.size())) {
    size_t index = metrics_percentiles[i] * samples.size();
    if (i > 0) {
      (*ss) << "; ";
    }
    (*ss) << (metrics_percentiles[i] * 100.0)
          << "%=" << data->Repr(samples[index].value);
  }
  (*ss) << std::endl;
}

// 发送计数器信息到输出流，包括名称和当前值
void EmitCounterInfo(
    const std::string& name,
    CounterData* data,
    std::stringstream* ss) {
  (*ss) << "Counter: " << name << std::endl;
  (*ss) << "  Value: " << data->Value() << std::endl;
}

// 将键值对插入到映射容器中，如果键已存在则不进行操作
template <typename T, typename G>
const typename T::mapped_type& MapInsert(
    T* cont,
    const typename T::key_type& key,
    const G& gen) {
  auto it = cont->find(key);  # 在容器 cont 中查找键为 key 的元素
  if (it == cont->end()) {    # 如果未找到匹配的元素
    it = cont->emplace(key, gen()).first;  # 使用键 key 和 gen() 生成的值插入到 cont 中，并返回插入的位置
  }
  return it->second;  # 返回键为 key 的元素的值引用
} // namespace

// 获取 MetricsArena 的单例实例
MetricsArena* MetricsArena::Get() {
  // 使用静态局部变量确保只创建一个 MetricsArena 实例
  static MetricsArena* arena = new MetricsArena();
  return arena;
}

// 重置所有计数器的计数值
void MetricsArena::ResetCounters() {
  // 遍历所有计数器，并调用其 Reset 方法进行重置
  for (auto& pair : counters_) {
    if (pair.second) {
      pair.second->Reset();
    }
  }
}

// 重置所有度量指标的数据
void MetricsArena::ResetMetrics() {
  // 遍历所有度量指标，并调用其 Reset 方法进行重置
  for (auto& pair : metrics_) {
    if (pair.second) {
      pair.second->Reset();
    }
  }
}

// 注册一个度量指标
void MetricsArena::RegisterMetric(
    const std::string& name,
    MetricReprFn repr_fn,
    size_t max_samples,
    std::shared_ptr<MetricData>* data) {
  std::lock_guard<std::mutex> lock(lock_);
  // 如果数据为空指针，则创建新的 MetricData 实例并插入到 metrics_ 中
  if (*data == nullptr) {
    *data = MapInsert(&metrics_, name, [&]() {
      return std::make_shared<MetricData>(std::move(repr_fn), max_samples);
    });
  }
}

// 注册一个计数器
void MetricsArena::RegisterCounter(
    const std::string& name,
    std::shared_ptr<CounterData>* data) {
  std::lock_guard<std::mutex> lock(lock_);
  // 如果数据为空指针，则创建新的 CounterData 实例并插入到 counters_ 中
  if (*data == nullptr) {
    *data = MapInsert(
        &counters_, name, []() { return std::make_shared<CounterData>(); });
  }
}

// 对每个度量指标执行给定函数
void MetricsArena::ForEachMetric(
    const std::function<void(const std::string&, MetricData*)>& metric_func) {
  std::lock_guard<std::mutex> lock(lock_);
  // 遍历每个度量指标，调用 metric_func 处理有效的度量指标
  for (auto& name_data : metrics_) {
    if (!name_data.second->IsValid()) {
      continue;
    }
    metric_func(name_data.first, name_data.second.get());
  }
}

// 对每个计数器执行给定函数
void MetricsArena::ForEachCounter(
    const std::function<void(const std::string&, CounterData*)>& counter_func) {
  std::lock_guard<std::mutex> lock(lock_);
  // 遍历每个计数器，调用 counter_func 处理有效的计数器
  for (auto& name_data : counters_) {
    if (!name_data.second->IsValid())
      continue;
    counter_func(name_data.first, name_data.second.get());
  }
}

// 获取所有度量指标的名称列表
std::vector<std::string> MetricsArena::GetMetricNames() {
  std::vector<std::string> names;
  // 调用 ForEachMetric 将每个度量指标的名称添加到 names 中
  ForEachMetric([&names](const std::string& name, MetricData* data) {
    names.push_back(name);
  });
  return names;
}

// 根据名称获取度量指标对象的指针
MetricData* MetricsArena::GetMetric(const std::string& name) {
  std::lock_guard<std::mutex> lock(lock_);
  // 查找指定名称的度量指标，如果找到且有效则返回其指针，否则返回空指针
  auto it = metrics_.find(name);
  if (it == metrics_.end()) {
    return nullptr;
  }
  return it->second->IsValid() ? it->second.get() : nullptr;
}

// 获取所有计数器的名称列表
std::vector<std::string> MetricsArena::GetCounterNames() {
  std::vector<std::string> names;
  // 调用 ForEachCounter 将每个计数器的名称添加到 names 中
  ForEachCounter([&names](const std::string& name, CounterData* data) {
    names.push_back(name);
  });
  return names;
}

// 根据名称获取计数器对象的指针
CounterData* MetricsArena::GetCounter(const std::string& name) {
  std::lock_guard<std::mutex> lock(lock_);
  // 查找指定名称的计数器，如果找到且有效则返回其指针，否则返回空指针
  auto it = counters_.find(name);
  if (it == counters_.end()) {
    return nullptr;
  }
  return it->second->IsValid() ? it->second.get() : nullptr;
}

// MetricData 的构造函数，初始化度量指标对象
MetricData::MetricData(MetricReprFn repr_fn, size_t max_samples)
    : repr_fn_(std::move(repr_fn)), samples_(max_samples) {}

// 向度量指标对象添加样本数据
void MetricData::AddSample(int64_t timestamp_ns, double value) {
  std::lock_guard<std::mutex> lock(lock_);
  // 计算样本数据的位置并添加到 samples_ 中，同时更新计数器和累加器
  size_t position = count_ % samples_.size();
  ++count_;
  accumulator_ += value;
  samples_[position] = Sample(timestamp_ns, value);
}
// 返回累加器的值，使用互斥锁确保线程安全
double MetricData::Accumulator() const {
  std::lock_guard<std::mutex> lock(lock_);
  return accumulator_;
}

// 返回总样本数，使用互斥锁确保线程安全
size_t MetricData::TotalSamples() const {
  std::lock_guard<std::mutex> lock(lock_);
  return count_;
}

// 返回样本数据，使用互斥锁确保线程安全
std::vector<Sample> MetricData::Samples(
    double* accumulator,
    size_t* total_samples) const {
  std::lock_guard<std::mutex> lock(lock_);
  std::vector<Sample> samples;
  if (count_ <= samples_.size()) {
    samples.insert(samples.end(), samples_.begin(), samples_.begin() + count_);
  } else {
    size_t position = count_ % samples_.size();
    samples.insert(samples.end(), samples_.begin() + position, samples_.end());
    samples.insert(
        samples.end(), samples_.begin(), samples_.begin() + position);
  }
  if (accumulator != nullptr) {
    *accumulator = accumulator_;
  }
  if (total_samples != nullptr) {
    *total_samples = count_;
  }
  return samples;
}

// 重置数据，使用互斥锁确保线程安全，将计数器归零，保留样本容器
void MetricData::Reset() {
  std::lock_guard<std::mutex> lock(lock_);
  count_ = 0;
  // 不清空样本容器，样本容器初始化时已包含占位符
  samples_ = std::vector<Sample>(samples_.size());
  accumulator_ = 0.0;
}

// Metric 类的构造函数，初始化名称、表示函数和最大样本数
Metric::Metric(std::string name, MetricReprFn repr_fn, size_t max_samples)
    : name_(std::move(name)),
      repr_fn_(std::move(repr_fn)),
      max_samples_(
          max_samples != 0 ? max_samples : FLAGS_torch_lazy_metrics_samples),
      data_(nullptr) {}

// 返回累加器的值，通过调用 MetricData 的 Accumulator 函数实现
double Metric::Accumulator() const {
  return GetData()->Accumulator();
}

// 添加一个时间戳和值的样本，通过调用 MetricData 的 AddSample 函数实现
void Metric::AddSample(int64_t timestamp_ns, double value) {
  GetData()->AddSample(timestamp_ns, value);
}

// 添加一个值的样本，时间戳使用当前时间，通过调用 MetricData 的 AddSample 函数实现
void Metric::AddSample(double value) {
  GetData()->AddSample(NowNs(), value);
}

// 返回样本数据，通过调用 MetricData 的 Samples 函数实现
std::vector<Sample> Metric::Samples(double* accumulator, size_t* total_samples)
    const {
  return GetData()->Samples(accumulator, total_samples);
}

// 返回值的字符串表示，通过调用 MetricData 的 Repr 函数实现
std::string Metric::Repr(double value) const {
  return GetData()->Repr(value);
}

// 获取 MetricData 对象的指针，确保在多线程环境下只创建一个 MetricData 对象
MetricData* Metric::GetData() const {
  MetricData* data = data_.load();
  if (C10_UNLIKELY(data == nullptr)) {
    // RegisterMetric() API 是同步点，确保只创建一个 MetricData 对象
    MetricsArena* arena = MetricsArena::Get();
    arena->RegisterMetric(name_, repr_fn_, max_samples_, &data_ptr_);
    // 多个线程即使进入此 IF 语句，也会获取并存储相同的值
    data = data_ptr_.get();
    data_.store(data);
  }
  return data;
}

// Counter 类的构造函数，初始化名称
Counter::Counter(std::string name) : name_(std::move(name)), data_(nullptr) {}

// 获取 CounterData 对象的指针，确保在多线程环境下只创建一个 CounterData 对象
CounterData* Counter::GetData() const {
  CounterData* data = data_.load();
  if (C10_UNLIKELY(data == nullptr)) {
    // RegisterCounter() API 是同步点，确保只创建一个 CounterData 对象
    MetricsArena* arena = MetricsArena::Get();
    arena->RegisterCounter(name_, &data_ptr_);
    // 多个线程即使进入此 IF 语句，也会获取并存储相同的值
    data = data_ptr_.get();
    data_.store(data);
  }
  return data;
}
    // 获取 data_ptr_ 指向的数据，并将其存储在变量 data 中
    data = data_ptr_.get();
    // 使用 data 存储的数值更新 data_ 中的内容
    data_.store(data);
  }
  // 返回最终存储在 data 中的数值
  return data;
}

// 定义一个函数，用于将给定的浮点数值转换为字符串，并保留两位小数
std::string MetricFnValue(double value) {
  std::stringstream ss;
  ss.precision(2);  // 设置输出精度为两位小数
  ss << std::fixed << value;  // 将浮点数按固定点格式输出到stringstream中
  return ss.str();  // 返回stringstream中的字符串表示
}

// 定义一个函数，将给定的浮点数值转换为字节单位的字符串表示
std::string MetricFnBytes(double value) {
  static const std::array<const char*, 6> kSizeSuffixes{
      "B", "KB", "MB", "GB", "TB", "PB"};  // 定义字节单位后缀数组
  unsigned sfix = 0;
  for (; (sfix + 1) < kSizeSuffixes.size() && value >= 1024.0; ++sfix) {
    value /= 1024.0;  // 循环计算适合的字节单位，并将值转换为对应单位下的数值
  }
  std::stringstream ss;
  ss.precision(2);  // 设置输出精度为两位小数
  ss << std::fixed << value << kSizeSuffixes[sfix];  // 将转换后的数值和单位后缀连接为字符串
  return ss.str();  // 返回stringstream中的字符串表示
}

// 定义一个函数，将给定的浮点数值转换为时间单位的字符串表示
std::string MetricFnTime(double value) {
  struct TimePart {
    const char* suffix;
    double scaler;
    int width;
    int precision;
    char fill;
  };
  static const std::array<TimePart, 6> time_parts{
      {{"d", 86400.0 * 1e9, 2, 0, '0'},  // 定义不同时间单位的相关信息
       {"h", 3600.0 * 1e9, 2, 0, '0'},
       {"m", 60.0 * 1e9, 2, 0, '0'},
       {"s", 1e9, 2, 0, '0'},
       {"ms", 1e6, 3, 0, '0'},
       {"us", 1e3, 7, 3, '0'}}};
  int count = 0;
  std::stringstream ss;
  for (const auto i : c10::irange(time_parts.size())) {
    const TimePart& part = time_parts[i];
    double ctime = value / part.scaler;
    if (ctime >= 1.0 || count > 0 || i + 1 == time_parts.size()) {
      ss.precision(part.precision);  // 设置输出精度
      ss.width(part.width);  // 设置输出宽度
      ss.fill(part.fill);  // 设置填充字符
      ss << std::fixed << ctime << part.suffix;  // 将转换后的时间和单位后缀连接为字符串
      value -= std::floor(ctime) * part.scaler;  // 更新剩余的时间值
      ++count;  // 计数加一
    }
  }
  return ss.str();  // 返回stringstream中的字符串表示
}

// 定义一个函数，生成完整的度量报告字符串，包括指标和计数器
std::string CreateMetricReport() {
  MetricsArena* arena = MetricsArena::Get();  // 获取度量竞技场对象的指针
  std::stringstream ss;  // 创建一个stringstream对象
  arena->ForEachMetric([&ss](const std::string& name, MetricData* data) {
    EmitMetricInfo(name, data, &ss);  // 对每个度量调用EmitMetricInfo函数输出信息到stringstream
  });
  arena->ForEachCounter([&ss](const std::string& name, CounterData* data) {
    EmitCounterInfo(name, data, &ss);  // 对每个计数器调用EmitCounterInfo函数输出信息到stringstream
  });

  // 添加后端度量报告到stringstream
  ss << getBackend()->CreateMetricReport();  // 调用getBackend()函数获取后端对象，生成度量报告并追加到stringstream
  return ss.str();  // 返回stringstream中的字符串表示
}

// 定义一个函数，生成指定计数器和度量的度量报告字符串
std::string CreateMetricReport(
    const std::vector<std::string>& counter_names,
    const std::vector<std::string>& metric_names) {
  MetricsArena* arena = MetricsArena::Get();  // 获取度量竞技场对象的指针
  std::stringstream ss;  // 创建一个stringstream对象
  std::set<std::string> metric_name_set(
      metric_names.begin(), metric_names.end());  // 将metric_names转换为集合
  arena->ForEachMetric(
      [&ss, &metric_name_set](const std::string& name, MetricData* data) {
        if (metric_name_set.find(name) != metric_name_set.end()) {
          EmitMetricInfo(name, data, &ss);  // 如果度量名称在集合中，则调用EmitMetricInfo函数输出信息到stringstream
        }
      });
  std::set<std::string> counter_name_set(
      counter_names.begin(), counter_names.end());  // 将counter_names转换为集合
  arena->ForEachCounter(
      [&ss, &counter_name_set](const std::string& name, CounterData* data) {
        if (counter_name_set.find(name) != counter_name_set.end()) {
          EmitCounterInfo(name, data, &ss);  // 如果计数器名称在集合中，则调用EmitCounterInfo函数输出信息到stringstream
        }
      });

  static std::string fall_back_counter_prefix = "aten::";  // 定义后备计数器名称的前缀
  arena->ForEachCounter([&ss](const std::string& name, CounterData* data) {
    // 对每个计数器调用EmitCounterInfo函数输出信息到stringstream
    // 这里忽略名称前缀为"aten::"的计数器
    ```
    if (name.substr(0, fall_back_counter_prefix.size()) != fall_back_counter_prefix)
    // 检查字符串 name 是否以 fall_back_counter_prefix 开头的位置是否为 0
    if (name.rfind(fall_back_counter_prefix, 0) == 0) {
      // 如果条件成立，说明 name 是以 fall_back_counter_prefix 开头的字符串
      // 这种情况下，可能会在 counter_names 中出现重复的计数器，但这种情况应该非常罕见。
      
      // 调用 EmitCounterInfo 函数，向输出流 ss 中发射（输出）计数器信息，传递 name、data 和 ss 的地址
      EmitCounterInfo(name, data, &ss);
    }
  });
  // 返回输出流 ss 的字符串表示形式
  return ss.str();
}

// 结束命名空间 torch
namespace torch {

// 结束命名空间 lazy
namespace lazy {

std::vector<std::string> GetMetricNames() {
  // 调用 MetricsArena 的静态方法 Get()，获取指向 MetricsArena 的指针，再调用 GetMetricNames() 返回指标名称向量
  return MetricsArena::Get()->GetMetricNames();
}

MetricData* GetMetric(const std::string& name) {
  // 调用 MetricsArena 的静态方法 Get()，获取指向 MetricsArena 的指针，再调用 GetMetric(name) 返回指定名称的指标数据指针
  return MetricsArena::Get()->GetMetric(name);
}

std::vector<std::string> GetCounterNames() {
  // 调用 MetricsArena 的静态方法 Get()，获取指向 MetricsArena 的指针，再调用 GetCounterNames() 返回计数器名称向量
  return MetricsArena::Get()->GetCounterNames();
}

CounterData* GetCounter(const std::string& name) {
  // 调用 MetricsArena 的静态方法 Get()，获取指向 MetricsArena 的指针，再调用 GetCounter(name) 返回指定名称的计数器数据指针
  return MetricsArena::Get()->GetCounter(name);
}

int64_t NowNs() {
  // 获取当前高精度时钟的当前时间点 now
  auto now = std::chrono::high_resolution_clock::now();
  // 计算当前时间点距离纪元的纳秒数，并返回
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             now.time_since_epoch())
      .count();
}

} // namespace lazy
} // namespace torch
```