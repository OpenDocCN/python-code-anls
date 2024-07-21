# `.\pytorch\torch\csrc\api\include\torch\data\datasets\map.h`

```
  /// 根据模板参数C的值，选择性地使用torch::optional包装类型T
  template <bool C, typename T>
  using optional_if_t = typename std::conditional<C, torch::optional<T>, T>::type;

/// `MapDataset`是对源数据集应用转换的数据集。
template <typename SourceDataset, typename AppliedTransform>
class MapDataset : public BatchDataset<
                       MapDataset<SourceDataset, AppliedTransform>,
                       detail::optional_if_t<
                           SourceDataset::is_stateful,
                           typename AppliedTransform::OutputBatchType>,
                       typename SourceDataset::BatchRequestType> {
 public:
  using DatasetType = SourceDataset;
  using TransformType = AppliedTransform;
  using BatchRequestType = typename SourceDataset::BatchRequestType;
  using OutputBatchType = detail::optional_if_t<
      SourceDataset::is_stateful,
      typename AppliedTransform::OutputBatchType>;

  /// 构造函数，初始化MapDataset对象
  MapDataset(DatasetType dataset, TransformType transform)
      : dataset_(std::move(dataset)), transform_(std::move(transform)) {}

  /// 从源数据集获取一个批次数据，并对其应用转换，返回结果。
  OutputBatchType get_batch(BatchRequestType indices) override {
    return get_batch_impl(std::move(indices));
  }

  /// 返回源数据集的大小。
  // NOLINTNEXTLINE(bugprone-exception-escape)
  optional<size_t> size() const noexcept override {
    return dataset_.size();
  }

  /// 调用底层数据集的`reset()`方法。
  /// 注意：无状态数据集没有`reset()`方法，因此仅适用于有状态数据集（具有`reset()`方法）。
  void reset() {
    dataset_.reset();
  }

  /// 返回底层数据集。
  const SourceDataset& dataset() noexcept {
    return dataset_;
  }

  /// 返回正在应用的转换。
  const AppliedTransform& transform() noexcept {
    return transform_;
  }

 private:
  /// 无状态情况下`get_batch()`的实现，简单地将转换应用于数据集的`get_batch()`输出。
  template <
      typename D = SourceDataset,
      typename = std::enable_if_t<!D::is_stateful>>
  OutputBatchType get_batch_impl(BatchRequestType indices) {
  // 调用 `transform_.apply_batch()` 来对数据集的批次进行转换，并返回转换后的结果
  return transform_.apply_batch(dataset_.get_batch(std::move(indices)));
}

/// `get_batch()` 的有状态情况下的实现。在这里，我们遵循许多函数式语言中 `Optional.map()` 的语义，
/// 当可选值包含一个值时，应用转换到可选值的内容，并返回一个新的可选值（可能是不同类型的），
/// 如果 `get_batch()` 返回的原始可选值为空。
template <typename D = SourceDataset>
std::enable_if_t<D::is_stateful, OutputBatchType> get_batch_impl(
    BatchRequestType indices) {
  // 调用 `dataset_.get_batch()` 获取数据集的批次
  if (auto batch = dataset_.get_batch(std::move(indices))) {
    // 如果批次不为空，应用 `transform_.apply_batch()` 转换批次并返回
    return transform_.apply_batch(std::move(*batch));
  }
  // 如果批次为空，返回一个空的 optional 对象
  return nullopt;
}

/// 被转换的基础数据集。
SourceDataset dataset_;

// 用于对从数据集接收到的批次应用的转换。
AppliedTransform transform_;
};

/// 结束了 `map` 函数的实现，该函数用于创建一个 `MapDataset` 对象，并接收一个数据集和一个变换函数作为参数。
template <typename DatasetType, typename TransformType>
MapDataset<DatasetType, TransformType> map(
    DatasetType dataset,
    TransformType transform) {
  // 静态断言，用于检查数据集的批处理类型是否与变换函数的输入批处理类型匹配
  static_assert(
      std::is_same<
          typename std::conditional<
              DatasetType::is_stateful,
              typename DatasetType::BatchType::value_type,
              typename DatasetType::BatchType>::type,
          typename TransformType::InputBatchType>::value,
      "BatchType type of dataset does not match input type of transform");
  // 返回一个新的 MapDataset 对象，使用给定的数据集和变换函数
  return {std::move(dataset), std::move(transform)};
}

} // namespace datasets
} // namespace data
} // namespace torch
```