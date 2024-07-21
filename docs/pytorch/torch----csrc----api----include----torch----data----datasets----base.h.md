# `.\pytorch\torch\csrc\api\include\torch\data\datasets\base.h`

```
#pragma once
/// 定义了一个模板，用于表示数据流数据集。它的模板参数包括：
/// - Self: 表示自身类型，通常用于引用类本身
/// - Batch: 表示批次数据的类型，默认为 Example<> 的向量
/// 该模板继承自 BatchDataset，并且使用 size_t 作为批次请求的类型。
template <typename Self, typename Batch = std::vector<Example<>>>
using StreamDataset = BatchDataset<Self, Batch, /*BatchRequest=*/size_t>;
} // namespace datasets
} // namespace data
} // namespace torch
```