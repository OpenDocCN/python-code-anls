# `.\pytorch\torch\csrc\jit\serialization\import_export_functions.h`

```
#pragma once
#include <ATen/core/ivalue.h>

// Functions that are used in both import and export processes

// 命名空间 torch::jit 下的 IValue 类型的别名，引用 c10::IValue
namespace torch::jit {
    using c10::IValue;

    // 从元组元素列表中获取指定名称和索引的字段值
    IValue expect_field(
        c10::ivalue::TupleElements& elements,  // 元组元素列表的引用
        const std::string& expected_name,      // 预期字段的名称
        size_t entry);                         // 元组中字段的索引

    // 返回一个字符串，描述操作符名称和重载名称的组合
    std::string operator_str(
        const std::string& name,        // 操作符的名称
        const std::string& overloadname // 操作符的重载名称
    );
} // namespace torch::jit
```