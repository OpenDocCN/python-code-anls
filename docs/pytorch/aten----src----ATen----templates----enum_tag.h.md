# `.\pytorch\aten\src\ATen\templates\enum_tag.h`

```py
#pragma once
// 声明这个文件只被编译一次，以防止头文件重复包含

// ${generated_comment}
// 自动生成的注释，可能包含有关此文件生成方式或相关信息的占位符

namespace at {
    // 命名空间 'at' 开始

    // Enum of valid tags obtained from the entries in tags.yaml
    // 从 tags.yaml 文件中获取的有效标签枚举
    enum class Tag {
        ${enum_of_valid_tags}
        // 枚举类定义，表示可能的有效标签
    };
    // 枚举类 'Tag' 结束
}
// 命名空间 'at' 结束
```