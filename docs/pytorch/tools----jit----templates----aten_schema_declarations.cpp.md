# `.\pytorch\tools\jit\templates\aten_schema_declarations.cpp`

```
namespace torch { namespace jit {
const char * schema_declarations = R"===(
  ${declarations}
)===";
}}



namespace torch { namespace jit {
    // 定义一个常量 C 字符串，用于存储包含在 "${declarations}" 中的内容
    const char * schema_declarations = R"===(
      ${declarations}
    )===";
}}
```