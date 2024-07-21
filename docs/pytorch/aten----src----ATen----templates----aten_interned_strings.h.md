# `.\pytorch\aten\src\ATen\templates\aten_interned_strings.h`

```py
#pragma once

// ${generated_comment}  // 描述：这是一个占位注释，将来会被生成的注释内容替换

#if defined(TORCH_ASSERT_NO_OPERATORS) || defined(TORCH_ASSERT_ONLY_METHOD_OPERATORS)
#error This change adds a dependency on native_functions.yaml,          \
  meaning the file will need to be re-compiled every time an operator   \
  is changed or added. Consider if including <ATen/core/symbol.h> for   \
  the c10::Symbol class would be sufficient, or if your change would be \
  better placed in another file.
#endif

// ATen symbols correspond exactly to operators defined in ATen. Every
// symbol here corresponds exactly to an ATen operation defined in
// native_functions.yaml; attributes are in one-to-one correspondence
// with their ATen name.

#define FORALL_ATEN_BASE_SYMBOLS(_) \  // 描述：定义一个宏，用于遍历所有的 ATen 基础符号

${aten_symbols}  // 描述：插入 ATen 的基础符号列表

#define FORALL_ATTR_BASE_SYMBOLS(_) \  // 描述：定义一个宏，用于遍历所有的属性基础符号

${attr_symbols}  // 描述：插入属性的基础符号列表
```