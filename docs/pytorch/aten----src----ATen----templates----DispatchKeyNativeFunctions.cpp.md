# `.\pytorch\aten\src\ATen\templates\DispatchKeyNativeFunctions.cpp`

```
// 在未命名的命名空间中定义辅助函数，这些函数在当前文件中可见但对外部不可见
namespace {
    // 插入生成的注释
    ${helper_fns}
} // namespace

// 插入命名空间的起始部分
${namespace_prologue}

// 插入原生函数的定义，这些函数是通过原生代码实现的，并在当前命名空间中可见
${native_function_definitions}

// 插入命名空间的结束部分
${namespace_epilogue}
```