# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\static_switch.h`

```py
// 宏定义：根据条件(COND)进行静态分支切换，使用给定的常量名(CONST_NAME)
#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      // 如果条件为真，设定 CONST_NAME 为 true 的 constexpr 常量
      constexpr static bool CONST_NAME = true;  \
      // 执行传入的代码块并返回结果
      return __VA_ARGS__();                     \
    } else {                                    \
      // 如果条件为假，设定 CONST_NAME 为 false 的 constexpr 常量
      constexpr static bool CONST_NAME = false; \
      // 执行传入的代码块并返回结果
      return __VA_ARGS__();                     \
    }                                           \
  }()

// 如果 FLASHATTENTION_DISABLE_DROPOUT 宏被定义，则禁用 DROPOUT_SWITCH 宏，否则使用 BOOL_SWITCH 宏
#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    // 定义 CONST_NAME 为 false 的 constexpr 常量，因为禁用了 dropout
    constexpr static bool CONST_NAME = false;   \
    // 执行传入的代码块并返回结果
    return __VA_ARGS__();                       \
  }()
#else
  // 如果未定义 FLASHATTENTION_DISABLE_DROPOUT 宏，则使用 BOOL_SWITCH 宏进行条件分支
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

// 如果 FLASHATTENTION_DISABLE_ALIBI 宏被定义，则禁用 ALIBI_SWITCH 宏，否则使用 BOOL_SWITCH 宏
#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    // 定义 CONST_NAME 为 false 的 constexpr 常量，因为禁用了 ALIBI
    constexpr static bool CONST_NAME = false;   \
    // 执行传入的代码块并返回结果
    return __VA_ARGS__();                       \
  }()
#else
  // 如果未定义 FLASHATTENTION_DISABLE_ALIBI 宏，则使用 BOOL_SWITCH 宏进行条件分支
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

// 如果 FLASHATTENTION_DISABLE_UNEVEN_K 宏被定义，则启用 EVENK_SWITCH 宏，否则使用 BOOL_SWITCH 宏
#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    // 定义 CONST_NAME 为 true 的 constexpr 常量，因为启用了 EVENK
    constexpr static bool CONST_NAME = true;    \
    // 执行传入的代码块并返回结果
    return __VA_ARGS__();                       \
  }()
#else
  // 如果未定义 FLASHATTENTION_DISABLE_UNEVEN_K 宏，则使用 BOOL_SWITCH 宏进行条件分支
  #define EVENK_SWITCH BOOL_SWITCH
#endif

// 如果 FLASHATTENTION_DISABLE_LOCAL 宏被定义，则禁用 LOCAL_SWITCH 宏，否则使用 BOOL_SWITCH 宏
#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    // 定义 CONST_NAME 为 false 的 constexpr 常量，因为禁用了 LOCAL
    constexpr static bool CONST_NAME = false;    \
    // 执行传入的代码块并返回结果
    return __VA_ARGS__();                       \
  }()
#else
  // 如果未定义 FLASHATTENTION_DISABLE_LOCAL 宏，则使用 BOOL_SWITCH 宏进行条件分支
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

// 宏定义：根据条件(COND)进行静态分支切换，使用给定的条件执行不同的代码
#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      // 如果条件为真，使用 cutlass::half_t 类型
      using elem_type = cutlass::half_t;     \
      // 执行传入的代码块并返回结果
      return __VA_ARGS__();                  \
    } else {                                 \
      // 如果条件为假，使用 cutlass::bfloat16_t 类型
      using elem_type = cutlass::bfloat16_t; \
      // 执行传入的代码块并返回结果
      return __VA_ARGS__();                  \
    }                                        \
  }()

// 宏定义：根据头维度(HEADDIM)的大小进行静态分支切换，选择不同的头维度并执行代码
#define HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM <= 32) {                   \
      // 如果头维度小于等于32，设定 kHeadDim 为32
      constexpr static int kHeadDim = 32;  \
      // 执行传入的代码块并返回结果
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 64) {            \
      // 如果头维度小于等于64，设定 kHeadDim 为64
      constexpr static int kHeadDim = 64;  \
      // 执行传入的代码块并返回结果
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 96) {            \  # 如果 HEADDIM 小于等于 96
      constexpr static int kHeadDim = 96;  \  # 设置 kHeadDim 常量为 96
      return __VA_ARGS__();                \  # 返回使用传入的可变参数的结果
    } else if (HEADDIM <= 128) {           \  # 如果 HEADDIM 小于等于 128
      constexpr static int kHeadDim = 128; \  # 设置 kHeadDim 常量为 128
      return __VA_ARGS__();                \  # 返回使用传入的可变参数的结果
    } else if (HEADDIM <= 160) {           \  # 如果 HEADDIM 小于等于 160
      constexpr static int kHeadDim = 160; \  # 设置 kHeadDim 常量为 160
      return __VA_ARGS__();                \  # 返回使用传入的可变参数的结果
    } else if (HEADDIM <= 192) {           \  # 如果 HEADDIM 小于等于 192
      constexpr static int kHeadDim = 192; \  # 设置 kHeadDim 常量为 192
      return __VA_ARGS__();                \  # 返回使用传入的可变参数的结果
    } else if (HEADDIM <= 224) {           \  # 如果 HEADDIM 小于等于 224
      constexpr static int kHeadDim = 224; \  # 设置 kHeadDim 常量为 224
      return __VA_ARGS__();                \  # 返回使用传入的可变参数的结果
    } else if (HEADDIM <= 256) {           \  # 如果 HEADDIM 小于等于 256
      constexpr static int kHeadDim = 256; \  # 设置 kHeadDim 常量为 256
      return __VA_ARGS__();                \  # 返回使用传入的可变参数的结果
    }                                      \  # 结束 if-else 块
  }()
```