# `.\pytorch\torch\csrc\THConcat.h`

```
#pragma once
// 宏定义：将两个参数 x 和 y 连接成一个字符串
#define TH_CONCAT_STRING_2(x, y) TH_CONCAT_STRING_2_EXPAND(x, y)
// 宏定义展开：将参数 x 和 y 连接成一个字符串
#define TH_CONCAT_STRING_2_EXPAND(x, y) #x #y

// 宏定义：将三个参数 x、y、z 连接成一个字符串
#define TH_CONCAT_STRING_3(x, y, z) TH_CONCAT_STRING_3_EXPAND(x, y, z)
// 宏定义展开：将参数 x、y、z 连接成一个字符串
#define TH_CONCAT_STRING_3_EXPAND(x, y, z) #x #y #z

// 宏定义：将四个参数 x、y、z、w 连接成一个字符串
#define TH_CONCAT_STRING_4(x, y, z, w) TH_CONCAT_STRING_4_EXPAND(x, y, z, w)
// 宏定义展开：将参数 x、y、z、w 连接成一个字符串
#define TH_CONCAT_STRING_4_EXPAND(x, y, z, w) #x #y #z #w

// 宏定义：将两个参数 x 和 y 连接成一个标识符
#define TH_CONCAT_2(x, y) TH_CONCAT_2_EXPAND(x, y)
// 宏定义展开：将参数 x 和 y 连接成一个标识符
#define TH_CONCAT_2_EXPAND(x, y) x##y

// 宏定义：将三个参数 x、y、z 连接成一个标识符
#define TH_CONCAT_3(x, y, z) TH_CONCAT_3_EXPAND(x, y, z)
// 宏定义展开：将参数 x、y、z 连接成一个标识符
#define TH_CONCAT_3_EXPAND(x, y, z) x##y##z

// 宏定义展开：将四个参数 x、y、z、w 连接成一个标识符
#define TH_CONCAT_4_EXPAND(x, y, z, w) x##y##z##w
// 宏定义：将四个参数 x、y、z、w 连接成一个标识符
#define TH_CONCAT_4(x, y, z, w) TH_CONCAT_4_EXPAND(x, y, z, w)
```