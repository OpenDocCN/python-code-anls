# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_config.h`

```py
#ifndef AGG_CONFIG_INCLUDED
#define AGG_CONFIG_INCLUDED

// 如果 AGG_CONFIG_INCLUDED 未定义，则定义 AGG_CONFIG_INCLUDED

//---------------------------------------
// 1. 默认的基本数据类型定义：
// 
// AGG_INT8
// AGG_INT8U
// AGG_INT16
// AGG_INT16U
// AGG_INT32
// AGG_INT32U
// AGG_INT64
// AGG_INT64U
//
// 如果需要重新定义这些类型，可以通过替换这个文件来实现。
// 例如，如果你的编译器不支持 64 位整数类型，
// 你可以定义如下：
//
// #define AGG_INT64  int
// #define AGG_INT64U unsigned
//
// 这样会导致在 16 位每分量的图像/模式重采样中溢出，
// 但不会导致崩溃，库的其余部分仍然可以完全正常工作。

//---------------------------------------
// 2. 默认的 rendering_buffer 类型。可以是：
//
// 提供了更快的大规模像素操作访问，比如模糊、图像滤波：
// #define AGG_RENDERING_BUFFER row_ptr_cache<int8u>
// 
// 提供了更便宜的创建和销毁（无需内存分配）：
// #define AGG_RENDERING_BUFFER row_accessor<int8u>
//
// 你可以在应用程序中同时使用这两种类型。
// 这个 #define 仅用于默认的 rendering_buffer 类型，
// 在像 pixfmt_rgba32 这样的简写 typedef 中使用。

#endif
```