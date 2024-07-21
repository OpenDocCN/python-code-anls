# `.\pytorch\aten\src\ATen\native\transformers\cuda\flash_attn\dropout.h`

```
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <ATen/native/transformers/cuda/flash_attn/philox.cuh>
#include <ATen/native/transformers/cuda/flash_attn/utils.h>

namespace pytorch_flash {

using namespace cute;

// 定义一个结构体 Dropout，用于实现 dropout 功能
struct Dropout {

    // 成员变量，种子值和偏移量
    const unsigned long long seed, offset;
    // dropout 概率，以 uint8_t 类型存储
    const uint8_t p_dropout_in_uint8_t;

    // 构造函数，初始化成员变量
    __forceinline__ __device__ Dropout(const unsigned long long seed, const unsigned long long offset,
                              const uint8_t p_dropout_in_uint8_t,
                              const int bid, const int hid, const int tid, const int nheads)
            : seed(seed)
            , offset(offset + (bid * nheads + hid) * 32 + tid % 32)
            , p_dropout_in_uint8_t(p_dropout_in_uint8_t) {
    }

    // 模板函数声明，用于编码 dropout 到符号位（sign bit），使用特定的引擎和布局
    template <bool encode_dropout_in_sign_bit=false, typename Engine, typename Layout>
    }

};

} // namespace pytorch_flash


这段代码定义了一个名为 `Dropout` 的结构体，其中包括构造函数和一个模板函数声明，用于实现特定的 dropout 功能。
```