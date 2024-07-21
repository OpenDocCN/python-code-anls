# `.\pytorch\torch\_numpy\_casting_dicts.py`

```
# mypy: ignore-errors

import torch

# 自动化脚本 autogen/gen_dtypes.py 自动生成以下两个字典，使用的是 numpy 版本 1.24.3。

# 定义了一个能够执行类型转换的字典 _can_cast_dict
_can_cast_dict = {
    # 空字典，暂无内容
}

# 定义了一个描述不同数据类型间结果类型的字典 _result_type_dict
_result_type_dict = {
    torch.float16: {
        torch.float16: torch.float16,
        torch.float32: torch.float32,
        torch.float64: torch.float64,
        torch.complex64: torch.complex64,
        torch.complex128: torch.complex128,
        torch.uint8: torch.float16,
        torch.uint16: torch.float32,
        torch.uint32: torch.float64,
        torch.uint64: torch.float64,
        torch.int8: torch.float16,
        torch.int16: torch.float32,
        torch.int32: torch.float64,
        torch.int64: torch.float64,
        torch.bool: torch.float16,
    },
    torch.float32: {
        torch.float16: torch.float32,
        torch.float32: torch.float32,
        torch.float64: torch.float64,
        torch.complex64: torch.complex64,
        torch.complex128: torch.complex128,
        torch.uint8: torch.float32,
        torch.uint16: torch.float32,
        torch.uint32: torch.float64,
        torch.uint64: torch.float64,
        torch.int8: torch.float32,
        torch.int16: torch.float32,
        torch.int32: torch.float64,
        torch.int64: torch.float64,
        torch.bool: torch.float32,
    },
    torch.float64: {
        torch.float16: torch.float64,
        torch.float32: torch.float64,
        torch.float64: torch.float64,
        torch.complex64: torch.complex128,
        torch.complex128: torch.complex128,
        torch.uint8: torch.float64,
        torch.uint16: torch.float64,
        torch.uint32: torch.float64,
        torch.uint64: torch.float64,
        torch.int8: torch.float64,
        torch.int16: torch.float64,
        torch.int32: torch.float64,
        torch.int64: torch.float64,
        torch.bool: torch.float64,
    },
    torch.complex64: {
        torch.float16: torch.complex64,
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
        torch.complex64: torch.complex64,
        torch.complex128: torch.complex128,
        torch.uint8: torch.complex64,
        torch.uint16: torch.complex64,
        torch.uint32: torch.complex128,
        torch.uint64: torch.complex128,
        torch.int8: torch.complex64,
        torch.int16: torch.complex64,
        torch.int32: torch.complex128,
        torch.int64: torch.complex128,
        torch.bool: torch.complex64,
    },
}
    {
        torch.complex128: {
            torch.float16: torch.complex128,   # 映射 torch.float16 到 torch.complex128
            torch.float32: torch.complex128,   # 映射 torch.float32 到 torch.complex128
            torch.float64: torch.complex128,   # 映射 torch.float64 到 torch.complex128
            torch.complex64: torch.complex128,  # 映射 torch.complex64 到 torch.complex128
            torch.complex128: torch.complex128,  # 映射 torch.complex128 到 torch.complex128
            torch.uint8: torch.complex128,     # 映射 torch.uint8 到 torch.complex128
            torch.uint16: torch.complex128,    # 映射 torch.uint16 到 torch.complex128
            torch.uint32: torch.complex128,    # 映射 torch.uint32 到 torch.complex128
            torch.uint64: torch.complex128,    # 映射 torch.uint64 到 torch.complex128
            torch.int8: torch.complex128,      # 映射 torch.int8 到 torch.complex128
            torch.int16: torch.complex128,     # 映射 torch.int16 到 torch.complex128
            torch.int32: torch.complex128,     # 映射 torch.int32 到 torch.complex128
            torch.int64: torch.complex128,     # 映射 torch.int64 到 torch.complex128
            torch.bool: torch.complex128,      # 映射 torch.bool 到 torch.complex128
        },
        torch.uint8: {
            torch.float16: torch.float16,      # 映射 torch.float16 到 torch.float16
            torch.float32: torch.float32,      # 映射 torch.float32 到 torch.float32
            torch.float64: torch.float64,      # 映射 torch.float64 到 torch.float64
            torch.complex64: torch.complex64,  # 映射 torch.complex64 到 torch.complex64
            torch.complex128: torch.complex128,  # 映射 torch.complex128 到 torch.complex128
            torch.uint8: torch.uint8,          # 映射 torch.uint8 到 torch.uint8
            torch.uint16: torch.uint16,        # 映射 torch.uint16 到 torch.uint16
            torch.uint32: torch.uint32,        # 映射 torch.uint32 到 torch.uint32
            torch.uint64: torch.uint64,        # 映射 torch.uint64 到 torch.uint64
            torch.int8: torch.int16,           # 映射 torch.int8 到 torch.int16
            torch.int16: torch.int16,          # 映射 torch.int16 到 torch.int16
            torch.int32: torch.int32,          # 映射 torch.int32 到 torch.int32
            torch.int64: torch.int64,          # 映射 torch.int64 到 torch.int64
            torch.bool: torch.uint8,           # 映射 torch.bool 到 torch.uint8
        },
        torch.uint16: {
            torch.float16: torch.float32,      # 映射 torch.float16 到 torch.float32
            torch.float32: torch.float32,      # 映射 torch.float32 到 torch.float32
            torch.float64: torch.float64,      # 映射 torch.float64 到 torch.float64
            torch.complex64: torch.complex64,  # 映射 torch.complex64 到 torch.complex64
            torch.complex128: torch.complex128,  # 映射 torch.complex128 到 torch.complex128
            torch.uint8: torch.uint16,         # 映射 torch.uint8 到 torch.uint16
            torch.uint16: torch.uint16,        # 映射 torch.uint16 到 torch.uint16
            torch.uint32: torch.uint32,        # 映射 torch.uint32 到 torch.uint32
            torch.uint64: torch.uint64,        # 映射 torch.uint64 到 torch.uint64
            torch.int8: torch.int32,           # 映射 torch.int8 到 torch.int32
            torch.int16: torch.int32,          # 映射 torch.int16 到 torch.int32
            torch.int32: torch.int32,          # 映射 torch.int32 到 torch.int32
            torch.int64: torch.int64,          # 映射 torch.int64 到 torch.int64
            torch.bool: torch.uint16,          # 映射 torch.bool 到 torch.uint16
        },
        torch.uint32: {
            torch.float16: torch.float64,      # 映射 torch.float16 到 torch.float64
            torch.float32: torch.float64,      # 映射 torch.float32 到 torch.float64
            torch.float64: torch.float64,      # 映射 torch.float64 到 torch.float64
            torch.complex64: torch.complex128,  # 映射 torch.complex64 到 torch.complex128
            torch.complex128: torch.complex128,  # 映射 torch.complex128 到 torch.complex128
            torch.uint8: torch.uint32,         # 映射 torch.uint8 到 torch.uint32
            torch.uint16: torch.uint32,        # 映射 torch.uint16 到 torch.uint32
            torch.uint32: torch.uint32,        # 映射 torch.uint32 到 torch.uint32
            torch.uint64: torch.uint64,        # 映射 torch.uint64 到 torch.uint64
            torch.int8: torch.int64,           # 映射 torch.int8 到 torch.int64
            torch.int16: torch.int64,          # 映射 torch.int16 到 torch.int64
            torch.int32: torch.int64,          # 映射 torch.int32 到 torch.int64
            torch.int64: torch.int64,          # 映射 torch.int64 到 torch.int64
            torch.bool: torch.uint32,          # 映射 torch.bool 到 torch.uint32
        },
        torch.uint64: {
            torch.float16: torch.float64,      # 映射 torch.float16 到 torch.float64
            torch.float32: torch.float64,      # 映射 torch.float32 到 torch.float64
            torch.float64: torch.float64,      # 映射 torch.float64 到 torch.float64
            torch.complex64: torch.complex128,  # 映射 torch.complex64 到 torch.complex128
            torch.complex128: torch.complex128,  # 映射 torch.complex128 到 torch.complex128
            torch.uint8: torch.uint64,         # 映射 torch.uint8 到 torch.uint64
            torch.uint16: torch.uint64,        # 映射 torch.uint16 到 torch.uint64
            torch.uint32: torch.uint64,        # 映射 torch.uint32 到 torch.uint64
            torch.uint64: torch.uint64,        # 映射 torch.uint64 到 torch.uint64
            torch.int8: torch.float64,         # 映射 torch.int8 到 torch.float64
            torch.int16: torch.float64,        # 映射 torch.int16 到 torch.float64
            torch.int32: torch.float64,        # 映射 torch.int32 到 torch.float64
            torch.int64: torch.float64,        # 映射 torch.int64 到 torch.float64
            torch.bool: torch.uint64,          # 映射 torch.bool 到 torch.uint64
        },
    }
    {
        torch.int8: {
            torch.float16: torch.float16,
            torch.float32: torch.float32,
            torch.float64: torch.float64,
            torch.complex64: torch.complex64,
            torch.complex128: torch.complex128,
            torch.uint8: torch.int16,
            torch.uint16: torch.int32,
            torch.uint32: torch.int64,
            torch.uint64: torch.float64,
            torch.int8: torch.int8,
            torch.int16: torch.int16,
            torch.int32: torch.int32,
            torch.int64: torch.int64,
            torch.bool: torch.int8,
        },
        torch.int16: {
            torch.float16: torch.float32,
            torch.float32: torch.float32,
            torch.float64: torch.float64,
            torch.complex64: torch.complex64,
            torch.complex128: torch.complex128,
            torch.uint8: torch.int16,
            torch.uint16: torch.int32,
            torch.uint32: torch.int64,
            torch.uint64: torch.float64,
            torch.int8: torch.int16,
            torch.int16: torch.int16,
            torch.int32: torch.int32,
            torch.int64: torch.int64,
            torch.bool: torch.int16,
        },
        torch.int32: {
            torch.float16: torch.float64,
            torch.float32: torch.float64,
            torch.float64: torch.float64,
            torch.complex64: torch.complex128,
            torch.complex128: torch.complex128,
            torch.uint8: torch.int32,
            torch.uint16: torch.int32,
            torch.uint32: torch.int64,
            torch.uint64: torch.float64,
            torch.int8: torch.int32,
            torch.int16: torch.int32,
            torch.int32: torch.int32,
            torch.int64: torch.int64,
            torch.bool: torch.int32,
        },
        torch.int64: {
            torch.float16: torch.float64,
            torch.float32: torch.float64,
            torch.float64: torch.float64,
            torch.complex64: torch.complex128,
            torch.complex128: torch.complex128,
            torch.uint8: torch.int64,
            torch.uint16: torch.int64,
            torch.uint32: torch.int64,
            torch.uint64: torch.float64,
            torch.int8: torch.int64,
            torch.int16: torch.int64,
            torch.int32: torch.int64,
            torch.int64: torch.int64,
            torch.bool: torch.int64,
        },
        torch.bool: {
            torch.float16: torch.float16,
            torch.float32: torch.float32,
            torch.float64: torch.float64,
            torch.complex64: torch.complex64,
            torch.complex128: torch.complex128,
            torch.uint8: torch.uint8,
            torch.uint16: torch.uint16,
            torch.uint32: torch.uint32,
            torch.uint64: torch.uint64,
            torch.int8: torch.int8,
            torch.int16: torch.int16,
            torch.int32: torch.int32,
            torch.int64: torch.int64,
            torch.bool: torch.bool,
        },
    }
    
    
    注释：
    
    
    {
        torch.int8: {
            # 定义 torch.int8 到其它数据类型的映射关系
            torch.float16: torch.float16,
            torch.float32: torch.float32,
            torch.float64: torch.float64,
            torch.complex64: torch.complex64,
            torch.complex128: torch.complex128,
            torch.uint8: torch.int16,
            torch.uint16: torch.int32,
            torch.uint32: torch.int64,
            torch.uint64: torch.float64,
            torch.int8: torch.int8,  # torch.int8 映射到 torch.int8
            torch.int16: torch.int16,
            torch.int32: torch.int32,
            torch.int64: torch.int64,
            torch.bool: torch.int8,  # torch.bool 映射到 torch.int8
        },
        torch.int16: {
            # 定义 torch.int16 到其它数据类型的映射关系
            torch.float16: torch.float32,
            torch.float32: torch.float32,
            torch.float64: torch.float64,
            torch.complex64: torch.complex64,
            torch.complex128: torch.complex128,
            torch.uint8: torch.int16,  # torch.uint8 映射到 torch.int16
            torch.uint16: torch.int32,
            torch.uint32: torch.int64,
            torch.uint64: torch.float64,
            torch.int8: torch.int16,
            torch.int16: torch.int16,  # torch.int16 映射到 torch.int16
            torch.int32: torch.int32,
            torch.int64: torch.int64,
            torch.bool: torch.int16,  # torch.bool 映射到 torch.int16
        },
        torch.int32: {
            # 定义 torch.int32 到其它数据类型的映射关系
            torch.float16: torch.float64,
            torch.float32: torch.float64,
            torch.float64: torch.float64,
            torch.complex64: torch.complex128,
            torch.complex128: torch.complex128,
            torch.uint8: torch.int32,
            torch.uint16: torch.int32,  # torch.uint16 映射到 torch.int32
            torch.uint32: torch.int64,
            torch.uint64: torch.float64,
            torch.int8: torch.int32,
            torch.int16: torch.int32,
            torch.int32: torch.int32,  # torch.int32 映射到 torch.int32
            torch.int64: torch.int64,
            torch.bool: torch.int32,  # torch.bool 映射到 torch.int32
        },
        torch.int64: {
            # 定义 torch.int64 到其它数据类型的映射关系
            torch.float16: torch.float64,
            torch.float32: torch.float64,
            torch.float64: torch.float64,
            torch.complex64: torch.complex128,
            torch.complex128: torch.complex128,
            torch.uint8: torch.int64,
            torch.uint16: torch.int64,
            torch.uint32: torch.int64,
            torch.uint64: torch.float64,
            torch.int8: torch.int64,
            torch.int16: torch.int64,
            torch.int32: torch.int64,
            torch.int64: torch.int64,  # torch.int64 映射到 torch.int64
            torch.bool: torch.int64,  # torch.bool 映射到 torch.int64
        },
        torch.bool: {
            # 定义 torch.bool 到其它数据类型的映射关系
            torch.float16: torch.float16,
            torch.float32: torch.float32,
            torch.float64: torch.float64,
            torch.complex64: torch.complex64,
            torch.complex128: torch.complex128,
            torch.uint8: torch.uint8,
            torch.uint16: torch.uint16,
            torch.uint32: torch.uint32,
            torch.uint64: torch.uint64,
            torch.int8: torch.int8,
            torch.int16: torch.int16,
            torch.int32: torch.int32,
            torch.int64: torch.int64,
            torch.bool: torch.bool,  # torch.bool 映射到 torch.bool
        },
    }
}


注释：


# 这是一个单独的右大括号，用于结束一个代码块或数据结构的定义。
```