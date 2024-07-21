# `.\pytorch\aten\src\ATen\core\ATen_pch.h`

```
// 在这里暂时禁用 TORCH_ASSERT_NO_OPERATORS 宏定义，以确保不依赖于 native_functions.yaml 文件，
// 这样增量构建就不会变得几乎无用。
#pragma push_macro("TORCH_ASSERT_NO_OPERATORS")
#define TORCH_ASSERT_NO_OPERATORS

// 如果 __STDC_FORMAT_MACROS 宏未定义，则在此处定义它，以确保后续包含 inttypes.h 文件时不会出现问题。
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <cinttypes>

// 下面列出的头文件是通过一个脚本生成的，该脚本查找高影响的头文件，然后手动调整以删除特定于操作系统的头文件
// 或重复的头文件（例如 <cassert> 和 <assert.h>），并删除 "impl" 头文件（例如 c10 中的 BFloat16-inl.h 或 complex_math.h）。

// 生成初始列表的步骤：
// 1. 禁用所有构建缓存重新从头构建 pytorch
// 2. 使用 ninjatracing（https://github.com/nico/ninjatracing）生成构建跟踪
//    $ ninjatracing /path/to/pytorch/build/.ninja_log > trace_all.json
// 3. 使用 https://github.com/peterbell10/build_analysis/ 中的 pch_gen.py 生成预编译头文件列表
//    $ python pch_gen.py --threshold .80 --target torch_cpu --build_dir /path/to/pytorch/build --trace trace_all.json
//    调整阈值直到包含 c10 和部分 ATen 核心但仍然通过 TORCH_ASSERT_NO_OPERATORS。

#include <cerrno>                 // C 标准错误码
#include <cmath>                  // 数学函数
#include <cstddef>                // 标准定义
#include <cstdint>                // 标准整数类型
#include <cstdlib>                // 标准库函数
#include <cstring>                // 字符串操作
#include <algorithm>              // 算法
#include <array>                  // 固定大小数组
#include <atomic>                 // 原子操作
#include <chrono>                 // 时间库
#include <complex>                // 复数
#include <deque>                  // 双端队列
#include <exception>              // 异常处理
#include <functional>             // 函数对象
#include <initializer_list>       // 初始化列表
#include <iomanip>                // 输入输出流格式控制
#include <iosfwd>                 // 输入输出流前置声明
#include <iterator>               // 迭代器
#include <limits>                 // 数值极限
#include <list>                   // 链表
#include <map>                    // 映射容器
#include <memory>                 // 内存管理
#include <mutex>                  // 互斥量
#include <new>                    // 动态内存管理
#include <numeric>                // 数值操作
#include <ostream>                // 输出流
#include <sstream>                // 字符串流
#include <stdexcept>              // 标准异常类
#include <string>                 // 字符串类
#include <tuple>                  // 元组
#include <type_traits>            // 类型特性
#include <typeindex>              // 类型索引
#include <typeinfo>               // 类型信息
#include <unordered_map>          // 无序映射容器
#include <unordered_set>          // 无序集合容器
#include <utility>                // 实用工具
#include <vector>                 // 向量容器

#include <c10/core/Allocator.h>
#include <c10/core/AutogradState.h>
#include <c10/core/Backend.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/OptionalRef.h>
#include <c10/core/QScheme.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/SymBool.h>
#include <c10/core/SymFloat.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/core/impl/SizesAndStrides.h>

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>

#include <c10/util/AlignOf.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/BFloat16.h>
#include <c10/util/C++17.h>
#include <c10/util/ConstexprCrc.h>
#include <c10/util/Deprecated.h>
#include <c10/util/DimVector.h>
#include <c10/util/Exception.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/Flags.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/FunctionRef.h>
#include <c10/util/Half.h>
#include <c10/util/IdWrapper.h>
#include <c10/util/Logging.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/Optional.h>
#include <c10/util/Registry.h>
#include <c10/util/SmallVector.h>
#include <c10/util/StringUtil.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <c10/util/Type.h>
#include <c10/util/TypeCast.h>
#include <c10/util/TypeIndex.h>
#include <c10/util/TypeList.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/TypeTraits.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/accumulate.h>
#include <c10/util/bit_cast.h>
#include <c10/util/bits.h>
#include <c10/util/complex.h>
#include <c10/util/floating_point_utils.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/python_stub.h>
#include <c10/util/qint32.h>
#include <c10/util/qint8.h>
#include <c10/util/quint2x4.h>
#include <c10/util/quint4x2.h>
#include <c10/util/quint8.h>
#include <c10/util/safe_numerics.h>
#include <c10/util/string_utils.h>
#include <c10/util/string_view.h>
#include <c10/util/typeid.h>

#include <ATen/StorageUtils.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/DeprecatedTypePropertiesRegistry.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/Dimname.h>
#include <ATen/core/Generator.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/core/QuantizerBase.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/core/TensorBase.h>
#include <ATen/core/symbol.h>

// 弹出之前定义的宏 "TORCH_ASSERT_NO_OPERATORS"
#pragma pop_macro("TORCH_ASSERT_NO_OPERATORS")
```