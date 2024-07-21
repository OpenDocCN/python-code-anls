# `.\pytorch\c10\core\DefaultDtype.h`

```py
#pragma once
// 使用预处理指令#pragma once，确保头文件只被编译一次

#include <c10/core/ScalarType.h>
// 包含c10库中的ScalarType头文件

#include <c10/macros/Export.h>
// 包含c10库中的Export.h头文件，用于导出符号定义

namespace caffe2 {
class TypeMeta;
} // namespace caffe2
// 定义caffe2命名空间，并声明TypeMeta类

namespace c10 {
// 定义c10命名空间

C10_API void set_default_dtype(caffe2::TypeMeta dtype);
// 使用C10_API导出函数set_default_dtype，设置默认数据类型，参数为caffe2::TypeMeta类型

C10_API const caffe2::TypeMeta get_default_dtype();
// 使用C10_API导出函数get_default_dtype，获取默认数据类型，返回值为const caffe2::TypeMeta

C10_API ScalarType get_default_dtype_as_scalartype();
// 使用C10_API导出函数get_default_dtype_as_scalartype，获取默认数据类型的标量类型

C10_API const caffe2::TypeMeta get_default_complex_dtype();
// 使用C10_API导出函数get_default_complex_dtype，获取默认复杂数据类型，返回值为const caffe2::TypeMeta

} // namespace c10
// 结束c10命名空间
```