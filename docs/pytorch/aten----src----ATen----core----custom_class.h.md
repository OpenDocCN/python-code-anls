# `.\pytorch\aten\src\ATen\core\custom_class.h`

```
#pragma once

#include <typeindex>  // 引入类型索引的标准头文件
#include <memory>     // 引入智能指针的标准头文件

#include <c10/macros/Export.h>  // 引入导出宏的头文件
#include <c10/macros/Macros.h>  // 引入宏定义的头文件
#include <c10/util/Exception.h> // 引入异常处理的头文件

namespace c10 {

struct ClassType;              // 声明结构体 ClassType
using ClassTypePtr = std::shared_ptr<ClassType>;  // 使用智能指针定义 ClassTypePtr 类型别名

TORCH_API c10::ClassTypePtr getCustomClassTypeImpl(const std::type_index &tindex);  // 声明获取自定义类类型的函数原型

template <typename T>
const c10::ClassTypePtr& getCustomClassType() {
  // 类型永远不会从 getCustomClassTypeMap 注销，并且哈希查找可能是一个热点路径，因此只需缓存。
  // 出于同样的原因，如果因为某种原因这会跨动态共享对象（DSO）边界复制，这也是可以接受的。
  static c10::ClassTypePtr cache = getCustomClassTypeImpl(
      std::type_index(typeid(T)));  // 使用类型索引初始化静态缓存，调用 getCustomClassTypeImpl 获取类型 T 对应的自定义类类型
  return cache;  // 返回缓存的自定义类类型指针
}

}
```