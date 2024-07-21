# `.\pytorch\torch\csrc\jit\passes\peephole_list_idioms.h`

```
#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// 对列表操作进行窥孔优化，如len(li)和li[1]。
// 1. 构造/解包优化
//    给定以下函数：
//    def foo(a, b):
//        li = [a, b]
//        x, y = li
//        return x, y
//    该优化在死代码消除后会生成：
//    def foo(a, b):
//        return a, b
//
//    仅适用于未被修改的列表。
//
// 2. 索引操作优化
//    给定以下函数：
//    def foo(a, b):
//        li = [a, b]
//        x = li[0]
//        return x
//    该优化在死代码消除后会生成：
//    def foo(a, b):
//        return a
//
//    仅当列表未被修改时才能进行此优化。
//
// 3. 长度操作优化
//    给定以下函数：
//    def foo():
//        li = [1, 2]
//        return len(li)
//    该优化在死代码消除后会生成：
//    def foo():
//        return 2
//
//    与索引操作优化具有相同的要求。
//
// 4. 列表构造 + 列表构造
//    给定以下函数：
//    def foo():
//        return [1, 2] + [3, 4]
//    该优化在死代码消除后会生成：
//    def foo():
//        return [1, 2, 3, 4]
//
//    仅适用于未被修改的列表。
//
// 5. 切片操作
//    给定以下函数：
//    def foo():
//        return [1, 2, 3, 4, 5][0:2]
//    该优化在死代码消除后会生成：
//    def foo():
//        return [1, 2]
//
//    目前作为PeepholeOptimize的一部分被调用。
//    如果图表被修改则返回true。
//    如果`refine_list_len`为true，将尝试通过长度比较和断言来精确化列表的长度。
//    这通常不会优化PyTorch程序，因此在PeepholeOptimize中默认情况下不会调用。
TORCH_API bool PeepholeOptimizeListIdioms(
    const std::shared_ptr<Graph>& graph,
    bool refine_list_len = false);

} // namespace jit
} // namespace torch
```