# `.\pytorch\test\edge\custom_ops.cpp`

```
# 引入 PyTorch 的 Tensor 类
#include <ATen/Tensor.h>

# 定义 custom 命名空间，并进入 native 命名空间
namespace custom {
namespace native {

    # 定义函数 add_3_out，接受三个常量引用和一个输出引用的 Tensor 参数
    at::Tensor& add_3_out(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, at::Tensor& out) {
        # 计算 a、b、c 三个 Tensor 的和，并将结果赋给输出 Tensor out
        out = a.add(b).add(c);
        # 返回输出 Tensor 的引用
        return out;
    }

}  // 结束 native 命名空间
}  // 结束 custom 命名空间
```