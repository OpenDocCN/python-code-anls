# `.\pytorch\aten\src\ATen\native\cudnn\BatchNorm.h`

```
# 定义了一个命名空间 `at::native`，用于组织与本地（native）实现相关的功能
namespace at::native {

# 定义了一个名为 `_get_cudnn_batch_norm_reserve_space_size` 的函数，用于获取 CUDNN 批归一化保留空间的大小
# 函数接受两个参数：
# - `input_t`：类型为 Tensor，表示输入张量
# - `training`：类型为 bool，表示是否处于训练模式

TORCH_API size_t
_get_cudnn_batch_norm_reserve_space_size(const Tensor& input_t, bool training);

} // namespace at::native
```