# `.\pytorch\aten\src\ATen\native\vulkan\impl\Common.cpp`

```py
namespace at {
namespace native {
namespace vulkan {

在 `at` 命名空间下，定义 `native` 命名空间，再在其中定义 `vulkan` 命名空间。


api::utils::uvec3 adaptive_work_group_size(
    const api::utils::uvec3& global_work_group) {

定义名为 `adaptive_work_group_size` 的函数，接受一个 `api::utils::uvec3` 类型的引用参数 `global_work_group`。


api::utils::uvec3 local_group_size = {4, 4, 4};

初始化 `local_group_size` 变量为 `{4, 4, 4}`，这是一个 `api::utils::uvec3` 类型的对象。


if (global_work_group.data[2u] == 1) {

如果 `global_work_group` 对象的 `data` 数组中索引为 `2u` 的元素等于 `1`，则执行以下代码块。


if (global_work_group.data[1u] < 8) {

如果 `global_work_group` 对象的 `data` 数组中索引为 `1u` 的元素小于 `8`，则执行以下代码块；否则执行另一个代码块。


local_group_size.data[0u] = 16;
local_group_size.data[1u] = 4;
local_group_size.data[2u] = 1;

设置 `local_group_size` 对象的 `data` 数组的不同索引位置的值为 `16`、`4` 和 `1`。


} else {

否则，执行以下代码块。


local_group_size.data[0u] = 8;
local_group_size.data[1u] = 8;
local_group_size.data[2u] = 1;

设置 `local_group_size` 对象的 `data` 数组的不同索引位置的值为 `8`、`8` 和 `1`。


}

结束条件判断语句的代码块。


return local_group_size;

返回 `local_group_size` 对象，该对象存储了根据条件设置后的局部工作组大小。


} // namespace vulkan
} // namespace native
} // namespace at

结束 `vulkan`、`native` 和 `at` 命名空间的定义。
```