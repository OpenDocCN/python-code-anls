# `.\pytorch\aten\src\ATen\native\Integration.cpp`

```
// 定义预处理指令，仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量操作的核心头文件
#include <ATen/core/Tensor.h>
// 包含维度向量的核心头文件
#include <ATen/core/DimVector.h>
// 包含张量操作的头文件
#include <ATen/TensorOperators.h>
// 包含维度包装工具的头文件
#include <ATen/WrapDimUtils.h>
// 包含异常处理的实用工具
#include <c10/util/Exception.h>
// 包含整数范围的头文件
#include <c10/util/irange.h>
// 包含标量类型的核心头文件
#include <c10/core/ScalarType.h>
// 包含标量值的核心头文件
#include <c10/core/Scalar.h>

// 如果未定义每个操作符的头文件，则包含以下功能头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了每个操作符的头文件，则包含以下特定函数头文件
#else
#include <ATen/ops/cumulative_trapezoid_native.h>
#include <ATen/ops/trapezoid_native.h>
#include <ATen/ops/trapz_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

// 命名空间内部的匿名函数定义开始

// 函数：使用梯形法则计算函数 y 关于 x 的估计积分，
// 样本点为 (y_1, ..., y_n)，各点间距为 (dx_1, ..., dx_{n-1})
Tensor do_trapezoid(const Tensor& y, const Tensor& dx, int64_t dim) {
    // 切片操作，获取 y 的左侧切片
    Tensor left = y.slice(dim, 0, -1);
    // 切片操作，获取 y 的右侧切片
    Tensor right = y.slice(dim, 1);
    // 如果 'dx' 的维度与 '(left + right)' 不匹配，则尝试广播操作
    return ((left + right) * dx).sum(dim) / 2.;
}

// 当 dx 为常数时，上述公式简化为 dx * [(\sum_{i=1}^n y_i) - (y_1 + y_n)/2]
Tensor do_trapezoid(const Tensor& y, double dx, int64_t dim) {
    return (y.sum(dim) - (y.select(dim, 0) + y.select(dim, -1)) * (0.5)) * dx;
}

// 创建一个与 'y' 维度相同的零张量，除了指定的维度 'dim' 外其它维度大小相同
Tensor zeros_like_except(const Tensor& y, int64_t dim) {
    // 获取 'y' 的符号化大小向量
    auto sizes = y.sym_sizes().vec();
    // 可能需要对维度 'dim' 进行包装
    dim = maybe_wrap_dim(dim, y.dim());
    // 移除 'sizes' 中位于 'dim' 位置的维度
    sizes.erase(sizes.begin() + dim);
    // 返回一个与 'y' 相同选项的零张量
    return at::zeros_symint(sizes, y.options());
}

// 使用累积梯形法则计算函数 y 关于 x 的估计积分，
// 样本点为 (y_1, ..., y_n)，各点间距为 (dx_1, ..., dx_{n-1})
Tensor do_cumulative_trapezoid(const Tensor& y, const Tensor& dx, int64_t dim) {
    // 切片操作，获取 y 的左侧切片
    Tensor left = y.slice(dim, 0, -1);
    // 切片操作，获取 y 的右侧切片
    Tensor right = y.slice(dim, 1);
    // 返回累积和：((left + right) * dx).cumsum(dim) / 2.
    return ((left + right) * dx).cumsum(dim) / 2.;
}

// 当 dx 为常数时，使用累积梯形法则简化为 dx / 2. * (left + right).cumsum(dim)
Tensor do_cumulative_trapezoid(const Tensor& y, double dx, int64_t dim) {
    // 切片操作，获取 y 的左侧切片
    Tensor left = y.slice(dim, 0, -1);
    // 切片操作，获取 y 的右侧切片
    Tensor right = y.slice(dim, 1);
    // 返回累积和：dx / 2. * (left + right).cumsum(dim)
    return (dx / 2. * (left + right)).cumsum(dim);
}

// 根据张量的当前形状和目标维度数目，
// 返回一个新的形状，保持原始形状的值，
// 但在开始处用 '1' 填充以匹配目标维度数目。
// 例如，curr_shape = (5,5,5)，target_n_dim = 6 ==> (1,1,1,5,5,5)
// 如果当前形状的维度数大于或等于目标维度数，不会添加填充。
SymDimVector add_padding_to_shape(SymIntArrayRef curr_shape, int64_t target_n_dim) {
    // 获取当前形状的大小
    const auto curr_size = static_cast<int64_t>(curr_shape.size());
    // 如果当前大小大于等于目标维度数，将目标维度数设置为当前大小
    if (curr_size >= target_n_dim){
        target_n_dim = curr_size;
    }
    // 创建新的形状向量，初始填充为 '1'
    SymDimVector new_shape(target_n_dim, 1);
    // 从后向前复制当前形状的维度值到新的形状向量中
    for (const auto i : c10::irange(curr_size)) {
        new_shape[target_n_dim-i-1] = curr_shape[curr_size-i-1];
    }
    // 返回填充后的新形状向量
    return new_shape;
}

// 命名空间内部的匿名函数定义结束

} // 结束命名空间 at::native
// 计算梯形积分，返回积分结果
Tensor trapezoid(const Tensor& y, const Tensor& x, int64_t dim) {
    // 确保维度在合理范围内
    dim = maybe_wrap_dim(dim, y);
    
    // 如果在指定维度上样本数为0，则返回与y相同形状的零张量
    if (y.sym_size(dim) == 0) {
        return zeros_like_except(y, dim);
    }
    
    // 检查输入张量x和y的数据类型，不支持布尔类型
    TORCH_CHECK(y.scalar_type() != kBool && x.scalar_type() != kBool, "trapezoid: received a bool input for `x` or `y`, but bool is not supported")
    
    Tensor x_viewed;
    
    // 如果x是一维张量，则根据y的维度和dim创建一个视图张量x_viewed
    if (x.dim() == 1) {
        // 确保x的长度与y在给定维度上的长度相同
        TORCH_CHECK(x.sym_size(0) == y.sym_size(dim), "trapezoid: There must be one `x` value for each sample point");
        
        // 根据y的维度创建新的大小向量，并在给定维度上设置为x的长度，其余维度设置为1
        SymDimVector new_sizes(y.dim(), 1);
        new_sizes[dim] = x.sym_size(0);
        
        // 将x视图化为新的形状
        x_viewed = x.view_symint(new_sizes);
    } else if (x.dim() < y.dim()) {
        // 当y的维度比x多时，在x的形状前面添加足够多的1，以匹配y的维度
        SymDimVector new_sizes = add_padding_to_shape(x.sym_sizes(), y.dim());
        x_viewed = x.view_symint(new_sizes);
    } else {
        x_viewed = x;
    }
    
    // 根据dim在x_viewed上做切片操作，得到左右两侧的坐标差
    Tensor x_left = x_viewed.slice(dim, 0, -1);
    Tensor x_right = x_viewed.slice(dim, 1);
    
    // 计算每个梯形的宽度
    Tensor dx = x_right - x_left;
    
    // 调用具体的梯形积分计算函数
    return do_trapezoid(y, dx, dim);
}

// 当dx是标量时的梯形积分计算
Tensor trapezoid(const Tensor& y, const Scalar& dx, int64_t dim) {
    // 见上述函数
    if (y.sym_size(dim) == 0) {
        return zeros_like_except(y, dim);
    }
    
    // 检查输入张量y的数据类型，不支持布尔类型
    TORCH_CHECK(y.scalar_type() != kBool, "trapezoid: received a bool input for `y`, but bool is not supported")
    
    // 检查dx不是复数或布尔类型
    TORCH_CHECK(!(dx.isComplex() ||  dx.isBoolean()), "trapezoid: Currently, we only support dx as a real number.");
    
    // 调用具体的梯形积分计算函数
    return do_trapezoid(y, dx.toDouble(), dim);
}

// 对外暴露的函数，计算张量y关于x的梯形积分
Tensor trapz(const Tensor& y, const Tensor& x, int64_t dim) {
    return at::native::trapezoid(y, x, dim);
}

// 对外暴露的函数，计算张量y关于等间距采样间隔dx的梯形积分
Tensor trapz(const Tensor& y, double dx, int64_t dim) {
    return at::native::trapezoid(y, dx, dim);
}

// 计算累积梯形积分，返回积分结果
Tensor cumulative_trapezoid(const Tensor& y, const Tensor& x, int64_t dim) {
    // 确保维度在合理范围内
    dim = maybe_wrap_dim(dim, y);
    
    // 检查输入张量x和y的数据类型，不支持布尔类型
    TORCH_CHECK(y.scalar_type() != kBool && x.scalar_type() != kBool, "cumulative_trapezoid: received a bool input for `x` or `y`, but bool is not supported")
    // 声明一个 Tensor 变量 x_viewed;
    Tensor x_viewed;
    
    // 如果 x 的维度为 1，则执行以下操作
    if (x.dim() == 1) {
        // 见 trapezoid 的实现说明
        // 检查 x 的符号尺寸（symbolic size）是否与 y 在指定维度上的符号尺寸相同，否则抛出错误信息
        TORCH_CHECK(x.sym_size(0) == y.sym_size(dim), "cumulative_trapezoid: There must be one `x` value for each sample point");
        
        // 创建一个新的尺寸向量 new_sizes，维度与 y 相同，但每个维度尺寸为 1
        SymDimVector new_sizes(y.dim(), 1); // shape = [1] * y.
        
        // 将指定维度 dim 的尺寸设置为 x 的符号尺寸大小
        new_sizes[dim] = x.sym_size(0); // shape[axis] = d.shape[0]
        
        // 将 x 按照新的尺寸 new_sizes 进行视图变换，并赋值给 x_viewed
        x_viewed = x.view_symint(new_sizes);
    } else if (x.dim() < y.dim()) {
        // 见 trapezoid 的实现说明
        // 创建一个新的尺寸向量 new_sizes，通过在 x 的符号尺寸后面添加填充来使其与 y 的维度相同
        SymDimVector new_sizes = add_padding_to_shape(x.sym_sizes(), y.dim());
        
        // 将 x 按照新的尺寸 new_sizes 进行视图变换，并赋值给 x_viewed
        x_viewed = x.view_symint(new_sizes);
    } else {
        // 否则，直接将 x 赋值给 x_viewed
        x_viewed = x;
    }
    
    // 根据指定的维度 dim，从 x_viewed 中切片得到 x_left 和 x_right
    Tensor x_left = x_viewed.slice(dim, 0, -1);
    Tensor x_right = x_viewed.slice(dim, 1);
    
    // 计算 x_right 和 x_left 之间的差值，并赋值给 dx
    Tensor dx = x_right - x_left;
    
    // 调用 do_cumulative_trapezoid 函数，传入参数 y, dx, dim，并返回其结果
    return do_cumulative_trapezoid(y, dx, dim);
}

// 结束 at::native 命名空间的定义

Tensor cumulative_trapezoid(const Tensor& y, const Scalar& dx, int64_t dim) {
    // 检查输入张量 y 的数据类型是否为 bool，若是则报错，因为不支持 bool 类型
    TORCH_CHECK(y.scalar_type() != kBool, "cumulative_trapezoid: received a bool input for `y`, but bool is not supported")
    // 检查 dx 是否为复数或布尔类型，若是则报错，因为目前只支持实数作为 dx
    TORCH_CHECK(!(dx.isComplex() || dx.isBoolean()), "cumulative_trapezoid: Currently, we only support dx as a real number.");

    // 调用 do_cumulative_trapezoid 函数计算梯形积分，返回结果张量
    return do_cumulative_trapezoid(y, dx.toDouble(), dim);
}

} // 结束命名空间 at::native
```