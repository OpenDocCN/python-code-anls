# `.\pytorch\torch\csrc\lazy\core\shape_inference.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/Tensor.h>
// 引入 ATen 库中的 Tensor 头文件

#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/SymNodeImpl.h>
// 引入 C10 核心库中的 ScalarType、SymInt、SymIntArrayRef 和 SymNodeImpl 头文件

#include <c10/macros/Export.h>
// 引入 C10 宏定义导出头文件

#include <c10/util/Optional.h>
// 引入 C10 工具库中的 Optional 头文件

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/core/tensor.h>
// 引入 Torch 懒执行模块中的 backend_data、ir、shape 和 tensor 头文件

#include <vector>
// 引入标准库中的 vector 头文件

namespace torch {
namespace lazy {
// 命名空间 torch::lazy

// 关闭 clang-format，因为我们依赖整个签名在一行上用于代码生成。
// clang-format off

TORCH_API std::vector<torch::lazy::Shape> compute_shape__adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size);
// 声明计算自适应平均池化二维形状的函数，接受输入张量和输出大小作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape__adaptive_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self);
// 声明计算自适应平均池化二维反向传播形状的函数，接受梯度输出张量和输入张量作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape__adaptive_avg_pool3d(const at::Tensor & self, at::IntArrayRef output_size);
// 声明计算自适应平均池化三维形状的函数，接受输入张量和输出大小作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape__adaptive_avg_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self);
// 声明计算自适应平均池化三维反向传播形状的函数，接受梯度输出张量和输入张量作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape_abs(const at::Tensor & self);
// 声明计算绝对值形状的函数，接受输入张量作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape_arange_out(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out);
// 声明计算 arange 函数输出形状的函数，接受起始、终止、步长和输出张量作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape_bernoulli(const at::Tensor & self, ::std::optional<at::Generator> generator);
// 声明计算 Bernoulli 分布形状的函数，接受输入张量和可选的随机数生成器作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape_bernoulli(const at::Tensor & self, double p, ::std::optional<at::Generator> generator);
// 声明计算 Bernoulli 分布形状的函数，接受输入张量、概率值和可选的随机数生成器作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape_binary_cross_entropy(const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction);
// 声明计算二元交叉熵形状的函数，接受输入张量、目标张量、权重张量和减少方式作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape_binary_cross_entropy_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction);
// 声明计算二元交叉熵反向传播形状的函数，接受梯度输出张量、输入张量、目标张量、权重张量和减少方式作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape_cat(at::TensorList tensors, int64_t dim);
// 声明计算 cat 函数形状的函数，接受张量列表和维度作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape_cholesky(const at::Tensor & self, bool upper);
// 声明计算 Cholesky 分解形状的函数，接受输入张量和是否上三角矩阵作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape_clamp_min(const at::Tensor & self, const at::Scalar & min);
// 声明计算 clamp_min 函数形状的函数，接受输入张量和最小值作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape_clone(const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format);
// 声明计算克隆形状的函数，接受输入张量和内存格式作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape_constant_pad_nd(const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value);
// 声明计算常数填充形状的函数，接受输入张量、填充大小和常数值作为参数

TORCH_API std::vector<torch::lazy::Shape> compute_shape_convolution(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups);
// 声明计算卷积形状的函数，接受输入张量、权重张量、偏置张量、步长、填充、膨胀、是否转置、输出填充和分组数作为参数
# 计算卷积反向传播的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_convolution_backward(
    const at::Tensor & grad_output,                 // 梯度输出张量
    const at::Tensor & input,                       // 输入张量
    const at::Tensor & weight,                      // 权重张量
    at::OptionalIntArrayRef bias_sizes,             // 可选的偏置大小
    at::IntArrayRef stride,                         // 步幅
    at::IntArrayRef padding,                        // 填充
    at::IntArrayRef dilation,                       // 膨胀
    bool transposed,                                // 是否转置
    at::IntArrayRef output_padding,                 // 输出填充
    int64_t groups,                                 // 组数
    ::std::array<bool,3> output_mask                // 输出掩码数组
);

# 计算嵌入层的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_embedding(
    const at::Tensor & weight,                      // 权重张量
    const at::Tensor & indices,                     // 索引张量
    int64_t padding_idx,                            // 填充索引
    bool scale_grad_by_freq,                        // 是否按频率缩放梯度
    bool sparse                                     // 是否稀疏
);

# 计算稠密嵌入层反向传播的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_embedding_dense_backward(
    const at::Tensor & grad_output,                 // 梯度输出张量
    const at::Tensor & indices,                     // 索引张量
    int64_t num_weights,                            // 权重数量
    int64_t padding_idx,                            // 填充索引
    bool scale_grad_by_freq                         // 是否按频率缩放梯度
);

# 计算张量扩展的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_expand(
    const at::Tensor & self,                        // 输入张量
    at::IntArrayRef size,                           // 尺寸
    bool implicit                                   // 是否隐式
);

# 计算张量扩展的形状（符号数组版本）
TORCH_API std::vector<torch::lazy::Shape> compute_shape_expand(
    const at::Tensor & self,                        // 输入张量
    c10::SymIntArrayRef size,                       // 符号数组尺寸
    bool implicit                                   // 是否隐式
);

# 计算张量翻转的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_flip(
    const at::Tensor & self,                        // 输入张量
    at::IntArrayRef dims                            // 维度
);

# 计算 GLU 层反向传播的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_glu_backward(
    const at::Tensor & grad_output,                 // 梯度输出张量
    const at::Tensor & self,                        // 输入张量
    int64_t dim                                     // 维度
);

# 计算 GLU 层的 JVP 形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_glu_jvp(
    const at::Tensor & glu,                         // GLU 输出张量
    const at::Tensor & x,                           // 输入张量
    const at::Tensor & dx,                          // 输入梯度张量
    int64_t dim                                     // 维度
);

# 计算二维网格采样器的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_grid_sampler_2d(
    const at::Tensor & input,                       // 输入张量
    const at::Tensor & grid,                        // 网格张量
    int64_t interpolation_mode,                     // 插值模式
    int64_t padding_mode,                           // 填充模式
    bool align_corners                              // 是否对齐角点
);

# 计算二维网格采样器反向传播的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_grid_sampler_2d_backward(
    const at::Tensor & grad_output,                 // 梯度输出张量
    const at::Tensor & input,                       // 输入张量
    const at::Tensor & grid,                        // 网格张量
    int64_t interpolation_mode,                     // 插值模式
    int64_t padding_mode,                           // 填充模式
    bool align_corners,                             // 是否对齐角点
    ::std::array<bool,2> output_mask                // 输出掩码数组
);

# 计算索引选择操作的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_index_select(
    const at::Tensor & self,                        // 输入张量
    int64_t dim,                                    // 维度
    const at::Tensor & index                        // 索引张量
);

# 计算矩阵的逆的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_inverse(
    const at::Tensor & self                         // 输入张量
);

# 计算张量是否为 NaN 的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_isnan(
    const at::Tensor & self                         // 输入张量
);

# 计算对数 Sigmoid 反向传播的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_log_sigmoid_backward(
    const at::Tensor & grad_output,                 // 梯度输出张量
    const at::Tensor & self,                        // 输入张量
    const at::Tensor & buffer                       // 缓冲张量
);

# 计算对数 Sigmoid 前向传播的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_log_sigmoid_forward(
    const at::Tensor & self                         // 输入张量
);

# 计算对数行列式的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_logdet(
    const at::Tensor & self                         // 输入张量
);

# 计算逻辑与操作的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_logical_and(
    const at::Tensor & self,                        // 输入张量
    const at::Tensor & other                        // 另一个张量
);

# 计算逻辑非操作的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_logical_not(
    const at::Tensor & self                         // 输入张量
);

# 计算逻辑或操作的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_logical_or(
    const at::Tensor & self,                        // 输入张量
    const at::Tensor & other                        // 另一个张量
);

# 计算逻辑异或操作的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_logical_xor(
    const at::Tensor & self,                        // 输入张量
    const at::Tensor & other                        // 另一个张量
);
# 计算填充掩码后的形状，返回一个由 lazy 模块中 Shape 对象组成的向量
TORCH_API std::vector<torch::lazy::Shape> compute_shape_masked_fill(const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value);

# 计算填充掩码后的形状，返回一个由 lazy 模块中 Shape 对象组成的向量，使用另一个张量作为填充值
TORCH_API std::vector<torch::lazy::Shape> compute_shape_masked_fill(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value);

# 计算张量的最大值形状，返回一个由 lazy 模块中 Shape 对象组成的向量
TORCH_API std::vector<torch::lazy::Shape> compute_shape_max(const at::Tensor & self);

# 计算张量的均值形状，返回一个由 lazy 模块中 Shape 对象组成的向量，可以指定数据类型
TORCH_API std::vector<torch::lazy::Shape> compute_shape_mean(const at::Tensor & self, ::std::optional<at::ScalarType> dtype);

# 计算张量的最小值形状，返回一个由 lazy 模块中 Shape 对象组成的向量
TORCH_API std::vector<torch::lazy::Shape> compute_shape_min(const at::Tensor & self);

# 计算张量与向量的形状，返回一个由 lazy 模块中 Shape 对象组成的向量
TORCH_API std::vector<torch::lazy::Shape> compute_shape_mv(const at::Tensor & self, const at::Tensor & vec);

# 计算原生批归一化的形状，返回一个由 lazy 模块中 Shape 对象组成的向量，支持训练状态、动量和 epsilon 参数
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_batch_norm(const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double momentum, double eps);

# 计算原生批归一化反向传播的形状，返回一个由 lazy 模块中 Shape 对象组成的向量，支持训练状态、epsilon 和输出掩码
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_batch_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, const ::std::optional<at::Tensor> & save_mean, const ::std::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask);

# 计算原生 dropout 的形状，返回一个由 lazy 模块中 Shape 对象组成的向量，支持概率和训练状态
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_dropout(const at::Tensor & input, double p, ::std::optional<bool> train);

# 计算原生 dropout 反向传播的形状，返回一个由 lazy 模块中 Shape 对象组成的向量，支持梯度输出、掩码和缩放比例
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double scale);

# 计算原生层归一化的形状，返回一个由 lazy 模块中 Shape 对象组成的向量，支持输入、归一化形状、权重、偏置和 epsilon 参数
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps);

# 计算原生层归一化反向传播的形状，返回一个由 lazy 模块中 Shape 对象组成的向量，支持梯度输出、输入、归一化形状、均值、反标准差、权重、偏置和输出掩码
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_layer_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask);

# 计算新建空的分步张量的形状，返回一个由 lazy 模块中 Shape 对象组成的向量，支持大小、步幅、数据类型、布局、设备和是否锁定内存
TORCH_API std::vector<torch::lazy::Shape> compute_shape_new_empty_strided(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory);

# 计算 2D 负对数似然损失反向传播的形状，返回一个由 lazy 模块中 Shape 对象组成的向量，支持梯度输出、自身、目标、权重、减少维度、忽略索引和总权重
TORCH_API std::vector<torch::lazy::Shape> compute_shape_nll_loss2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight);
# 计算二维 NLL 损失函数的前向传播的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_nll_loss2d_forward(const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index);

# 计算张量中非零元素的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_nonzero(const at::Tensor & self);

# 计算正态分布函数操作的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_normal_functional(const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator);

# 计算随机张量的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_random(const at::Tensor & self, ::std::optional<at::Generator> generator);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_random(const at::Tensor & self, int64_t to, ::std::optional<at::Generator> generator);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_random(const at::Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<at::Generator> generator);

# 计算 ReLU 激活函数的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_relu(const at::Tensor & self);

# 计算张量重复操作的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_repeat(const at::Tensor & self, at::IntArrayRef repeats);

# 计算张量行列式的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_slogdet(const at::Tensor & self);

# 计算 Smooth L1 损失函数的反向传播的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_smooth_l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta);

# 计算张量排序的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_sort(const at::Tensor & self, int64_t dim, bool descending);

# 计算张量堆叠操作的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_stack(at::TensorList tensors, int64_t dim);

# 计算张量标准差的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_std(const at::Tensor & self, bool unbiased);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_std(const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_std(const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim);

# 计算张量求和的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_sum(const at::Tensor & self, ::std::optional<at::ScalarType> dtype);

# 复制张量并返回其副本的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape__to_copy(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, bool non_blocking, ::std::optional<at::MemoryFormat> memory_format);

# 计算从张量中取值的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_take(const at::Tensor & self, const at::Tensor & index);

# 计算张量的迹操作的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_trace(const at::Tensor & self);

# 计算全零张量的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_zero(const at::Tensor & self);

# 计算对张量进行缩窄复制的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_narrow_copy_symint(const at::Tensor & self, int64_t dim, int64_t start, c10::SymInt length);
// 定义一个函数，计算 Hardswish 操作的输出形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_hardswish(const at::Tensor & self);

// 定义一个函数，计算 Hardswish 反向传播操作的输出形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_hardswish_backward(const at::Tensor & grad_output, const at::Tensor & self);

// 定义一个函数，计算 SELU 操作的输出形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_selu(const at::Tensor & self);

// 定义一个函数，计算生成指定范围均匀分布的张量的形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_uniform(const at::Tensor & self, double from, double to, ::std::optional<at::Generator> generator);

// 定义一个函数，计算标量值的形状
TORCH_API std::vector<Shape> compute_shape_scalar(const at::Scalar& value, const at::ScalarType& type);

// 定义一个函数，计算扩展操作的输出形状
TORCH_API std::vector<Shape> compute_shape_expand(const Output& input0, const std::vector<int64_t>& size, const bool& is_scalar_expand);

// 定义一个函数，计算视图操作的输出形状
TORCH_API std::vector<Shape> compute_shape_view(const Output& input0, const std::vector<int64_t>& output_sizes);

// 定义一个函数，计算类型转换操作的输出形状
TORCH_API std::vector<Shape> compute_shape_cast(const Output& input0, const at::ScalarType& dtype, const ::std::optional<at::ScalarType>& stype);

// 定义一个函数，计算 as_strided 视图更新操作的输出形状
TORCH_API std::vector<Shape> compute_shape_as_strided_view_update(const Output& target, const Output& input, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, const int64_t& storage_offset);

// 定义一个函数，计算 as_strided 操作的输出形状
TORCH_API std::vector<Shape> compute_shape_as_strided(const Output& input, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, const int64_t& storage_offset);

// 定义一个函数，计算对角线视图更新操作的输出形状
TORCH_API std::vector<Shape> compute_shape_diagonal_view_update(const Output& target, const Output& input, const int64_t& offset, const int64_t& dim1, const int64_t& dim2);

// 定义一个函数，计算对角线操作的输出形状
TORCH_API std::vector<Shape> compute_shape_diagonal(const Output& input, const int64_t& offset, const int64_t& dim1, const int64_t& dim2);

// 定义一个函数，计算 narrow 视图更新操作的输出形状
TORCH_API std::vector<Shape> compute_shape_narrow_view_update(const Output& input, const Output& source, const std::vector<int64_t>& base_indices);

// 定义一个函数，计算 narrow 操作的输出形状
TORCH_API std::vector<Shape> compute_shape_narrow(const Output& input, const std::vector<int64_t>& base_indices, const std::vector<int64_t>& sizes);

// 定义一个函数，计算维度置换操作的输出形状
TORCH_API std::vector<Shape> compute_shape_permute(const Output& input, const std::vector<int64_t>& dims);

// 定义一个函数，计算调整大小操作的输出形状
TORCH_API std::vector<Shape> compute_shape_resize(const Output& input, const std::vector<int64_t>& size);

// 定义一个函数，计算 select 视图更新操作的输出形状
TORCH_API std::vector<Shape> compute_shape_select_view_update(const Output& target, const Output& source, const int64_t& dim, const int64_t& start, const int64_t& end, const int64_t& stride);

// 定义一个函数，计算 select 操作的输出形状
TORCH_API std::vector<Shape> compute_shape_select(const Output& input, const int64_t& dim, const int64_t& start, const int64_t& end, const int64_t& stride);

// 定义一个函数，计算挤压操作的输出形状
TORCH_API std::vector<Shape> compute_shape_squeeze(const Output& input, const int& dim);

// 定义一个函数，计算展开操作的输出形状
TORCH_API std::vector<Shape> compute_shape_unsqueeze(const Output& input, const int& dim);

// 定义一个函数，计算选择散列操作的输出形状
TORCH_API std::vector<torch::lazy::Shape> compute_shape_select_scatter(const at::Tensor & self, const at::Tensor & src, int64_t dim, int64_t index);
// 定义一个命名空间 lazy，其包含了与形状计算相关的函数
namespace torch {
namespace lazy {

// 计算对角线散布的形状，返回一个包含形状的向量
TORCH_API std::vector<torch::lazy::Shape> compute_shape_diagonal_scatter(const at::Tensor & self, const at::Tensor & src, int64_t offset, int64_t dim1, int64_t dim2);

// 计算切片散布的对称整数形状，返回一个包含形状的向量
TORCH_API std::vector<torch::lazy::Shape> compute_shape_slice_scatter_symint(const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step);

// 计算作为扩展的对称整数形状散布，返回一个包含形状的向量
TORCH_API std::vector<torch::lazy::Shape> compute_shape_as_strided_scatter_symint(const at::Tensor & self, const at::Tensor & src, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset);

// 命名空间 lazy 结束
} // namespace lazy

// 命名空间 torch 结束
} // namespace torch
```