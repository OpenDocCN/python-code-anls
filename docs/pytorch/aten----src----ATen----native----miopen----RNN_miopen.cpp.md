# `.\pytorch\aten\src\ATen\native\miopen\RNN_miopen.cpp`

```py
// 定义宏，用于在 ATen 库中只包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库中与 RNN 相关的头文件
#include <ATen/native/RNN.h>
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/MatrixRef.h>
#include <ATen/TensorUtils.h>

// 包含 ATen CUDA 配置和异常处理相关的头文件
#include <ATen/cuda/CUDAConfig.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

// 根据宏定义条件编译不同的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/miopen_rnn.h>
#include <ATen/ops/miopen_rnn_native.h>
#include <ATen/ops/miopen_rnn_backward_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

// 根据不同的条件编译，选择不同的命名空间和函数定义
#if !AT_ROCM_ENABLED()

namespace at { namespace native {

    // 在没有 MIOpen 支持的情况下，定义 miopen_rnn 函数返回错误信息
    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> miopen_rnn(
            const Tensor& input_r, TensorList weight, int64_t weight_stride0,
            const Tensor& hx, const std::optional<Tensor>& cx_opt,
            int64_t fn_mode, int64_t fn_hidden_size, int64_t fn_num_layers,
            bool batch_first, double fn_dropout, bool fn_train, bool fn_bidirectional,
            IntArrayRef fn_batch_sizes, const std::optional<Tensor>& fn_dropout_state_opt
            ) {
        AT_ERROR("miopen_rnn : ATen not compiled with MIOpen support.");
    }

    // 在没有 MIOpen 支持的情况下，定义 miopen_rnn_backward 函数返回错误信息
    std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> miopen_rnn_backward(
            const Tensor& input, TensorList weight, int64_t weight_stride0, const Tensor& weight_buf, const Tensor& hx, const std::optional<Tensor>& cx_opt,
            const Tensor& output, const std::optional<Tensor>& grad_output_r_opt, const std::optional<Tensor>& grad_hy_r_opt, const std::optional<Tensor>& grad_cy_r_opt, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first,
            double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const std::optional<Tensor>& dropout_state_opt,
            const Tensor& reserve, std::array<bool, 4> output_mask
            ) {
        AT_ERROR("miopen_rnn_backward: ATen not compiled with MIOpen support.");
    }

}} //namespace at::native

#else // AT_ROCM_ENABLED()

// 在 ROCm 平台上启用 MIOpen 支持，包含相关头文件
#include <ATen/miopen/miopen-wrapper.h>
#include <ATen/miopen/Descriptors.h>
#include <ATen/miopen/Types.h>
#include <ATen/miopen/Utils.h>

#include <ATen/TensorUtils.h>

#include <functional>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

namespace at { namespace native {

// 定义 RNNDescriptorParams 结构体，描述 RNN 的参数
struct RNNDescriptorParams {
    int64_t hidden_size;                    // RNN 隐藏层的大小
    int64_t num_layers;                     // RNN 层数
    miopenRNNDirectionMode_t direction;     // RNN 方向模式
    miopenRNNMode_t rnn_mode;               // RNN 模式
    miopenDataType_t datatype;              // 数据类型
    miopenRNNAlgo_t algo = miopenRNNdefault;    // RNN 算法，默认为默认算法
    miopenRNNInputMode_t input_mode = miopenRNNlinear;   // RNN 输入模式，默认为线性输入
    miopenRNNBiasMode_t bias_mode = miopenRNNNoBias;      // RNN 偏置模式，默认为无偏置

    // 返回 RNN 的方向数目，如果是双向的返回 2，单向返回 1
    int64_t num_directions() const {
        return (direction == miopenRNNbidirection) ? 2 : 1;
    }
    // 设置循环神经网络的方向（双向或单向）
    void set_bidirectional(bool fn_bidirectional) {
        // 根据输入的布尔值设置方向类型
        direction = fn_bidirectional ? miopenRNNbidirection : miopenRNNunidirection;
    }

    // 设置循环神经网络的算法
    void set_algo(miopenRNNAlgo_t algo) {
        // 将传入的算法参数赋给当前对象的算法变量
        this->algo = algo;
    }

    // 设置循环神经网络的工作模式
    void set_mode(int64_t fn_mode) {
        // 根据传入的整数选择不同的工作模式
        switch (fn_mode) {
            case 0:
                rnn_mode = miopenRNNRELU;
                break;
            case 1:
                rnn_mode = miopenRNNTANH;
                break;
            case 2:
                rnn_mode = miopenLSTM;
                break;
            case 3:
                rnn_mode = miopenGRU;
                break;
            default:
                {
                    // 若传入的模式不在已定义的范围内，抛出错误信息
                    std::ostringstream oss;
                    oss << "unrecognized miopen RNN mode " << fn_mode;
                    AT_ERROR(oss.str());
                }
        }
    }

    // 设置循环神经网络的各项参数
    void set(int64_t mode, int64_t hidden_size, int64_t num_layers, bool bidirectional, miopenDataType_t datatype, miopenRNNBiasMode_t bias_mode) {
        // 依次设置模式、隐藏层大小、层数、方向、数据类型和偏置模式
        this->set_mode(mode);
        this->hidden_size = hidden_size;
        this->num_layers = num_layers;
        this->set_bidirectional(bidirectional);
        this->datatype = datatype;
        this->bias_mode = bias_mode;
    }

    // 获取循环神经网络的描述符
    RNNDescriptor descriptor() const {
        // 创建一个新的RNN描述符对象，并设置其参数
        RNNDescriptor rnn_desc;
        rnn_desc.set(hidden_size, num_layers, input_mode, direction, rnn_mode, bias_mode, algo, datatype);
        // 返回设置好参数的RNN描述符对象
        return rnn_desc;
    }
};

// 结构体定义结束

// 创建一个返回RNN序列描述符的函数，接收张量和批次大小数组作为参数
std::vector<TensorDescriptor> rnn_descriptor_sequence(const Tensor& tensor, IntArrayRef batch_sizes) {
    // 创建描述符向量，大小为批次大小数组的大小
    std::vector<TensorDescriptor> descriptors(batch_sizes.size());
    // 初始化索引 i
    size_t i = 0;

    // 获取张量的尺寸向量
    auto batch_tensor_size = tensor.sizes().vec();
    // 遍历批次大小数组中的每个批次大小
    for (auto batch_size : batch_sizes) {
        // 更新张量尺寸向量的第一个维度为当前批次大小
        batch_tensor_size[0] = batch_size;

        // 使用张量的数据类型、更新后的张量尺寸、步长和维度创建描述符
        descriptors[i].set(getMiopenDataType(tensor), batch_tensor_size, tensor.strides(), 3);
        // 更新索引
        i++;
    }

    // 返回描述符向量
    return descriptors;
}

// 创建一个返回RNN描述符的函数，接收张量和整数 N 作为参数
std::vector<TensorDescriptor> rnn_descriptor(const Tensor& tensor, int64_t N) {
    // 创建描述符向量，大小为 N
    std::vector<TensorDescriptor> descriptors(N);
    // 对于 0 到 N-1 的每个索引 i
    for (const auto i : c10::irange(N)) {
        // 使用张量和数字 5 创建描述符
        descriptors[i].set(tensor, 5);
    }

    // 返回描述符向量
    return descriptors;
}

// 定义一个结构体 TensorDescriptorListParams
struct TensorDescriptorListParams {
    // 定义成员变量
    IntArrayRef batch_sizes;
    int64_t seq_length;
    int64_t mini_batch;
    int64_t input_size;
    int64_t batch_sizes_sum;

    // 检查是否输入数据已经打包的函数
    bool is_input_packed() const {
        return batch_sizes.size() != 0;
    }

    // 设置函数，接收输入大小、批次大小数组和是否批次优先作为参数
    void set(IntArrayRef input_sizes, IntArrayRef batch_sizes_, bool batch_first) {
        // 将批次大小数组设置为传入的批次大小数组
        batch_sizes = batch_sizes_;
        // 如果输入数据已经打包
        if (is_input_packed()) {
            // 设置序列长度为批次大小数组的大小
            seq_length = batch_sizes.size();
            // 设置最小批次为批次大小数组的第一个元素
            mini_batch = batch_sizes[0];
            // 设置批次大小总和为输入尺寸的第一个元素
            batch_sizes_sum = input_sizes[0];
            // 设置输入大小为输入尺寸的第二个元素
            input_size = input_sizes[1];
        } else {
            // 如果不是批次优先
            if (batch_first) {
                // 设置序列长度为输入尺寸的第二个元素
                seq_length = input_sizes[1];
                // 设置最小批次为输入尺寸的第一个元素
                mini_batch = input_sizes[0];
            } else {
                // 设置序列长度为输入尺寸的第一个元素
                seq_length = input_sizes[0];
                // 设置最小批次为输入尺寸的第二个元素
                mini_batch = input_sizes[1];
            }
            // 设置输入大小为输入尺寸的第三个元素
            input_size = input_sizes[2];
            // 设置批次大小总和为 -1（未定义）
            batch_sizes_sum = -1;
        }
    }

    // 描述符函数，接收张量 x 作为参数
    std::vector<TensorDescriptor> descriptors(Tensor x) const {
        // 检查输入数据是否已经打包
        auto is_input_packed = batch_sizes.size() != 0;
        // 如果输入数据已经打包
        if (is_input_packed) {
            // 返回 RNN 序列描述符
            return rnn_descriptor_sequence(x, batch_sizes);
        } else {
            // 否则返回 RNN 描述符
            return rnn_descriptor(x[0], seq_length);
        }
    }
};

// 定义结构体 RNNParams
struct RNNParams {
    // 包含 RNN 描述符参数和张量描述符列表参数
    RNNDescriptorParams rnn;
    TensorDescriptorListParams tensors;
};

// 定义结构体 RNNDescriptors
struct RNNDescriptors {
    // 包含 RNN 描述符、输入描述符向量、输出描述符向量、隐藏状态描述符等
    RNNDescriptor rnn_desc;
    std::vector<TensorDescriptor> x_descs;
    std::vector<TensorDescriptor> y_descs;
    TensorDescriptor hx_desc;
    TensorDescriptor hy_desc;
    TensorDescriptor cx_desc;
    TensorDescriptor cy_desc;

    // 构造函数，接收 RNN 参数、miopen 句柄、张量 x、y、隐藏状态 hx、细胞状态 cx 作为参数
    RNNDescriptors(const RNNParams& fn, miopenHandle_t handle, Tensor x, Tensor y, Tensor hx, Tensor cx) {
        // 初始化 RNN 描述符
        rnn_desc = fn.rnn.descriptor();
        // 初始化输入描述符向量
        x_descs = fn.tensors.descriptors(x);
        // 初始化输出描述符向量
        y_descs = fn.tensors.descriptors(y);
        // 初始化隐藏状态描述符
        hx_desc.set(hx, 5);
        // 初始化隐藏状态描述符
        hy_desc.set(hx, 5);
        // 初始化细胞状态描述符
        cx_desc.set(hx, 5);
        // 初始化细胞状态描述符
        cy_desc.set(hx, 5);
    }

    // 获取描述符函数，接收描述符向量 descs 作为参数
    std::vector<miopenTensorDescriptor_t> get_descs(const std::vector<TensorDescriptor>& descs) {
        // 创建 miopenTensorDescriptor_t 类型的描述符向量 r
        std::vector<miopenTensorDescriptor_t> r;
        // 预留描述符向量 r 的大小
        r.reserve(descs.size());
        // 对于每个描述符 desc 在描述符向量 descs 中
        for (auto& desc : descs) {
            // 将描述符 desc 的描述符添加到 r 中
            r.emplace_back(desc.desc());
        }
        // 返回描述符向量 r
        return r;
    }
}
    // 返回 x_descs 的描述符向量，通过调用 get_descs 函数实现
    std::vector<miopenTensorDescriptor_t> get_x_descs() {
        return get_descs(x_descs);
    }
    
    // 返回 y_descs 的描述符向量，通过调用 get_descs 函数实现
    std::vector<miopenTensorDescriptor_t> get_y_descs() {
        return get_descs(y_descs);
    }
};

// 对权重张量进行重新排列以适应 MIOpen 库的要求
Tensor permute_wei_for_miopen(Tensor wei, int64_t mode)
{
    // 如果模式小于2，直接返回原始权重张量
    if (mode < 2)
        return wei;

    Tensor permuted_wei;
    // 如果模式为2，表示 LSTM
    if(mode == 2) { // LSTM
        // 将权重张量按照指定的方式切片
        auto sliced_tensor = wei.chunk(4, 0);
        // 按照特定顺序连接切片后的张量块
        permuted_wei = at::cat({sliced_tensor[0], sliced_tensor[1], sliced_tensor[3], sliced_tensor[2]});
    }
    // 如果模式为3，表示 GRU
    else if(mode == 3) {    // GRU
        // 将权重张量按照指定的方式切片
        auto sliced_tensor = wei.chunk(3, 0);
        // 按照特定顺序连接切片后的张量块
        permuted_wei = at::cat({sliced_tensor[1], sliced_tensor[0], sliced_tensor[2]});
    }
    // 返回重新排列后的权重张量
    return permuted_wei;
}

// 将源参数视图复制到目标参数
void _viewOrCopyParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to, bool copy) {
    // 检查每一层的参数数量是否匹配
    TORCH_CHECK(params_from.size(0) == params_to.size(0), "number of layers mismatch");
    for (const auto i : c10::irange(params_from.size(0))) {
        auto layer_params_from = params_from[i];
        auto layer_params_to = params_to[i];
        // 遍历每一层的参数
        for (auto a = layer_params_from.begin(), b = layer_params_to.begin();
                a != layer_params_from.end() && b != layer_params_to.end();
                ++a, ++b) {
            auto param_from = *a, param_to = *b;
            // 检查参数类型是否匹配
            TORCH_CHECK(param_from.type() == param_to.type(), "parameter types mismatch");
            // 如果需要复制，则执行复制操作，否则执行视图操作
            if (copy) {
                param_to.copy_(param_from.view_as(param_to));
            } else {
                param_from.resize_as_(param_to);
            }
        }
    }
}

// 复制并根据模式重新排列参数
void _copyParams_and_permute(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to, int64_t mode) {
    // 检查每一层的参数数量是否匹配
    TORCH_CHECK(params_from.size(0) == params_to.size(0), "number of layers mismatch");
    for (const auto i : c10::irange(params_from.size(0))) {
        auto layer_params_from = params_from[i];
        auto layer_params_to = params_to[i];
        // 遍历每一层的参数
        for (auto a = layer_params_from.begin(), b = layer_params_to.begin();
                a != layer_params_from.end() && b != layer_params_to.end();
                ++a, ++b) {
            auto param_from = *a, param_to = *b;
            // 检查参数类型是否匹配
            TORCH_CHECK(param_from.type() == param_to.type(), "parameter types mismatch");
            // 重新排列权重并复制到目标参数
            auto tmp = permute_wei_for_miopen(param_from, mode);
            param_to.copy_(tmp.view_as(param_to));
        }
    }
}

// 将源参数复制到目标参数
void _copyParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to) {
    // 调用视图或复制函数，执行复制操作
    _viewOrCopyParams(params_from, params_to, true);
}

// 将源参数视图到目标参数
void _viewParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to) {
    // 调用视图或复制函数，执行视图操作
    _viewOrCopyParams(params_from, params_to, false);
}

// 获取 RNN 的权重数量
int64_t get_num_weights(miopenHandle_t handle, const RNNDescriptor& rnn_desc,
        const TensorDescriptor& x_desc, miopenDataType_t datatype)
{
    size_t weight_size;
    // 查询 RNN 的参数大小
    MIOPEN_CHECK(miopenGetRNNParamsSize(handle, rnn_desc.desc(), x_desc.desc(), &weight_size, datatype));
    // 计算数据类型的每个元素大小
    auto element_size = dataSize(datatype);
    # 使用 TORCH_CHECK 函数检查 weight_size 是否能被 element_size 整除，否则抛出异常信息 "miopenGetRNNParamsSize returned nonsensical weight_size."
    TORCH_CHECK(weight_size % element_size == 0, "miopenGetRNNParamsSize returned nonsensical weight_size.")
    # 返回 weight_size 除以 element_size 的结果，即权重大小除以元素大小的商
    return weight_size / element_size;
}

// 根据不同的 miopen RNN 模式返回线性层数量
int64_t _num_linear_layers(miopenRNNMode_t mode) {
    switch(mode) {
        case miopenLSTM:
            return 8;
        case miopenGRU:
            return 6;
        case miopenRNNRELU:
            return 2;
        case miopenRNNTANH:
            return 2;
        default:
            AT_ERROR("Unknown miopen RNN mode : ", mode);
    }
}

// 获取 RNN 模型的参数及其数量
std::pair<std::vector<Tensor>, size_t> get_parameters(miopenHandle_t handle, const RNNDescriptorParams& rnn,
                    const RNNDescriptor& rnn_desc, const TensorDescriptor& x_desc, const FilterDescriptor& w_desc,
                    const Tensor& weight_buf)
{
    std::vector<Tensor> params;
    int64_t num_linear_layers = _num_linear_layers(rnn.rnn_mode); // 获取线性层数量
    int64_t num_layers = rnn.num_directions() * rnn.num_layers; // 计算总层数量
    size_t cur_offset = 0; // 当前偏移量
    size_t global_layer_params_count = 0; // 全局层参数计数
    auto elem_size = dataSize(getMiopenDataType(weight_buf)); // 获取数据元素大小
    auto bias_mode = rnn.bias_mode; // 获取偏置模式

    } // layer
    return std::make_pair(params, global_layer_params_count); // 返回参数列表及全局层参数计数
}

// 根据输入张量描述参数返回输入大小
std::vector<int64_t> _input_size(const TensorDescriptorListParams& tensors) {
    if (tensors.is_input_packed()) {
        return {tensors.batch_sizes_sum, tensors.input_size}; // 如果输入是打包的，则返回打包后的批次大小与输入大小
    } else {
        return {tensors.seq_length, tensors.mini_batch, tensors.input_size}; // 否则返回序列长度、迷你批次大小和输入大小
    }
}

// 根据 RNN 描述参数及张量描述参数返回隐藏层大小
std::vector<int64_t> _hidden_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
    return {rnn.num_layers * rnn.num_directions(), tensors.mini_batch, rnn.hidden_size}; // 返回隐藏层的大小
}

// 根据 RNN 描述参数及张量描述参数返回输出大小
std::vector<int64_t> _output_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
    if (tensors.is_input_packed()) {
        return {tensors.batch_sizes_sum, rnn.hidden_size * rnn.num_directions()}; // 如果输入是打包的，则返回打包后的批次大小和输出大小
    } else {
        return {tensors.seq_length, tensors.mini_batch, rnn.hidden_size * rnn.num_directions()}; // 否则返回序列长度、迷你批次大小和输出大小
    }
}

// 调用 miopen_rnn 的函数，返回多个张量作为结果
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> miopen_rnn(
        const Tensor& input_r, TensorList weight, int64_t weight_stride0,
        const Tensor& hx, const std::optional<Tensor>& cx_opt,
        int64_t fn_mode, int64_t fn_hidden_size, int64_t fn_num_layers,
        bool batch_first, double fn_dropout, bool fn_train, bool fn_bidirectional,
        IntArrayRef fn_batch_sizes, const std::optional<Tensor>& fn_dropout_state_opt
        ) {
    // See [Note: hacky wrapper removal for optional tensor]
    c10::MaybeOwned<Tensor> cx_maybe_owned = at::borrow_from_optional_tensor(cx_opt); // 将可选的张量 cx_opt 转换为 MaybeOwned 类型
    const Tensor& cx = *cx_maybe_owned; // 获取有效的 cx 张量
    const Tensor& fn_dropout_state = c10::value_or_else(fn_dropout_state_opt, [] {return Tensor();}); // 获取丢弃状态张量或者创建一个空张量

    check_attributes(input_r, weight, {hx, cx}); // 检查输入张量、权重张量及隐藏状态张量的属性
    auto input = input_r; // 复制输入张量

    RNNParams fn; // 定义 RNN 参数对象
    auto datatype = getMiopenDataType(input); // 获取输入张量的数据类型
    miopenRNNBiasMode_t bias_mode = (weight_stride0 == 4) ? miopenRNNwithBias : miopenRNNNoBias; // 根据权重步长确定偏置模式
    fn.rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional, datatype, bias_mode); // 设置 RNN 参数
    fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first); // 设置张量参数
    // 检查当前 RNN 的模式是否为 miopenLSTM，如果不是，要求 cx 未定义，否则抛出异常
    if (fn.rnn.rnn_mode != miopenLSTM) {
        TORCH_CHECK(!cx.defined(), "miopen_rnn: illegal defined cx for non-LSTM RNN.");
    }

    // 检查是否输入数据已经打包，如果 batch_first 为 true 且输入未打包，则转置输入张量
    auto is_input_packed = fn.tensors.batch_sizes.size() != 0;
    if (batch_first && !is_input_packed) {
        input = input.transpose(0, 1);
    }

    // 计算隐藏状态大小和输出大小
    auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
    auto output_size = _output_size(fn.rnn, fn.tensors);

    // 检查 hx 张量是否是连续的，如果不是，抛出异常
    TORCH_CHECK(hx.is_contiguous(), "miopen_rnn : hx is not contiguous.");
    // 检查 cx 是否定义，如果定义则检查其是否是连续的，否则为非 LSTM RNN 设置空张量
    TORCH_CHECK(!cx.defined() || cx.is_contiguous(), "miopen_rnn : cx is not contiguous.");

    // 将输入张量变为连续张量
    auto x = input.contiguous();
    // 创建空的输出张量和隐藏状态张量
    auto output = at::empty(output_size, input.options());
    auto hy = at::empty(hidden_size, hx.options());
    Tensor cy;
    // 如果 cx 已定义，则创建与隐藏状态大小相同的空张量，否则创建大小为 {0} 的空张量
    if (cx.defined()) {
        cy = at::empty(hidden_size, cx.options());
    } else {
        cy = at::empty({0}, hx.options());
    }

    // 将输出张量赋值给 y
    auto y = output;
    // 获取 miopen 句柄和默认的 RNN 算法
    auto handle = getMiopenHandle();
    miopenRNNAlgo_t algo = miopenRNNdefault;
    // 设置 RNN 算法
    fn.rnn.set_algo(algo);

    // 创建 RNN 描述符
    RNNDescriptors descs(fn, handle, x, y, hx, cx);

    // 创建权重描述符和权重缓冲区，初始化权重缓冲区为零
    FilterDescriptor w_desc;
    auto num_weights = get_num_weights(handle, descs.rnn_desc, descs.x_descs[0], datatype);
    auto weight_buf = at::empty(num_weights, x.options());
    w_desc.set(weight_buf, 3);
    weight_buf.zero_();
    
    // 获取 RNN 参数并将其复制到 weight 张量中，根据 fn_mode 决定复制方式
    auto [params, params_stride0] = get_parameters(handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, weight_buf);
    if (fn_mode < 2)
        _copyParams(MatrixRef<Tensor>{weight, static_cast<size_t>(weight_stride0)},
                MatrixRef<Tensor>{params, params_stride0});
    else
        _copyParams_and_permute(MatrixRef<Tensor>{weight, static_cast<size_t>(weight_stride0)},
                    MatrixRef<Tensor>{params, params_stride0}, fn_mode);

    // 检查 cx 是否定义，如果定义则检查其大小是否与隐藏状态大小相等，否则抛出异常
    TORCH_CHECK(!cx.defined() || cx.sizes().equals(hidden_size), "Expected cell size ", IntArrayRef{hidden_size}, ", got", cx.sizes());

    // 计算工作空间大小，为 RNN 分配工作空间
    size_t workspace_size;
    auto x_descs_arr = descs.get_x_descs();
    auto y_descs_arr = descs.get_y_descs();
    MIOPEN_CHECK(miopenGetRNNWorkspaceSize(handle, descs.rnn_desc.desc(), fn.tensors.seq_length, x_descs_arr.data(), &workspace_size));
    auto workspace = at::empty(workspace_size, input.options().dtype(kByte));

    // 初始化保留张量
    Tensor reserve;
    // 如果是训练阶段
    if (fn_train) { //Train.
        // 定义用于保留状态的缓冲区大小变量
        size_t reserver_size;
        // 查询并获取 RNN 训练时需要的状态保留空间大小
        MIOPEN_CHECK(miopenGetRNNTrainingReserveSize(handle, descs.rnn_desc.desc(), fn.tensors.seq_length, x_descs_arr.data(), &reserver_size));
        // 创建一个空的 Tensor 作为状态保留的缓冲区
        reserve = at::empty(reserver_size, input.options().dtype(kByte));
        // 执行 RNN 训练过程的前向计算
        MIOPEN_CHECK(miopenRNNForwardTraining(handle, descs.rnn_desc.desc(), fn.tensors.seq_length,
                x_descs_arr.data(), x.data_ptr(),
                descs.hx_desc.desc(), hx.data_ptr(),
                descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
                w_desc.desc(), weight_buf.data_ptr(),
                y_descs_arr.data(), y.data_ptr(),
                descs.hy_desc.desc(), hy.data_ptr(),
                descs.cy_desc.desc(), cy.defined() ? cy.data_ptr() : nullptr,
                workspace.data_ptr(), workspace_size, reserve.mutable_data_ptr(), reserver_size ));
    } else { //Inference.
        // 对于推断阶段，设置空的状态保留缓冲区
        reserve = at::empty({0}, input.options().dtype(kByte));
        // 执行 RNN 推断过程的前向计算
        MIOPEN_CHECK(miopenRNNForwardInference(handle, descs.rnn_desc.desc(), fn.tensors.seq_length,
                x_descs_arr.data(), x.data_ptr(),
                descs.hx_desc.desc(), hx.data_ptr(),
                descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
                w_desc.desc(), weight_buf.data_ptr(),
                y_descs_arr.data(), y.data_ptr(),
                descs.hy_desc.desc(), hy.data_ptr(),
                descs.cy_desc.desc(), cy.defined() ? cy.data_ptr() : nullptr,
                workspace.data_ptr(), workspace_size));
    }

    // 如果 batch_first 为真且输入未打包
    if (batch_first && !is_input_packed) {
        // 转置输出 Tensor 的第 0 和 1 维度
        output.transpose_(0, 1);
    }

    // 返回输出、最终隐藏状态、最终细胞状态、状态保留、以及权重缓冲区的元组
    return std::make_tuple(output, hy, cy, reserve, weight_buf);
    // 定义函数 miopen_rnn_backward_input，接收多个张量作为输入和输出，并返回多个张量作为结果
    std::tuple<Tensor, Tensor, Tensor, Tensor> miopen_rnn_backward_input(
        const Tensor& input_r,                   // 输入张量 input_r
        const Tensor& weight_buf,                // 权重缓冲张量 weight_buf
        const Tensor& hx,                        // 隐藏状态张量 hx
        const Tensor& cx,                        // 细胞状态张量 cx
        const Tensor& output_r,                  // 输出张量 output_r
        const Tensor& grad_output_r,             // 梯度输出张量 grad_output_r
        const Tensor& grad_hy,                   // 梯度隐藏状态张量 grad_hy
        const Tensor& grad_cy,                   // 梯度细胞状态张量 grad_cy
        int64_t fn_mode,                         // RNN 模式参数 fn_mode
        int64_t fn_hidden_size,                  // 隐藏层大小参数 fn_hidden_size
        int64_t fn_num_layers,                   // RNN 层数参数 fn_num_layers
        bool batch_first,                        // 是否以 batch_first 形式组织数据的布尔值
        double fn_dropout,                       // dropout 率参数 fn_dropout
        bool fn_train,                           // 是否处于训练模式的布尔值 fn_train
        bool fn_bidirectional,                   // 是否双向 RNN 的布尔值 fn_bidirectional
        IntArrayRef fn_batch_sizes,              // batch 尺寸的整数数组引用 fn_batch_sizes
        const Tensor& fn_dropout_state,          // dropout 状态张量 fn_dropout_state
        const Tensor& fn_reserve,                // 预留张量 fn_reserve
        std::array<bool, 3> output_mask          // 输出掩码数组 output_mask
        ) {
    // 复制输入张量到局部变量 input
    auto input = input_r;
    // 复制梯度输出张量到局部变量 grad_output
    auto grad_output = grad_output_r;
    // 复制输出张量到局部变量 output
    auto output = output_r;

    // 定义 RNN 参数对象 fn
    RNNParams fn;
    // 获取输入数据的 Miopen 数据类型
    auto datatype = getMiopenDataType(input);
    // 使用输入数据的大小和 batch 尺寸设置 RNN 参数 fn 的张量
    fn.rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional, datatype, miopenRNNwithBias);
    fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

    // 获取 Miopen 句柄
    auto handle = getMiopenHandle();

    // 如果 RNN 模式不是 LSTM，确保细胞状态张量 cx 未定义
    if(fn.rnn.rnn_mode != miopenLSTM) {
        TORCH_CHECK(!cx.defined(), "rnn: illegal defined cx for non-LSTM RNN");
    }

    // 检查输入是否被打包，如果是 batch_first 形式且未打包，则转置输入、梯度输出和输出张量
    auto is_input_packed = fn_batch_sizes.size() != 0;
    if (batch_first && !is_input_packed) {
        input = input.transpose(0, 1);
        grad_output = grad_output.transpose(0, 1);
        output = output.transpose(0, 1);
    }

    // 获取输入、隐藏状态和输出的大小
    auto input_size = _input_size(fn.tensors);
    auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
    auto output_size = _output_size(fn.rnn, fn.tensors);

    // 检查隐藏状态张量 hx 是否连续
    TORCH_CHECK(hx.is_contiguous(), "rnn: hx is not contiguous");
    // 检查细胞状态张量 cx 是否连续，仅对 LSTM 有效
    TORCH_CHECK(!cx.defined() || cx.is_contiguous(), "rnn: cx is not contiguous");

    // 复制输入、梯度输出和输出张量到局部变量，保证连续性
    auto x = input.contiguous();
    auto dy = grad_output.contiguous();
    auto y = output;
    auto w = weight_buf;
    auto dx = at::empty(input.sizes(), input.options());
    auto dhy = grad_hy.contiguous().view(hidden_size);
    auto dcy = grad_cy.defined() ? grad_cy.contiguous().view(hidden_size) : Tensor();
    auto dhx = at::empty(hidden_size, hx.options());
    // 如果细胞状态张量 cx 已定义，则创建一个与隐藏状态大小相同的空张量，保证连续性
    auto dcx = cx.defined() ? at::empty(hidden_size, cx.options()) : Tensor();

    // 检查是否处于训练模式
    TORCH_CHECK(fn_train, "miopen RNN backward can only be called in training mode");

    // 检查输入大小是否符合预期
    TORCH_CHECK(input.sizes().equals(input_size),
        "Expected input size ", IntArrayRef{input_size}, ", got ", input.sizes());
    // 检查输出大小是否符合预期
    TORCH_CHECK(output.sizes().equals(output_size),
        "Expected output size ", IntArrayRef{output_size}, ", got ", output.sizes());

    // 检查隐藏状态大小是否符合预期
    TORCH_CHECK(!hx.defined() || hx.sizes().equals(hidden_size),
        "Expected hidden size ", IntArrayRef{hidden_size}, ", got ", hx.sizes());
    // 检查细胞状态大小是否符合预期，仅对 LSTM 有效
    TORCH_CHECK(!cx.defined() || cx.sizes().equals(hidden_size),
        "Expected cell size ", IntArrayRef{hidden_size}, ", got ", cx.sizes());
    // 检查是否定义了 dhy 并且其尺寸与 hidden_size 相符，如果不相符则抛出异常
    TORCH_CHECK(!dhy.defined() || dhy.sizes().equals(hidden_size),
        "Expected d_hidden size ", IntArrayRef{hidden_size}, ", got ", dhy.sizes());
    // 检查是否定义了 dcy 并且其尺寸与 hidden_size 相符，如果不相符则抛出异常
    TORCH_CHECK(!dcy.defined() || dcy.sizes().equals(hidden_size),
        "Expected d_cell size ", IntArrayRef{hidden_size}, ", got ", dcy.sizes());

    // 检查 dhy、dy 是否在 CUDA 上，并且如果定义了 dcy，则其也必须在 CUDA 上，否则抛出异常
    TORCH_CHECK(dhy.is_cuda() && dy.is_cuda() && (!dcy.defined() || dcy.is_cuda()),
        "Gradients aren't HIP tensors");

    // 设置 miopenRNN 算法为默认算法
    miopenRNNAlgo_t algo = miopenRNNdefault;
    fn.rnn.set_algo(algo);
    
    // 根据函数 fn、句柄 handle、输入 x、输出 y、初始隐藏状态 hx、初始细胞状态 cx 创建 RNN 描述符
    RNNDescriptors descs(fn, handle, x, y, hx, cx);

    // 设置权重描述符 w_desc，使用 weight_buf 和大小 3
    FilterDescriptor w_desc;
    w_desc.set(weight_buf, 3);

    // 计算所需的 workspace 大小，并创建对应大小的空 Tensor
    size_t workspace_size;
    auto x_descs_arr = descs.get_x_descs();
    auto y_descs_arr = descs.get_y_descs();
    MIOPEN_CHECK(miopenGetRNNWorkspaceSize(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        &workspace_size
        ));
    auto workspace = at::empty(workspace_size, input.options().dtype(kByte));

    // 执行 RNN 反向数据传播计算
    MIOPEN_CHECK(miopenRNNBackwardData(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        y_descs_arr.data(), y.data_ptr(),
        y_descs_arr.data(), dy.data_ptr(),
        descs.hy_desc.desc(), dhy.data_ptr(),
        descs.cy_desc.desc(), cx.defined() ? dcy.data_ptr() : nullptr,
        w_desc.desc(), w.data_ptr(),
        descs.hx_desc.desc(), hx.data_ptr(),
        descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
        x_descs_arr.data(), dx.data_ptr(),
        descs.hx_desc.desc(), dhx.data_ptr(),
        descs.cx_desc.desc(), cx.defined() ? dcx.data_ptr() : nullptr,
        workspace.data_ptr(), workspace.size(0),
        fn_reserve.data_ptr(), fn_reserve.size(0)
        ));

    // 如果 batch_first 为真且输入未打包，则对 dx 进行转置
    if(batch_first && !is_input_packed) {
        dx = dx.transpose_(0, 1);
    }

    // 返回包含 dx、dhx、dcx 和 workspace 的元组
    return std::make_tuple(dx, dhx, dcx, workspace);
}

// 定义 miopen_rnn_backward_weight 函数，用于计算 Miopen RNN 权重的反向传播
std::vector<Tensor> miopen_rnn_backward_weight(
        const Tensor& input_r, TensorList weight_arr, int64_t weight_stride0,
        const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
        const Tensor& output_r,
        int64_t fn_mode, int64_t fn_hidden_size,
        int64_t fn_num_layers, bool batch_first, double fn_dropout,
        bool fn_train, bool fn_bidirectional, IntArrayRef fn_batch_sizes,
        const Tensor& fn_dropout_state, const Tensor& fn_reserve, const Tensor& fn_workspace
        ) {
    // 将 weight_arr 转换为 MatrixRef<Tensor> 类型，使用 weight_stride0 控制步长
    MatrixRef<Tensor> weight{ weight_arr, static_cast<size_t>(weight_stride0) };

    auto input = input_r;  // 复制输入张量
    auto output = output_r;  // 复制输出张量

    RNNParams fn;  // 创建 RNN 参数结构体
    auto datatype = getMiopenDataType(input);  // 获取输入张量的数据类型
    // 根据 weight_stride0 的值确定是否包含偏置项
    miopenRNNBiasMode_t bias_mode = (weight_stride0 == 4) ? miopenRNNwithBias : miopenRNNNoBias;
    fn.rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional, datatype, bias_mode);  // 设置 RNN 参数
    fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);  // 设置 RNN 张量参数

    auto handle = getMiopenHandle();  // 获取 Miopen 的句柄

    // 对于非 LSTM 模式，检查 cx 是否未定义
    if (fn.rnn.rnn_mode != miopenLSTM) {
        TORCH_CHECK(!cx.defined(), "rnn: illegal defined cx for non-LSTM RNN");
    }

    auto is_input_packed = fn_batch_sizes.size() != 0;  // 检查是否对输入进行了打包
    // 如果 batch_first 为真且未对输入进行打包，则转置输入和输出张量
    if (batch_first && !is_input_packed) {
        input = input.transpose(0, 1);
        output = output.transpose(0, 1);
    }

    auto input_size = _input_size(fn.tensors);  // 计算输入大小
    auto hidden_size = _hidden_size(fn.rnn, fn.tensors);  // 计算隐藏状态大小

    // 检查是否处于训练模式
    TORCH_CHECK(fn_train, "miopen RNN backward can only be called in training mode");

    // 检查输入张量的尺寸是否正确
    TORCH_CHECK(input.sizes().equals(input_size),
        "Expected input size ", IntArrayRef{input_size}, ", got ", input.sizes());
    // 检查隐藏状态张量 hx 是否未定义或尺寸是否正确
    TORCH_CHECK(!hx.defined() || hx.sizes().equals(hidden_size),
        "Expected hidden size ", IntArrayRef{hidden_size}, ", got ", hx.sizes());

    // 检查隐藏状态张量 hx 和 cx 是否是连续的
    TORCH_CHECK(hx.is_contiguous(), "rnn: hx is not contiguous");
    TORCH_CHECK(!cx.defined() || cx.is_contiguous(), "rnn: cx is not contiguous");

    auto x = input.contiguous();  // 确保输入张量 x 是连续的
    const auto& y = output;  // 将输出张量 y 定义为常量引用
    auto dw = at::zeros(weight_buf.sizes(), weight_buf.options());  // 创建一个与 weight_buf 尺寸相同的全零张量 dw

    miopenRNNAlgo_t algo = miopenRNNdefault;  // 设置 Miopen RNN 的算法为默认算法
    fn.rnn.set_algo(algo);  // 设置 RNN 算法

    // 创建 RNN 描述符对象
    RNNDescriptors descs(fn, handle, x, y, hx, cx);

    // 设置权重描述符对象
    FilterDescriptor w_desc;
    w_desc.set(weight_buf, 3);

    // 获取输入描述符数组和输出描述符数组
    auto x_descs_arr = descs.get_x_descs();
    auto y_descs_arr = descs.get_y_descs();

    // 执行 Miopen RNN 权重的反向传播计算
    MIOPEN_CHECK(miopenRNNBackwardWeights(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(), x.data_ptr(),
        descs.hx_desc.desc(), hx.data_ptr(),
        y_descs_arr.data(), y.data_ptr(),
        w_desc.desc(), dw.data_ptr(),
        fn_workspace.data_ptr(), fn_workspace.size(0),
        fn_reserve.data_ptr(), fn_reserve.size(0)
        ));

    // 获取梯度参数数组和步长参数
    auto [grad_params_arr, grad_params_stride0] = get_parameters(handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, dw);
    # 检查梯度参数的第一个维度是否等于权重的第一个维度
    if (grad_params_stride0 == static_cast<size_t>(weight_stride0)) {
        # 如果相等，调用_viewParams函数，传入梯度参数和权重数组的引用，然后返回梯度参数数组
        _viewParams(MatrixRef<Tensor>{grad_params_arr, grad_params_stride0},
            MatrixRef<Tensor>{weight_arr, static_cast<size_t>(weight_stride0)});
        return grad_params_arr;
    } else {
        # 如果不相等，创建一个空的Tensor数组grad_weight_arr，预留足够的空间以容纳所有权重的梯度
        std::vector<Tensor> grad_weight_arr;
        grad_weight_arr.reserve( weight.numel() );
        # 遍历权重数组weight_arr，为每个权重创建一个与其大小和选项相同的空Tensor，并加入grad_weight_arr
        for (const auto& w : weight_arr) {
            grad_weight_arr.emplace_back(at::empty(w.sizes(), w.options()));
        }
        # 调用_copyParams函数，传入梯度参数和grad_weight_arr的引用，然后返回grad_weight_arr
        _copyParams(MatrixRef<Tensor>{grad_params_arr, grad_params_stride0},
            MatrixRef<Tensor>{grad_weight_arr, static_cast<size_t>(weight_stride0)});
        return grad_weight_arr;
    }
// 解包隐藏状态张量，返回隐藏状态张量和一个空张量
std::tuple<Tensor, Tensor> unpack_hidden(const Tensor& hidden) {
    return std::make_tuple(hidden, at::Tensor{});  // 返回输入的隐藏状态张量和一个空张量
}

// 解包隐藏状态元组，直接返回输入的隐藏状态元组
std::tuple<Tensor, Tensor> unpack_hidden(const std::tuple<Tensor, Tensor>& hidden) {
    return hidden;  // 返回输入的隐藏状态元组
}

// 根据给定的隐藏类型，将隐藏状态张量打包为隐藏类型对象
template<typename hidden_type>
hidden_type pack_hidden(const Tensor& hx, const Tensor& cx) {
    // 使用静态断言检查 hidden_type 是否为 void 类型，若不是则编译失败并输出错误信息 "pack_hidden not implemented for this type"
    static_assert(std::is_same<hidden_type, void>::value, "pack_hidden not implemented for this type");
    // 抛出一个错误，提示 "NOT IMPLEMENTED"
    AT_ERROR("NOT IMPLEMENTED");
// 特化模板函数，用于将隐藏状态打包成单个 Tensor，对于 LSTM 网络，要求 cx 必须为空
template<>
Tensor pack_hidden<Tensor>(const Tensor& hx, const Tensor& cx) {
    AT_ASSERT(cx.numel() == 0);  // 断言，确保 cx 张量为空
    return hx;  // 返回输入的隐藏状态 hx
}

// 特化模板函数，用于将隐藏状态打包成包含两个 Tensor 的元组
template<>
std::tuple<Tensor, Tensor> pack_hidden<std::tuple<Tensor, Tensor>>(const Tensor& hx, const Tensor& cx) {
    return std::make_tuple(hx, cx);  // 返回包含 hx 和 cx 的元组
}

// miopen 实现函数，用于调用 miopen_rnn 运行 RNN 操作
template<typename hidden_type>
std::pair<Tensor, hidden_type> _miopen_impl(
    const Tensor& input, const Tensor& _batch_sizes, const hidden_type& hidden,
    TensorList params, bool has_biases, miopenRNNMode_t mode,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
    auto [hx, cx] = unpack_hidden(hidden);  // 解包隐藏状态 hidden 得到 hx 和 cx
    int64_t hidden_size = hx.size(2);  // 获取隐藏状态的维度大小

    TORCH_CHECK(_batch_sizes.dim() == 1, "batch_sizes tensor should be 1D");  // 检查 batch_sizes 张量是否为一维

    // 创建 IntArrayRef 对象，用于表示 batch_sizes 张量的引用
    IntArrayRef batch_sizes { _batch_sizes.data_ptr<int64_t>(), static_cast<size_t>(_batch_sizes.size(0)) };

    // 创建空的 dropout_state 张量，与输入张量的选项相同
    Tensor dropout_state = at::empty({0}, input.options());

    // 调用 miopen_rnn 函数执行 RNN 操作
    auto miopen_output = at::miopen_rnn(
        input, params, has_biases ? 4 : 2,
        hx, cx, static_cast<int>(mode), hidden_size, num_layers, /*batch_first=*/false,
        dropout_p, train, bidirectional, batch_sizes, dropout_state);

    // 返回 miopen_output 的第一个元素和打包后的隐藏状态
    return {std::get<0>(miopen_output),
        pack_hidden<hidden_type>(std::get<1>(miopen_output), std::get<2>(miopen_output))};
}

// miopen 实现函数的重载版本，用于支持 batch_first 参数
template<typename hidden_type>
std::pair<Tensor, hidden_type> _miopen_impl(
    const Tensor& input, const hidden_type& hidden,
    TensorList params, bool has_biases, miopenRNNMode_t mode,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
    auto [hx, cx] = unpack_hidden(hidden);  // 解包隐藏状态 hidden 得到 hx 和 cx
    int64_t hidden_size = hx.size(2);  // 获取隐藏状态的维度大小

    // 创建空的 dropout_state 张量，与输入张量的选项相同
    Tensor dropout_state = at::empty({0}, input.options());

    // 调用 miopen_rnn 函数执行 RNN 操作
    auto miopen_output = at::miopen_rnn(
        input, params, has_biases ? 4 : 2,
        hx, cx, static_cast<int>(mode), hidden_size, num_layers, batch_first, dropout_p,
        train, bidirectional, /*batch_sizes=*/{}, dropout_state);

    // 返回 miopen_output 的第一个元素和打包后的隐藏状态
    return {std::get<0>(miopen_output),
        pack_hidden<hidden_type>(std::get<1>(miopen_output), std::get<2>(miopen_output))};
}

// 宏定义，用于简化函数定义，生成具有指定名称和模式的 miopen 函数
#define ONE_HIDDEN_RNN(NAME, MODE)                                             \
void NAME##_miopen(Tensor& output, Tensor& hy,                                 \
      const Tensor& input, const Tensor& hx,                                   \
      TensorList params, bool has_biases,                                      \
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) { \
  // 调用 _miopen_impl 函数执行 RNN 操作
  std::tie(output, hy) = _miopen_impl(input, hx, params, has_biases,           \
      MODE, num_layers, dropout_p, train, bidirectional, batch_first);         \
}                                                                              \
                                                                               \
// 定义一个名为 NAME##_packed_miopen 的函数，接受多个张量作为参数，并调用 _miopen_impl 函数处理这些参数，
// 将输出结果绑定到 output 和 hy 上
void NAME##_packed_miopen(Tensor& output, Tensor& hy,
      const Tensor& data, const Tensor& batch_sizes, const Tensor& hx,
      TensorList params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
  // 调用 _miopen_impl 函数处理输入数据和参数，获取结果并解构赋值给 output 和 hy
  std::tie(output, hy) = _miopen_impl(data, batch_sizes, hx, params,
      has_biases, MODE, num_layers, dropout_p, train, bidirectional);
}

// 将 NAME##_miopen_stub 注册为 CUDA 分发函数，并将其指向 NAME##_miopen 函数
REGISTER_CUDA_DISPATCH(NAME##_miopen_stub, &NAME##_miopen);

// 将 NAME##_packed_miopen_stub 注册为 CUDA 分发函数，并将其指向 NAME##_packed_miopen 函数
REGISTER_CUDA_DISPATCH(NAME##_packed_miopen_stub, &NAME##_packed_miopen);

// 定义宏 ONE_HIDDEN_RNN，分别注册三个不同 RNN 类型的 miopen 实现函数
ONE_HIDDEN_RNN(gru, miopenGRU)
ONE_HIDDEN_RNN(rnn_tanh, miopenRNNTANH)
ONE_HIDDEN_RNN(rnn_relu, miopenRNNRELU)

// 定义一个名为 lstm_miopen 的函数，用于处理 LSTM 模型的推理过程
void lstm_miopen(Tensor& output, Tensor& hy, Tensor& cy,
      const Tensor& input, TensorList hx,
      TensorList params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
    // 调用 _miopen_impl 函数处理输入数据和参数，获取结果并解构赋值给 output、hy 和 cy
    auto result = _miopen_impl(input, std::make_tuple(hx[0], hx[1]), params, has_biases,
        miopenLSTM, num_layers, dropout_p, train, bidirectional, batch_first);
    output = result.first;
    hy = std::get<0>(result.second);
    cy = std::get<1>(result.second);
}

// 定义一个名为 lstm_packed_miopen 的函数，用于处理带有 packed 数据的 LSTM 模型的推理过程
void lstm_packed_miopen(Tensor& output, Tensor& hy, Tensor& cy,
      const Tensor& data, const Tensor& batch_sizes, TensorList hx,
      TensorList params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
    // 调用 _miopen_impl 函数处理输入数据和参数，获取结果并解构赋值给 output、hy 和 cy
    auto result = _miopen_impl(data, batch_sizes, std::make_tuple(hx[0], hx[1]),
        params, has_biases, miopenLSTM, num_layers, dropout_p, train, bidirectional);
    output = result.first;
    hy = std::get<0>(result.second);
    cy = std::get<1>(result.second);
}

// 将 lstm_miopen_stub 注册为 CUDA 分发函数，并将其指向 lstm_miopen 函数
REGISTER_CUDA_DISPATCH(lstm_miopen_stub, &lstm_miopen);

// 将 lstm_packed_miopen_stub 注册为 CUDA 分发函数，并将其指向 lstm_packed_miopen 函数
REGISTER_CUDA_DISPATCH(lstm_packed_miopen_stub, &lstm_packed_miopen);

// 匿名命名空间结束
} // anonymous namespace

// 结束 native 命名空间
}} //namespace native.

// 结束头文件条件编译
#endif
```