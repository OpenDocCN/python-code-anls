# `.\pytorch\aten\src\ATen\native\Histogram.cpp`

```
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏，用于条件编译
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库中的 Tensor 类定义和 Dispatch 头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

// 包含 ATen 库中的直方图处理和调整大小的相关头文件
#include <ATen/native/Histogram.h>
#include <ATen/native/Resize.h>

// 根据条件编译选择是否包含操作符相关的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_histogramdd_bin_edges.h>
#include <ATen/ops/_histogramdd_bin_edges_native.h>
#include <ATen/ops/_histogramdd_from_bin_cts.h>
#include <ATen/ops/_histogramdd_from_bin_cts_native.h>
#include <ATen/ops/_histogramdd_from_bin_tensors.h>
#include <ATen/ops/_histogramdd_from_bin_tensors_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/histc_native.h>
#include <ATen/ops/histogram_native.h>
#include <ATen/ops/histogramdd_native.h>
#include <ATen/ops/linspace.h>
#endif

// 包含用于数值操作、元组、向量等的标准库头文件
#include <numeric>
#include <tuple>
#include <vector>
#include <functional>

// 包含 C10 库中的 ArrayRef 和 ScalarType 定义头文件
#include <c10/util/ArrayRef.h>
#include <c10/core/ScalarType.h>

// 包含 C10 库中的默认数据类型头文件和范围迭代器头文件
#include <c10/core/DefaultDtype.h>
#include <c10/util/irange.h>


这段代码是一个 C++ 头文件，其中包含了对 ATen（PyTorch C++ 前端）和 C10 库的各种头文件的引用，以及一些条件编译的宏定义。
/* Implements a numpy-like histogramdd function running on cpu
 * https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html
 *
 * See the docstr for torch.histogramdd in torch/functional.py for further explanation.
 *
 * - torch.histogramdd(input, bins, range=None, weight=None, density=False)
 *   input     - tensor with shape (M, N). input is interpreted as M coordinates in N-dimensional space.
 *               If a tensor with more than 2 dimensions is passed, all but the last dimension will be flattened.
 *   bins      - int[] of length N or tensor list of length N. If int[], defines the number of equal-width bins
 *               in each dimension. If tensor list, defines the sequences of bin edges, including rightmost edges,
 *               for each dimension.
 *   range     - float[] of length 2 * N, optional. If specified, defines the leftmost and rightmost bin edges
 *               for each dimension.
 *   weight    - tensor, optional. If provided, weight should have the same shape as input excluding its last dimension.
 *               Each N-dimensional value in input contributes its associated weight towards its bin's result.
 *               If weight is not specified, each value has weight 1 by default.
 *   density   - bool, optional. If false (default), the result will contain the total count (weight) in each bin.
 *               If True, each count (weight) is divided by the total count (total weight), then divided by the
 *               volume of its associated bin.
 *
 * Returns:
 *   hist      - N-dimensional tensor containing the values of the histogram.
 *   bin_edges - tensor list of length N containing the edges of the histogram bins in each dimension.
 *               Bins include their left edge and exclude their right edge, with the exception of the
 *               rightmost bin in each dimension which includes both of its edges.
 *
 * Restrictions are defined in histogram_check_inputs() and in select_outer_bin_edges().
 */

namespace at::native {

DEFINE_DISPATCH(histogramdd_stub);
DEFINE_DISPATCH(histogramdd_linear_stub);
DEFINE_DISPATCH(histogram_select_outer_bin_edges_stub);

namespace {

/* Checks properties of input tensors input, bins, and weight.
 */
void histogramdd_check_inputs(const Tensor& input, const TensorList& bins, const std::optional<Tensor>& weight) {
    // Check that input tensor has at least 2 dimensions
    TORCH_CHECK(input.dim() >= 2, "torch.histogramdd: input tensor should have at least 2 dimensions, but got ", input.dim());

    // Number of dimensions N is determined by the last dimension size of input
    const int64_t N = input.size(-1);

    // Check that number of bins sequences matches the number of dimensions
    TORCH_CHECK(static_cast<int64_t>(bins.size()) == N, "torch.histogramdd: expected ", N, " sequences of bin edges for a ", N,
                "-dimensional histogram but got ", bins.size());

    // Get the data type of the input tensor
    auto input_dtype = input.dtype();
    // 遍历从 0 到 N-1 的维度索引
    for (const auto dim : c10::irange(N)) {
        // 获取当前维度的 bins 引用
        const Tensor& dim_bins = bins[dim];

        // 获取 bins 张量的数据类型
        auto bins_dtype = dim_bins.dtype();
        // 检查输入张量和 bins 张量的数据类型是否相同
        TORCH_CHECK(input_dtype == bins_dtype, "torch.histogramdd: input tensor and bins tensors should",
                " have the same dtype, but got input with dtype ", input_dtype,
                " and bins for dimension ", dim, " with dtype ", bins_dtype);

        // 获取 bins 张量的维度数
        const int64_t dim_bins_dim = dim_bins.dim();
        // 检查 bins 张量的维度数是否为 1
        TORCH_CHECK(dim_bins_dim == 1, "torch.histogramdd: bins tensor should have one dimension,",
                " but got ", dim_bins_dim, " dimensions in the bins tensor for dimension ", dim);

        // 获取 bins 张量的元素个数
        const int64_t numel = dim_bins.numel();
        // 检查 bins 张量的元素个数是否大于 0
        TORCH_CHECK(numel > 0, "torch.histogramdd: bins tensor should have at least 1 element,",
                " but got ", numel, " elements in the bins tensor for dimension ", dim);
    }

    // 如果 weight 存在
    if (weight.has_value()) {
        // 检查输入张量和权重张量的数据类型是否相同
        TORCH_CHECK(input.dtype() == weight.value().dtype(), "torch.histogramdd: if weight tensor is provided,"
                " input tensor and weight tensor should have the same dtype, but got input(", input.dtype(), ")",
                ", and weight(", weight.value().dtype(), ")");

        /* 如果提供了权重张量，我们期望其形状与输入张量相匹配，但排除最内层的维度 N。*/
        // 获取输入张量去除最内层维度 N 后的形状
        auto input_sizes = input.sizes().vec();
        input_sizes.pop_back();

        // 获取权重张量的形状
        auto weight_sizes = weight.value().sizes().vec();
        if (weight_sizes.empty()) {
            // 处理标量的情况
            weight_sizes = {1};
        }

        // 检查权重张量的形状是否与输入张量去除最内层维度后的形状相同
        TORCH_CHECK(input_sizes == weight_sizes, "torch.histogramdd: if weight tensor is provided it should have"
                " the same shape as the input tensor excluding its innermost dimension, but got input with shape ",
                input.sizes(), " and weight with shape ", weight.value().sizes());
    }
}

/* 检查输出张量 hist 和 bin_edges 的属性，然后调整它们的大小。
 */
void histogramdd_prepare_out(const Tensor& input, const std::vector<int64_t>& bin_ct,
        const Tensor& hist, const TensorList& bin_edges) {
    const int64_t N = input.size(-1);

    // 内部断言，确保 bin_ct 和 bin_edges 的大小与输入维度 N 相匹配
    TORCH_INTERNAL_ASSERT((int64_t)bin_ct.size() == N);
    TORCH_INTERNAL_ASSERT((int64_t)bin_edges.size() == N);

    // 检查输入张量和 hist 张量的数据类型是否一致
    TORCH_CHECK(input.dtype() == hist.dtype(), "torch.histogram: input tensor and hist tensor should",
            " have the same dtype, but got input ", input.dtype(), " and hist ", hist.dtype());

    // 遍历每个维度，检查输入张量和对应的 bin_edges 张量的数据类型是否一致，并检查 bin_ct 是否大于 0
    for (const auto dim : c10::irange(N)) {
        TORCH_CHECK(input.dtype() == bin_edges[dim].dtype(), "torch.histogram: input tensor and bin_edges tensor should",
                " have the same dtype, but got input ", input.dtype(), " and bin_edges ", bin_edges[dim].dtype(),
                " for dimension ", dim);

        TORCH_CHECK(bin_ct[dim] > 0,
                "torch.histogram(): bins must be > 0, but got ", bin_ct[dim], " for dimension ", dim);

        // 调整 bin_edges[dim] 的大小，确保有 bin_ct[dim] + 1 个元素
        at::native::resize_output(bin_edges[dim], bin_ct[dim] + 1);
    }

    // 调整 hist 的大小，确保有 bin_ct.size() 个元素
    at::native::resize_output(hist, bin_ct);
}

// 根据 bins 的数量确定 bin_ct，然后调用 histogramdd_prepare_out 函数
void histogramdd_prepare_out(const Tensor& input, TensorList bins,
        const Tensor& hist, const TensorList& bin_edges) {
    // 计算每个 bin 的数量
    std::vector<int64_t> bin_ct(bins.size());
    std::transform(bins.begin(), bins.end(), bin_ct.begin(), [](Tensor t) { return t.numel() - 1; });
    histogramdd_prepare_out(input, bin_ct, hist, bin_edges);
}

/* 确定最外层的 bin edges。为了在调用 aminmax 时简化操作，
 * 假设输入已经被重塑为 (M, N) 的形状。
 */
std::pair<std::vector<double>, std::vector<double>>
select_outer_bin_edges(const Tensor& input, std::optional<c10::ArrayRef<double>> range) {
    // 内部断言，确保输入张量 input 的维度为 2
    TORCH_INTERNAL_ASSERT(input.dim() == 2, "expected input to have shape (M, N)");
    const int64_t N = input.size(-1);

    // 默认的空输入情况下，与 numpy.histogram 的默认行为匹配的范围
    std::vector<double> leftmost_edges(N, 0.);
    std::vector<double> rightmost_edges(N, 1.);

    if (range.has_value()) {
        // 如果指定了范围
        TORCH_CHECK((int64_t)range.value().size() == 2 * N, "torch.histogramdd: for a ", N, "-dimensional histogram",
                " range should have ", 2 * N, " elements, but got ", range.value().size());

        // 根据范围设置左右边界
        for (const auto dim : c10::irange(N)) {
            leftmost_edges[dim] = range.value()[2 * dim];
            rightmost_edges[dim] = range.value()[2 * dim + 1];
        }
    } else if (input.numel() > 0) {
        // 非空输入情况下，调用特定的外层 bin edges 选择函数
        histogram_select_outer_bin_edges_stub(input.device().type(), input, N, leftmost_edges, rightmost_edges);
    }
    // 遍历维度范围的循环
    for (const auto dim : c10::irange(N)) {
        // 获取当前维度的最左边界和最右边界
        double leftmost_edge = leftmost_edges[dim];
        double rightmost_edge = rightmost_edges[dim];

        // 检查边界是否有限，否则报错
        TORCH_CHECK(std::isfinite(leftmost_edge) && std::isfinite(rightmost_edge),
                "torch.histogramdd: dimension ", dim, "'s range [",
                leftmost_edge, ", ", rightmost_edge, "] is not finite");

        // 检查最左边界是否小于等于最右边界，否则报错
        TORCH_CHECK(leftmost_edge <= rightmost_edge, "torch.histogramdd: min should not exceed max, but got",
                " min ", leftmost_edge, " max ", rightmost_edge, " for dimension ", dim);

        // 若最左边界等于最右边界，则扩展范围以匹配 numpy 行为并避免归一化时除以 0
        if (leftmost_edge == rightmost_edge) {
            // 调整最左边界和最右边界，以避免出现空范围
            leftmost_edges[dim] -= 0.5;
            rightmost_edges[dim] += 0.5;
        }
    }

    // 返回左边界和右边界组成的 pair
    return std::make_pair(leftmost_edges, rightmost_edges);
}

/* histc's version of the logic for outermost bin edges.
 */
// 定义函数histc_select_outer_bin_edges，接收输入张量和最小、最大标量，返回左右边界的pair
std::pair<double, double> histc_select_outer_bin_edges(const Tensor& input,
        const Scalar& min, const Scalar& max) {
    // 将最小和最大标量转换为double类型的左右边界
    double leftmost_edge = min.to<double>();
    double rightmost_edge = max.to<double>();

    // 如果左右边界相等且输入张量元素数量大于0
    if (leftmost_edge == rightmost_edge && input.numel() > 0) {
        // 计算输入张量的最小和最大值
        auto extrema = aminmax(input);
        // 更新左右边界为最小和最大值的double表示
        leftmost_edge = std::get<0>(extrema).item<double>();
        rightmost_edge = std::get<1>(extrema).item<double>();
    }

    // 如果左右边界仍然相等，则向左减小1，向右增加1
    if (leftmost_edge == rightmost_edge) {
        leftmost_edge -= 1;
        rightmost_edge += 1;
    }

    // 检查左右边界是否为无穷大或NaN，如果是则抛出错误
    TORCH_CHECK(!(std::isinf(leftmost_edge) || std::isinf(rightmost_edge) ||
            std::isnan(leftmost_edge) || std::isnan(rightmost_edge)),
            "torch.histc: range of [", leftmost_edge, ", ", rightmost_edge, "] is not finite");

    // 检查左边界是否小于右边界，如果不是则抛出错误
    TORCH_CHECK(leftmost_edge < rightmost_edge, "torch.histc: max must be larger than min");

    // 返回左右边界的pair
    return std::make_pair(leftmost_edge, rightmost_edge);
}

} // namespace

// 静态函数allocate_bin_edges_tensors，为输入张量分配空的边界张量
static std::vector<Tensor> allocate_bin_edges_tensors(const Tensor& self) {
    // 检查输入张量维度至少为2
    TORCH_CHECK(self.dim() >= 2, "torch.histogramdd: input tensor should have at least 2 dimensions");
    // 获取最后一个维度的大小N
    const int64_t N = self.size(-1);
    // 创建大小为N的空张量向量bin_edges_out
    std::vector<Tensor> bin_edges_out(N);
    // 对每个维度进行循环
    for (const auto dim : c10::irange(N)) {
        // 在每个维度上创建空的张量，选项与输入张量相同，内存格式为连续
        bin_edges_out[dim] = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    }
    // 返回创建的边界张量向量
    return bin_edges_out;
}

/* Versions of histogramdd in which bins is a Tensor[] defining the sequences of bin edges.
 */
// 静态函数histogramdd_out，其中bins是定义边界序列的张量数组
static Tensor& histogramdd_out(const Tensor& self, TensorList bins,
        const std::optional<Tensor>& weight, bool density,
        Tensor& hist, TensorList& bin_edges) {
    // 检查输入的直方图参数
    histogramdd_check_inputs(self, bins, weight);
    // 准备直方图输出和边界
    histogramdd_prepare_out(self, bins, hist, bin_edges);

    // 将bins数组中的每个张量复制到bin_edges数组中
    for (const auto dim : c10::irange(bins.size())) {
        bin_edges[dim].copy_(bins[dim]);
    }

    // 调用直方图函数的底层实现
    histogramdd_stub(self.device().type(), self, weight, density, hist, bin_edges);
    // 返回计算得到的直方图张量
    return hist;
}

// 函数_histogramdd，其中bins是定义边界序列的张量数组
Tensor _histogramdd(const Tensor& self, TensorList bins,
        const std::optional<Tensor>& weight, bool density) {
    // 创建空的直方图张量hist，选项与输入张量相同，内存格式为连续
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    // 为输入张量分配空的边界张量向量
    std::vector<Tensor> bin_edges_out = allocate_bin_edges_tensors(self);
    // 将边界张量向量转换为张量列表
    TensorList bin_edges_out_tl(bin_edges_out);

    // 调用直方图函数的输出版本，传入参数为self、bins、weight、density、hist和bin_edges_out_tl
    histogramdd_out(self, bins, weight, density, hist, bin_edges_out_tl);
    // 返回计算得到的直方图张量hist
    return hist;
}

/* Versions of histogramdd in which bins is an int[]
 * defining the number of bins in each dimension.
 */
// 静态函数histogramdd_bin_edges_out，其中bins是每个维度中bin数目的整数数组
static std::vector<Tensor>& histogramdd_bin_edges_out(const Tensor& self, IntArrayRef bin_ct,
        std::optional<c10::ArrayRef<double>> range,
        const std::optional<Tensor>& weight, bool density,
        std::vector<Tensor>& bin_edges_out) {
    // 将bin_edges_out转换为张量列表
    TensorList bin_edges_out_tl(bin_edges_out);

    // 获取最后一个维度的大小N
    const int64_t N = self.size(-1);
    // 计算张量 self 中除最后一个维度外所有维度的乘积 M
    const int64_t M = std::accumulate(self.sizes().begin(), self.sizes().end() - 1,
            (int64_t)1, std::multiplies<int64_t>());
    // 将 self 重塑为形状 { M, N }
    Tensor reshaped_self = self.reshape({ M, N });

    // 选择外部边界 bin_edges
    auto outer_bin_edges = select_outer_bin_edges(reshaped_self, range);

    // 获取 bin_ct 的大小，应与 N 相等
    const int64_t bin_size = bin_ct.size();
    // 检查 N 是否与 bin_ct 的大小相等，若不相等则抛出错误信息
    TORCH_CHECK(
        N == bin_size,
        "histogramdd: The size of bins must be equal to the innermost dimension of the input.");
    
    // 对于每个维度 dim 在范围 [0, N) 中
    for (const auto dim : c10::irange(N)) {
        // 在指定范围内创建一个均匀间隔的张量，并将结果存储在 bin_edges_out[dim] 中
        at::linspace_out(bin_edges_out[dim], outer_bin_edges.first[dim], outer_bin_edges.second[dim],
                bin_ct[dim] + 1);
    }

    // 返回 bin_edges_out 张量数组
    return bin_edges_out;
}

// 根据给定的张量和 bin 数量、范围、权重以及密度标志，计算多维直方图的 bin 边界
std::vector<Tensor> histogramdd_bin_edges(const Tensor& self, IntArrayRef bin_ct,
        std::optional<c10::ArrayRef<double>> range,
        const std::optional<Tensor>& weight, bool density) {
    // 为 bin 边界张量分配内存
    std::vector<Tensor> bin_edges_out = allocate_bin_edges_tensors(self);
    // 调用内部函数计算多维直方图的 bin 边界
    return histogramdd_bin_edges_out(self, bin_ct, range, weight, density, bin_edges_out);
}

// 在输出张量中计算多维直方图，将 bin 边界存储在 bin_edges 中
static Tensor& histogramdd_out(const Tensor& self, IntArrayRef bin_ct,
        std::optional<c10::ArrayRef<double>> range,
        const std::optional<Tensor>& weight, bool density,
        Tensor& hist, TensorList& bin_edges) {
    // 获取多维直方图的 bin 边界
    std::vector<Tensor> bins = histogramdd_bin_edges(self, bin_ct, range, weight, density);

    // 检查输入参数的有效性
    histogramdd_check_inputs(self, bins, weight);
    // 准备输出直方图及其对应的 bin 边界
    histogramdd_prepare_out(self, bins, hist, bin_edges);

    // 将计算得到的 bin 边界复制到输出 bin_edges 中
    for (const auto dim : c10::irange(bins.size())) {
        bin_edges[dim].copy_(bins[dim]);
    }

    // 调用线性插值函数计算直方图
    histogramdd_linear_stub(self.device().type(), self, weight, density, hist, bin_edges, true);
    return hist;
}

// 计算多维直方图，返回直方图张量
Tensor _histogramdd(const Tensor& self, IntArrayRef bin_ct,
        std::optional<c10::ArrayRef<double>> range,
        const std::optional<Tensor>& weight, bool density) {
    // 创建空的直方图张量
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    // 为 bin 边界张量分配内存
    std::vector<Tensor> bin_edges_out = allocate_bin_edges_tensors(self);
    // 将 bin 边界张量封装为 TensorList
    TensorList bin_edges_out_tl(bin_edges_out);

    // 计算多维直方图并存储在 hist 中
    histogramdd_out(self, bin_ct, range, weight, density, hist, bin_edges_out_tl);
    return hist;
}

/* Versions of histogram in which bins is a Tensor defining the sequence of bin edges.
 */
// 计算直方图的变体，其中 bins 是定义 bin 边界序列的张量
std::tuple<Tensor&, Tensor&>
histogram_out(const Tensor& self, const Tensor& bins,
        const std::optional<Tensor>& weight, bool density,
        Tensor& hist, Tensor& bin_edges) {
    // 将输入张量重塑为二维张量
    Tensor reshaped_self = self.reshape({ self.numel(), 1 });
    // 如果权重张量存在，则将其重塑为一维张量
    std::optional<Tensor> reshaped_weight = weight.has_value()
        ? weight.value().reshape({ weight.value().numel() }) : weight;
    // 将输入 bin 边界和输出 bin 边界封装为 TensorList
    TensorList bins_in = bins;
    TensorList bins_out = bin_edges;

    // 调用多维直方图计算函数
    histogramdd_out(reshaped_self, bins_in, reshaped_weight, density, hist, bins_out);

    // 返回计算结果的引用
    return std::forward_as_tuple(hist, bin_edges);
}

// 计算直方图，返回直方图张量和 bin 边界张量
std::tuple<Tensor, Tensor>
histogram(const Tensor& self, const Tensor& bins,
        const std::optional<Tensor>& weight, bool density) {
    // 创建空的直方图张量和 bin 边界张量
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    Tensor bin_edges = at::empty({0}, bins.options(), MemoryFormat::Contiguous);
    // 调用直方图计算函数
    return histogram_out(self, bins, weight, density, hist, bin_edges);
}

/* Versions of histogram in which bins is an integer specifying the number of equal-width bins.
 */
// 计算直方图的变体，其中 bins 是指定等宽 bin 数量的整数
std::tuple<Tensor&, Tensor&>
histogram_out(const Tensor& self, int64_t bin_ct, std::optional<c10::ArrayRef<double>> range,
        const std::optional<Tensor>& weight, bool density,
        Tensor& hist, Tensor& bin_edges) {
    // 将输入张量重塑为二维张量
    Tensor reshaped_self = self.reshape({ self.numel(), 1 });
    // 尝试对权重进行重塑，如果权重存在，则将其重塑为一维张量，否则保持不变
    std::optional<Tensor> reshaped_weight = weight.has_value()
        ? weight.value().reshape({ weight.value().numel() }) : weight;
    
    // 复制直方图边界到本地变量 bins_in 和 bins_out
    TensorList bins_in = bin_edges;
    TensorList bins_out = bin_edges;

    // 准备输出参数以进行直方图计算，设置输出直方图 hist 和输出边界 bins_out
    histogramdd_prepare_out(reshaped_self, std::vector<int64_t>{bin_ct}, hist, bins_out);
    
    // 选择外部边界用于直方图计算，并在 bin_edges 中生成等间距的 bin_ct + 1 个边界
    auto outer_bin_edges = select_outer_bin_edges(reshaped_self, range);
    at::linspace_out(bin_edges, outer_bin_edges.first[0], outer_bin_edges.second[0], bin_ct + 1);

    // 检查输入数据的有效性，包括 reshaped_self、bins_in 和 reshaped_weight
    histogramdd_check_inputs(reshaped_self, bins_in, reshaped_weight);

    // 调用直方图计算的线性代理函数，计算直方图并存储在 hist 中
    histogramdd_linear_stub(reshaped_self.device().type(), reshaped_self, reshaped_weight, density, hist, bin_edges, true);
    
    // 返回直方图 hist 和边界 bin_edges 的元组
    return std::forward_as_tuple(hist, bin_edges);
}

std::tuple<Tensor, Tensor>
histogram(const Tensor& self, int64_t bin_ct, std::optional<c10::ArrayRef<double>> range,
        const std::optional<Tensor>& weight, bool density) {
    // 创建一个空的张量用于存储直方图数据
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    // 创建一个空的张量用于存储直方图的边界
    Tensor bin_edges_out = at::empty({0}, self.options());
    // 调用具体的直方图计算函数，将结果返回
    return histogram_out(self, bin_ct, range, weight, density, hist, bin_edges_out);
}

/* Narrowed interface for the legacy torch.histc function.
 */
Tensor& histogram_histc_out(const Tensor& self, int64_t bin_ct,
        const Scalar& min, const Scalar& max, Tensor& hist) {
    // 创建一个空的张量用于存储直方图的边界
    Tensor bin_edges = at::empty({0}, self.options());
    
    // 将输入张量形状变换为 (numel(self), 1)
    Tensor reshaped = self.reshape({ self.numel(), 1 });
    // 准备直方图计算所需的输入边界
    TensorList bins_in = bin_edges;
    TensorList bins_out = bin_edges;
    
    // 准备直方图计算的输出
    histogramdd_prepare_out(reshaped, std::vector<int64_t>{bin_ct}, hist, bins_out);
    
    // 选择外部边界用于直方图计算
    auto outer_bin_edges = histc_select_outer_bin_edges(self, min, max);
    // 在指定范围内生成均匀分布的边界
    at::linspace_out(bin_edges, outer_bin_edges.first, outer_bin_edges.second, bin_ct + 1);
    
    // 检查直方图计算的输入参数是否有效
    histogramdd_check_inputs(reshaped, bins_in, {});
    
    // 执行线性插值的直方图计算
    histogramdd_linear_stub(reshaped.device().type(), reshaped,
            std::optional<Tensor>(), false, hist, bin_edges, false);
    // 返回计算后的直方图
    return hist;
}

Tensor histogram_histc(const Tensor& self, int64_t bin_ct,
        const Scalar& min, const Scalar& max) {
    // 创建一个空的张量用于存储直方图数据
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    // 调用具体的直方图计算函数，并返回结果
    return histogram_histc_out(self, bin_ct, min, max, hist);
}

std::tuple<Tensor, std::vector<Tensor>> histogramdd(
    const Tensor &self, TensorList bins, std::optional<ArrayRef<double>> /*range*/,
    const std::optional<Tensor> &weight, bool density) {
  // 调用内部函数计算多维直方图，返回结果及其边界
  auto hist = at::_histogramdd_from_bin_tensors(self, bins, weight, density);
  return std::tuple<Tensor, std::vector<Tensor>>{
      std::move(hist), bins.vec()};
}

std::tuple<Tensor, std::vector<Tensor>> histogramdd(
    const Tensor &self, IntArrayRef bins, std::optional<ArrayRef<double>> range,
    const std::optional<Tensor> &weight, bool density) {
  // 调用内部函数计算多维直方图的边界
  auto bin_edges = at::_histogramdd_bin_edges(self, bins, range, weight, density);
  // 调用内部函数计算多维直方图，返回结果及其边界
  auto hist = at::_histogramdd_from_bin_cts(self, bins, range, weight, density);
  return std::tuple<Tensor, std::vector<Tensor>>{
      std::move(hist), std::move(bin_edges)};
}

std::tuple<Tensor, std::vector<Tensor>> histogramdd(
    const Tensor &self, int64_t bins, std::optional<ArrayRef<double>> range,
    const std::optional<Tensor> &weight, bool density) {
  // 创建维度向量，用于指定每个维度的直方图区间数量
  DimVector bins_v(self.size(-1), bins);
  // 调用内部函数计算多维直方图，返回结果及其边界
  return at::native::histogramdd(self, bins_v, range, weight, density);
}

} // namespace at::native
```