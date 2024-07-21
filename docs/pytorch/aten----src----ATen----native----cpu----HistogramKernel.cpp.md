# `.\pytorch\aten\src\ATen\native\cpu\HistogramKernel.cpp`

```py
// 定义宏，仅启用操作符方法的断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含直方图相关的头文件
#include <ATen/native/Histogram.h>

// 包含张量相关的核心头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>

// 如果未定义每个操作符的头文件
#ifndef AT_PER_OPERATOR_HEADERS
// 包含通用操作函数的头文件
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
// 包含特定操作函数的头文件
#include <ATen/ops/aminmax.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like_ops.h>
#endif

// 包含标准库头文件
#include <algorithm>
#include <numeric>
#include <functional>

// 定义 ATen 的本地命名空间
namespace at::native {

// 匿名命名空间，用于实现局部函数或变量
namespace {

// 定义直方图计算的默认分块大小
constexpr int64_t HISTOGRAM_GRAIN_SIZE = 200;
/*
 * The main algorithm. Expects that the input tensor has shape (N, D).
 * Expects that bin_edges contains D one-dimensional tensors, each specifying
 * an increasing sequences of bin edges.
 *
 * Interprets the input as N different D-dimensional coordinates and maps them
 * into the D-dimensional bins defined by bin_edges, accumulating a D-dimensional
 * histogram in the hist tensor.
 *
 * Accepts a template argument of type BIN_SELECTION_ALGORITHM specifying how
 * the scalars in each dimension should be mapped into the dimension's bins:
 *
 *     - LINEAR_INTERPOLATION: each bin edge sequence must form a linear progression.
 *       Scalars are mapped to bins by computing
 *           (element - leftmost_edge)/(rightmost_edge - leftmost_edge) * bin_ct
 *       and truncating the result to an integer.
 *
 *       This is the fastest option, but its results may not be perfectly consistent
 *       with the boundaries specified in bin_edges due to precision issues.
 *
 *       Used by torch.histc, which doesn't need consistency with bin_edges as it does not
 *       return bin_edges. Additionally, this implementation is identical to the legacy histc
 *       implementation, which was replaced when histogram was implemented.
 *
 *     - LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH: Also expects that each bin edge sequence
 *       forms a linear progression. For each scalar, if 'pos' is the bin selected by the
 *       LINEAR_INTERPOLATION approach, this approach inspects the boundaries in bin_edges to
 *       place the scalar into pos - 1, pos, or pos + 1. The "local search" over neighboring
 *       bins allows for correction of misclassifications due to precision issues (a scalar
 *       very close to a bin_edge may be misclassified by LINEAR_INTERPOLATION).
 *
 *       Should produce the same output as the general case BINARY_SEARCH, but run about
 *       3x faster asymptotically.
 *
 *       Used by torch.histogram for cases in which bin_edges is constructed using
 *       torch.linspace. The behavior of LINEAR_INTERPOLATION may not perfectly align
 *       with linspace bin_edges due to precision issues. torch.histogram returns both
 *       the hist and bin_edges tensors as output, so the "local search" is needed to
 *       keep its output internally consistent.
 *
 *     - BINARY_SEARCH: Handles torch.histogram's general case by searching over the
 *       elements of bin_edges. Implemented using std::upper_bound.
 *
 * See discussion at https://github.com/pytorch/pytorch/pull/58780#discussion_r648604866
 * for further details on relative performance of the bin selection algorithms.
 */
enum BIN_SELECTION_ALGORITHM {
    LINEAR_INTERPOLATION,
    LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH,
    BINARY_SEARCH,
};

/*
 * Template function that computes a D-dimensional histogram based on the input data,
 * using specified binning strategies defined by BIN_SELECTION_ALGORITHM.
 *
 * Parameters:
 * - hist: Reference to the output tensor where the histogram will be accumulated.
 * - bin_edges: List of tensors, each containing bin edges for one dimension.
 * - input: Input tensor containing N coordinates in D dimensions to be binned.
 * - weight: Optional tensor representing weights associated with each input coordinate.
 *
 * The function uses template argument 'algorithm' to determine the binning strategy,
 * which affects how input coordinates are mapped to bins defined by bin_edges.
 */
template<typename input_t, BIN_SELECTION_ALGORITHM algorithm>
void histogramdd_cpu_contiguous(Tensor& hist, const TensorList& bin_edges,
        const Tensor& input, const std::optional<Tensor>& weight) {
    // 确保输入张量的维度为2
    TORCH_INTERNAL_ASSERT(input.dim() == 2);

    // 获取输入张量的第一维大小
    const int64_t N = input.size(0);

    // 如果权重张量有值，则确保其维度为1且元素数量与N相同
    if (weight.has_value()) {
        TORCH_INTERNAL_ASSERT(weight.value().dim() == 1 && weight.value().numel() == N);
    }

    // 获取输入张量的第二维大小
    const int64_t D = input.size(1);

    // 确保二进制边界的数量与第二维大小D相等
    TORCH_INTERNAL_ASSERT(int64_t(bin_edges.size()) == D);

    // 对于每个维度，确保二进制边界是连续的，并且直方图的大小加1等于二进制边界的元素数量
    for (const auto dim : c10::irange(D)) {
        TORCH_INTERNAL_ASSERT(bin_edges[dim].is_contiguous());
        TORCH_INTERNAL_ASSERT(hist.size(dim) + 1 == bin_edges[dim].numel());
    }

    // 如果D为0，说明直方图张量为空，直接返回
    if (D == 0) {
        // 在这种情况下，直方图张量是空的，没有需要执行的操作
        return;
    }

    // 使用TensorAccessor创建一个输入张量的常量访问器
    TensorAccessor<const input_t, 2> accessor_in = input.accessor<const input_t, 2>();

    /* 如果可选的权重张量有值，则构造一个包含访问器的std::optional<TensorAccessor> */
    const auto accessor_wt = weight.has_value()
            ? std::optional<TensorAccessor<const input_t, 1>>(weight.value().accessor<const input_t, 1>())
            : std::optional<TensorAccessor<const input_t, 1>>();

    // 创建存储指针的数组，以及存储边界数量和边界值的向量
    std::vector<input_t*> bin_seq(D);
    std::vector<int64_t> num_bin_edges(D);
    std::vector<input_t> leftmost_edge(D), rightmost_edge(D);

    // 对于每个维度，填充存储指针数组和边界信息向量
    for (const auto dim : c10::irange(D)) {
        bin_seq[dim] = bin_edges[dim].data_ptr<input_t>();
        num_bin_edges[dim] = bin_edges[dim].numel();
        leftmost_edge[dim] = bin_seq[dim][0];
        rightmost_edge[dim] = bin_seq[dim][num_bin_edges[dim] - 1];
    }

    // 计算GRAIN_SIZE，用于控制并行处理的粒度
    int64_t GRAIN_SIZE = std::max(int64_t(1), HISTOGRAM_GRAIN_SIZE / D);

    /* 使用at::parallel_for并行处理输入数据。
     * 每个线程将结果累加到thread_histograms的自己的切片中，
     * 最后将这些切片合并到一起。
     */
    const auto num_threads = at::get_num_threads();
    const auto hist_sizes = hist.sizes();
    DimVector thread_hist_sizes(hist_sizes.size() + 1);
    thread_hist_sizes[0] = num_threads;
    std::copy(hist_sizes.begin(), hist_sizes.end(),
              thread_hist_sizes.begin() + 1);
    Tensor thread_histograms = at::zeros(thread_hist_sizes, hist.dtype());
    TORCH_INTERNAL_ASSERT(thread_histograms.is_contiguous());

    // 将各个线程的thread_histograms累加到hist中的指定维度上
    });

    at::sum_out(hist, thread_histograms, /*dim=*/{0});
/* Some pre- and post- processing steps for the main algorithm.
 * 初始化 hist 为 0，调用主算法，并在必要时对输出进行归一化。
 */
template<BIN_SELECTION_ALGORITHM bin_algorithm>
void histogramdd_out_cpu_template(const Tensor& self, const std::optional<Tensor>& weight, bool density,
        Tensor& hist, const TensorList& bin_edges) {
    // 将 hist 张量填充为 0
    hist.fill_(0);

    // 获取 self 张量的最后一个维度大小 N 和除最后一个维度外的所有维度的乘积 M
    const int64_t N = self.size(-1);
    const int64_t M = std::accumulate(self.sizes().begin(), self.sizes().end() - 1,
            (int64_t)1, std::multiplies<int64_t>());

    // 将 self 张量重塑为形状为 {M, N} 的张量
    const Tensor reshaped_input = self.reshape({M, N});

    // 如果 weight 存在，将其重塑为形状为 {M} 的张量，否则为空
    const auto reshaped_weight = weight.has_value()
            ? std::optional<Tensor>(weight.value().reshape({M}))
            : std::optional<Tensor>();

    // 将 bin_edges 的每个元素都连续化，并存储在 bin_edges_contig 中
    std::vector<Tensor> bin_edges_contig(bin_edges.size());
    for (const auto dim : c10::irange(bin_edges_contig.size())) {
        bin_edges_contig[dim] = bin_edges[dim].contiguous();
    }

    // 根据 self 张量的数据类型调度到适当的浮点类型处理函数
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(), "histogram_cpu", [&]() {
        // 调用直方图计算函数 histogramdd_cpu_contiguous，使用指定的算法 bin_algorithm
        histogramdd_cpu_contiguous<scalar_t, bin_algorithm>(
                hist, bin_edges_contig, reshaped_input, reshaped_weight);
    });

    /* Divides each bin's value by the total count/weight in all bins,
     * and by the bin's volume.
     */
    // 如果 density 为 true，则进行归一化处理
    if (density) {
        // 计算 hist 张量所有元素之和
        const auto hist_sum = hist.sum().item();
        // 将 hist 中每个元素除以 hist_sum
        hist.div_(hist_sum);

         /* For each dimension, divides each bin's value
          * by the bin's length in that dimension.
          */
        // 遍历每个维度，将 hist 中每个 bin 的值除以该维度的 bin 长度
        for (const auto dim : c10::irange(N)) {
            // 获取当前维度的 bin_edges 中每个 bin 的长度
            const auto bin_lengths = bin_edges[dim].diff();

            // 用于将 bin_lengths 重塑为与 hist 相应维度对齐的形状
            std::vector<int64_t> shape(N, 1);
            shape[dim] = bin_lengths.numel();

            // 将 hist 中每个 bin 的值除以 bin_lengths
            hist.div_(bin_lengths.reshape(shape));
        }
    }
}

/* The general implementation of the histogram kernel. Maps each element of the input tensor
 * to its corresponding bin by performing a binary search over the elements of bin_edges.
 *
 * Refer to histogramdd_out_cpu_template for more details.
 */
// 直方图核心算法的通用实现。通过对 bin_edges 中的元素执行二分查找，将输入张量的每个元素映射到相应的 bin 中。
static void histogramdd_kernel_impl(const Tensor& self, const std::optional<Tensor>& weight, bool density,
        Tensor& hist, const TensorList& bin_edges) {
    // 调用通用模板函数 histogramdd_out_cpu_template，使用 BINARY_SEARCH 算法
    histogramdd_out_cpu_template<BINARY_SEARCH>(self, weight, density, hist, bin_edges);
}

/* A faster version of the histogram kernel for cases in which bin_edges are known
 * to form a linear progression.
 *
 * Refer to histogramdd_out_cpu_template for more details.
 */
// 在 bin_edges 形成线性序列的情况下，直方图核心算法的快速版本。
static void histogramdd_linear_kernel_impl(const Tensor& self, const std::optional<Tensor>& weight,
        bool density, Tensor& hist, const TensorList& bin_edges, bool local_search) {
    if (local_search) {
        // 如果 local_search 为真，则执行以下代码块：
        // 使用 histogramdd_out_cpu_template 函数，采用 LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH 模板，
        // 将 self, weight, density, hist, bin_edges 作为参数传递，并最终返回 hist 和 bin_edges。
        // 这保证了 hist 和 bin_edges 在输出中保持一致。
        histogramdd_out_cpu_template<LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH>(
              self, weight, density, hist, bin_edges);
    } else {
        // 如果 local_search 不为真，则执行以下代码块：
        // 使用 histogramdd_out_cpu_template 函数，采用 LINEAR_INTERPOLATION 模板，
        // 将 self, weight, density, hist, bin_edges 作为参数传递，并返回 hist。
        // 注意，bin_edges 不会作为输出返回给调用者。
        histogramdd_out_cpu_template<LINEAR_INTERPOLATION>(
              self, weight, density, hist, bin_edges);
    }
}

template<typename scalar_t>
void infer_bin_edges_from_input(const Tensor& input, const int64_t N,
        std::vector<double> &leftmost_edges, std::vector<double> &rightmost_edges) {
    // 从输入张量中计算指定维度（dim=0）的最小值和最大值
    Tensor min, max;
    std::tie(min, max) = aminmax(input, 0);

    // 断言最小值和最大值张量是连续的
    TORCH_INTERNAL_ASSERT(min.is_contiguous() && max.is_contiguous());

    // 获取最小值张量的数据指针，复制到左边缘数组中
    const scalar_t *min_data = min.const_data_ptr<scalar_t>();
    std::copy(min_data, min_data + N, leftmost_edges.begin());

    // 获取最大值张量的数据指针，复制到右边缘数组中
    const scalar_t *max_data = max.const_data_ptr<scalar_t>();
    std::copy(max_data, max_data + N, rightmost_edges.begin());
}

static void histogram_select_outer_bin_edges_impl(const Tensor& input, const int64_t N,
        std::vector<double> &leftmost_edges, std::vector<double> &rightmost_edges) {
    // 根据输入张量的数据类型，分发到对应的模板函数 infer_bin_edges_from_input
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "histogramdd", [&]() {
        infer_bin_edges_from_input<scalar_t>(input, N, leftmost_edges, rightmost_edges);
    });
}

} // namespace

// 将 histogramdd_kernel_impl 函数注册到 histogramdd_stub
REGISTER_DISPATCH(histogramdd_stub, &histogramdd_kernel_impl);
// 将 histogramdd_linear_kernel_impl 函数注册到 histogramdd_linear_stub
REGISTER_DISPATCH(histogramdd_linear_stub, &histogramdd_linear_kernel_impl);
// 将 histogram_select_outer_bin_edges_impl 函数注册到 histogram_select_outer_bin_edges_stub
REGISTER_DISPATCH(histogram_select_outer_bin_edges_stub, &histogram_select_outer_bin_edges_impl);

} // namespace at::native
```