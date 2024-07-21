# `.\pytorch\torch\csrc\jit\runtime\register_prim_ops_fulljit.cpp`

```
// 包含 Torch 库的头文件，用于 JIT 编译器和运行时操作的注册和工具函数
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>

// 包含 ATen 库的头文件，用于处理 IValue、时间计算和范围计算等工具
#include <ATen/core/ivalue.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/irange.h>

// 包含 Torch 自动求导和性能分析的相关头文件
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/jit/frontend/tracer.h>

// 包含标准 C++ 库的相关头文件
#include <algorithm>
#include <bitset>
#include <cctype>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Torch JIT 命名空间
namespace torch::jit {

// 匿名命名空间，用于隐藏实现细节和局部变量
namespace {

// 注册自定义操作符到 Torch JIT
RegisterOperators reg({
    // 注册 prim::profile 操作符，用于运行时性能分析
    Operator(
        prim::profile,
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            // 抛出错误信息，提示必须降低到 Interpreter 的 PROFILE 指令
            AT_ERROR(
                "Must be lowered to Interpreter's PROFILE instruction"); // NOLINT
          };
        },
        aliasAnalysisSpecialCase()),

    // 注册 prim::profile_ivalue 操作符，用于 IValue 的运行时性能分析
    Operator(
        prim::profile_ivalue,
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            // 抛出错误信息，提示必须降低到 Interpreter 的 PROFILE 指令
            AT_ERROR(
                "Must be lowered to Interpreter's PROFILE instruction"); // NOLINT
          };
        },
        aliasAnalysisSpecialCase()),

    // 注册 prim::FusionGroup 操作符，用于运行时的融合组操作
    Operator(
        prim::FusionGroup,
        [](const Node* node) -> Operation {
          // 在函数运行期间记录融合组操作的功能，用空的 IValue 向量作为参数
          const auto key = registerFusion(node);
          return [key](Stack& stack) {
            RECORD_FUNCTION("FusionGroup", std::vector<c10::IValue>());
            // 运行注册的融合组操作
            runFusion(key, stack);
          };
        },
        aliasAnalysisSpecialCase()),

    // 注册 prim::RequiresGradCheck 操作符，用于检查张量是否需要梯度
    Operator(
        prim::RequiresGradCheck /* (...)  -> (..., bool) */,
        [](const Node* node) -> Operation {
          // 从节点中提取类型，并返回是否需要梯度的布尔值向量
          std::vector<bool> rg_props =
              fmap(node->tys(attr::types), [](const TypePtr& t) {
                // 如果梯度属性变化，假设张量确实需要梯度，设置在 guardDifferentiableGraph 中
                TORCH_INTERNAL_ASSERT(
                    t->castRaw<TensorType>()->requiresGrad().has_value());
                return *t->castRaw<TensorType>()->requiresGrad();
              });
          return [rg_props](Stack& stack) {
            auto num_inputs = rg_props.size();
            // 检查每个输入张量的形状是否与预期的形状匹配
            for (const auto i : c10::irange(num_inputs)) {
              auto& input = peek(stack, i, num_inputs);
              const auto& t = input.toTensor();
              if (rg_props[i] != t.requires_grad()) {
                push(stack, false);
                return;
              }
            }

            push(stack, true);
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        prim::ConstantChunk,
        [](const Node* node) -> Operation {
          // 从节点属性中获取要分块的数量和维度
          int64_t chunks = node->i(attr::chunks);
          int64_t dim = node->i(attr::dim);
          // 检查每个输出是否被使用，并存储结果
          auto outputs_used = fmap(node->outputs(), [](const Value* v) {
            return !v->uses().empty();
          });
          // 返回一个 lambda 函数，实现 chunk 操作
          return [=](Stack& stack) {
            // 记录函数调用信息，标记为 chunk
            RECORD_FUNCTION("chunk", last(stack, 1));

            // 从栈中弹出张量 t
            at::Tensor t;
            pop(stack, t);
            // 对张量 t 进行 chunk 操作，得到结果
            auto result = at::chunk(t, chunks, dim);
            // 将结果插入到栈的末尾
            stack.insert(
                stack.end(),
                std::make_move_iterator(result.begin()),
                std::make_move_iterator(result.end()));
            // 注意：chunk 有时可能返回比期望少的输出
            int64_t num_results = result.size();
            if (num_results != chunks) {
              // 如果实际输出数量不等于期望的数量
              if (num_results > chunks) {
                // 如果实际输出多于期望，报错
                TORCH_CHECK(
                    num_results == chunks,
                    "Expected chunk to return ",
                    chunks,
                    " outputs, but got ",
                    num_results);
              }
              // 对于实际输出少于期望的情况，处理未使用的输出
              for (const auto i : c10::irange(num_results, chunks)) {
                TORCH_CHECK(
                    !outputs_used[i],
                    "Expected chunk to return at least ",
                    chunks,
                    " outputs, but got only ",
                    num_results);
                // 由于输出未被使用，可以推送任意内容到栈上
                stack.emplace_back();
              }
            }
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        prim::ChunkSizes,
        [](const Node* node) -> Operation {
          // 获取节点属性中的维度和分块数量
          int64_t raw_dim = node->i(attr::dim);
          int64_t chunks = node->i(attr::chunks);
          // 返回一个 lambda 函数，实现 ChunkSizes 操作
          return [raw_dim, chunks](Stack& stack) {
            // 从栈中弹出张量的形状信息
            c10::List<int64_t> shape = pop(stack).toIntList();
            // 复制形状信息以备后用
            c10::List<int64_t> regular_shape = shape.copy();
            c10::List<int64_t> last_shape = shape.copy();
            // 确定有效的维度索引
            int64_t dim = at::maybe_wrap_dim(raw_dim, shape.size());
            // 检查维度是否在合理范围内
            TORCH_CHECK(
                dim < (int64_t)regular_shape.size(),
                "Dimension out of range for chunk");
            // 计算分块后每块的大小
            int64_t split_size = (regular_shape[dim] + chunks - 1) / chunks;
            regular_shape[dim] = split_size;
            // 如果形状在维度上可以被均分
            if (shape[dim] % chunks == 0) {
              last_shape[dim] = split_size;
            } else {
              // 如果不能均分，计算最后一块的大小
              int64_t num_splits = std::max<int64_t>(
                  (shape[dim] + split_size - 1) / split_size, 1);
              last_shape[dim] =
                  split_size - (split_size * num_splits - shape[dim]);
              AT_ASSERT(last_shape[dim] >= 0);
            }
            // 将计算结果推送到栈上
            push(stack, std::move(regular_shape));
            push(stack, std::move(last_shape));
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)",
        [](Stack& stack) {
          RECORD_FUNCTION("_grad_sum_to_size", std::vector<c10::IValue>());
          IValue self, size;
          pop(stack, self, size);
          if (size.isNone()) {
            // 如果 size 是 None，则直接将 self 压回栈顶
            push(stack, std::move(self));
          } else {
            // 否则，调用 at::sum_to 函数计算 self 对应尺寸的和，并将结果压入栈顶
            push(stack, at::sum_to(self.toTensor(), size.toDimVector()));
          }
        },
        aliasAnalysisFromSchema()),
    // This operator is generated inside the compiler for indexing into
    // ModuleDict without a statically determinable key. Accordingly,
    // self must be a ModuleType and the output must be an InterfaceType.
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "prim::ModuleContainerIndex.dict(Any self, str ind) -> Any"),
        [](Stack& stack) {
          // 从栈中弹出索引值 ind 和 ModuleDict 对象 module_dict
          IValue ind = pop(stack);
          IValue module_dict = pop(stack);
          // 通过 ind 获取 module_dict 中对应的属性，并将结果压入栈顶
          push(stack, module_dict.toModule().attr(ind.toStringRef()));
        },
        aliasAnalysisFromSchema()),
    Operator(
        prim::TypeCheck /* (...)  -> (..., bool) */,
        [](const Node* /* node */) -> Operation {
          return [](Stack& /* stack */) {
            // 抛出错误，暂时未实现 prim::TypeCheck 操作
            AT_ERROR("prim::TypeCheck not yet implemented"); // NOLINT
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        prim::FallbackGraph,
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            // 抛出错误，要求将 prim::FallbackGraph 转换为 prim::FunctionCall
            AT_ERROR(
                "Must be converted to prim::FunctionCall by replaceFallbackGraphWithFallbackFunction"); // NOLINT
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        "prim::Guard(Tensor(a) t) -> Tensor(a)",
        [](Stack& stack) { AT_ERROR("Should be replaced by prim::BailOut"); },
        aliasAnalysisFromSchema()),
    Operator(
        "prim::BailOut(...) -> Tensor(a)",
        [](Stack& /* stack */) {
          // 抛出错误，暂时未实现 prim::BailOut 操作
          AT_ERROR("prim::BailOut not yet implemented"); // NOLINT
        },
        aliasAnalysisFromSchema()),
    Operator(
        "prim::BailoutTemplate() -> int",
        [](Stack& stack) {
          // TODO: 今天，我们在前面放置了一个单一的 bailout 模板，
          // 用于携带未优化的图形以供 bailout 节点使用。理想情况下，这
          // 应该永远不会运行，但我们还没有编写删除它的代码。
          // TORCH_INTERNAL_ASSERT(false);

          // 将整数 1 压入栈顶，以便进行图形遍历
          push(stack, 1);
        },
        aliasAnalysisFromSchema()),
    // 定义名为 "aten::grad" 的运算符，接受输出、输入、梯度输出、保留计算图标志、是否创建计算图、是否允许未使用张量的参数
    Operator(
        "aten::grad(Tensor[] outputs, Tensor[] inputs, Tensor?[]? grad_outputs=None, bool? retain_graph=None, bool create_graph=False, bool allow_unused=False) -> Tensor?[]",
        [](Stack& stack) {
          // 从堆栈中弹出并转换是否允许未使用张量标志
          bool allow_unused = pop(stack).toBool();
          // 从堆栈中弹出并转换是否创建计算图标志
          bool create_graph = pop(stack).toBool();
          // 从堆栈中弹出并转换可选的保留计算图标志
          auto retain_graph = pop(stack).toOptional<bool>();
          // 从堆栈中弹出梯度输出
          auto grad_outputs = pop(stack);
          // 从堆栈中弹出输入张量列表
          auto inputs = pop(stack).toTensorList();
          // 从堆栈中弹出输出张量列表
          auto outputs = pop(stack).toTensorList();
          // 创建输入变量的变量向量
          std::vector<torch::autograd::Variable> input_vars(
              inputs.begin(), inputs.end());
          // 创建输出变量的变量向量
          std::vector<torch::autograd::Variable> output_vars(
              outputs.begin(), outputs.end());
          // 创建梯度变量的变量向量
          std::vector<torch::autograd::Variable> gradients;
    
          // 如果梯度输出不为空
          if (!grad_outputs.isNone()) {
            // 遍历梯度输出的列表引用
            for (const IValue& v : grad_outputs.toListRef()) {
              // 如果当前值是空，则添加一个空的张量；否则转换为张量并添加
              gradients.emplace_back(v.isNone() ? at::Tensor() : v.toTensor());
            }
          }
    
          // 调用 PyTorch 的自动求导函数 grad
          auto res = torch::autograd::grad(
              output_vars,
              input_vars,
              gradients,
              retain_graph,
              create_graph,
              allow_unused);
    
          // 创建包含可选张量类型的通用列表 res_list
          c10::impl::GenericList res_list{OptionalType::ofTensor()};
          // 遍历结果张量列表 res
          for (const at::Tensor& t : res) {
            // 如果张量已定义，则添加到列表中；否则添加空值
            res_list.emplace_back(t.defined() ? t : IValue());
          }
          // 将结果列表推入堆栈
          push(stack, res_list);
        },
        // 使用从架构推断的别名分析
        aliasAnalysisFromSchema()),
    
    // 注意：反向操作可能会写入计算图中每个输入张量，分析叶节点更加昂贵，有时会保留整个梯度在 Autograd 图中的每个张量上，所以我们对这两个操作使用 aliasAnalysisConservative
    // 定义名为 "aten::backward.TensorList" 的运算符，接受张量列表、梯度张量列表、保留计算图标志、是否创建计算图标志
    Operator(
        "aten::backward.TensorList(Tensor[] tensors, Tensor?[]? grad_tensors=None, bool? retain_graph=None, bool create_graph=False) -> ()",
        [](Stack& stack) {
          // 从堆栈中弹出并转换是否创建计算图标志
          bool create_graph = pop(stack).toBool();
          // 从堆栈中弹出并转换可选的保留计算图标志
          auto retain_graph = pop(stack).toOptional<bool>();
          // 从堆栈中弹出梯度张量列表
          auto grad_tensors = pop(stack);
          // 从堆栈中弹出输出张量列表
          auto outputs = pop(stack).toTensorList();
          // 创建输出变量的变量向量
          std::vector<torch::autograd::Variable> output_vars(
              outputs.begin(), outputs.end());
          // 创建梯度变量的变量向量
          std::vector<torch::autograd::Variable> gradients;
    
          // 如果梯度张量列表不为空
          if (!grad_tensors.isNone()) {
            // 遍历梯度张量列表的引用
            for (const IValue& v : grad_tensors.toListRef()) {
              // 如果当前值是空，则添加一个空的张量；否则转换为张量并添加
              gradients.emplace_back(v.isNone() ? at::Tensor() : v.toTensor());
            }
          }
    
          // 调用 PyTorch 的自动求导函数 backward
          torch::autograd::backward(
              output_vars, gradients, retain_graph, create_graph);
        },
        // 使用保守的别名分析策略
        aliasAnalysisConservative()),
    Operator(
        "aten::save(t item, str filename) -> ()",
        [](Stack& stack) {
          // 从栈中弹出文件名，并转换为字符串引用
          auto filename = pop(stack).toStringRef();
          // 从栈中弹出要保存的值
          auto ivalue = pop(stack);

          // 使用 JIT 将值序列化为 pickle 格式的数据
          auto data = jit::pickle_save(ivalue);

          // 打开文件流，以二进制写模式写入数据
          std::fstream output(filename, std::ios::out | std::ios::binary);
          output.write(data.data(), data.size());
        },
        aliasAnalysisFromSchema()),

    Operator(
        "prim::IgnoredPythonOp(...) -> None",
        [](Stack& stack) {
          // 抛出 JIT 异常，指示此 Python 函数被忽略
          throw JITException(
              "This Python function is annotated to be ignored"
              " and cannot be and has not been included in the exported"
              " binary, meaning that it cannot be executed now."
              " Make sure that ignored operations are never executed after"
              " import");
        },
        aliasAnalysisFromSchema()),

    Operator(
        "aten::wait(Future(t) self) -> t",
        [](Stack& stack) {
          // 抛出 TORCH_CHECK 异常，表示等待操作在解释器中直接实现
          TORCH_CHECK(false, "wait is implemented directly in the interpreter");
        },
        aliasAnalysisSpecialCase()),

    Operator(
        "prim::awaitable_wait(Await(t) self) -> t",
        [](Stack& stack) {
          // 获取当前栈顶的 awaitable 对象
          auto aw = stack.back().toAwait();
          // 等待该 awaitable 对象完成
          aw->wait();
          // 弹出栈顶对象，将完成的值推入栈顶
          stack.pop_back();
          stack.emplace_back(aw->value());
        },
        aliasAnalysisSpecialCase()),

    Operator(
        "prim::awaitable_nowait(t self) -> Await(t)",
        [](Stack& stack) {
          // 创建一个新的 awaitable 对象，用栈顶的值标记为完成
          auto aw = c10::make_intrusive<c10::ivalue::Await>(stack.back().type());
          aw->markCompleted(pop(stack));
          // 将新创建的 awaitable 对象推入栈顶
          push(stack, std::move(aw));
        },
        aliasAnalysisSpecialCase()),
});

// 注册自定义运算符 logging_operators
RegisterOperators logging_operators(
    {
        Operator(
            "prim::AddStatValue(str key, int val) -> ()",
            [](Stack& stack) {
                // 从堆栈中弹出整数值
                auto val = pop(stack).toInt();
                // 从堆栈中弹出字符串键
                auto key = pop(stack).toString();

                // 解析操作的模式匹配模式
                auto schema =
                    parseSchema("prim::AddStatValue(str key, int val) -> ()");

                // 如果正在追踪，则创建节点并添加输入
                if (jit::tracer::isTracing()) {
                    const auto& graph = tracer::getTracingState()->graph;
                    Node* node = graph->create(prim::AddStatValue, /*num_outputs=*/0);
                    tracer::recordSourceLocation(node);
                    node->addInput(insertConstant(*graph, key));
                    tracer::addInputs(node, "val", val);
                    graph->insertNode(node);
                }

                // 将键值对添加到日志记录器
                torch::jit::logging::getLogger()->addStatValue(*key, val);
            },
            aliasAnalysisFromSchema()),
        Operator(
            "prim::TimePoint() -> int",
            [](Stack& stack) {
                // 解析操作的模式匹配模式
                auto schema = parseSchema("prim::TimePoint() -> int");
                Node* node = nullptr;

                // 如果正在追踪，则创建节点
                if (jit::tracer::isTracing()) {
                    const auto& graph = tracer::getTracingState()->graph;
                    node = graph->create(prim::TimePoint, /*num_outputs=*/0);
                    tracer::recordSourceLocation(node);
                    graph->insertNode(node);
                }

                // 获取当前时间戳，并将其推入堆栈
                auto output = c10::getTime(/*allow_monotonic=*/true);
                push(stack, output);

                // 如果正在追踪，则将输出添加到节点
                if (jit::tracer::isTracing()) {
                    jit::tracer::addOutput(node, output);
                }
            },
            aliasAnalysisFromSchema())
    });

// 未使用的函数，用于计算值的哈希值
C10_UNUSED void hashValue(Stack& stack) {
    auto value = pop(stack);
    push(stack, value.hash());
}

// 参考：torch/nn/functional.py 中的 _output_size 函数
// size 可以是 None、int 或 intlist，scale_factors 可以是 None、float 或 floatlist
std::vector<int64_t> _output_size(
    const at::Tensor& input,
    size_t dim,
    const IValue& size,
    const IValue& scale_factors) {
    
    // 如果 size 不是 None，则根据类型返回相应的向量
    if (!size.isNone()) {
        if (size.isInt()) {
            std::vector<int64_t> repeated(dim, size.toInt());
            return repeated;
        } else {
            return size.toIntVector();
        }
    }
    
    // 如果 scale_factors 是 double 类型，则生成相应维度的重复向量
    std::vector<double> scale_repeated;
    if (scale_factors.isDouble()) {
        scale_repeated = std::vector<double>(dim, scale_factors.toDouble());
    } else {
        scale_repeated = scale_factors.toDoubleVector();
    }
    
    // 计算并返回输出大小的向量
    std::vector<int64_t> ret;
    for (const auto i : c10::irange(dim)) {
        ret.push_back(std::floor(input.size(i + 2) * scale_repeated[i]));
    }
    return ret;
}

// 如果 v 是实数 float，则返回 true，否则返回 false
bool _is_floating_value(double v) {
    return std::floor(v) != v;
}

// 参考：torch/nn/functional.py 中的 interpolate 函数
// size 可以是 None、int 或 intlist，scale_factors 可以是 None、float 或 floatlist
at::Tensor interpolate(
    // 定义一个函数，接受多个参数：输入张量、尺寸、缩放因子、插值模式、对齐角点选项、重新计算缩放因子选项
    void interpolate(const at::Tensor& input,
                     const IValue& size,
                     const IValue& scale_factors,
                     const std::string& mode,
                     std::optional<bool> align_corners,
                     std::optional<bool> recompute_scale_factor) {
      // 如果插值模式是 "nearest" 或 "area"
      if ((mode == "nearest" || mode == "area")) {
        // 如果指定了 align_corners 参数，抛出运行时错误
        if (align_corners != c10::nullopt) {
          throw std::runtime_error(
              "align_corners option can only be set with the "
              "interpolating modes: linear | bilinear | bicubic | trilinear");
        }
      } else {
        // 如果插值模式不是 "nearest" 或 "area"
        // 如果 align_corners 参数未指定
        if (align_corners == c10::nullopt) {
          // 发出警告，说明默认的上采样行为已更改
          TORCH_WARN(
              "Default upsampling behavior when mode=",
              mode,
              " is changed "
              "to align_corners=False since 0.4.0. Please specify align_corners=True "
              "if the old behavior is desired. See the documentation of nn.Upsample for details");
          // 设置 align_corners 为 false
          align_corners = false;
        }
      }
    
      // 初始化缩放因子的三个变量
      double scale_factors_1 = -1.0;
      double scale_factors_2 = -1.0;
      double scale_factors_3 = -1.0;
    
      // 如果 scale_factors 不是 None 并且未指定 recompute_scale_factor
      if (!scale_factors.isNone() && recompute_scale_factor == c10::nullopt) {
        // 设置 recompute_scale_factor 为 true
        recompute_scale_factor = true;
        // 初始化一个警告标志
        bool warn_recompute_scale_factor = false;
    
        // 如果 scale_factors 是一个 double
        if (scale_factors.isDouble()) {
          // 当缩放因子具有浮点值时发出警告，因为对于整数，无论是否重新计算缩放因子，结果都相同
          if (_is_floating_value(scale_factors.toDouble())) {
            warn_recompute_scale_factor = true;
          }
        } else if (scale_factors.isDoubleList()) {
          // 如果 scale_factors 是一个双精度浮点数列表
          auto scale_factors_list = scale_factors.toDoubleList();
    
          // 遍历列表中的每个缩放因子
          for (const auto& scales : scale_factors_list) {
            // 当缩放因子具有浮点值时发出警告
            if (_is_floating_value(scales)) {
              warn_recompute_scale_factor = true;
              break;
            }
          }
        }
    
        // 如果需要警告重新计算缩放因子
        if (warn_recompute_scale_factor) {
          // 发出警告，说明在 1.5.0 版本中，默认行为将更改以与其他框架/库一致，并直接使用缩放因子而不是依赖计算的输出尺寸
          TORCH_WARN(
              "The default behavior for interpolate/upsample with float scale_factor will change "
              "in 1.5.0 to align with other frameworks/libraries, and use scale_factor directly, "
              "instead of relying on the computed output size. "
              "If you wish to keep the old behavior, please set recompute_scale_factor=True. "
              "See the documentation of nn.Upsample for details.");
        }
      }
    
      // 如果不重新计算缩放因子
      if (recompute_scale_factor == false) {
        // 如果 scale_factors 是一个 double
        if (scale_factors.isDouble()) {
          // 将三个缩放因子都设置为相同的值
          scale_factors_1 = scale_factors.toDouble();
          scale_factors_2 = scale_factors.toDouble();
          scale_factors_3 = scale_factors.toDouble();
        } else if (scale_factors.isDoubleList()) {
          // 如果 scale_factors 是一个双精度浮点数列表
          auto scale_factors_list = scale_factors.toDoubleList();
          // 设置第一个缩放因子
          scale_factors_1 = scale_factors_list[0];
          // 如果列表中有第二个缩放因子
          if (scale_factors_list.size() >= 2) {
            scale_factors_2 = scale_factors_list[1];
            // 如果列表中有第三个缩放因子
            if (scale_factors_list.size() >= 3) {
              scale_factors_3 = scale_factors_list[2];
            }
          }
  }
  // 结束函数定义

  const auto dim1d = 3;
  const auto dim2d = 4;
  const auto dim3d = 5;

  auto input_dim = input.dim();
  // 获取输入张量的维度

  if (input_dim == dim1d && mode == "nearest")
    // 如果输入是一维且模式是最近邻插值
    return at::upsample_nearest1d(
        input,
        _output_size(input, 1, size, scale_factors),
        c10::make_optional(scale_factors_1));
  // 调用一维最近邻插值函数，并返回结果

  if (input_dim == dim2d && mode == "nearest")
    // 如果输入是二维且模式是最近邻插值
    return at::upsample_nearest2d(
        input,
        _output_size(input, 2, size, scale_factors),
        scale_factors_1,
        scale_factors_2);
  // 调用二维最近邻插值函数，并返回结果

  if (input_dim == dim3d && mode == "nearest")
    // 如果输入是三维且模式是最近邻插值
    return at::upsample_nearest3d(
        input,
        _output_size(input, 3, size, scale_factors),
        scale_factors_1,
        scale_factors_2,
        scale_factors_3);
  // 调用三维最近邻插值函数，并返回结果

  if (input_dim == dim1d && mode == "area")
    // 如果输入是一维且模式是区域插值
    return at::adaptive_avg_pool1d(
        input, _output_size(input, 1, size, scale_factors));
  // 调用一维自适应平均池化函数，并返回结果

  if (input_dim == dim2d && mode == "area")
    // 如果输入是二维且模式是区域插值
    return at::adaptive_avg_pool2d(
        input, _output_size(input, 2, size, scale_factors));
  // 调用二维自适应平均池化函数，并返回结果

  if (input_dim == dim3d && mode == "area")
    // 如果输入是三维且模式是区域插值
    return at::adaptive_avg_pool3d(
        input, _output_size(input, 3, size, scale_factors));
  // 调用三维自适应平均池化函数，并返回结果

  if (input_dim == dim1d && mode == "linear")
    // 如果输入是一维且模式是线性插值
    return at::upsample_linear1d(
        input,
        _output_size(input, 1, size, scale_factors),
        *align_corners,
        c10::make_optional(scale_factors_1));
  // 调用一维线性插值函数，并返回结果

  if (input_dim == dim1d && mode == "bilinear")
    // 如果输入是一维且模式是双线性插值
    throw std::runtime_error("Got 3D input, but bilinear mode needs 4D input");
  // 抛出运行时错误，因为双线性插值需要四维输入

  if (input_dim == dim1d && mode == "bicubic")
    // 如果输入是一维且模式是双三次插值
    throw std::runtime_error("Got 3D input, but bicubic mode needs 4D input");
  // 抛出运行时错误，因为双三次插值需要四维输入

  if (input_dim == dim1d && mode == "trilinear")
    // 如果输入是一维且模式是三线性插值
    throw std::runtime_error("Got 3D input, but trilinear mode needs 5D input");
  // 抛出运行时错误，因为三线性插值需要五维输入

  if (input_dim == dim2d && mode == "linear")
    // 如果输入是二维且模式是线性插值
    throw std::runtime_error("Got 4D input, but linear mode needs 3D input");
  // 抛出运行时错误，因为线性插值需要三维输入

  if (input_dim == dim2d && mode == "bilinear")
    // 如果输入是二维且模式是双线性插值
    return at::upsample_bilinear2d(
        input,
        _output_size(input, 2, size, scale_factors),
        *align_corners,
        scale_factors_1,
        scale_factors_2);
  // 调用二维双线性插值函数，并返回结果

  if (input_dim == dim2d && mode == "bicubic")
    // 如果输入是二维且模式是双三次插值
    return at::upsample_bicubic2d(
        input,
        _output_size(input, 2, size, scale_factors),
        *align_corners,
        scale_factors_1,
        scale_factors_2);
  // 调用二维双三次插值函数，并返回结果

  if (input_dim == dim2d && mode == "trilinear")
    // 如果输入是二维且模式是三线性插值
    throw std::runtime_error("Got 4D input, but trilinear mode needs 5D input");
  // 抛出运行时错误，因为三线性插值需要五维输入

  if (input_dim == dim3d && mode == "linear")
    // 如果输入是三维且模式是线性插值
    throw std::runtime_error("Got 5D input, but linear mode needs 3D input");
  // 抛出运行时错误，因为线性插值需要三维输入

  if (input_dim == dim3d && mode == "bilinear")
    // 如果输入是三维且模式是双线性插值
    throw std::runtime_error("Got 5D input, but bilinear mode needs 4D input");
  // 抛出运行时错误，因为双线性插值需要四维输入

  if (input_dim == dim3d && mode == "bicubic")
    // 如果输入是三维且模式是双三次插值
    throw std::runtime_error("Got 5D input, but bicubic mode needs 4D input");
  // 抛出运行时错误，因为双三次插值需要四维输入

  if (input_dim == dim3d && mode == "trilinear")
    // 如果输入是三维且模式是三线性插值
    # 使用 trilinear 方法对 3D 张量进行上采样
    return at::upsample_trilinear3d(
        input,
        # 调用 _output_size 函数计算上采样后的输出尺寸
        _output_size(input, 3, size, scale_factors),
        # 对齐角点参数，是否保持角点对齐
        *align_corners,
        # 沿各个维度的缩放因子
        scale_factors_1,
        scale_factors_2,
        scale_factors_3);

  # 报错处理：仅支持 3D、4D 和 5D 的输入张量
  AT_ERROR(
      "Input Error: Only 3D, 4D and 5D input Tensors supported",
      # 显示当前输入张量的维度
      " (got ",
      input_dim,
      "D) for the modes: nearest | linear | bilinear | trilinear",
      # 显示当前使用的插值模式
      " (got ",
      mode,
      ") ");
}

// 定义了一个名为 interpolate_op 的函数，接受一个名为 stack 的参数栈
void interpolate_op(Stack& stack) {
  // 声明变量
  at::Tensor input;
  IValue size;
  IValue scale_factors;
  std::string mode;
  IValue align_corners;
  IValue recompute_scale_factor;
  bool antialias = false;
  // 从栈中弹出多个值，并分别赋给对应的变量
  pop(stack,
      input,
      size,
      scale_factors,
      mode,
      align_corners,
      recompute_scale_factor,
      antialias);
  // 如果 antialias 为真，抛出运行时错误
  if (antialias) {
    throw std::runtime_error("Antialias is not yet supported");
  }
  // 调用 interpolate 函数进行插值操作，得到结果 tensor res
  at::Tensor res = interpolate(
      input,
      size,
      scale_factors,
      mode,
      align_corners.toOptional<bool>(),
      recompute_scale_factor.toOptional<bool>());
  // 将结果 tensor 压入栈中
  push(stack, std::move(res));
}

// 将整型或整型数组转换为双精度浮点数
IValue convert_scale_factor_to_double(const IValue& int_ivalue) {
  // 声明变量 scale_factor_double
  IValue scale_factor_double;
  // 根据输入值的类型进行处理
  if (int_ivalue.isInt()) {
    // 如果是单个整数，转换为双精度浮点数
    scale_factor_double = static_cast<double>(int_ivalue.toInt());
  } else if (int_ivalue.isIntList()) {
    // 如果是整数列表，将每个整数转换为双精度浮点数，组成双精度浮点数数组
    auto int_list = int_ivalue.toDimVector();
    std::vector<double> double_vec(int_list.begin(), int_list.end());
    scale_factor_double = double_vec;
  } else if (int_ivalue.isNone()) {
    // 如果是空值，直接返回空值
    return IValue();
  } else {
    // 如果类型不符合预期，抛出运行时错误
    std::stringstream ss;
    ss << "Expecting optional int or int list arg for scale factor, got"
       << int_ivalue;
    throw std::runtime_error(ss.str());
  }
  // 返回转换后的结果
  return scale_factor_double;
}

// 定义了一个名为 upsample_nearest_op 的函数，接受一个名为 stack 的参数栈
void upsample_nearest_op(Stack& stack) {
  // 声明变量
  at::Tensor input;
  IValue size;
  IValue scale_factor_int;
  // 从栈中弹出多个值，并分别赋给对应的变量
  pop(stack, input, size, scale_factor_int);
  // 调用 convert_scale_factor_to_double 函数将 scale_factor_int 转换为双精度浮点数
  IValue scale_factor_double = convert_scale_factor_to_double(scale_factor_int);
  // 调用 interpolate 函数进行最近邻插值操作，得到结果 tensor res
  at::Tensor res = interpolate(
      input, size, scale_factor_double, "nearest", c10::nullopt, c10::nullopt);
  // 将结果 tensor 压入栈中
  push(stack, std::move(res));
}

// 定义了一个名为 upsample_op 的函数，接受一个名为 stack 的参数栈
void upsample_op(Stack& stack) {
  // 声明变量
  at::Tensor input;
  IValue size;
  IValue scale_factor_int;
  std::string mode;
  IValue align_corners;
  // 从栈中弹出多个值，并分别赋给对应的变量
  pop(stack, input, size, scale_factor_int, mode, align_corners);
  // 调用 convert_scale_factor_to_double 函数将 scale_factor_int 转换为双精度浮点数
  IValue scale_factor_double = convert_scale_factor_to_double(scale_factor_int);
  // 调用 interpolate 函数进行插值操作，得到结果 tensor res
  at::Tensor res = interpolate(
      input,
      size,
      scale_factor_double,
      mode,
      align_corners.toOptional<bool>(),
      c10::nullopt);
  // 将结果 tensor 压入栈中
  push(stack, std::move(res));
}

// 定义了一个名为 upsample_bilinear_op 的函数，接受一个名为 stack 的参数栈
void upsample_bilinear_op(Stack& stack) {
  // 声明变量
  at::Tensor input;
  IValue size;
  IValue scale_factor_int;
  // 从栈中弹出多个值，并分别赋给对应的变量
  pop(stack, input, size, scale_factor_int);
  // 调用 convert_scale_factor_to_double 函数将 scale_factor_int 转换为双精度浮点数
  IValue scale_factor_double = convert_scale_factor_to_double(scale_factor_int);
  // 调用 interpolate 函数进行双线性插值操作，得到结果 tensor res
  at::Tensor res = interpolate(
      input, size, scale_factor_double, "bilinear", true, c10::nullopt);
  // 将结果 tensor 压入栈中
  push(stack, std::move(res));
}

// 这些操作不再生成，但为了向后兼容性而保留在此处
RegisterOperators reg3({
    Operator(
        "aten::__interpolate.scale_list(Tensor input, int? size = None, float[]? scale_factor = None, str mode = 'nearest', bool? align_corners = None, bool? recompute_scale_factor = None, bool antialias = False) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),
    
    Operator(
        "aten::__interpolate.size_list_scale_list(Tensor input, int[]? size = None, float[]? scale_factor = None, str mode = 'nearest', bool? align_corners = None, bool? recompute_scale_factor = None, bool antialias = False) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),
    
    Operator(
        "aten::__interpolate(Tensor input, int? size = None, float? scale_factor = None, str mode = 'nearest', bool? align_corners = None, bool? recompute_scale_factor = None, bool antialias = False) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),
    
    Operator(
        "aten::__interpolate.size_list(Tensor input, int[]? size = None, float? scale_factor = None, str mode = 'nearest', bool? align_corners = None, bool? recompute_scale_factor = None, bool antialias = False) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),
    
    Operator(
        "aten::__upsample_nearest(Tensor input, int? size = None, int? scale_factor = None) -> Tensor",
        upsample_nearest_op,
        aliasAnalysisFromSchema()),
    
    Operator(
        "aten::__upsample_nearest.size_list(Tensor input, int[]? size = None, int? scale_factor = None) -> Tensor",
        upsample_nearest_op,
        aliasAnalysisFromSchema()),
    
    Operator(
        "aten::__upsample(Tensor input, int? size = None, int? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        upsample_op,
        aliasAnalysisFromSchema()),
    
    Operator(
        "aten::__upsample.size_list(Tensor input, int[]? size = None, int? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        upsample_op,
        aliasAnalysisFromSchema()),
    
    Operator(
        "aten::__upsample_bilinear(Tensor input, int? size = None, int? scale_factor = None) -> Tensor",
        upsample_bilinear_op,
        aliasAnalysisFromSchema()),
    
    Operator(
        "aten::__upsample_bilinear.size_list(Tensor input, int[]? size = None, int? scale_factor = None) -> Tensor",
        upsample_bilinear_op,
        aliasAnalysisFromSchema()),
    
    Operator(
        "aten::__upsample_bilinear.scale_list(Tensor input, int? size = None, int[]? scale_factor = None) -> Tensor",
        upsample_bilinear_op,
        aliasAnalysisFromSchema()),
    
    Operator(
        "aten::__upsample_bilinear.size_list_scale_list(Tensor input, int[]? size = None, int[]? scale_factor = None) -> Tensor",
        upsample_bilinear_op,
        aliasAnalysisFromSchema()),
    
    
    
    # 第一段注释：
    创建一个表示插值操作的Operator对象，接受Tensor输入，可以根据参数进行尺寸调整和插值操作。
    
    # 第二段注释：
    创建一个表示插值操作的Operator对象，接受Tensor输入和尺寸列表，可以根据参数进行尺寸调整和插值操作。
    
    # 第三段注释：
    创建一个表示插值操作的Operator对象，接受Tensor输入和比例因子，可以根据参数进行尺寸调整和插值操作。
    
    # 第四段注释：
    创建一个表示插值操作的Operator对象，接受Tensor输入、尺寸列表和比例因子，可以根据参数进行尺寸调整和插值操作。
    
    # 第五段注释：
    创建一个表示最近邻上采样操作的Operator对象，接受Tensor输入，可以根据参数进行尺寸调整和上采样操作。
    
    # 第六段注释：
    创建一个表示最近邻上采样操作的Operator对象，接受Tensor输入和尺寸列表，可以根据参数进行尺寸调整和上采样操作。
    
    # 第七段注释：
    创建一个表示上采样操作的Operator对象，接受Tensor输入和比例因子，可以根据参数进行尺寸调整和上采样操作，支持不同插值模式。
    
    # 第八段注释：
    创建一个表示上采样操作的Operator对象，接受Tensor输入、尺寸列表和比例因子，可以根据参数进行尺寸调整和上采样操作，支持不同插值模式。
    
    # 第九段注释：
    创建一个表示双线性上采样操作的Operator对象，接受Tensor输入和尺寸列表，可以根据参数进行尺寸调整和双线性上采样操作。
    
    # 第十段注释：
    创建一个表示双线性上采样操作的Operator对象，接受Tensor输入、比例因子列表，可以根据参数进行尺寸调整和双线性上采样操作。
    
    # 第十一段注释：
    创建一个表示双线性上采样操作的Operator对象，接受Tensor输入、尺寸列表和比例因子列表，可以根据参数进行尺寸调整和双线性上采样操作。
});

// 定义一个函数，实现带泄漏整流操作的激活函数
at::Tensor leaky_relu(const at::Tensor& tensor, double scalar) {
    // 调用 PyTorch 的 leaky_relu 函数，返回带泄漏整流的张量
    return at::leaky_relu(tensor, scalar);
}

// 定义一个函数，将输入张量列表连接成一个张量
at::Tensor cat(const c10::List<at::Tensor>& tensors) {
    // 调用 PyTorch 的 cat 函数，将输入张量列表连接成一个张量
    return at::cat(tensors.vec());
}

// 定义一个函数，获取嵌套列表中的第一个字符串
std::string get_first(const c10::List<c10::List<std::string>>& strings) {
    // 获取嵌套列表中的第一个列表，并返回其第一个字符串
    return strings.get(0).get(0);
}

// 定义静态变量 reg4，注册自定义操作到 PyTorch 框架中
static auto reg4 =
    torch::RegisterOperators()
        // 注册名为 "_test::leaky_relu" 的自定义操作，映射到 leaky_relu 函数
        .op("_test::leaky_relu(Tensor self, float v=0.01) -> Tensor", &leaky_relu)
        // 注册名为 "_test::cat" 的自定义操作，映射到 cat 函数
        .op("_test::cat(Tensor[] inputs) -> Tensor", &cat)
        // 注册名为 "_test::get_first" 的自定义操作，映射到 get_first 函数
        .op("_test::get_first", &get_first);

// 结束命名空间定义
} // namespace
} // namespace torch::jit
```