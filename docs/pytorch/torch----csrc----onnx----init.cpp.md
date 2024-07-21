# `.\pytorch\torch\csrc\onnx\init.cpp`

```py
// 包含 ONNX 相关的头文件
#include <onnx/onnx_pb.h>
#include <torch/csrc/onnx/back_compat.h>
#include <torch/csrc/onnx/init.h>
#include <torch/csrc/onnx/onnx.h>
#include <torch/version.h>

// 包含异常处理相关的头文件
#include <torch/csrc/Exceptions.h>

// 包含 JIT Passes 相关的头文件
#include <torch/csrc/jit/passes/onnx.h>
#include <torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.h>
#include <torch/csrc/jit/passes/onnx/constant_fold.h>
#include <torch/csrc/jit/passes/onnx/deduplicate_initializers.h>
#include <torch/csrc/jit/passes/onnx/eliminate_unused_items.h>
#include <torch/csrc/jit/passes/onnx/eval_peephole.h>
#include <torch/csrc/jit/passes/onnx/fixup_onnx_controlflow.h>
#include <torch/csrc/jit/passes/onnx/function_extraction.h>
#include <torch/csrc/jit/passes/onnx/function_substitution.h>
#include <torch/csrc/jit/passes/onnx/list_model_parameters.h>
#include <torch/csrc/jit/passes/onnx/naming.h>
#include <torch/csrc/jit/passes/onnx/onnx_log.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/autograd_function_process.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/pattern_conversion.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/pattern_encapsulation.h>
#include <torch/csrc/jit/passes/onnx/peephole.h>
#include <torch/csrc/jit/passes/onnx/prepare_division_for_onnx.h>
#include <torch/csrc/jit/passes/onnx/preprocess_for_onnx.h>
#include <torch/csrc/jit/passes/onnx/remove_inplace_ops_for_onnx.h>
#include <torch/csrc/jit/passes/onnx/scalar_type_analysis.h>
#include <torch/csrc/jit/passes/onnx/shape_type_inference.h>
#include <torch/csrc/jit/passes/onnx/unpack_quantized_weights.h>
#include <torch/csrc/jit/serialization/export.h>

// 定义命名空间 torch::onnx
namespace torch::onnx {

// 使用 torch::jit 命名空间中的内容
using namespace torch::jit;

} // namespace torch::onnx
```