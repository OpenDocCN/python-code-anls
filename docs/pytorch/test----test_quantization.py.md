# `.\pytorch\test\test_quantization.py`

```py
# Owner(s): ["oncall: quantization"]

import logging  # 导入日志模块，用于记录程序运行时的信息
from torch.testing._internal.common_utils import run_tests  # 导入测试工具函数

# Quantization core tests. These include tests for
# - quantized kernels
# - quantized functional operators
# - quantized workflow modules
# - quantized workflow operators
# - quantized tensor

# 1. Quantized Kernels
# TODO: merge the different quantized op tests into one test class
from quantization.core.test_quantized_op import TestQuantizedOps  # 导入测试量化操作的类，用于测试量化内核操作
from quantization.core.test_quantized_op import TestQNNPackOps  # 导入测试 QNNPack 操作的类
from quantization.core.test_quantized_op import TestQuantizedLinear  # 导入测试量化线性操作的类
from quantization.core.test_quantized_op import TestQuantizedConv  # 导入测试量化卷积操作的类
from quantization.core.test_quantized_op import TestDynamicQuantizedOps  # 导入测试动态量化操作的类
from quantization.core.test_quantized_op import TestComparatorOps  # 导入测试比较器操作的类
from quantization.core.test_quantized_op import TestPadding  # 导入测试填充操作的类
from quantization.core.test_quantized_op import TestQuantizedEmbeddingOps  # 导入测试量化嵌入操作的类

# 2. Quantized Functional/Workflow Ops
from quantization.core.test_quantized_functional import TestQuantizedFunctionalOps  # 导入测试量化功能操作的类
from quantization.core.test_workflow_ops import TestFakeQuantizeOps  # 导入测试虚假量化操作的类
from quantization.core.test_workflow_ops import TestFusedObsFakeQuant  # 导入测试融合观察者虚假量化操作的类

# 3. Quantized Tensor
from quantization.core.test_quantized_tensor import TestQuantizedTensor  # 导入测试量化张量的类

# 4. Modules
from quantization.core.test_workflow_module import TestFakeQuantize  # 导入测试虚假量化模块的类
from quantization.core.test_workflow_module import TestObserver  # 导入测试观察者模块的类
from quantization.core.test_quantized_module import TestStaticQuantizedModule  # 导入测试静态量化模块的类
from quantization.core.test_quantized_module import TestDynamicQuantizedModule  # 导入测试动态量化模块的类
from quantization.core.test_quantized_module import TestReferenceQuantizedModule  # 导入测试参考量化模块的类
from quantization.core.test_workflow_module import TestRecordHistogramObserver  # 导入测试记录直方图观察者模块的类
from quantization.core.test_workflow_module import TestHistogramObserver  # 导入测试直方图观察者模块的类
from quantization.core.test_workflow_module import TestDistributed  # 导入测试分布式模块的类
from quantization.core.test_workflow_module import TestFusedObsFakeQuantModule  # 导入测试融合观察者虚假量化模块的类
from quantization.core.test_backend_config import TestBackendConfig  # 导入测试后端配置的类
from quantization.core.test_utils import TestUtils  # 导入测试工具函数

try:
    # This test has extra data dependencies, so in some environments, e.g. Meta internal
    # Buck, it has its own test runner.
    from quantization.core.test_docs import TestQuantizationDocs  # 导入量化文档测试类，可能因额外数据依赖在某些环境中有独立的测试运行器
except ImportError as e:
    logging.warning(e)  # 如果导入失败，记录警告信息到日志中

# Eager Mode Workflow. Tests for the functionality of APIs and different features implemented
# using eager mode.

# 1. Eager mode post training quantization
from quantization.eager.test_quantize_eager_ptq import TestQuantizeEagerPTQStatic  # 导入测试急切模式下后训练量化的类
# 导入需要的模块用于量化测试
from quantization.eager.test_quantize_eager_ptq import TestQuantizeEagerPTQDynamic  # noqa: F401
from quantization.eager.test_quantize_eager_ptq import TestQuantizeEagerOps  # noqa: F401

# Eager 模式量化感知训练
from quantization.eager.test_quantize_eager_qat import TestQuantizeEagerQAT  # noqa: F401
from quantization.eager.test_quantize_eager_qat import TestQuantizeEagerQATNumerics  # noqa: F401

# Eager 模式融合传递测试
from quantization.eager.test_fuse_eager import TestFuseEager  # noqa: F401

# 测试量化模型数值精度（量化模型与 FP32 模型之间）
from quantization.eager.test_model_numerics import TestModelNumericsEager  # noqa: F401

# 工具：numeric_suite 的测试
from quantization.eager.test_numeric_suite_eager import TestNumericSuiteEager  # noqa: F401

# Equalization 和 Bias Correction 测试
from quantization.eager.test_equalize_eager import TestEqualizeEager  # noqa: F401
from quantization.eager.test_bias_correction_eager import TestBiasCorrectionEager  # noqa: F401


# FX GraphModule 图模式量化。测试使用 FX 量化实现的 API 功能和不同特性
try:
    from quantization.fx.test_quantize_fx import TestFuseFx  # noqa: F401
    from quantization.fx.test_quantize_fx import TestQuantizeFx  # noqa: F401
    from quantization.fx.test_quantize_fx import TestQuantizeFxOps  # noqa: F401
    from quantization.fx.test_quantize_fx import TestQuantizeFxModels  # noqa: F401
    from quantization.fx.test_subgraph_rewriter import TestSubgraphRewriter  # noqa: F401
except ImportError as e:
    # 在 FBCode 中，为了开发速度的考虑，我们将 FX 分离到一个单独的目标中，名为 `quantization_fx`
    logging.warning(e)

# PyTorch 2 Export 量化
try:
    # 以后将移到编译器端
    from quantization.pt2e.test_graph_utils import TestGraphUtils  # noqa: F401
    from quantization.pt2e.test_duplicate_dq import TestDuplicateDQPass  # noqa: F401
    from quantization.pt2e.test_metadata_porting import TestMetaDataPorting  # noqa: F401
    from quantization.pt2e.test_generate_numeric_debug_handle import TestGenerateNumericDebugHandle  # noqa: F401
    from quantization.pt2e.test_quantize_pt2e import TestQuantizePT2E  # noqa: F401
    from quantization.pt2e.test_representation import TestPT2ERepresentation  # noqa: F401
    from quantization.pt2e.test_xnnpack_quantizer import TestXNNPACKQuantizer  # noqa: F401
    from quantization.pt2e.test_xnnpack_quantizer import TestXNNPACKQuantizerModels  # noqa: F401
    from quantization.pt2e.test_x86inductor_quantizer import TestQuantizePT2EX86Inductor  # noqa: F401
    # TODO: 找出一种方法将所有 QAT 测试合并为一个 TestCase
    from quantization.pt2e.test_quantize_pt2e_qat import TestQuantizePT2EQAT_ConvBn1d  # noqa: F401
    from quantization.pt2e.test_quantize_pt2e_qat import TestQuantizePT2EQAT_ConvBn2d  # noqa: F401
    # 导入 quantization.pt2e.test_quantize_pt2e_qat 模块中的 TestQuantizePT2EQATModels 类
    from quantization.pt2e.test_quantize_pt2e_qat import TestQuantizePT2EQATModels  # noqa: F401
# 如果导入 quantization.fx.test_numeric_suite_fx 中的测试模块失败，则记录警告信息
except ImportError as e:
    # 在 FBCode 中，我们为了开发速度将 PT2 单独拆分为一个独立的目标
    # 这些内容会由独立的测试目标 `quantization_pt2e` 覆盖
    logging.warning(e)

# 导入 quantization.fx.test_numeric_suite_fx 中的多个测试类或函数，忽略 F401 错误
try:
    from quantization.fx.test_numeric_suite_fx import TestFXGraphMatcher  # noqa: F401
    from quantization.fx.test_numeric_suite_fx import TestFXGraphMatcherModels  # noqa: F401
    from quantization.fx.test_numeric_suite_fx import TestFXNumericSuiteCoreAPIs  # noqa: F401
    from quantization.fx.test_numeric_suite_fx import TestFXNumericSuiteNShadows  # noqa: F401
    from quantization.fx.test_numeric_suite_fx import TestFXNumericSuiteCoreAPIsModels  # noqa: F401
except ImportError as e:
    # 如果导入失败，记录警告信息
    logging.warning(e)

# 测试模型报告模块
try:
    from quantization.fx.test_model_report_fx import TestFxModelReportDetector  # noqa: F401
    from quantization.fx.test_model_report_fx import TestFxModelReportObserver  # noqa: F401
    from quantization.fx.test_model_report_fx import TestFxModelReportDetectDynamicStatic  # noqa: F401
    from quantization.fx.test_model_report_fx import TestFxModelReportClass  # noqa: F401
    from quantization.fx.test_model_report_fx import TestFxDetectInputWeightEqualization  # noqa: F401
    from quantization.fx.test_model_report_fx import TestFxDetectOutliers  # noqa: F401
    from quantization.fx.test_model_report_fx import TestFxModelReportVisualizer  # noqa: F401
except ImportError as e:
    # 如果导入失败，记录警告信息
    logging.warning(e)

# FX 模式下的均衡化
try:
    from quantization.fx.test_equalize_fx import TestEqualizeFx  # noqa: F401
except ImportError as e:
    # 如果导入失败，记录警告信息
    logging.warning(e)

# 向后兼容性。测试量化模块的序列化和 BC（Backward Compatibility）。
try:
    from quantization.bc.test_backward_compatibility import TestSerialization  # noqa: F401
except ImportError as e:
    # 如果导入失败，记录警告信息
    logging.warning(e)

# JIT 图模式量化
from quantization.jit.test_quantize_jit import TestQuantizeJit  # noqa: F401
from quantization.jit.test_quantize_jit import TestQuantizeJitPasses  # noqa: F401
from quantization.jit.test_quantize_jit import TestQuantizeJitOps  # noqa: F401
from quantization.jit.test_quantize_jit import TestQuantizeDynamicJitPasses  # noqa: F401
from quantization.jit.test_quantize_jit import TestQuantizeDynamicJitOps  # noqa: F401

# 量化特定的融合 passes
from quantization.jit.test_fusion_passes import TestFusionPasses  # noqa: F401
from quantization.jit.test_deprecated_jit_quant import TestDeprecatedJitQuantized  # noqa: F401

# AO 迁移测试
from quantization.ao_migration.test_quantization import TestAOMigrationQuantization  # noqa: F401
from quantization.ao_migration.test_ao_migration import TestAOMigrationNNQuantized  # noqa: F401
from quantization.ao_migration.test_ao_migration import TestAOMigrationNNIntrinsic  # noqa: F401
try:
    from quantization.ao_migration.test_quantization_fx import TestAOMigrationQuantizationFx  # noqa: F401
except ImportError as e:
    # 如果导入失败，记录警告信息
    logging.warning(e)
    logging.warning(e)


注释：


# 使用 logging 模块记录警告级别的日志，记录异常 e 的信息
logging.warning(e)


这行代码使用了 Python 的 logging 模块，将异常 `e` 记录为警告级别的日志。
# Experimental functionality

# 尝试导入 CPU 环境下的 TestBitsCPU 类
try:
    from quantization.core.experimental.test_bits import TestBitsCPU  # noqa: F401
# 如果导入失败，记录警告信息
except ImportError as e:
    logging.warning(e)

# 尝试导入 CUDA 环境下的 TestBitsCUDA 类
try:
    from quantization.core.experimental.test_bits import TestBitsCUDA  # noqa: F401
# 如果导入失败，记录警告信息
except ImportError as e:
    logging.warning(e)

# 尝试导入 CPU 环境下的 TestFloat8DtypeCPU 类
try:
    from quantization.core.experimental.test_float8 import TestFloat8DtypeCPU  # noqa: F401
# 如果导入失败，记录警告信息
except ImportError as e:
    logging.warning(e)

# 尝试导入 CUDA 环境下的 TestFloat8DtypeCUDA 类
try:
    from quantization.core.experimental.test_float8 import TestFloat8DtypeCUDA  # noqa: F401
# 如果导入失败，记录警告信息
except ImportError as e:
    logging.warning(e)

# 尝试导入仅限 CPU 环境下的 TestFloat8DtypeCPUOnlyCPU 类
try:
    from quantization.core.experimental.test_float8 import TestFloat8DtypeCPUOnlyCPU  # noqa: F401
# 如果导入失败，记录警告信息
except ImportError as e:
    logging.warning(e)

# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == '__main__':
    run_tests()
```