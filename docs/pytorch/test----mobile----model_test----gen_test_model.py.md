# `.\pytorch\test\mobile\model_test\gen_test_model.py`

```
import io  # 导入io模块，用于处理字节流
import sys  # 导入sys模块，用于系统相关操作

import yaml  # 导入yaml模块，用于读取和写入YAML格式的数据
from android_api_module import AndroidAPIModule  # 导入AndroidAPIModule模块，处理Android API相关功能
from builtin_ops import TSBuiltinOpsModule, TSCollectionOpsModule  # 导入TSBuiltinOpsModule和TSCollectionOpsModule模块，内置操作和集合操作
from math_ops import (  # 导入数学运算相关模块
    BlasLapackOpsModule,  # BLAS和LAPACK相关操作模块
    ComparisonOpsModule,  # 比较操作模块
    OtherMathOpsModule,  # 其他数学操作模块
    PointwiseOpsModule,  # 逐点操作模块
    ReductionOpsModule,  # 减少操作模块
    SpectralOpsModule,  # 谱操作模块
)
from nn_ops import (  # 导入神经网络操作相关模块
    NNActivationModule,  # 激活函数相关模块
    NNConvolutionModule,  # 卷积操作模块
    NNDistanceModule,  # 距离函数操作模块
    NNDropoutModule,  # Dropout操作模块
    NNLinearModule,  # 线性层操作模块
    NNLossFunctionModule,  # 损失函数操作模块
    NNNormalizationModule,  # 归一化操作模块
    NNPaddingModule,  # 填充操作模块
    NNPoolingModule,  # 池化操作模块
    NNRecurrentModule,  # 循环神经网络操作模块
    NNShuffleModule,  # 洗牌操作模块
    NNSparseModule,  # 稀疏矩阵操作模块
    NNTransformerModule,  # Transformer操作模块
    NNUtilsModule,  # 神经网络工具模块
    NNVisionModule,  # 视觉处理模块
)
from quantization_ops import (  # 导入量化操作相关模块
    FusedQuantModule,  # 融合量化操作模块
    GeneralQuantModule,  # 通用量化操作模块
    # DynamicQuantModule,  # 动态量化操作模块（已注释）
    StaticQuantModule,  # 静态量化操作模块
)
from sampling_ops import SamplingOpsModule  # 导入采样操作模块
from tensor_ops import (  # 导入张量操作相关模块
    TensorCreationOpsModule,  # 张量创建操作模块
    TensorIndexingOpsModule,  # 张量索引操作模块
    TensorOpsModule,  # 张量操作模块
    TensorTypingOpsModule,  # 张量类型操作模块
    TensorViewOpsModule,  # 张量视图操作模块
)
from torchvision_models import (  # 导入torchvision中的模型相关模块
    MobileNetV2Module,  # MobileNetV2模型模块
    MobileNetV2VulkanModule,  # MobileNetV2 Vulkan模块
    Resnet18Module,  # ResNet-18模型模块
)

import torch  # 导入torch模块，PyTorch深度学习库
from torch.jit.mobile import _load_for_lite_interpreter  # 导入_load_for_lite_interpreter函数，用于加载用于Lite解释器的模型

test_path_ios = "ios/TestApp/models/"  # 定义iOS测试路径
test_path_android = "android/pytorch_android/src/androidTest/assets/"  # 定义Android测试路径

production_ops_path = "test/mobile/model_test/model_ops.yaml"  # 定义生产操作路径
coverage_out_path = "test/mobile/model_test/coverage.yaml"  # 定义覆盖输出路径

all_modules = {  # 定义所有模块字典
    # math ops
    "pointwise_ops": PointwiseOpsModule(),  # 点对点操作模块实例化
    "reduction_ops": ReductionOpsModule(),  # 减少操作模块实例化
    "comparison_ops": ComparisonOpsModule(),  # 比较操作模块实例化
    "spectral_ops": SpectralOpsModule(),  # 谱操作模块实例化
    "other_math_ops": OtherMathOpsModule(),  # 其他数学操作模块实例化
    "blas_lapack_ops": BlasLapackOpsModule(),  # BLAS和LAPACK操作模块实例化
    # sampling
    "sampling_ops": SamplingOpsModule(),  # 采样操作模块实例化
    # tensor ops
    "tensor_general_ops": TensorOpsModule(),  # 张量通用操作模块实例化
    "tensor_creation_ops": TensorCreationOpsModule(),  # 张量创建操作模块实例化
    "tensor_indexing_ops": TensorIndexingOpsModule(),  # 张量索引操作模块实例化
    "tensor_typing_ops": TensorTypingOpsModule(),  # 张量类型操作模块实例化
    "tensor_view_ops": TensorViewOpsModule(),  # 张量视图操作模块实例化
    # nn ops
    "convolution_ops": NNConvolutionModule(),  # 卷积操作模块实例化
    "pooling_ops": NNPoolingModule(),  # 池化操作模块实例化
    "padding_ops": NNPaddingModule(),  # 填充操作模块实例化
    "activation_ops": NNActivationModule(),  # 激活函数操作模块实例化
    "normalization_ops": NNNormalizationModule(),  # 归一化操作模块实例化
    "recurrent_ops": NNRecurrentModule(),  # 循环神经网络操作模块实例化
    "transformer_ops": NNTransformerModule(),  # Transformer操作模块实例化
    "linear_ops": NNLinearModule(),  # 线性层操作模块实例化
    "dropout_ops": NNDropoutModule(),  # Dropout操作模块实例化
    "sparse_ops": NNSparseModule(),  # 稀疏矩阵操作模块实例化
    "distance_function_ops": NNDistanceModule(),  # 距离函数操作模块实例化
    "loss_function_ops": NNLossFunctionModule(),  # 损失函数操作模块实例化
    "vision_function_ops": NNVisionModule(),  # 视觉处理操作模块实例化
    "shuffle_ops": NNShuffleModule(),  # 洗牌操作模块实例化
    "nn_utils_ops": NNUtilsModule(),  # 神经网络工具操作模块实例化
    # quantization ops
    "general_quant_ops": GeneralQuantModule(),  # 通用量化操作模块实例化
    # TODO(sdym@fb.com): fix and re-enable dynamic_quant_ops
    # "dynamic_quant_ops": DynamicQuantModule(),  # 动态量化操作模块实例化（已注释）
    "static_quant_ops": StaticQuantModule(),  # 静态量化操作模块实例化
    "fused_quant_ops": FusedQuantModule(),  # 融合量化操作模块实例化
    # TorchScript buildin ops
    # 创建名为 "torchscript_builtin_ops" 的项，并调用 TSBuiltinOpsModule() 初始化其值
    "torchscript_builtin_ops": TSBuiltinOpsModule(),
    # 创建名为 "torchscript_collection_ops" 的项，并调用 TSCollectionOpsModule() 初始化其值
    "torchscript_collection_ops": TSCollectionOpsModule(),
    # 创建名为 "mobilenet_v2" 的项，并调用 MobileNetV2Module() 初始化其值
    "mobilenet_v2": MobileNetV2Module(),
    # 创建名为 "mobilenet_v2_vulkan" 的项，并调用 MobileNetV2VulkanModule() 初始化其值
    "mobilenet_v2_vulkan": MobileNetV2VulkanModule(),
    # 创建名为 "resnet18" 的项，并调用 Resnet18Module() 初始化其值
    "resnet18": Resnet18Module(),
    # 创建名为 "android_api_module" 的项，并调用 AndroidAPIModule() 初始化其值
    "android_api_module": AndroidAPIModule(),
}

models_need_trace = [
    "static_quant_ops",
]


# 计算操作覆盖率
def calcOpsCoverage(ops):
    # 打开生产操作 YAML 文件并加载为字典
    with open(production_ops_path) as input_yaml_file:
        production_ops_dict = yaml.safe_load(input_yaml_file)

    # 提取生产操作的集合
    production_ops = set(production_ops_dict["root_operators"].keys())
    # 提取所有生成操作的集合
    all_generated_ops = set(ops)
    # 计算覆盖的操作集合
    covered_ops = production_ops.intersection(all_generated_ops)
    # 计算未覆盖的操作集合
    uncovered_ops = production_ops - covered_ops
    # 计算操作覆盖率（以百分比形式）
    coverage = round(100 * len(covered_ops) / len(production_ops), 2)

    # 加权覆盖率（考虑操作出现次数）
    total_occurrences = sum(production_ops_dict["root_operators"].values())
    covered_ops_dict = {
        op: production_ops_dict["root_operators"][op] for op in covered_ops
    }
    uncovered_ops_dict = {
        op: production_ops_dict["root_operators"][op] for op in uncovered_ops
    }
    covered_occurrences = sum(covered_ops_dict.values())
    occurrences_coverage = round(100 * covered_occurrences / total_occurrences, 2)

    # 打印未覆盖的操作和相关信息
    print(f"\n{len(uncovered_ops)} uncovered ops: {uncovered_ops}\n")
    print(f"Generated {len(all_generated_ops)} ops")
    print(
        f"Covered {len(covered_ops)}/{len(production_ops)} ({coverage}%) production ops"
    )
    print(
        f"Covered {covered_occurrences}/{total_occurrences} ({occurrences_coverage}%) occurrences"
    )
    print(f"pytorch ver {torch.__version__}\n")

    # 将覆盖率信息写入输出路径的 YAML 文件
    with open(coverage_out_path, "w") as f:
        yaml.safe_dump(
            {
                "_covered_ops": len(covered_ops),
                "_production_ops": len(production_ops),
                "_generated_ops": len(all_generated_ops),
                "_uncovered_ops": len(uncovered_ops),
                "_coverage": round(coverage, 2),
                "uncovered_ops": uncovered_ops_dict,
                "covered_ops": covered_ops_dict,
                "all_generated_ops": sorted(all_generated_ops),
            },
            f,
        )


# 根据模型名获取模块
def getModuleFromName(model_name):
    # 如果模型名不在所有模块中，则打印错误信息并返回空
    if model_name not in all_modules:
        print("Cannot find test model for " + model_name)
        return None, []

    # 获取模块对象
    module = all_modules[model_name]
    # 如果模块不是 torch.nn.Module 的实例，则调用其 getModule 方法
    if not isinstance(module, torch.nn.Module):
        module = module.getModule()

    # 判断模型是否有捆绑输入
    has_bundled_inputs = False  # module.find_method("get_all_bundled_inputs")

    # 如果模型名在需要跟踪的模型列表中，则使用 torch.jit.trace 进行跟踪
    if model_name in models_need_trace:
        module = torch.jit.trace(module, [])
    else:
        # 否则使用 torch.jit.script 进行脚本化
        module = torch.jit.script(module)

    # 导出模块中的操作名称
    ops = torch.jit.export_opnames(module)
    print(ops)

    # 尝试运行模型
    runModule(module)

    return module, ops


# 运行模块
def runModule(module):
    # 将模块保存到字节流中，并加载为 lite 解释器
    buffer = io.BytesIO(module._save_to_buffer_for_lite_interpreter())
    buffer.seek(0)
    lite_module = _load_for_lite_interpreter(buffer)
    # 如果 lite_module 包含方法 "get_all_bundled_inputs"，则使用第一个捆绑输入运行
    if lite_module.find_method("get_all_bundled_inputs"):
        input = lite_module.run_method("get_all_bundled_inputs")[0]
        lite_module.forward(*input)
    else:
        # 否则假定模型无输入，直接调用 lite_module
        lite_module()
# 根据给定文件夹中的模型生成所有模型。
# 如果处于“on the fly”模式下，将模型文件名添加 "_temp" 后缀。
def generateAllModels(folder, on_the_fly=False):
    # 初始化操作列表为空
    all_ops = []
    # 遍历所有模块名
    for name in all_modules:
        # 从模块名获取模块对象和操作列表
        module, ops = getModuleFromName(name)
        # 将当前模块的操作列表添加到总操作列表中
        all_ops = all_ops + ops
        # 构建模型保存路径，包括文件夹路径、模块名和文件后缀
        path = folder + name + ("_temp.ptl" if on_the_fly else ".ptl")
        # 使用轻量级解释器保存模块
        module._save_for_lite_interpreter(path)
        # 打印保存路径信息
        print("model saved to " + path)
    # 计算所有操作的覆盖率
    calcOpsCoverage(all_ops)


# 为给定的模型名称生成或更新存储版本
def generateModel(name):
    # 从模块名称获取模块对象和操作列表
    module, ops = getModuleFromName(name)
    # 如果模块对象为空，返回
    if module is None:
        return
    # 构建存储版本模型保存路径（iOS）
    path_ios = test_path_ios + name + ".ptl"
    # 构建存储版本模型保存路径（Android）
    path_android = test_path_android + name + ".ptl"
    # 使用轻量级解释器保存模块（iOS）
    module._save_for_lite_interpreter(path_ios)
    # 使用轻量级解释器保存模块（Android）
    module._save_for_lite_interpreter(path_android)
    # 打印保存路径信息（iOS 和 Android）
    print("model saved to " + path_ios + " and " + path_android)


# 主函数，根据命令行参数生成相应的模型
def main(argv):
    # 如果参数为空或参数个数不为1，打印帮助信息并返回
    if argv is None or len(argv) != 1:
        print(
            """
This script generate models for mobile test. For each model we have a "storage" version
and an "on-the-fly" version. The "on-the-fly" version will be generated during test,and
should not be committed to the repo.
The "storage" version is for back compatibility # test (a model generated today should
run on master branch in the next 6 months). We can use this script to update a model that
is no longer supported.
- use 'python gen_test_model.py android-test' to generate on-the-fly models for android
- use 'python gen_test_model.py ios-test' to generate on-the-fly models for ios
- use 'python gen_test_model.py android' to generate checked-in models for android
- use 'python gen_test_model.py ios' to generate on-the-fly models for ios
- use 'python gen_test_model.py <model_name_no_suffix>' to update the given storage model
"""
        )
        return

    # 根据不同的命令行参数调用相应的生成模型函数
    if argv[0] == "android":
        generateAllModels(test_path_android, on_the_fly=False)
    elif argv[0] == "ios":
        generateAllModels(test_path_ios, on_the_fly=False)
    elif argv[0] == "android-test":
        generateAllModels(test_path_android, on_the_fly=True)
    elif argv[0] == "ios-test":
        generateAllModels(test_path_ios, on_the_fly=True)
    else:
        generateModel(argv[0])


# 如果当前脚本作为主程序运行，则调用主函数并传入命令行参数（去除第一个参数，即脚本本身）
if __name__ == "__main__":
    main(sys.argv[1:])
```