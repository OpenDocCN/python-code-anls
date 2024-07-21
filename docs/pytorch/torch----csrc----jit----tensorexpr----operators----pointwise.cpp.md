# `.\pytorch\torch\csrc\jit\tensorexpr\operators\pointwise.cpp`

```py
// 引入 Torch 的 Tensor Expression 库中的头文件
#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <torch/csrc/jit/tensorexpr/operators/pointwise.h>

// Torch 的 JIT Tensor Expression 命名空间
namespace torch {
namespace jit {
namespace tensorexpr {

// 使用 Torch 的 JIT Tensor Expression 命名空间
using namespace torch::jit::tensorexpr;

// 定义一个函数 computeSign，用于计算输入的符号函数
Tensor computeSign(
    const std::vector<ArgValue>& inputValues,  // 输入参数列表
    const std::vector<ExprHandle>& outputShape,  // 输出的形状描述
    std::optional<std::vector<ExprHandle>> outputStrides) {  // 可选的输出步长
  // 调用 Compute 函数来创建一个 Tensor 计算
  return Compute(
      "aten_sign",  // 计算的名称
      outputShape,  // 输出的形状描述
      outputStrides,  // 输出的步长描述
      [&](ParameterList& axes) {  // 使用 lambda 表达式定义计算逻辑
        // 将参数列表转换为表达式列表
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        // 构造输入的表达式列表
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices)};  // 获取第一个输入并转换为张量或常量
        auto inp = inputs[0];  // 获取第一个输入表达式
        auto zero = ExprHandle(immLike(inp, 0.0f));  // 创建一个与输入类型相同的零常量表达式
        auto res = (zero < inp) - (inp < zero);  // 计算符号函数的表达式
        return promoteToDtype(res, inp.dtype().scalar_type());  // 将结果升级为与输入相同的数据类型
      });
}

// 定义一个函数 computeOneOperand，用于计算单操作数的表达式
Tensor computeOneOperand(
    const std::string& name,  // 计算的名称
    const std::vector<ArgValue>& inputValues,  // 输入参数列表
    const std::vector<ExprHandle>& outputShape,  // 输出的形状描述
    const std::vector<ExprHandle>& outputStrides,  // 输出的步长描述
    const std::optional<ScalarType>& outputType,  // 可选的输出数据类型
    const std::function<ExprHandle(const ExprHandle&)>& innerExpr,  // 内部表达式的函数
    const int checkParamTypes) {  // 检查参数类型的标志
  // 调用 Compute 函数来创建一个 Tensor 计算
  return Compute(
      name,  // 计算的名称
      outputShape,  // 输出的形状描述
      outputStrides,  // 输出的步长描述
      [inputValues, outputType, innerExpr, checkParamTypes](
          const std::vector<VarHandle>& axes) {  // 使用 lambda 表达式定义计算逻辑
        // 将参数列表转换为表达式列表
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        // 构造输入的表达式列表
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices)};  // 获取第一个输入并转换为张量或常量
        promoteInputs(inputs, checkParamTypes);  // 根据参数类型标志升级输入
        ExprHandle compute = innerExpr(inputs[0]);  // 应用内部表达式函数到第一个输入
        return demoteOutput(compute, outputType);  // 将计算结果降级为指定的输出数据类型
      });
}

// 定义一个函数 computeTwoOperand，用于计算双操作数的表达式
Tensor computeTwoOperand(
    const std::string& name,  // 计算的名称
    const std::vector<ArgValue>& inputValues,  // 输入参数列表
    const std::vector<ExprHandle>& outputShape,  // 输出的形状描述
    const std::vector<ExprHandle>& outputStrides,  // 输出的步长描述
    const std::optional<ScalarType>& outputType,  // 可选的输出数据类型
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr) {  // 内部表达式的函数
  // 调用 Compute 函数来创建一个 Tensor 计算
  return Compute(
      name,  // 计算的名称
      outputShape,  // 输出的形状描述
      outputStrides,  // 输出的步长描述
      [inputValues, outputType, innerExpr](
          const std::vector<VarHandle>& axes) {  // 使用 lambda 表达式定义计算逻辑
        // 将参数列表转换为表达式列表
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        // 构造输入的表达式列表
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices),  // 获取第一个输入并转换为张量或常量
            tensorOrConstant(inputValues[1], indices),  // 获取第二个输入并转换为张量或常量
        };

        promoteInputs(inputs);  // 升级输入的数据类型
        ExprHandle compute = innerExpr(inputs[0], inputs[1]);  // 应用内部表达式函数到两个输入
        return demoteOutput(compute, outputType);  // 将计算结果降级为指定的输出数据类型
      });
}

// 定义一个函数 computeTwoOperandWithAlpha，用于计算带有 alpha 参数的双操作数表达式
Tensor computeTwoOperandWithAlpha(
    const std::string& name,  // 计算的名称
    const std::vector<ArgValue>& inputValues,  // 输入参数列表
    const std::vector<ExprHandle>& outputShape,  // 输出的形状描述
    const std::vector<ExprHandle>& outputStrides,  // 输出的步长描述
    const std::optional<ScalarType>& outputType,


这段代码定义了几个函数，每个函数都使用了Torch的Tensor Expression库中的Compute函数来创建Tensor计算。每个函数的注释解释了它们的功能和每一行代码的作用。
    // 定义一个函数，接受一个名为 innerExpr 的函数对象作为参数，并返回一个表达式句柄
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr) {
      // 调用 Compute 函数，传入以下参数：name、outputShape、outputStrides 和一个 lambda 函数
      return Compute(
          name,
          outputShape,
          outputStrides,
          // lambda 函数接受一个 axes 向量作为参数
          [inputValues, outputType, innerExpr](const std::vector<VarHandle>& axes) {
            // 将 axes 向量转换为表达式句柄向量 indices
            std::vector<ExprHandle> indices(axes.begin(), axes.end());
            // 创建一个包含三个元素的 inputs 向量，每个元素是一个表达式句柄
            std::vector<ExprHandle> inputs = {
                // 调用 tensorOrConstant 函数，将 inputValues 中的每个输入张量或常量转换为表达式句柄
                tensorOrConstant(inputValues[0], indices),
                tensorOrConstant(inputValues[1], indices),
                tensorOrConstant(inputValues[2], indices),
            };
    
            // 提升 inputs 向量中的元素类型，确保它们具有相同的类型和形状
            promoteInputs(inputs);
            // 计算表达式，调用 innerExpr 函数对象，传入两个表达式句柄作为参数
            ExprHandle compute = innerExpr(inputs[0], inputs[2] * inputs[1]);
            // 将计算结果转换为指定的输出类型 outputType
            return demoteOutput(compute, outputType);
          });
}

// 计算具有两个操作数的条件
Tensor computeConditionWithTwoOperand(
    const std::string& name,                            // 函数名称
    const std::vector<ArgValue>& inputValues,           // 输入参数值的向量
    const std::vector<ExprHandle>& outputShape,         // 输出形状的表达式向量
    const std::vector<ExprHandle>& outputStrides,       // 输出步长的表达式向量
    const std::optional<ScalarType>& outputType,        // 输出类型的可选参数
    const std::function<                              // 内部表达式的函数对象
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        innerExpr) {
  return Compute(
      name,                                             // 调用 Compute 函数计算张量
      outputShape,                                      // 输出形状
      outputStrides,                                    // 输出步长
      [inputValues, outputType, innerExpr](             // Lambda 函数，捕获输入值、输出类型和内部表达式
          const std::vector<VarHandle>& axes) {         // Lambda 函数的参数 axes，变量句柄向量
        std::vector<ExprHandle> indices(axes.begin(), axes.end());  // 根据 axes 创建索引表达式向量
        std::vector<ExprHandle> inputs = {              // 输入表达式向量，包含三个表达式
            tensorOrConstant(inputValues[1], indices),  // 第一个操作数，根据输入值和索引创建张量或常量表达式
            tensorOrConstant(inputValues[2], indices),  // 第二个操作数
        };

        promoteInputs(inputs);                          // 如果需要，提升输入表达式的类型
        // 第一个表达式是条件，我们不对其进行类型提升
        inputs.emplace(                                 // 在 inputs 的开头插入第一个条件表达式
            inputs.begin(), tensorOrConstant(inputValues[0], indices));
        ExprHandle compute = innerExpr(inputs[0], inputs[1], inputs[2]);  // 计算内部表达式的结果
        return demoteOutput(compute, outputType);        // 将计算结果降级到指定的输出类型
      });
}

// 计算具有三个操作数的张量
Tensor computeThreeOperand(
    const std::string& name,                            // 函数名称
    const std::vector<ArgValue>& inputValues,           // 输入参数值的向量
    const std::vector<ExprHandle>& outputShape,         // 输出形状的表达式向量
    const std::vector<ExprHandle>& outputStrides,       // 输出步长的表达式向量
    const std::optional<ScalarType>& outputType,        // 输出类型的可选参数
    const std::function<                              // 内部表达式的函数对象
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        innerExpr,
    bool promote_inputs) {                             // 是否需要提升输入的布尔参数
  return Compute(
      name,                                             // 调用 Compute 函数计算张量
      outputShape,                                      // 输出形状
      outputStrides,                                    // 输出步长
      [inputValues, outputType, innerExpr, promote_inputs](  // Lambda 函数，捕获输入值、输出类型、内部表达式和提升输入标志
          const std::vector<VarHandle>& axes) {         // Lambda 函数的参数 axes，变量句柄向量
        std::vector<ExprHandle> indices(axes.begin(), axes.end());  // 根据 axes 创建索引表达式向量
        std::vector<ExprHandle> inputs = {              // 输入表达式向量，包含三个表达式
            tensorOrConstant(inputValues[0], indices),  // 第一个操作数，根据输入值和索引创建张量或常量表达式
            tensorOrConstant(inputValues[1], indices),  // 第二个操作数
            tensorOrConstant(inputValues[2], indices),  // 第三个操作数
        };

        if (promote_inputs) {                           // 如果需要提升输入类型
          promoteInputs(inputs);                        // 提升输入表达式的类型
        }
        ExprHandle compute = innerExpr(inputs[0], inputs[1], inputs[2]);  // 计算内部表达式的结果
        return demoteOutput(compute, outputType);        // 将计算结果降级到指定的输出类型
      });
}
Tensor computeFourOperand(
    const std::string& name,                            // 函数名称
    const std::vector<ArgValue>& inputValues,           // 输入参数值的向量
    const std::vector<ExprHandle>& outputShape,         // 输出形状的表达式向量
    const std::vector<ExprHandle>& outputStrides,       // 输出步长的表达式向量
    const std::optional<ScalarType>& outputType,
    const std::function<ExprHandle(
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&)>& innerExpr) {
  // 返回一个 Compute 对象，该对象使用指定的 innerExpr 函数对输入进行计算
  return Compute(
      name,
      outputShape,
      outputStrides,
      [inputValues, outputType, innerExpr](const std::vector<VarHandle>& axes) {
        // 根据传入的轴变量创建索引表达式数组
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        // 根据输入值和索引创建输入表达式数组
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices),
            tensorOrConstant(inputValues[1], indices),
            tensorOrConstant(inputValues[2], indices),
            tensorOrConstant(inputValues[3], indices),
        };

        // 提升输入表达式的数据类型
        promoteInputs(inputs);
        // 使用 innerExpr 函数计算输出表达式
        ExprHandle compute =
            innerExpr(inputs[0], inputs[1], inputs[2], inputs[3]);
        // 将计算结果按照输出类型降级并返回
        return demoteOutput(compute, outputType);
      });
} // 结束命名空间 torch

// 计算一个操作数的函数，执行复制操作
Tensor computeNoop(
    const std::vector<ArgValue>& inputValues,  // 输入值的向量
    const std::vector<ExprHandle>& outputShape,  // 输出形状的向量
    const std::vector<ExprHandle>& outputStrides,  // 输出步长的向量
    const std::optional<ScalarType>& outputType,  // 可选的输出类型
    at::Device device) {  // 设备类型参数
  return computeOneOperand(
      "copy",  // 计算操作的名称
      inputValues,  // 输入值向量
      outputShape,  // 输出形状向量
      outputStrides,  // 输出步长向量
      outputType,  // 输出类型
      [](const ExprHandle& a) { return a; });  // lambda 函数，执行直接返回输入的操作
}

// 计算标量操作的函数
Tensor computeScalar(
    const std::string& name,  // 操作的名称
    const std::vector<ArgValue>& inputValues,  // 输入值的向量
    const std::vector<ExprHandle>& outputShape,  // 输出形状的向量
    const std::vector<ExprHandle>& outputStrides,  // 输出步长的向量
    const std::optional<ScalarType>& outputType,  // 可选的输出类型
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr) {  // 内部表达式函数
  auto dt = Dtype(*outputType);  // 根据输出类型创建数据类型
  VarPtr let_var = alloc<Var>(name + "_var", dt);  // 分配一个变量指针
  std::vector<ExprHandle> inputs = {  // 输入表达式向量
      scalarOrConstant(inputValues[0]),  // 第一个输入值或常数
      scalarOrConstant(inputValues[1])};  // 第二个输入值或常数
  promoteInputs(inputs);  // 提升输入表达式的类型
  ExprHandle compute = innerExpr(inputs[0], inputs[1]);  // 计算内部表达式
  StmtPtr let_stmt =
      Let::make(VarHandle(let_var), demoteOutput(compute, outputType));  // 创建 let 语句，将计算结果降级为输出类型
  std::vector<ExprPtr> dims;  // 维度向量
  BufPtr buf = alloc<Buf>(let_var, dims, dt);  // 分配一个缓冲区指针
  return Tensor(buf, let_stmt);  // 返回张量对象
}

} // 结束命名空间 jit
} // 结束命名空间 tensorexpr
```