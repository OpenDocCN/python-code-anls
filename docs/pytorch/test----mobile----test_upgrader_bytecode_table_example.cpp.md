# `.\pytorch\test\mobile\test_upgrader_bytecode_table_example.cpp`

```py
/**
 * @generated
 * This is an auto-generated file. Please do not modify it by hand.
 * To re-generate, please run:
 * cd ~/pytorch && python torch/csrc/jit/mobile/upgrader_mobile.cpp
 */

#include <torch/csrc/jit/mobile/upgrader_mobile.h>

#include <ATen/core/ivalue.h>
#include <caffe2/serialize/versions.h>
#include <torch/csrc/jit/mobile/type_parser.h>

namespace c10 {
TypePtr parseType(const std::string& pythonStr);
} // namespace c10

namespace torch {
namespace jit {

// clang-format off

// From operator_versions_map

/**
 * 获取移动端操作符版本映射表
 */
const std::unordered_map<std::string, std::vector<Upgrader>>
getOperatorVersionMapForMobile() {
  static std::unordered_map<std::string, std::vector<Upgrader>>
        operatorVersionMapForMobile({
                {std::string("aten::div.Scalar"),
                    std::vector<Upgrader>({
                        Upgrader({0, 3, "div_Scalar_0_3", 0})
                    })},
                {std::string("aten::div.Tensor"),
                    std::vector<Upgrader>({
                        Upgrader({0, 3, "div_Tensor_0_3", 1})
                    })},
                {std::string("aten::div.out"),
                    std::vector<Upgrader>({
                        Upgrader({0, 3, "div_out_0_3", 4})
                    })},
                {std::string("aten::div_.Scalar"),
                    std::vector<Upgrader>({
                        Upgrader({0, 3, "div__Scalar_0_3", 2})
                    })},
                {std::string("aten::div_.Tensor"),
                    std::vector<Upgrader>({
                        Upgrader({0, 3, "div__Tensor_0_3", 3})
                    })},
                {std::string("aten::linspace"),
                    std::vector<Upgrader>({
                        Upgrader({0, 7, "linspace_0_7", 7})
                    })},
                {std::string("aten::linspace.out"),
                    std::vector<Upgrader>({
                        Upgrader({0, 7, "linspace_out_0_7", 8})
                    })},
      });
  return operatorVersionMapForMobile;
}

/**
 * 获取升级字节码函数列表
 */
const std::vector<ByteCodeFunctionWithOperator>& getUpgraderBytecodeList() {
  auto generate_upgrader_bytecode_list = []() {
    for (const auto& upgrader_function : upgrader_function_list) {
      for (const auto& op : upgrader_function.operators) {
        upgrader_function.function.append_operator(
            op.name,
            op.overload_name,
            op.num_specified_args,
            caffe2::serialize::kMaxSupportedFileFormatVersion);
      }
    }
    return upgrader_function_list;
  };
  static std::vector<ByteCodeFunctionWithOperator> upgraderBytecodeList =
      generate_upgrader_bytecode_list();
  return upgraderBytecodeList;
}

// clang-format on

} // namespace jit
} // namespace torch
```