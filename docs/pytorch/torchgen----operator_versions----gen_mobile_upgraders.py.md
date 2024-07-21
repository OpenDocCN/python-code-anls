# `.\pytorch\torchgen\operator_versions\gen_mobile_upgraders.py`

```
#!/usr/bin/env python3

from __future__ import annotations  # 允许使用后续版本的类型注释

import os  # 导入操作系统功能模块
from enum import Enum  # 导入枚举类型模块
from operator import itemgetter  # 导入操作符模块中的itemgetter函数
from pathlib import Path  # 导入路径操作模块
from typing import Any  # 导入类型提示模块中的Any类型

import torch  # 导入PyTorch深度学习库
from torch.jit.generate_bytecode import generate_upgraders_bytecode  # 导入生成升级程序字节码的函数
from torchgen.code_template import CodeTemplate  # 导入代码模板生成模块
from torchgen.operator_versions.gen_mobile_upgraders_constant import (
    MOBILE_UPGRADERS_HEADER_DESCRIPTION,  # 导入移动设备升级器常量
)


class ByteCode(Enum):  # 定义字节码枚举类
    instructions = 1  # 指令类型
    constants = 2  # 常量类型
    types = 3  # 类型定义
    operators = 4  # 操作符定义
    register_size = 5  # 寄存器大小


EXCLUDED_OP_SET = [  # 不包含的操作符集合列表
    "aten::full.names",
    "aten::full.out",
    "aten::full",
]

EXCLUE_UPGRADER_SET = ["full_0_4", "full_out_0_4"]  # 不包含的升级程序集合列表

ONE_INSTRUCTION = CodeTemplate(  # 单条指令的代码模板
    """
    Instruction{OpCode::${operator_name}, ${X}, ${N}},"""
)

INSTRUCTION_LIST = CodeTemplate(  # 指令列表的代码模板
    """std::vector<Instruction>({
        ${instruction_list}
    }), // instructions list"""
)

ONE_CONSTANT = CodeTemplate(  # 单个常量的代码模板
    """
    c10::IValue(${constant}),"""
)

CONSTANT_LIST = CodeTemplate(  # 常量列表的代码模板
    """std::vector<c10::IValue>({
        ${constant_list}
    }), // constants list"""
)

CONSTANTS_LIST_EMPTY = """std::vector<c10::IValue>(), // constants list"""  # 空常量列表的定义

ONE_TYPE = CodeTemplate(  # 单个类型的代码模板
    """c10::parseType("${type_str}"),"""
)

TYPE_LIST = CodeTemplate(  # 类型列表的代码模板
    """std::vector<c10::TypePtr>({
        ${type_list}
    }), // types list"""
)

TYPE_LIST_EMPTY = """std::vector<c10::TypePtr>(), // types list"""  # 空类型列表的定义

ONE_OPERATOTR_STRING = CodeTemplate(  # 单个操作符字符串的代码模板
    """
    OperatorString({"${operator_name}", "${overload_name}", ${num_of_args}}),"""
)

OPERATOR_STRING_LIST = CodeTemplate(  # 操作符字符串列表的代码模板
    """
    std::vector<OperatorString>({
        ${operator_string_list}
    }), // operators list"""
)

ONE_UPGRADER_FUNCTION = CodeTemplate(  # 单个升级程序函数的代码模板
    """
    mobile::Function::registerFunc(
        "${upgrader_name}",
        ${instruction_list},
        ${constant_list},
        ${type_list},
        ${register_size}
    )"""
)

ONE_UPGRADER_SRC = CodeTemplate(  # 单个升级程序源码的代码模板
    """
    ByteCodeFunctionWithOperator({
        ${bytecode_function},
        ${operator_string_list}
    }),"""
)


ONE_UPGRADER_IN_VERSION_MAP = CodeTemplate(  # 单个升级程序在版本映射中的代码模板
    """Upgrader({${upgrader_min_version}, ${upgrader_max_version}, "${upgrader_name}", ${bytecode_func_index}})"""
)  # noqa: E501

ONE_OPERATOR_IN_VERSION_MAP = CodeTemplate(  # 单个操作符在版本映射中的代码模板
    """
    {std::string("${operator_name}"),
        std::vector<Upgrader>({
            ${upgrader_list_in_version_map}
        })},"""
)


OPERATOR_VERSION_MAP = CodeTemplate(  # 操作符版本映射的代码模板
    """
const std::unordered_map<std::string, std::vector<Upgrader>>
getOperatorVersionMapForMobile() {
  static std::unordered_map<std::string, std::vector<Upgrader>>
        operatorVersionMapForMobile({
            ${operator_list_in_version_map}
      });
  return operatorVersionMapForMobile;
}
"""
)


UPGRADER_CPP_SRC = CodeTemplate(  # 升级程序C++源码的代码模板
    MOBILE_UPGRADERS_HEADER_DESCRIPTION
    + """
#include <caffe2/serialize/versions.h>
#include <torch/csrc/jit/mobile/upgrader_mobile.h>

namespace c10 {

"""
)
// 定义一个函数 parseType，接受一个常量引用的 std::string 参数 pythonStr，返回 TypePtr 类型
TypePtr parseType(const std::string& pythonStr);
} // namespace c10

// 命名空间 torch 下的命名空间 jit
namespace torch {
namespace jit {

// clang-format off

// 从 operator_versions_map 导入数据
${operator_version_map}

// 定义函数 getUpgraderBytecodeList，返回一个常量引用的 std::vector<ByteCodeFunctionWithOperator>
const std::vector<ByteCodeFunctionWithOperator>& getUpgraderBytecodeList() {
  // 定义 lambda 函数 generate_upgrader_bytecode_list，返回一个 std::vector<ByteCodeFunctionWithOperator>
  auto generate_upgrader_bytecode_list = []() {
    // 初始化 upgrader_function_list，包含 ${upgrader_bytecode} 所指定的内容
    std::vector<ByteCodeFunctionWithOperator> upgrader_function_list({
               ${upgrader_bytecode}
            });
    // 遍历 upgrader_function_list 中的每个 upgrader_function
    for (const auto& upgrader_function : upgrader_function_list) {
      // 遍历每个 upgrader_function 中的 operators
      for (const auto& op : upgrader_function.operators) {
        // 将 op 的信息添加到 upgrader_function 的 function 中
        upgrader_function.function.append_operator(
            op.name,
            op.overload_name,
            op.num_specified_args);
      }
    }
    // 返回 upgrader_function_list
    return upgrader_function_list;
  };
  // 定义静态变量 upgraderBytecodeList，初始化为 generate_upgrader_bytecode_list 的结果
  static std::vector<ByteCodeFunctionWithOperator> upgraderBytecodeList =
      generate_upgrader_bytecode_list();
  // 返回 upgraderBytecodeList
  return upgraderBytecodeList;
}

// clang-format on

} // namespace jit
} // namespace torch

// UPGRADER_MOBILE_FILE_NAME 常量，表示 upgrader_mobile.cpp 的文件名
UPGRADER_MOBILE_FILE_NAME = "upgrader_mobile.cpp"

// 定义 UPGRADER_ELEMENT 模板，用于生成 Upgrader 对象的字符串表示
UPGRADER_ELEMENT = CodeTemplate(
    """\
Upgrader({${min_version}, ${max_version}, ${operator_name}, ${index}}),
"""
)

// 定义 PER_OPERATOR_UPGRADER_LIST 模板，用于生成每个操作符对应的 Upgrader 列表的字符串表示
PER_OPERATOR_UPGRADER_LIST = CodeTemplate(
    """\
{
  std::string(${operator_name}),
  std::vector<Upgrader>({${upgrader_list}});
}
"""
)

// 定义函数 construct_instruction，接受一个 list[Any] 类型的参数 instruction_list_from_yaml，返回一个字符串
def construct_instruction(instruction_list_from_yaml: list[Any]) -> str:
    // 初始化 instruction_list_part 为空列表
    instruction_list_part = []
    // 遍历 instruction_list_from_yaml 中的每个 instruction
    for instruction in instruction_list_from_yaml:
        // 将当前 instruction 的信息添加到 instruction_list_part 中
        instruction_list_part.append(
            ONE_INSTRUCTION.substitute(
                operator_name=instruction[0],
                X=instruction[1],
                N=instruction[2],
            )
        )
    // 返回拼接后的指令列表的字符串表示
    return INSTRUCTION_LIST.substitute(
        instruction_list="".join(instruction_list_part).lstrip("\n")
    )

// 定义函数 construct_constants，接受一个 list[Any] 类型的参数 constants_list_from_yaml，返回一个字符串
def construct_constants(constants_list_from_yaml: list[Any]) -> str:
    // 初始化 constants_list_part 为空列表
    constants_list_part = []
    // 遍历 constants_list_from_yaml 中的每个 constant_from_yaml
    for constant_from_yaml in constants_list_from_yaml:
        // 根据 constant_from_yaml 的类型进行转换并添加到 constants_list_part 中
        convert_constant = None
        if isinstance(constant_from_yaml, str):
            // 如果是字符串类型，在常量两端添加引号
            convert_constant = f'"{constant_from_yaml}"'
        elif isinstance(constant_from_yaml, bool):
            // 如果是布尔类型，转换为字符串形式
            convert_constant = "true" if constant_from_yaml else "false"
        elif constant_from_yaml is None:
            // 如果是 None 类型，置为空字符串
            convert_constant = ""
        elif isinstance(constant_from_yaml, int):
            // 如果是整数类型，转换为字符串形式
            convert_constant = str(constant_from_yaml)
        else:
            // 抛出异常，提示未知类型
            raise ValueError(
                f"The type of {constant_from_yaml} is {type(constant_from_yaml)}. "
                "Please add change in construct_constants function in gen_mobile_upgraders.py."
            )
        // 将转换后的常量添加到 constants_list_part 中
        constants_list_part.append(ONE_CONSTANT.substitute(constant=convert_constant))
    // 如果 constants_list_part 为空，返回空常量列表的字符串表示
    if len(constants_list_part) == 0:
        return CONSTANTS_LIST_EMPTY
    // 返回拼接后的常量列表的字符串表示
    return CONSTANT_LIST.substitute(
        constant_list="".join(constants_list_part).lstrip("\n")
    )

// 定义函数 construct_operators，接受一个 list[Any] 类型的参数 operator_list_from_yaml，返回一个字符串
def construct_operators(operator_list_from_yaml: list[Any]) -> str:
    # 创建一个空列表，用于存储处理后的运算符字符串部分
    operator_list_part = []
    
    # 遍历从 YAML 文件中获取的运算符列表
    for operator in operator_list_from_yaml:
        # 使用字符串模板替换操作符名称、重载名称和参数数量，生成一个运算符字符串
        operator_list_part.append(
            ONE_OPERATOTR_STRING.substitute(
                operator_name=operator[0],
                overload_name=operator[1],
                num_of_args=operator[2],
            )
        )
    
    # 将所有生成的运算符字符串连接成一个完整的字符串，并去除开头的换行符
    # 然后使用字符串模板将其作为参数填充到 OPERATOR_STRING_LIST 字符串中
    return OPERATOR_STRING_LIST.substitute(
        operator_string_list="".join(operator_list_part).lstrip("\n")
    )
# 构造类型列表字符串，根据从 YAML 中读取的类型列表
def construct_types(types_tr_list_from_yaml: list[Any]) -> str:
    # 初始化一个空列表，用于存放类型转换后的部分字符串
    types_tr_list_part = []
    # 遍历 YAML 中的类型列表
    for types_tr in types_tr_list_from_yaml:
        # 使用 ONE_TYPE 模板替换每个类型字符串，并添加到列表中
        types_tr_list_part.append(ONE_TYPE.substitute(type_str=types_tr))
    # 如果生成的部分列表为空，则返回类型列表为空的字符串常量
    if len(types_tr_list_part) == 0:
        return TYPE_LIST_EMPTY
    # 否则，返回连接后的类型列表字符串，并去除开头的换行符
    return TYPE_LIST.substitute(type_list="".join(types_tr_list_part).lstrip("\n"))


# 构造注册大小字符串，根据从 YAML 中读取的注册大小
def construct_register_size(register_size_from_yaml: int) -> str:
    # 如果输入的注册大小不是整数，则引发 ValueError 异常
    if not isinstance(register_size_from_yaml, int):
        raise ValueError(
            f"Input register size is {register_size_from_yaml} and"
            "it's type is {type(register_size_from_yaml)}. An int type is expected."
        )
    # 否则，将注册大小转换为字符串并返回
    return str(register_size_from_yaml)


# 构造版本映射字符串，根据操作符的字节码函数到索引的映射
def construct_version_maps(
    upgrader_bytecode_function_to_index_map: dict[str, Any]
) -> str:
    # 获取 Torch 的操作符版本映射
    version_map = torch._C._get_operator_version_map()
    # 按键排序版本映射
    sorted_version_map_ = sorted(version_map.items(), key=itemgetter(0))  # type: ignore[no-any-return]
    sorted_version_map = dict(sorted_version_map_)

    # 初始化操作符列表部分的空列表
    operator_list_in_version_map_part = []
    # 遍历排序后的版本映射中的操作符名
    for op_name in sorted_version_map:
        # 初始化当前操作符版本映射中的升级器列表部分
        upgraders_in_version_map_part = []
        
        # TODO: 在修复这两个操作符模式后，移除跳过逻辑
        if op_name in EXCLUDED_OP_SET:
            continue
        
        # 获取当前操作符的升级器范围和升级器条目
        upgrader_ranges = torch._C._get_upgrader_ranges(op_name)
        upgrader_entries = sorted_version_map[op_name]
        assert len(upgrader_ranges) == len(upgrader_entries)
        
        # 遍历当前操作符的升级器条目
        for idx, upgrader_entry in enumerate(upgrader_entries):
            upgrader_name = upgrader_entry.upgrader_name
            # 根据升级器名称获取其字节码函数索引
            bytecode_function_index = upgrader_bytecode_function_to_index_map[
                upgrader_name
            ]
            # 使用 ONE_UPGRADER_IN_VERSION_MAP 模板替换生成当前升级器在版本映射中的条目，并添加到列表中
            upgraders_in_version_map_part.append(
                ONE_UPGRADER_IN_VERSION_MAP.substitute(
                    upgrader_min_version=upgrader_ranges[idx].min_version,
                    upgrader_max_version=upgrader_ranges[idx].max_version,
                    upgrader_name=upgrader_name,
                    bytecode_func_index=bytecode_function_index,
                )
            )
        
        # 使用 ONE_OPERATOR_IN_VERSION_MAP 模板替换生成当前操作符在版本映射中的条目，并添加到操作符列表部分列表中
        operator_list_in_version_map_part.append(
            ONE_OPERATOR_IN_VERSION_MAP.substitute(
                operator_name=op_name,
                upgrader_list_in_version_map="".join(upgraders_in_version_map_part),
            )
        )
    
    # 使用 OPERATOR_VERSION_MAP 模板替换生成整个操作符版本映射的字符串，并返回去除开头的换行符的结果
    return OPERATOR_VERSION_MAP.substitute(
        operator_list_in_version_map="".join(operator_list_in_version_map_part).lstrip(
            "\n"
        )
    )


# 获取升级器字节码函数到索引的映射，根据从 YAML 中读取的升级器字典列表
def get_upgrader_bytecode_function_to_index_map(
    upgrader_dict: list[dict[str, Any]]
) -> dict[str, Any]:
    # 初始化空字典，用于存放升级器字节码函数到索引的映射
    upgrader_bytecode_function_to_index_map = {}
    index = 0
    # 遍历升级器字典列表
    for upgrader_bytecode in upgrader_dict:
        for upgrader_name in upgrader_bytecode.keys():
            # 如果升级器名称在排除集合中，则跳过当前升级器
            if upgrader_name in EXCLUE_UPGRADER_SET:
                continue
            # 将升级器名称与当前索引添加到映射字典中，并增加索引计数
            upgrader_bytecode_function_to_index_map[upgrader_name] = index
            index += 1
    # 返回升级器字节码函数到索引的映射字典
    return upgrader_bytecode_function_to_index_map
    # 返回函数 upgrader_bytecode_function_to_index_map，该函数映射了字节码函数到索引的关系
    return upgrader_bytecode_function_to_index_map
# 定义一个函数，用于将升级器的字节码写入到指定的 C++ 文件中
def write_cpp(cpp_path: str, upgrader_dict: list[dict[str, Any]]) -> None:
    # 初始化一个空列表，用于存储生成的 C++ 文件的各个部分内容
    body_parts = []

    # 调用函数获取升级器字节码函数名称到索引的映射
    upgrader_bytecode_function_to_index_map = (
        get_upgrader_bytecode_function_to_index_map(upgrader_dict)
    )

    # 构建版本映射源码
    version_map_src = construct_version_maps(upgrader_bytecode_function_to_index_map)

    # 初始化一个空列表，用于存储所有升级器的源码字符串
    all_upgrader_src_string = []

    # 遍历每个升级器的字节码数据字典
    for upgrader_bytecode in upgrader_dict:
        for upgrader_name, bytecode in upgrader_bytecode.items():
            # TODO: remove the skip after these two operators schemas are fixed
            # 如果升级器名称在排除集合 EXCLUE_UPGRADER_SET 中，则跳过当前循环
            if upgrader_name in EXCLUE_UPGRADER_SET:
                continue

            # 初始化字符串变量，用于存储各类字节码表的内容
            instruction_list_str = ""
            constant_list_str = ""
            type_list_str = ""
            register_size_str = ""
            operator_list_str = ""

            # 遍历字节码字典中的每个表名和内容
            for table_name, contents in bytecode.items():
                element = ByteCode[table_name]
                body_string = ""

                # 根据表名选择构建不同类型的字符串内容
                if element is ByteCode.instructions:
                    instruction_list_str = construct_instruction(contents)
                elif element is ByteCode.constants:
                    constant_list_str = construct_constants(contents)
                elif element is ByteCode.operators:
                    operator_list_str = construct_operators(contents)
                elif element is ByteCode.types:
                    type_list_str = construct_types(contents)
                elif element is ByteCode.register_size:
                    register_size_str = construct_register_size(contents)

            # 使用模板 ONE_UPGRADER_FUNCTION 构建单个升级器函数的字符串
            one_upgrader_function_string = ONE_UPGRADER_FUNCTION.substitute(
                upgrader_name=upgrader_name,
                instruction_list=instruction_list_str,
                constant_list=constant_list_str,
                type_list=type_list_str,
                register_size=register_size_str,
            )

            # 使用模板 ONE_UPGRADER_SRC 构建单个升级器源码字符串
            one_upgrader_src_string = ONE_UPGRADER_SRC.substitute(
                bytecode_function=one_upgrader_function_string.lstrip("\n"),
                operator_string_list=operator_list_str.lstrip("\n"),
            )

            # 将生成的升级器源码字符串添加到列表中
            all_upgrader_src_string.append(one_upgrader_src_string)

    # 使用模板 UPGRADER_CPP_SRC 构建完整的升级器 C++ 源码字符串
    upgrader_file_content = UPGRADER_CPP_SRC.substitute(
        operator_version_map=version_map_src,
        upgrader_bytecode="".join(all_upgrader_src_string).lstrip("\n"),
    )

    # 将构建好的 C++ 源码字符串添加到 body_parts 列表中
    body_parts.append(upgrader_file_content)

    # 打印将要写入的文件路径
    print("writing file to : ", cpp_path + "/" + UPGRADER_MOBILE_FILE_NAME)

    # 打开文件并写入最终的 C++ 源码内容
    with open(os.path.join(cpp_path, UPGRADER_MOBILE_FILE_NAME), "wb") as out_file:
        final_output = "".join(body_parts)
        # 将最终的 C++ 源码内容以 UTF-8 编码写入文件
        out_file.write(upgrader_file_content.encode("utf-8"))


# 定义一个函数，用于对升级器列表按照升级器名称进行排序
def sort_upgrader(upgrader_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 使用 lambda 表达式根据字典中第一个键的值（升级器名称）对列表进行排序
    sorted_upgrader_list = sorted(
        upgrader_list, key=lambda one_upgrader: next(iter(one_upgrader))
    )
    return sorted_upgrader_list


# 定义主函数，用于程序入口和调用生成升级器字节码函数
def main() -> None:
    upgrader_list = generate_upgraders_bytecode()
    # 对升级器列表进行排序，返回排序后的列表
    sorted_upgrader_list = sort_upgrader(upgrader_list)
    
    # 遍历排序后的升级器列表
    for up in sorted_upgrader_list:
        # 打印每个升级器的第一个元素
        print("after sort upgrader : ", next(iter(up)))
    
    # 获取当前文件的父目录的父目录，即 PyTorch 的根目录
    pytorch_dir = Path(__file__).resolve().parents[2]
    
    # 构建升级器路径，指向 PyTorch 源码中的移动端 JIT 的目录
    upgrader_path = pytorch_dir / "torch" / "csrc" / "jit" / "mobile"
    
    # 将排序后的升级器列表写入指定路径的 CPP 文件
    write_cpp(str(upgrader_path), sorted_upgrader_list)
# 如果当前脚本作为主程序执行（而不是被导入到其他模块中执行），则执行 main() 函数
if __name__ == "__main__":
    main()
```