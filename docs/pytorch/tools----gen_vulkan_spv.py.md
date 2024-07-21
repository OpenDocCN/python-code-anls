# `.\pytorch\tools\gen_vulkan_spv.py`

```py
# 指定当前脚本使用 Python3 解释器

from __future__ import annotations  # 允许在注释中使用类型注释的未来特性

import argparse  # 解析命令行参数的模块
import array  # 提供数组（array）对象的模块
import codecs  # 提供编解码器注册和查询的模块
import copy  # 提供浅拷贝和深拷贝操作的模块
import glob  # 提供通过通配符搜索文件路径的模块
import io  # 提供对文件和流进行处理的核心工具集
import os  # 提供与操作系统进行交互的模块
import re  # 提供正则表达式操作的模块
import sys  # 提供对 Python 解释器相关功能的访问

from itertools import product  # 提供用于迭代器操作的模块

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # 将上级目录添加到模块搜索路径中
import subprocess  # 提供运行子进程的模块
import textwrap  # 提供文本包装和填充操作的模块
from dataclasses import dataclass  # 提供用于创建数据类的装饰器
from typing import Any  # 引入用于类型提示的通用类型

import yaml  # 提供 YAML 格式的读取和写入功能
from yaml.constructor import ConstructorError  # 提供用于处理 YAML 构造函数错误的异常
from yaml.nodes import MappingNode  # 提供 YAML 节点的映射操作

try:
    from yaml import CLoader as Loader  # 尝试使用 C 编写的高效 YAML 加载器
except ImportError:
    from yaml import Loader  # 若导入失败，使用纯 Python 实现的 YAML 加载器

CPP_H_NAME = "spv.h"  # 定义 C++ 头文件名常量
CPP_SRC_NAME = "spv.cpp"  # 定义 C++ 源文件名常量

DEFAULT_ENV: dict[str, Any] = {  # 定义默认环境变量字典，指定精度和图像格式
    "PRECISION": "highp",
    "FLOAT_IMAGE_FORMAT": "rgba16f",
    "INT_IMAGE_FORMAT": "rgba32i",
    "UINT_IMAGE_FORMAT": "rgba32ui",
}

TYPES_ENV: dict[str, Any] = {  # 定义类型环境字典，包含不同图像格式和类型的映射
    "IMAGE_FORMAT": {
        "float": "rgba32f",
        "half": "rgba16f",
        "int": "rgba32i",
        "uint": "rgba32ui",
        "int8": "rgba8i",
        "uint8": "rgba8ui",
    },
    "IMAGE_T": {
        3: {
            "float": "image3D",
            "half": "image3D",
            "int": "iimage3D",
            "uint": "uimage3D",
        },
        2: {
            "float": "image2D",
            "half": "image2D",
            "int": "iimage2D",
            "uint": "uimage2D",
        },
    },
    "SAMPLER_T": {
        3: {
            "float": "sampler3D",
            "half": "sampler3D",
            "int": "isampler3D",
            "uint": "usampler3D",
        },
        2: {
            "float": "sampler2D",
            "half": "sampler2D",
            "int": "isampler2D",
            "uint": "usampler2D",
        },
    },
    "VEC4_T": {
        "float": "vec4",
        "half": "vec4",
        "int": "ivec4",
        "uint": "uvec4",
        "int8": "vec4",
        "uint8": "uvec4",
    },
    "T": {
        "float": "float",
        "half": "float",
        "int": "int",
        "uint": "uint",
        "int8": "int",
        "uint8": "uint8",
    },
}

FUNCS_ENV: dict[str, Any] = {  # 定义函数环境字典，包含获取位置的不同维度的 lambda 函数
    "GET_POS": {
        3: lambda pos: pos,
        2: lambda pos: f"{pos}.xy",
    }
}


def extract_filename(path: str, keep_ext: bool = True) -> Any:  # 定义从路径中提取文件名的函数
    if keep_ext:
        return os.path.basename(path)  # 若保留扩展名，则返回路径中的基本名称
    else:
        return os.path.basename(path).split(".")[0]  # 否则返回去除扩展名的基本名称
    # 定义一个方法，用于构建映射关系。参数包括节点和是否深度构建的标志。
    # type: ignore[no-untyped-def] 表示忽略类型检查，因为该方法未显式定义返回类型。
    def construct_mapping(self, node, deep=False):
        # 如果节点不是映射节点，则抛出构造器错误。
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )
        # 初始化空映射
        mapping = {}
        # 遍历映射节点中的键值对
        for key_node, value_node in node.value:
            # 构造键对象，支持深度构建
            key = self.construct_object(key_node, deep=deep)
            # 尝试对键进行哈希，如果失败则抛出构造器错误
            try:
                hash(key)
            except TypeError as e:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unacceptable key ",
                    key_node.start_mark,
                ) from e
            # 检查是否存在重复键
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found duplicate key",
                    key_node.start_mark,
                )
            # 构造值对象，支持深度构建，并添加到映射中
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        # 返回构建完成的映射
        return mapping
# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def extract_leading_whitespace(line: str) -> str:
    # 匹配行首的空白字符序列，返回匹配结果（如果有的话）
    match = re.match(r"\s*", line)
    return match.group(0) if match else ""


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def escape(line: str) -> str:
    # 初始化输出部分的列表
    output_parts = []
    # 当字符串中包含 "${" 时执行循环
    while "${" in line:
        # 找到 "${" 的起始位置
        start_pos = line.index("${")
        # 找到 "}" 的结束位置，这个位置从 start_pos + 2 开始找
        end_pos = line.index("}", start_pos + 2)
        # 如果起始位置不是 0，则将起始位置前的内容加入输出列表，并转义双引号
        if start_pos != 0:
            output_parts.append('"' + line[:start_pos].replace('"', '\\"') + '"')
        # 将 "${" 和 "}" 中间的内容作为字符串加入输出列表
        output_parts.append("str(" + line[start_pos + 2 : end_pos] + ")")
        # 更新 line，去除已经处理过的部分
        line = line[end_pos + 1 :]
    # 如果还有剩余的内容，将其作为字符串加入输出列表，同时转义双引号
    if line:
        output_parts.append('"' + line.replace('"', '\\"') + '"')
    # 将输出列表中的所有部分以 " + " 连接成最终的字符串并返回
    return " + ".join(output_parts)


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def preprocess(
    input_text: str, variables: dict[str, Any], input_path: str = "codegen"
) -> str:
    # 将输入文本按行分割成列表
    input_lines = input_text.splitlines()
    # 初始化 Python 代码行的列表
    python_lines = []

    # 空白行计数器
    blank_lines = 0

    # 上一次的缩进字符串
    last_indent = ""

    # 缩进栈，每个元素是元组 (总缩进字符串, Python 缩进字符串)
    indent_stack = [("", "")]

    # 表示当前是否是 Python 代码块的第一行
    python_block_start = True
    for i, input_line in enumerate(input_lines):
        # 如果是空行，增加空白行计数并继续下一次循环
        if input_line == "":
            blank_lines += 1
            continue
        # 如果行中包含 "LINT"，则跳过该行
        if "LINT" in input_line:
            continue

        # 提取行首的空白字符序列
        input_indent = extract_leading_whitespace(input_line)
        # 如果是 Python 代码块的第一行
        if python_block_start:
            # 断言当前行的缩进应该以上一行缩进开头
            assert input_indent.startswith(last_indent)
            # 计算额外的 Python 缩进
            extra_python_indent = input_indent[len(last_indent) :]
            # 计算当前 Python 缩进，并将其压入缩进栈
            python_indent = indent_stack[-1][1] + extra_python_indent
            indent_stack.append((input_indent, python_indent))
            # 断言当前行的缩进应该以栈顶元组中的总缩进字符串开头
            assert input_indent.startswith(indent_stack[-1][0])
        else:
            # 当不是 Python 代码块的第一行时，根据缩进栈调整栈的深度
            while not input_indent.startswith(indent_stack[-1][0]):
                del indent_stack[-1]
        # 标记当前不再是 Python 代码块的第一行
        python_block_start = False

        # 获取当前的 Python 缩进
        python_indent = indent_stack[-1][1]
        # 去除行两端的空白字符后的内容
        stripped_input_line = input_line.strip()
        # 如果行以 "$" 开头且不以 "${" 开头
        if stripped_input_line.startswith("$") and not stripped_input_line.startswith(
            "${"
        ):
            # 如果行以 ":" 结尾，表示这是一个 Python 代码块的开始
            if stripped_input_line.endswith(":"):
                python_block_start = True
            # 处理空白行计数
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            # 将去除 "$" 后的内容加入 Python 代码行列表
            python_lines.append(python_indent + stripped_input_line.replace("$", ""))
        else:
            # 断言当前行应该以 Python 缩进开头
            assert input_line.startswith(python_indent)
            # 处理空白行计数
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            # 将经过转义的内容加入 Python 代码行列表
            python_lines.append(
                python_indent
                + f"print({escape(input_line[len(python_indent) :])}, file=OUT_STREAM)"
            )
        # 更新上一次的缩进字符串
        last_indent = input_indent
    # 当还有空行需要处理时，向 python_lines 列表添加打印语句，将结果输出到 OUT_STREAM
    while blank_lines != 0:
        python_lines.append(python_indent + "print(file=OUT_STREAM)")
        blank_lines -= 1

    # 将 variables 字典中的内容复制到 exec_globals 字典中
    exec_globals = dict(variables)
    # 创建一个字符串缓冲区对象，用于捕获执行后的输出
    output_stream = io.StringIO()
    # 将输出流对象 OUT_STREAM 加入到 exec_globals 字典中
    exec_globals["OUT_STREAM"] = output_stream

    # 将 python_lines 列表中的代码组合成一个 Python 字节码对象
    python_bytecode = compile("\n".join(python_lines), input_path, "exec")
    # 在 exec_globals 的上下文中执行编译后的字节码
    exec(python_bytecode, exec_globals)

    # 返回输出流对象中的所有捕获内容作为字符串
    return output_stream.getvalue()
class SPVGenerator:
    def __init__(
        self,
        src_dir_paths: str | list[str],  # 初始化方法，接收源目录路径或路径列表、环境变量字典、glslc路径
        env: dict[Any, Any],
        glslc_path: str | None,
    ) -> None:
        # 如果src_dir_paths是字符串，将其转换为列表形式
        if isinstance(src_dir_paths, str):
            self.src_dir_paths = [src_dir_paths]
        else:
            self.src_dir_paths = src_dir_paths

        self.env = env  # 初始化环境变量字典
        self.glslc_path = glslc_path  # 初始化glslc路径

        self.glsl_src_files: dict[str, str] = {}  # 初始化存储GLSL源文件的字典，键为文件名，值为文件路径
        self.template_yaml_files: list[str] = []  # 初始化存储模板yaml文件路径的列表

        # 调用addSrcAndYamlFiles方法，将src_dir_paths中的文件加入到相应的列表或字典中
        self.addSrcAndYamlFiles(self.src_dir_paths)

        self.shader_template_params: dict[Any, Any] = {}  # 初始化存储着shader模板参数的字典
        # 遍历template_yaml_files列表中的每个文件，解析其中的模板yaml文件内容
        for yaml_file in self.template_yaml_files:
            self.parseTemplateYaml(yaml_file)

        self.output_shader_map: dict[str, tuple[str, dict[str, str]]] = {}  # 初始化存储输出shader映射的字典
        # 构建输出shader映射
        self.constructOutputMap()

    def addSrcAndYamlFiles(self, src_dir_paths: list[str]) -> None:
        # 遍历src_dir_paths列表中的每个路径
        for src_path in src_dir_paths:
            # 收集GLSL源文件
            glsl_files = glob.glob(
                os.path.join(src_path, "**", "*.glsl*"), recursive=True
            )
            # 遍历glsl_files列表中的每个文件路径
            for file in glsl_files:
                # 如果文件路径长度大于1，将文件名（不含扩展名）和文件路径加入到glsl_src_files字典中
                if len(file) > 1:
                    self.glsl_src_files[extract_filename(file, keep_ext=False)] = file
            # 收集模板yaml文件
            yaml_files = glob.glob(
                os.path.join(src_path, "**", "*.yaml"), recursive=True
            )
            # 遍历yaml_files列表中的每个文件路径
            for file in yaml_files:
                # 如果文件路径长度大于1，将文件路径加入到template_yaml_files列表中
                if len(file) > 1:
                    self.template_yaml_files.append(file)

    def generateVariantCombinations(
        self,
        iterated_params: dict[str, Any],  # 生成变体组合的方法，接收迭代参数字典和可选的排除参数集合
        exclude_params: set[str] | None = None,
    ) -> list[Any]:
        # 如果exclude_params为None，将其设为一个空集合
        if exclude_params is None:
            exclude_params = set()
        all_iterated_params = []
        # 遍历iterated_params字典中的每个参数名和对应的值列表
        for param_name, value_list in iterated_params.items():
            # 如果param_name不在exclude_params集合中
            if param_name not in exclude_params:
                param_values = []
                # 遍历value_list中的每个值
                for value in value_list:
                    # 获取值的后缀（如果有）、值本身，加入到param_values列表中
                    suffix = value.get("SUFFIX", value["VALUE"])
                    param_values.append((param_name, suffix, value["VALUE"]))
                all_iterated_params.append(param_values)

        # 返回所有可能的参数组合的笛卡尔积列表
        return list(product(*all_iterated_params))
    def parseTemplateYaml(self, yaml_file: str) -> None:
        # 打开并读取 YAML 文件
        with open(yaml_file) as f:
            # 使用 UniqueKeyLoader 加载 YAML 内容
            contents = yaml.load(f, Loader=UniqueKeyLoader)
            # 遍历 YAML 文件中的每个模板名和参数字典
            for template_name, params_dict in contents.items():
                # 如果模板名已存在于 self.shader_template_params 中，则抛出 KeyError
                if template_name in self.shader_template_params:
                    raise KeyError(f"{template_name} params file is defined twice")

                # 获取默认参数字典
                default_params = params_dict["parameter_names_with_default_values"]
                # 获取参数名集合，包括默认参数和固定参数 "NAME"
                params_names = set(default_params.keys()).union({"NAME"})

                # 初始化模板名对应的参数列表
                self.shader_template_params[template_name] = []

                # 获取生成所有变体的默认参数
                default_iterated_params = params_dict.get(
                    "generate_variant_forall", None
                )

                # 遍历每个着色器变体
                for variant in params_dict["shader_variants"]:
                    # 获取当前变体的参数名集合
                    variant_params_names = set(variant.keys())
                    # 检查当前变体中是否存在无效键
                    invalid_keys = (
                        variant_params_names
                        - params_names
                        - {"generate_variant_forall"}
                    )
                    assert len(invalid_keys) == 0

                    # 获取当前变体的迭代参数
                    iterated_params = variant.get(
                        "generate_variant_forall", default_iterated_params
                    )

                    # 如果存在迭代参数，则生成所有变体组合
                    if iterated_params is not None:
                        variant_combinations = self.generateVariantCombinations(
                            iterated_params, variant_params_names
                        )

                        # 遍历每个组合
                        for combination in variant_combinations:
                            # 深拷贝默认参数字典
                            default_params_copy = copy.deepcopy(default_params)
                            # 更新变体中的参数值到默认参数副本中
                            for key in variant:
                                if key != "generate_variant_forall":
                                    default_params_copy[key] = variant[key]

                            # 构建变体名称
                            variant_name = variant["NAME"]
                            for param_value in combination:
                                default_params_copy[param_value[0]] = param_value[2]
                                if len(param_value[1]) > 0:
                                    variant_name = f"{variant_name}_{param_value[1]}"

                            default_params_copy["NAME"] = variant_name

                            # 将构建好的参数字典添加到模板参数列表中
                            self.shader_template_params[template_name].append(
                                default_params_copy
                            )
                    else:
                        # 没有迭代参数时，直接深拷贝默认参数字典并添加到模板参数列表中
                        default_params_copy = copy.deepcopy(default_params)
                        for key in variant:
                            default_params_copy[key] = variant[key]

                        self.shader_template_params[template_name].append(
                            default_params_copy
                        )

    def create_shader_params(
        self, variant_params: dict[str, Any] | None = None
    # 定义方法 constructOutputMap，用于构建输出映射关系，无返回值
    def constructOutputMap(self) -> None:
        # 遍历 shader_template_params 字典，其中 key 是 shader_name，value 是 params 列表
        for shader_name, params in self.shader_template_params.items():
            # 遍历 params 列表中的每个 variant 字典
            for variant in params:
                # 获取 shader_name 对应的 GLSL 源文件路径
                source_glsl = self.glsl_src_files[shader_name]

                # 将 variant["NAME"] 映射到一个元组，包括 source_glsl 和 create_shader_params(variant) 的返回值
                self.output_shader_map[variant["NAME"]] = (
                    source_glsl,
                    self.create_shader_params(variant),
                )

        # 遍历 glsl_src_files 字典，其中 key 是 shader_name，value 是 GLSL 源文件路径
        for shader_name, source_glsl in self.glsl_src_files.items():
            # 如果 shader_name 不在 shader_template_params 中
            if shader_name not in self.shader_template_params:
                # 将 shader_name 映射到一个元组，包括 source_glsl 和 create_shader_params() 的返回值
                self.output_shader_map[shader_name] = (
                    source_glsl,
                    self.create_shader_params(),
                )
    # 生成着色器程序文件和关联的输出路径字典
    def generateSPV(self, output_dir: str) -> dict[str, str]:
        # 初始化输出文件路径字典
        output_file_map = {}
        # 遍历每个着色器程序的名称
        for shader_name in self.output_shader_map:
            # 获取当前着色器程序的源文件路径和着色器参数
            source_glsl = self.output_shader_map[shader_name][0]
            shader_params = self.output_shader_map[shader_name][1]

            # 使用utf-8编码打开源GLSL文件
            with codecs.open(source_glsl, "r", encoding="utf-8") as input_file:
                # 读取输入文件的全部内容
                input_text = input_file.read()
                # 对GLSL源代码进行预处理，传入着色器参数，获取预处理后的输出文本
                output_text = preprocess(input_text, shader_params)

            # 构建输出的GLSL文件路径
            glsl_out_path = os.path.join(output_dir, f"{shader_name}.glsl")
            # 使用utf-8编码打开GLSL输出文件，写入预处理后的输出文本
            with codecs.open(glsl_out_path, "w", encoding="utf-8") as output_file:
                output_file.write(output_text)

            # 如果指定了GLSL编译器路径，则生成SPIR-V文件
            if self.glslc_path is not None:
                # 构建SPIR-V输出文件路径
                spv_out_path = os.path.join(output_dir, f"{shader_name}.spv")

                # 构建调用GLSL编译器的命令行参数列表
                cmd = [
                    self.glslc_path,                                # 指定GLSL编译器路径
                    "-fshader-stage=compute",                       # 指定着色器阶段为compute
                    glsl_out_path,                                  # 输入的GLSL文件路径
                    "-o", spv_out_path,                             # 输出的SPIR-V文件路径
                    "--target-env=vulkan1.0",                       # 指定目标环境为Vulkan 1.0
                    "-Werror",                                      # 将所有警告视为错误
                ] + [
                    arg                                             # 添加所有源代码目录作为包含路径参数
                    for src_dir_path in self.src_dir_paths
                    for arg in ["-I", src_dir_path]
                ]

                # 打印生成的GLSL编译命令
                print("glslc cmd:", cmd)
                # 使用subprocess调用生成的GLSL编译命令，等待其完成
                subprocess.check_call(cmd)

                # 将生成的SPIR-V文件路径和对应的GLSL文件路径加入输出文件路径字典
                output_file_map[spv_out_path] = glsl_out_path

        # 返回输出文件路径字典
        return output_file_map
##############################################
#  Shader Info and Shader Registry Handling  #
##############################################

# 定义一个数据类，用于保存着色器信息
@dataclass
class ShaderInfo:
    tile_size: list[int]                # 列表，保存着色器的瓦片尺寸
    layouts: list[str]                  # 列表，保存着色器的布局信息
    weight_storage_type: str = ""       # 字符串，保存权重存储类型，默认为空
    bias_storage_type: str = ""         # 字符串，保存偏置存储类型，默认为空
    register_for: tuple[str, list[str]] | None = None  
                                        # 可选元组，保存注册信息，包括着色器名和注册键列表

# 获取文件路径的基本名称，并将斜杠和点替换为下划线，作为名称返回
def getName(filePath: str) -> str:
    return os.path.basename(filePath).replace("/", "_").replace(".", "_")

# 检查给定的字符串是否是描述符行，返回布尔值
def isDescriptorLine(lineStr: str) -> bool:
    descriptorLineId = r"^layout\(set"
    return re.search(descriptorLineId, lineStr) is not None

# 检查给定的字符串是否是瓦片尺寸行，返回布尔值
def isTileSizeLine(lineStr: str) -> bool:
    tile_size_id = r"^ \* TILE_SIZE = \("
    return re.search(tile_size_id, lineStr) is not None

# 从字符串中查找瓦片尺寸信息并返回整数列表
def findTileSizes(lineStr: str) -> list[int]:
    tile_size_id = r"^ \* TILE_SIZE = \(([0-9]+), ([0-9]+), ([0-9]+)\)"
    matches = re.search(tile_size_id, lineStr)
    if matches is None:
        raise AssertionError("matches is None in findTileSizes")
    return [int(matches.group(1)), int(matches.group(2)), int(matches.group(3))]

# 检查给定的字符串是否是权重存储类型行，返回布尔值
def isWeightStorageTypeLine(lineStr: str) -> bool:
    weight_storage_id = r"^ \* WEIGHT_STORAGE = "
    return re.search(weight_storage_id, lineStr) is not None

# 从字符串中获取权重存储类型，并返回字符串
def getWeightStorageType(lineStr: str) -> str:
    weight_storage_id = r"^ \* WEIGHT_STORAGE = ([a-zA-Z]+_\dD)"
    matches = re.search(weight_storage_id, lineStr)
    if matches is None:
        raise AssertionError("matches is None in getWeightStorageType")
    return matches.group(1)

# 检查给定的字符串是否是偏置存储类型行，返回布尔值
def isBiasStorageTypeLine(lineStr: str) -> bool:
    weight_storage_id = r"^ \* BIAS_STORAGE = "
    return re.search(weight_storage_id, lineStr) is not None

# 从字符串中获取偏置存储类型，并返回字符串
def getBiasStorageType(lineStr: str) -> str:
    weight_storage_id = r"^ \* BIAS_STORAGE = ([a-zA-Z]+_\dD)"
    matches = re.search(weight_storage_id, lineStr)
    if matches is None:
        raise AssertionError("matches is None in getBiasStorageType")
    return matches.group(1)

# 检查给定的字符串是否是注册信息行，返回布尔值
def isRegisterForLine(lineStr: str) -> bool:
    # 检查是否包含着色器名和至少一个注册键的列表
    register_for_id = (
        r"^ \* REGISTER_FOR = \('([A-Za-z0-9_]+)'\s*,\s*\['([A-Za-z0-9_]+)'.*\]\)"
    )
    return re.search(register_for_id, lineStr) is not None

# 从字符串中解析注册信息，返回包含着色器名和注册键列表的元组
def findRegisterFor(lineStr: str) -> tuple[str, list[str]]:
    register_for_pattern = r"'([A-Za-z0-9_]+)'"
    matches = re.findall(register_for_pattern, lineStr)
    if matches is None:
        raise AssertionError("matches is None in findRegisterFor")
    matches_list = list(matches)
    return (matches_list[0], matches_list[1:])

# 映射用于识别类型的正则表达式到其对应的枚举值
typeIdMapping = {
    r"image[123]D\b": "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
    r"sampler[123]D\b": "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
    r"\bbuffer\b": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
    r"\buniform\b": "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER",
}

# 映射存储类型到其对应的枚举值
storageTypeToEnum = {
    "TEXTURE_2D": "api::StorageType::TEXTURE_2D",
    "TEXTURE_3D": "api::StorageType::TEXTURE_3D",
}
    # 键为 "BUFFER"，值为 "api::StorageType::BUFFER"
    "BUFFER": "api::StorageType::BUFFER",
    # 键为空字符串，值为 "api::StorageType::UNKNOWN"
    "": "api::StorageType::UNKNOWN",
}

# 根据输入的字符串 lineStr 确定其描述符类型，并返回类型编号
def determineDescriptorType(lineStr: str) -> str:
    for identifier, typeNum in typeIdMapping.items():
        if re.search(identifier, lineStr):
            return typeNum
    # 如果没有找到匹配的描述符类型，则引发断言错误
    raise AssertionError(
        "No matching descriptor type for " + lineStr + " in determineDescriptorType"
    )


# 根据给定的源文件路径 srcFilePath 获取着色器信息对象 ShaderInfo
def getShaderInfo(srcFilePath: str) -> ShaderInfo:
    # 初始化一个空的 ShaderInfo 对象
    shader_info = ShaderInfo([], [], "")
    # 打开源文件并循环读取每一行
    with open(srcFilePath) as srcFile:
        for line in srcFile:
            # 如果当前行是描述符行，则确定描述符类型并添加到 layouts 列表中
            if isDescriptorLine(line):
                shader_info.layouts.append(determineDescriptorType(line))
            # 如果当前行是瓦片尺寸行，则查找并设置瓦片尺寸
            if isTileSizeLine(line):
                shader_info.tile_size = findTileSizes(line)
            # 如果当前行是权重存储类型行，则获取并设置权重存储类型
            if isWeightStorageTypeLine(line):
                shader_info.weight_storage_type = getWeightStorageType(line)
            # 如果当前行是偏置存储类型行，则获取并设置偏置存储类型
            if isBiasStorageTypeLine(line):
                shader_info.bias_storage_type = getBiasStorageType(line)
            # 如果当前行是寄存器分配行，则查找并设置寄存器分配信息
            if isRegisterForLine(line):
                shader_info.register_for = findRegisterFor(line)

    # 返回最终的 ShaderInfo 对象
    return shader_info


##########################
#  C++ File Generation  #
#########################

# C++ 文件模板，包含注册函数和着色器信息注册
cpp_template = """
#include <ATen/native/vulkan/api/ShaderRegistry.h>
#include <stdint.h>
#include <vector>

using namespace at::native::vulkan;

namespace at {{
namespace native {{
namespace vulkan {{

namespace {{

{spv_bin_arrays}

}}

// 注册函数，用于注册所有的着色器信息
static void register_fn() {{

{register_shader_infos}

{shader_info_registry}

}}

// 使用静态变量进行注册函数的调用
static const api::ShaderRegisterInit register_shaders(&register_fn);

}}
}}
}}

"""


# 根据给定的 SPV 文件路径 spvPath 和着色器名称 name 生成 SPV 二进制字符串
def generateSpvBinStr(spvPath: str, name: str) -> tuple[int, str]:
    with open(spvPath, "rb") as fr:
        next_bin = array.array("I", fr.read())
        sizeBytes = 4 * len(next_bin)
        spv_bin_str = "const uint32_t {}_bin[] = {{\n{}\n}};".format(
            name,
            textwrap.indent(",\n".join(str(x) for x in next_bin), "  "),
        )

    # 返回二进制数组的字节大小和生成的 SPV 二进制字符串
    return sizeBytes, spv_bin_str


# 根据给定的 ShaderInfo 对象 shader_info、着色器名称 name 和二进制大小 sizeBytes 生成着色器信息字符串
def generateShaderInfoStr(shader_info: ShaderInfo, name: str, sizeBytes: int) -> str:
    # 根据瓦片尺寸生成对应的字符串表示
    tile_size = (
        f"{{{', '.join(str(x) for x in shader_info.tile_size)}}}"
        if (len(shader_info.tile_size) > 0)
        else "std::vector<uint32_t>()"
    )

    # 将 layouts 列表转换为字符串形式
    shader_info_layouts = "{{{}}}".format(",\n ".join(shader_info.layouts))

    # 构建 shader_info_args 列表，包含生成 ShaderInfo 所需的各个参数
    shader_info_args = [
        f'"{name}"',
        f"{name}_bin",
        str(sizeBytes),
        shader_info_layouts,
        tile_size,
        storageTypeToEnum[shader_info.weight_storage_type],
        storageTypeToEnum[shader_info.bias_storage_type],
    ]

    # 使用 textwrap.indent 进行格式化，生成 ShaderInfo 对象的字符串表示
    shader_info_str = textwrap.indent(
        "api::shader_registry().register_shader(\n  api::ShaderInfo(\n{args}));\n".format(
            args=textwrap.indent(",\n".join(shader_info_args), "     "),
        ),
        "    ",
    )

    # 返回生成的着色器信息字符串
    return shader_info_str


# 根据给定的 ShaderInfo 对象 shader_info 和着色器名称 name 生成着色器调度字符串
def generateShaderDispatchStr(shader_info: ShaderInfo, name: str) -> str:
    # 如果 register_for 为 None，则返回空字符串
    if shader_info.register_for is None:
        return ""

    # 否则，从 shader_info 中获取操作名称和注册键
    (op_name, registry_keys) = shader_info.register_for
    # 遍历注册表中的每个注册键
    for registry_key in registry_keys:
        # 构建注册调度字符串，注册指定操作名、注册键、和名称到着色器注册表
        shader_dispatch_str = textwrap.indent(
            f'api::shader_registry().register_op_dispatch("{op_name}", api::DispatchKey::{registry_key.upper()}, "{name}");',
            "    ",
        )
    
    # 返回最后一个构建的着色器调度字符串
    return shader_dispatch_str
# 生成C++文件的函数，接受SPV文件字典、C++头文件路径和C++源文件路径作为参数
def genCppFiles(
    spv_files: dict[str, str], cpp_header_path: str, cpp_src_file_path: str
) -> None:
    # 存储SPV二进制字符串的列表
    spv_bin_strs = []
    # 存储注册着色器信息字符串的列表
    register_shader_info_strs = []
    # 存储着色器注册信息字符串的列表
    shader_registry_strs = []

    # 遍历SPV文件字典中的每一对路径
    for spvPath, srcPath in spv_files.items():
        # 获取SPV文件的名称，并去掉"_spv"后缀
        name = getName(spvPath).replace("_spv", "")

        # 生成SPV二进制字符串，并获取其字节大小
        sizeBytes, spv_bin_str = generateSpvBinStr(spvPath, name)
        spv_bin_strs.append(spv_bin_str)

        # 获取源文件对应的着色器信息
        shader_info = getShaderInfo(srcPath)

        # 生成着色器信息字符串，并添加到列表中
        register_shader_info_strs.append(
            generateShaderInfoStr(shader_info, name, sizeBytes)
        )

        # 如果需要注册着色器，则生成着色器分发字符串并添加到列表中
        if shader_info.register_for is not None:
            shader_registry_strs.append(generateShaderDispatchStr(shader_info, name))

    # 将SPV二进制字符串列表连接成一个大字符串，每个字符串间以换行符分隔
    spv_bin_arrays = "\n".join(spv_bin_strs)
    # 将注册着色器信息字符串列表连接成一个大字符串，每个字符串间以换行符分隔
    register_shader_infos = "\n".join(register_shader_info_strs)
    # 将着色器注册信息字符串列表连接成一个大字符串，每个字符串间以换行符分隔
    shader_info_registry = "\n".join(shader_registry_strs)

    # 使用格式化字符串替换C++模板中的占位符，生成最终的C++代码字符串
    cpp = cpp_template.format(
        spv_bin_arrays=spv_bin_arrays,
        register_shader_infos=register_shader_infos,
        shader_info_registry=shader_info_registry,
    )

    # 将生成的C++代码字符串写入到指定的C++源文件路径中
    with open(cpp_src_file_path, "w") as fw:
        fw.write(cpp)


##########
#  Main  #
##########


# 解析命令行参数中的环境变量设置，返回一个字典
def parse_arg_env(items: dict[Any, Any]) -> dict[Any, Any]:
    d = {}
    # 如果命令行参数中有环境变量设置
    if items:
        # 遍历每个环境变量设置项
        for item in items:
            # 根据等号分割键和值，并去除两端的空白字符
            tokens = item.split("=")
            key = tokens[0].strip()
            value = tokens[1].strip()
            # 将键值对添加到字典中
            d[key] = value
    return d


# 主函数，处理命令行参数并调用相关函数生成SPV文件和C++代码
def main(argv: list[str]) -> int:
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="")
    # 添加解析命令行参数的选项
    parser.add_argument(
        "-i",
        "--glsl-paths",
        nargs="+",
        help='List of paths to look for GLSL source files, separated by spaces. Ex: --glsl-paths "path1 path2 path3"',
        default=["."],
    )
    parser.add_argument("-c", "--glslc-path", required=True, help="")
    parser.add_argument("-t", "--tmp-dir-path", required=True, help="/tmp")
    parser.add_argument("-o", "--output-path", required=True, help="")
    parser.add_argument(
        "--env", metavar="KEY=VALUE", nargs="*", help="Set a number of key-value pairs"
    )
    # 解析命令行参数
    options = parser.parse_args()

    # 更新默认环境变量字典
    DEFAULT_ENV.update(TYPES_ENV)
    DEFAULT_ENV.update(FUNCS_ENV)
    # 将更新后的默认环境变量赋值给env
    env = DEFAULT_ENV

    # 解析命令行参数中的环境变量设置，更新到env字典中
    for key, value in parse_arg_env(options.env).items():
        env[key] = value

    # 如果输出路径不存在，则创建
    if not os.path.exists(options.output_path):
        os.makedirs(options.output_path)

    # 如果临时目录路径不存在，则创建
    if not os.path.exists(options.tmp_dir_path):
        os.makedirs(options.tmp_dir_path)

    # 创建SPV生成器对象，传入GLSL文件路径、环境变量和glslc路径
    shader_generator = SPVGenerator(options.glsl_paths, env, options.glslc_path)
    # 生成SPV文件，返回生成的SPV文件字典
    output_spv_files = shader_generator.generateSPV(options.tmp_dir_path)

    # 调用生成C++文件的函数，传入生成的SPV文件字典和输出的C++头文件路径、C++源文件路径
    genCppFiles(
        output_spv_files,
        f"{options.output_path}/{CPP_H_NAME}",
        f"{options.output_path}/{CPP_SRC_NAME}",
    )

    return 0


# 调用主函数并退出程序
def invoke_main() -> None:
    sys.exit(main(sys.argv))


# 如果当前脚本作为主程序运行，则调用invoke_main函数
if __name__ == "__main__":
    invoke_main()  # pragma: no cover
```