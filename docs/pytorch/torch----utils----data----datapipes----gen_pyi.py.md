# `.\pytorch\torch\utils\data\datapipes\gen_pyi.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union


def materialize_lines(lines: List[str], indentation: int) -> str:
    # 初始化空字符串用于存储输出内容
    output = ""
    # 创建新行字符串，带有指定缩进
    new_line_with_indent = "\n" + " " * indentation
    # 遍历输入的每一行
    for i, line in enumerate(lines):
        # 如果不是第一行，则添加带缩进的新行
        if i != 0:
            output += new_line_with_indent
        # 将行中的换行符替换为带缩进的新行字符串
        output += line.replace("\n", new_line_with_indent)
    return output


def gen_from_template(
    dir: str,
    template_name: str,
    output_name: str,
    replacements: List[Tuple[str, Any, int]],
):
    # 构建模板文件路径和输出文件路径
    template_path = os.path.join(dir, template_name)
    output_path = os.path.join(dir, output_name)

    # 打开模板文件并读取内容
    with open(template_path) as f:
        content = f.read()
    # 遍历替换元组列表中的每一项
    for placeholder, lines, indentation in replacements:
        # 打开输出文件，写入替换后的内容
        with open(output_path, "w") as f:
            content = content.replace(
                placeholder, materialize_lines(lines, indentation)
            )
            f.write(content)


def find_file_paths(dir_paths: List[str], files_to_exclude: Set[str]) -> Set[str]:
    """
    When given a path to a directory, returns the paths to the relevant files within it.

    This function does NOT recursive traverse to subdirectories.
    """
    # 初始化一个空集合，用于存储文件路径
    paths: Set[str] = set()
    # 遍历输入的目录路径列表
    for dir_path in dir_paths:
        # 获取目录下的所有文件名列表
        all_files = os.listdir(dir_path)
        # 筛选出Python文件名集合
        python_files = {fname for fname in all_files if fname.endswith(".py")}
        # 根据排除列表，进一步筛选文件名
        filter_files = {
            fname for fname in python_files if fname not in files_to_exclude
        }
        # 将符合条件的文件路径添加到集合中
        paths.update({os.path.join(dir_path, fname) for fname in filter_files})
    return paths


def extract_method_name(line: str) -> str:
    """Extract method name from decorator in the form of "@functional_datapipe({method_name})"."""
    # 根据不同的引号类型确定起始和结束标记
    if '("' in line:
        start_token, end_token = '("', '")'
    elif "('" in line:
        start_token, end_token = "('", "')"
    else:
        # 如果未找到合适的方法名，则引发运行时错误
        raise RuntimeError(
            f"Unable to find appropriate method name within line:\n{line}"
        )
    # 提取方法名的起始和结束位置，并返回提取的方法名字符串
    start, end = line.find(start_token) + len(start_token), line.find(end_token)
    return line[start:end]


def extract_class_name(line: str) -> str:
    """Extract class name from class definition in the form of "class {CLASS_NAME}({Type}):"."""
    # 定义类名的起始和结束标记
    start_token = "class "
    end_token = "("
    # 提取类名的起始和结束位置，并返回提取的类名字符串
    start, end = line.find(start_token) + len(start_token), line.find(end_token)
    return line[start:end]


def parse_datapipe_file(
    file_path: str,
) -> Tuple[Dict[str, str], Dict[str, str], Set[str], Dict[str, List[str]]]:
    """Given a path to file, parses the file and returns a dictionary of method names to function signatures."""
    # 初始化存储方法名到函数签名、方法名到类名、特殊输出类型和文档字符串字典的变量
    method_to_signature, method_to_class_name, special_output_type = {}, {}, set()
    doc_string_dict = defaultdict(list)
    # 打开文件并读取内容，使用上下文管理器确保文件在处理完成后自动关闭
    with open(file_path) as f:
        # 计数开放的圆括号数量，用于处理函数签名
        open_paren_count = 0
        # 存储方法名、类名和签名的变量初始化
        method_name, class_name, signature = "", "", ""
        # 是否跳过处理文档字符串的标志
        skip = False
        # 逐行读取文件内容进行处理
        for line in f:
            # 如果当前行包含奇数个双引号，则切换跳过标志
            if line.count('"""') % 2 == 1:
                skip = not skip
            # 如果需要跳过当前行（处理文档字符串），则将该行添加到对应方法名的文档字符串列表中
            if skip or '"""' in line:
                doc_string_dict[method_name].append(line)
                continue
            # 如果当前行包含 @functional_datapipe，则提取方法名并初始化其文档字符串列表
            if "@functional_datapipe" in line:
                method_name = extract_method_name(line)
                doc_string_dict[method_name] = []
                continue
            # 如果已经提取了方法名，并且当前行包含 "class "，则提取类名
            if method_name and "class " in line:
                class_name = extract_class_name(line)
                continue
            # 如果已经提取了方法名，并且当前行包含 "__init__" 或 "__new__"，则处理方法签名
            if method_name and ("def __init__(" in line or "def __new__(" in line):
                # 如果是 "__new__"，则标记为特殊输出类型
                if "def __new__(" in line:
                    special_output_type.add(method_name)
                # 计数开放的圆括号
                open_paren_count += 1
                # 提取方法签名的起始位置并截取签名内容
                start = line.find("(") + len("(")
                line = line[start:]
            # 如果正在处理方法签名
            if open_paren_count > 0:
                # 统计当前行中的圆括号数量变化
                open_paren_count += line.count("(")
                open_paren_count -= line.count(")")
                # 如果圆括号数量归零，则完成方法签名的提取和处理
                if open_paren_count == 0:
                    end = line.rfind(")")
                    signature += line[:end]
                    # 处理提取的方法签名并存储到相应的字典中
                    method_to_signature[method_name] = process_signature(signature)
                    method_to_class_name[method_name] = class_name
                    method_name, class_name, signature = "", "", ""
                # 如果圆括号数量小于零，则抛出运行时错误
                elif open_paren_count < 0:
                    raise RuntimeError(
                        "open parenthesis count < 0. This shouldn't be possible."
                    )
                else:
                    # 继续积累方法签名的行内容
                    signature += line.strip("\n").strip(" ")
    # 返回最终收集到的结果字典和列表
    return (
        method_to_signature,
        method_to_class_name,
        special_output_type,
        doc_string_dict,
    )
# 生成功能性 DataPipes 过程的 .pyi 文件

# 1. 确定要处理的文件（排除不需要的文件）
# 2. 解析方法名和签名
# 3. 移除 self 后的第一个参数（除非是 "*datapipes"），默认参数以及空格
def get_method_definitions(
    file_path: Union[str, List[str]],
    files_to_exclude: Set[str],
    deprecated_files: Set[str],
    default_output_type: str,
    method_to_special_output_type: Dict[str, str],
    root: str = "",
) -> List[str]:
    """
    #.pyi generation for functional DataPipes Process.

    # 1. Find files that we want to process (exclude the ones who don't)
    # 2. Parse method name and signature
    # 3. Remove first argument after self (unless it is "*datapipes"), default args, and spaces
    """
    # 如果 root 为空，则使用当前文件的父目录的绝对路径
    if root == "":
        root = str(pathlib.Path(__file__).parent.resolve())
    # 如果 file_path 是字符串，则将其转换为单元素列表；否则，保持不变
    file_path = [file_path] if isinstance(file_path, str) else file_path
    # 使用列表推导式，将 root 和每个 file_path 组合成完整的文件路径列表
    file_path = [os.path.join(root, path) for path in file_path]
    # 调用 find_file_paths 函数查找所有文件路径，排除 files_to_exclude 和 deprecated_files 中的文件
    file_paths = find_file_paths(
        file_path, files_to_exclude=files_to_exclude.union(deprecated_files)
    )

    # 解析 data pipe 文件，获取方法及其签名、类名、特殊输出类型及其方法和文档字符串的映射
    (
        methods_and_signatures,
        methods_and_class_names,
        methods_w_special_output_types,
        methods_and_doc_strings,
    ) = parse_datapipe_files(file_paths)

    # 将需要特殊输出类型的方法添加到 methods_w_special_output_types 集合中
    for fn_name in method_to_special_output_type:
        if fn_name not in methods_w_special_output_types:
            methods_w_special_output_types.add(fn_name)

    # 初始化空列表，用于存储方法定义字符串
    method_definitions = []
    # 遍历方法及其签名的字典
    for method_name, arguments in methods_and_signatures.items():
        # 获取方法所属的类名
        class_name = methods_and_class_names[method_name]
        # 如果方法名在 methods_w_special_output_types 集合中，则使用特殊输出类型；否则使用默认输出类型
        if method_name in methods_w_special_output_types:
            output_type = method_to_special_output_type[method_name]
        else:
            output_type = default_output_type
        # 获取方法的文档字符串，如果为空，则设置默认文档字符串
        doc_string = "".join(methods_and_doc_strings[method_name])
        if doc_string == "":
            doc_string = "    ...\n"
        # 构造方法定义字符串，包括类名、方法名、参数、返回类型和文档字符串
        method_definitions.append(
            f"# Functional form of '{class_name}'\n"
            f"def {method_name}({arguments}) -> {output_type}:\n"
            f"{doc_string}"
        )

    # 根据方法名的第二行（即第一行为注释行）对方法定义字符串列表进行排序
    method_definitions.sort(
        key=lambda s: s.split("\n")[1]
    )

    # 返回排序后的方法定义字符串列表
    return method_definitions
# 定义在 main() 之外，以便 TorchData 可以导入使用

# 文件路径，用于迭代数据管道
iterDP_file_path: str = "iter"
# 需要排除的文件集合，这些文件不会被处理
iterDP_files_to_exclude: Set[str] = {"__init__.py", "utils.py"}
# 废弃的文件集合，目前为空
iterDP_deprecated_files: Set[str] = set()
# 方法到特殊输出类型的映射，用于迭代数据管道
iterDP_method_to_special_output_type: Dict[str, str] = {
    "demux": "List[IterDataPipe]",
    "fork": "List[IterDataPipe]",
}

# 文件路径，用于映射数据管道
mapDP_file_path: str = "map"
# 需要排除的文件集合，这些文件不会被处理
mapDP_files_to_exclude: Set[str] = {"__init__.py", "utils.py"}
# 废弃的文件集合，目前为空
mapDP_deprecated_files: Set[str] = set()
# 方法到特殊输出类型的映射，用于映射数据管道
mapDP_method_to_special_output_type: Dict[str, str] = {"shuffle": "IterDataPipe"}


def main() -> None:
    """
    主函数，生成数据管道的接口定义文件。

    TODO: 当前脚本仅生成内置方法的接口。若要生成用户定义数据管道的接口，请考虑修改 `IterDataPipe.register_datapipe_as_function`。
    """
    # 获取迭代数据管道的方法定义
    iter_method_definitions = get_method_definitions(
        iterDP_file_path,
        iterDP_files_to_exclude,
        iterDP_deprecated_files,
        "IterDataPipe",
        iterDP_method_to_special_output_type,
    )

    # 获取映射数据管道的方法定义
    map_method_definitions = get_method_definitions(
        mapDP_file_path,
        mapDP_files_to_exclude,
        mapDP_deprecated_files,
        "MapDataPipe",
        mapDP_method_to_special_output_type,
    )

    # 获取当前文件所在目录的绝对路径
    path = pathlib.Path(__file__).parent.resolve()
    # 替换模板文件中的占位符
    replacements = [
        ("${IterDataPipeMethods}", iter_method_definitions, 4),
        ("${MapDataPipeMethods}", map_method_definitions, 4),
    ]
    # 从模板生成文件
    gen_from_template(
        dir=str(path),
        template_name="datapipe.pyi.in",
        output_name="datapipe.pyi",
        replacements=replacements,
    )


# 如果当前脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```