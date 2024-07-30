# `.\yolov8\docs\build_reference.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Helper file to build Ultralytics Docs reference section. Recursively walks through ultralytics dir and builds an MkDocs
reference section of *.md files composed of classes and functions, and also creates a nav menu for use in mkdocs.yaml.

Note: Must be run from repository root directory. Do not run from docs directory.
"""

import re  # 导入正则表达式模块
import subprocess  # 导入子进程模块
from collections import defaultdict  # 导入 defaultdict 集合
from pathlib import Path  # 导入 Path 模块

# Constants
hub_sdk = False  # 设置 hub_sdk 常量为 False
if hub_sdk:
    PACKAGE_DIR = Path("/Users/glennjocher/PycharmProjects/hub-sdk/hub_sdk")  # 设置 PACKAGE_DIR 变量为指定路径
    REFERENCE_DIR = PACKAGE_DIR.parent / "docs/reference"  # 设置 REFERENCE_DIR 变量为参考文档路径
    GITHUB_REPO = "ultralytics/hub-sdk"  # 设置 GitHub 仓库路径
else:
    FILE = Path(__file__).resolve()  # 获取当前脚本文件的路径
    PACKAGE_DIR = FILE.parents[1] / "ultralytics"  # 设置 PACKAGE_DIR 变量为指定路径
    REFERENCE_DIR = PACKAGE_DIR.parent / "docs/en/reference"  # 设置 REFERENCE_DIR 变量为参考文档路径
    GITHUB_REPO = "ultralytics/ultralytics"  # 设置 GitHub 仓库路径


def extract_classes_and_functions(filepath: Path) -> tuple:
    """Extracts class and function names from a given Python file."""
    content = filepath.read_text()  # 读取文件内容为文本
    class_pattern = r"(?:^|\n)class\s(\w+)(?:\(|:)"  # 定义匹配类名的正则表达式模式
    func_pattern = r"(?:^|\n)def\s(\w+)\("  # 定义匹配函数名的正则表达式模式

    classes = re.findall(class_pattern, content)  # 使用正则表达式从内容中查找类名列表
    functions = re.findall(func_pattern, content)  # 使用正则表达式从内容中查找函数名列表

    return classes, functions  # 返回类名列表和函数名列表的元组


def create_markdown(py_filepath: Path, module_path: str, classes: list, functions: list):
    """Creates a Markdown file containing the API reference for the given Python module."""
    md_filepath = py_filepath.with_suffix(".md")  # 将 Python 文件路径转换为 Markdown 文件路径
    exists = md_filepath.exists()  # 检查 Markdown 文件是否已存在

    # Read existing content and keep header content between first two ---
    header_content = ""
    if exists:
        existing_content = md_filepath.read_text()  # 读取现有 Markdown 文件的内容
        header_parts = existing_content.split("---")  # 使用 --- 分割内容为头部部分
        for part in header_parts:
            if "description:" in part or "comments:" in part:
                header_content += f"---{part}---\n\n"  # 将符合条件的头部部分添加到 header_content 中
    if not any(header_content):
        header_content = "---\ndescription: TODO ADD DESCRIPTION\nkeywords: TODO ADD KEYWORDS\n---\n\n"

    module_name = module_path.replace(".__init__", "")  # 替换模块路径中的特定字符串
    module_path = module_path.replace(".", "/")  # 将模块路径中的点号替换为斜杠
    url = f"https://github.com/{GITHUB_REPO}/blob/main/{module_path}.py"  # 构建 GitHub 文件链接
    edit = f"https://github.com/{GITHUB_REPO}/edit/main/{module_path}.py"  # 构建 GitHub 编辑链接
    pretty = url.replace("__init__.py", "\\_\\_init\\_\\_.py")  # 替换文件名以更好地显示 __init__.py
    title_content = (
        f"# Reference for `{module_path}.py`\n\n"  # 创建 Markdown 文件的标题部分
        f"!!! Note\n\n"
        f"    This file is available at [{pretty}]({url}). If you spot a problem please help fix it by [contributing]"
        f"(https://docs.ultralytics.com/help/contributing/) a [Pull Request]({edit}) 🛠️. Thank you 🙏!\n\n"
    )
    md_content = ["<br>\n"] + [f"## ::: {module_name}.{class_name}\n\n<br><br><hr><br>\n" for class_name in classes]
    # 创建 Markdown 文件的内容部分，包含每个类的标题
    # 使用列表推导式生成 Markdown 内容的标题部分，每个函数名都以特定格式添加到列表中
    md_content.extend(f"## ::: {module_name}.{func_name}\n\n<br><br><hr><br>\n" for func_name in functions)
    
    # 移除最后一个元素中的水平线标记，确保 Markdown 内容格式正确
    md_content[-1] = md_content[-1].replace("<hr><br>", "")
    
    # 将标题、内容和生成的 Markdown 内容合并为一个完整的 Markdown 文档
    md_content = header_content + title_content + "\n".join(md_content)
    
    # 如果 Markdown 文件内容末尾不是换行符，则添加一个换行符
    if not md_content.endswith("\n"):
        md_content += "\n"

    # 根据指定路径创建 Markdown 文件的父目录，如果父目录不存在则递归创建
    md_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 将 Markdown 内容写入到指定路径的文件中
    md_filepath.write_text(md_content)

    # 如果 Markdown 文件是新创建的：
    if not exists:
        # 将新创建的 Markdown 文件添加到 Git 的暂存区中
        print(f"Created new file '{md_filepath}'")
        subprocess.run(["git", "add", "-f", str(md_filepath)], check=True, cwd=PACKAGE_DIR)

    # 返回 Markdown 文件相对于其父目录的路径
    return md_filepath.relative_to(PACKAGE_DIR.parent)
def nested_dict() -> defaultdict:
    """Creates and returns a nested defaultdict."""
    # 创建并返回一个嵌套的 defaultdict 对象
    return defaultdict(nested_dict)


def sort_nested_dict(d: dict) -> dict:
    """Sorts a nested dictionary recursively."""
    # 递归地对嵌套字典进行排序
    return {key: sort_nested_dict(value) if isinstance(value, dict) else value for key, value in sorted(d.items())}


def create_nav_menu_yaml(nav_items: list, save: bool = False):
    """Creates a YAML file for the navigation menu based on the provided list of items."""
    # 创建一个嵌套的 defaultdict 作为导航树的基础结构
    nav_tree = nested_dict()

    # 遍历传入的导航项列表
    for item_str in nav_items:
        # 将每个导航项解析为路径对象
        item = Path(item_str)
        # 获取路径的各个部分
        parts = item.parts
        # 初始化当前层级为导航树的 "reference" 键对应的值
        current_level = nav_tree["reference"]
        # 遍历路径的部分，跳过前两个部分（docs 和 reference）和最后一个部分（文件名）
        for part in parts[2:-1]:
            # 将当前层级深入到下一级
            current_level = current_level[part]

        # 提取 Markdown 文件名（去除扩展名）
        md_file_name = parts[-1].replace(".md", "")
        # 将 Markdown 文件名与路径项关联存储到导航树中
        current_level[md_file_name] = item

    # 对导航树进行递归排序
    nav_tree_sorted = sort_nested_dict(nav_tree)

    def _dict_to_yaml(d, level=0):
        """Converts a nested dictionary to a YAML-formatted string with indentation."""
        # 初始化空的 YAML 字符串
        yaml_str = ""
        # 计算当前层级的缩进
        indent = "  " * level
        # 遍历字典的键值对
        for k, v in d.items():
            # 如果值是字典类型，则递归调用该函数处理
            if isinstance(v, dict):
                yaml_str += f"{indent}- {k}:\n{_dict_to_yaml(v, level + 1)}"
            else:
                # 如果值不是字典，则将键值对格式化为 YAML 行并追加到 yaml_str
                yaml_str += f"{indent}- {k}: {str(v).replace('docs/en/', '')}\n"
        return yaml_str

    # 打印更新后的 YAML 参考部分
    print("Scan complete, new mkdocs.yaml reference section is:\n\n", _dict_to_yaml(nav_tree_sorted))

    # 如果设置了保存标志，则将更新后的 YAML 参考部分写入文件
    if save:
        (PACKAGE_DIR.parent / "nav_menu_updated.yml").write_text(_dict_to_yaml(nav_tree_sorted))


def main():
    """Main function to extract class and function names, create Markdown files, and generate a YAML navigation menu."""
    # 初始化导航项列表
    nav_items = []

    # 遍历指定目录下的所有 Python 文件
    for py_filepath in PACKAGE_DIR.rglob("*.py"):
        # 提取文件中的类和函数列表
        classes, functions = extract_classes_and_functions(py_filepath)

        # 如果文件中存在类或函数，则处理该文件
        if classes or functions:
            # 计算相对于包目录的路径
            py_filepath_rel = py_filepath.relative_to(PACKAGE_DIR)
            # 构建 Markdown 文件路径
            md_filepath = REFERENCE_DIR / py_filepath_rel
            # 构建模块路径字符串
            module_path = f"{PACKAGE_DIR.name}.{py_filepath_rel.with_suffix('').as_posix().replace('/', '.')}"
            # 创建 Markdown 文件，并返回相对路径
            md_rel_filepath = create_markdown(md_filepath, module_path, classes, functions)
            # 将 Markdown 文件的相对路径添加到导航项列表中
            nav_items.append(str(md_rel_filepath))

    # 创建导航菜单的 YAML 文件
    create_nav_menu_yaml(nav_items)


if __name__ == "__main__":
    main()
```