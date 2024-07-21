# `.\pytorch\torch\package\file_structure_representation.py`

```
# mypy: allow-untyped-defs
# 引入必要的类型注解
from typing import Dict, List

# 从当前目录下的 glob_group 模块导入 GlobGroup 和 GlobPattern 类
from .glob_group import GlobGroup, GlobPattern

# 声明模块中公开的类和函数，这里只公开了 Directory 类
__all__ = ["Directory"]

# 表示文件结构的类
class Directory:
    """A file structure representation. Organized as Directory nodes that have lists of
    their Directory children. Directories for a package are created by calling
    :meth:`PackageImporter.file_structure`."""

    def __init__(self, name: str, is_dir: bool):
        # 初始化 Directory 实例的名称和类型（文件夹或文件）
        self.name = name
        self.is_dir = is_dir
        # 子目录的字典，键为子目录名，值为对应的 Directory 实例
        self.children: Dict[str, Directory] = {}

    def _get_dir(self, dirs: List[str]) -> "Directory":
        """Builds path of Directories if not yet built and returns last directory
        in list.

        Args:
            dirs (List[str]): List of directory names that are treated like a path.

        Returns:
            :class:`Directory`: The last Directory specified in the dirs list.
        """
        # 如果 dirs 列表为空，则返回当前实例
        if len(dirs) == 0:
            return self
        # 取出当前应该处理的目录名
        dir_name = dirs[0]
        # 如果目录名不在当前目录的子目录中，则创建一个新的 Directory 实例并添加到子目录中
        if dir_name not in self.children:
            self.children[dir_name] = Directory(dir_name, True)
        # 递归调用获取下一个目录
        return self.children[dir_name]._get_dir(dirs[1:])

    def _add_file(self, file_path: str):
        """Adds a file to a Directory.

        Args:
            file_path (str): Path of file to add. Last element is added as a file while
                other paths items are added as directories.
        """
        # 使用 "/" 分割文件路径，最后一个元素是文件名，前面的是目录路径
        *dirs, file = file_path.split("/")
        # 调用 _get_dir 方法获取文件所在的目录实例
        dir = self._get_dir(dirs)
        # 将文件名作为键，创建一个新的 Directory 实例作为值，添加到当前目录的子目录中
        dir.children[file] = Directory(file, False)

    def has_file(self, filename: str) -> bool:
        """Checks if a file is present in a :class:`Directory`.

        Args:
            filename (str): Path of file to search for.
        Returns:
            bool: If a :class:`Directory` contains the specified file.
        """
        # 使用 "/" 分割文件路径，第一个元素是子目录名，第二个元素是文件名
        lineage = filename.split("/", maxsplit=1)
        child = lineage[0]
        grandchildren = lineage[1] if len(lineage) > 1 else None
        # 检查当前目录的子目录中是否存在指定的子目录
        if child in self.children.keys():
            # 如果指定了文件名，则递归调用 has_file 方法在子目录中查找文件
            if grandchildren is None:
                return True
            else:
                return self.children[child].has_file(grandchildren)
        return False

    def __str__(self):
        str_list: List[str] = []
        self._stringify_tree(str_list)
        return "".join(str_list)

    def _stringify_tree(
        self,
        str_list: List[str],
        preamble: str = "",
        dir_ptr: str = "\u2500\u2500\u2500 ",
    ):
        # 辅助方法，递归生成目录树的字符串表示
        pass
        """
        Recursive method to generate print-friendly version of a Directory.
        This method constructs a tree representation of a directory structure,
        including files and subdirectories, using ASCII characters for branches,
        junctions, and endings.
        """
        space = "    "  # Define the space used for indentation in the tree representation
        branch = "\u2502   "  # Branch character for representing intermediate levels in the tree
        tee = "\u251c\u2500\u2500 "  # Tee character for representing branches in the tree
        last = "\u2514\u2500\u2500 "  # Last character for representing the end of branches in the tree

        # Add the current directory's representation to the string list
        str_list.append(f"{preamble}{dir_ptr}{self.name}\n")

        # Add representations of children directories and files
        if dir_ptr == tee:
            preamble = preamble + branch  # Update preamble with branch character if not the last directory
        else:
            preamble = preamble + space  # Update preamble with space if it's the last directory

        file_keys: List[str] = []  # List to store keys of files in the current directory
        dir_keys: List[str] = []   # List to store keys of subdirectories in the current directory

        # Categorize children into directories and files
        for key, val in self.children.items():
            if val.is_dir:
                dir_keys.append(key)
            else:
                file_keys.append(key)

        # Recursively call _stringify_tree for each subdirectory
        for index, key in enumerate(sorted(dir_keys)):
            if (index == len(dir_keys) - 1) and len(file_keys) == 0:
                self.children[key]._stringify_tree(str_list, preamble, last)  # Last directory in the list
            else:
                self.children[key]._stringify_tree(str_list, preamble, tee)   # Intermediate directories

        # Add files in the current directory to the string list
        for index, file in enumerate(sorted(file_keys)):
            pointer = last if (index == len(file_keys) - 1) else tee  # Determine pointer character
            str_list.append(f"{preamble}{pointer}{file}\n")
def _create_directory_from_file_list(
    filename: str,
    file_list: List[str],
    include: "GlobPattern" = "**",
    exclude: "GlobPattern" = (),
) -> Directory:
    """Return a :class:`Directory` file structure representation created from a list of files.

    Args:
        filename (str): The name given to the top-level directory that will be the
            relative root for all file paths found in the file_list.

        file_list (List[str]): List of files to add to the top-level directory.

        include (Union[List[str], str]): An optional pattern that limits what is included from the file_list to
            files whose name matches the pattern.

        exclude (Union[List[str], str]): An optional pattern that excludes files whose name match the pattern.

    Returns:
        :class:`Directory`: a :class:`Directory` file structure representation created from a list of files.
    """
    # 使用 include 和 exclude 参数创建 GlobGroup 对象，用于匹配文件名
    glob_pattern = GlobGroup(include, exclude=exclude, separator="/")

    # 创建一个顶层目录对象，表示文件结构的根目录
    top_dir = Directory(filename, True)

    # 遍历文件列表中的每个文件
    for file in file_list:
        # 如果文件名匹配 glob_pattern 的规则
        if glob_pattern.matches(file):
            # 将文件添加到顶层目录对象中
            top_dir._add_file(file)

    # 返回表示文件结构的顶层目录对象
    return top_dir
```