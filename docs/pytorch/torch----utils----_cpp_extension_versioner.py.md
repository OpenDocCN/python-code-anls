# `.\pytorch\torch\utils\_cpp_extension_versioner.py`

```py
# mypy: allow-untyped-defs
# 导入 collections 模块，用于创建命名元组
import collections

# 定义一个命名元组 Entry，包含 version 和 hash 两个字段
Entry = collections.namedtuple('Entry', 'version, hash')

# 定义一个函数 update_hash，用于更新哈希值，实现类似于 boost::hash_combine 的功能
def update_hash(seed, value):
    # Good old boost::hash_combine
    # https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
    return seed ^ (hash(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2))

# 定义函数 hash_source_files，计算源文件的哈希值
def hash_source_files(hash_value, source_files):
    # 遍历源文件列表
    for filename in source_files:
        # 打开文件
        with open(filename) as file:
            # 读取文件内容并更新哈希值
            hash_value = update_hash(hash_value, file.read())
    return hash_value

# 定义函数 hash_build_arguments，计算构建参数的哈希值
def hash_build_arguments(hash_value, build_arguments):
    # 遍历构建参数列表
    for group in build_arguments:
        if group:
            for argument in group:
                # 更新哈希值
                hash_value = update_hash(hash_value, argument)
    return hash_value

# 定义类 ExtensionVersioner，用于管理不同扩展的版本号和哈希值
class ExtensionVersioner:
    def __init__(self):
        # 初始化 entries 字典，用于存储每个扩展名对应的版本号和哈希值
        self.entries = {}

    # 获取指定扩展名的版本号
    def get_version(self, name):
        entry = self.entries.get(name)
        return None if entry is None else entry.version

    # 如果源文件或构建参数有变化，则增加扩展的版本号
    def bump_version_if_changed(self,
                                name,
                                source_files,
                                build_arguments,
                                build_directory,
                                with_cuda,
                                is_python_module,
                                is_standalone):
        hash_value = 0

        # 计算源文件的哈希值
        hash_value = hash_source_files(hash_value, source_files)
        # 计算构建参数的哈希值
        hash_value = hash_build_arguments(hash_value, build_arguments)
        # 更新哈希值，考虑构建目录、CUDA 支持、是否 Python 模块、是否独立构建等因素
        hash_value = update_hash(hash_value, build_directory)
        hash_value = update_hash(hash_value, with_cuda)
        hash_value = update_hash(hash_value, is_python_module)
        hash_value = update_hash(hash_value, is_standalone)

        # 获取当前扩展名的条目
        entry = self.entries.get(name)
        # 如果条目为空，说明是新的扩展名，初始化条目；否则，比较哈希值是否有变化，更新版本号
        if entry is None:
            self.entries[name] = entry = Entry(0, hash_value)
        elif hash_value != entry.hash:
            self.entries[name] = entry = Entry(entry.version + 1, hash_value)

        # 返回更新后的版本号
        return entry.version
```