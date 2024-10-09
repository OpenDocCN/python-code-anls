# `.\MinerU\tests\test_cli\lib\common.py`

```
# 模块说明
"""common definitions."""
# 导入操作系统接口模块
import os
# 导入高级文件操作模块
import shutil


# 检查给定的 shell 命令是否成功执行
def check_shell(cmd):
    """shell successful."""
    # 执行 shell 命令并将结果存储在 res 中
    res = os.system(cmd)
    # 确保命令执行返回值为 0，表示成功
    assert res == 0


# 检查指定路径下的文件夹数量及其内容
def cli_count_folders_and_check_contents(file_path):
    """" count cli files."""
    # 如果给定的文件路径存在
    if os.path.exists(file_path):
        # 遍历文件夹中的每个文件
        for files in os.listdir(file_path):
            # 获取每个文件的大小
            folder_count = os.path.getsize(os.path.join(file_path, files))
            # 确保文件大小大于 0
            assert folder_count > 0
    # 确保文件夹中的文件数量大于 5
    assert len(os.listdir(file_path)) > 5

# 检查指定路径下的文件夹数量及其内容
def sdk_count_folders_and_check_contents(file_path):
    """count folders."""
    # 如果给定的文件路径存在
    if os.path.exists(file_path):
        # 获取路径的文件大小
        file_count = os.path.getsize(file_path)
        # 确保文件大小大于 0
        assert file_count > 0
    # 如果路径不存在，则退出程序
    else:
        exit(1)


# 删除指定路径的文件或文件夹
def delete_file(path):
    """delete file."""
    # 如果路径不存在
    if not os.path.exists(path):
        # 检查路径是否为文件
        if os.path.isfile(path):
            try:
                # 尝试删除文件
                os.remove(path)
                # 输出文件删除成功的信息
                print(f"File '{path}' deleted.")
            # 捕获删除文件时的类型错误
            except TypeError as e:
                # 输出错误信息
                print(f"Error deleting file '{path}': {e}")
    # 如果路径是一个目录
    elif os.path.isdir(path):
        try:
            # 尝试递归删除目录及其内容
            shutil.rmtree(path)
            # 输出目录删除成功的信息
            print(f"Directory '{path}' and its contents deleted.")
        # 捕获删除目录时的类型错误
        except TypeError as e:
            # 输出错误信息
            print(f"Error deleting directory '{path}': {e}")
```