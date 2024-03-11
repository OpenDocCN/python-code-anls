# `.\Langchain-Chatchat\copy_config_example.py`

```
# 导入必要的库
import os  # 导入操作系统相关的库
import shutil  # 导入文件操作相关的库

# 主程序入口
if __name__ == "__main__":
    # 获取configs目录下的所有文件列表
    files = os.listdir("configs")

    # 筛选出所有以.example结尾的文件，并构建源文件路径列表
    src_files = [os.path.join("configs", file) for file in files if ".example" in file]

    # 遍历源文件路径列表
    for src_file in src_files:
        # 构建目标文件路径，去掉.example后缀
        tar_file = src_file.replace(".example", "")
        # 复制源文件到目标文件
        shutil.copy(src_file, tar_file)
```