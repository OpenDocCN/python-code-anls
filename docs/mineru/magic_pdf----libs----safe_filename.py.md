# `.\MinerU\magic_pdf\libs\safe_filename.py`

```
# 导入操作系统相关的模块
import os


# 定义一个函数，用于清理文件名，将无效字符替换为指定的字符
def sanitize_filename(filename, replacement="_"):
    # 检查操作系统名称，如果是 Windows 系统
    if os.name == 'nt':
        # 定义 Windows 系统中无效的字符
        invalid_chars = '<>:"|?*'

        # 遍历无效字符列表
        for char in invalid_chars:
            # 将文件名中的无效字符替换为指定的替换字符
            filename = filename.replace(char, replacement)

    # 返回处理后的文件名
    return filename
```