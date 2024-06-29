# `.\numpy\tools\ci\test_all_newsfragments_used.py`

```
#!/usr/bin/env python3
# 导入系统相关的库
import sys
# 导入 TOML 格式解析库
import toml
# 导入操作系统相关功能的库
import os

# 主函数定义
def main():
    # 从 pyproject.toml 文件中加载配置路径
    path = toml.load("pyproject.toml")["tool"]["towncrier"]["directory"]

    # 获取指定路径下的所有文件和目录列表
    fragments = os.listdir(path)
    # 移除特定文件名以保证正确性
    fragments.remove("README.rst")
    fragments.remove("template.rst")

    # 如果仍有未找到的文件
    if fragments:
        # 打印未被 towncrier 找到的文件列表
        print("The following files were not found by towncrier:")
        print("    " + "\n    ".join(fragments))
        # 以错误状态退出程序
        sys.exit(1)

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 调用主函数
    main()
```