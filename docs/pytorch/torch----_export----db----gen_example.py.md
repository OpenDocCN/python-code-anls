# `.\pytorch\torch\_export\db\gen_example.py`

```
# 导入必要的标准库模块
import os
import sys

# 导入需要使用的自定义模块
import torch._export.db.examples as examples

# 定义一个模板字符串，用于生成导出函数的 Python 脚本
TEMPLATE = '''import torch

def {case_name}(x):
    """
    """

    return
'''

# 主程序入口，检查命令行参数数量是否为2
if __name__ == "__main__":
    assert len(sys.argv) == 2  # 断言确保命令行参数数量为2，即脚本名和一个参数名
    # 根据模块 examples 的名称生成一个文件路径，替换模块名中的点号为斜杠
    root_dir = examples.__name__.replace(".", "/")
    assert os.path.exists(root_dir)  # 断言确保生成的目录路径存在
    
    # 打开一个文件以写入模板生成的代码，文件名为从命令行参数获得的参数名加上 '.py'
    with open(os.path.join(root_dir, sys.argv[1] + ".py"), "w") as f:
        print("Writing to", f.name, "...")
        # 将模板代码写入文件，使用命令行参数中的名称替换模板中的 {case_name}
        f.write(TEMPLATE.format(case_name=sys.argv[1]))
```