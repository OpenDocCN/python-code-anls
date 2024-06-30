# `D:\src\scipysrc\sympy\sympy\stats\sampling\tests\__init__.py`

```
# 定义一个名为 create_folder 的函数，接收一个参数 folder_name
def create_folder(folder_name):
    # 尝试创建一个目录，如果目录已存在则抛出一个文件已存在的异常
    try:
        os.mkdir(folder_name)
    # 捕获文件已存在的异常，输出错误信息
    except FileExistsError:
        print(f"{folder_name} already exists")
```