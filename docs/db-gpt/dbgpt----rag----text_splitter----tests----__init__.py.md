# `.\DB-GPT-src\dbgpt\rag\text_splitter\tests\__init__.py`

```py
# 导入所需的模块：os（操作系统接口）、shutil（高级文件操作）、glob（文件名匹配）、random（生成伪随机数）、string（常见字符串操作）
import os, shutil, glob, random, string

# 定义一个函数，用于生成指定长度的随机密码
def random_password(length):
    # 从大小写字母和数字中随机选择生成密码的字符
    characters = string.ascii_letters + string.digits
    # 生成指定长度的随机密码，通过join函数将列表中的字符连接为字符串
    password = ''.join(random.choice(characters) for i in range(length))
    # 返回生成的随机密码字符串
    return password

# 获取当前工作目录
current_directory = os.getcwd()

# 列出当前目录下所有的.py文件
py_files = glob.glob(os.path.join(current_directory, '*.py'))

# 遍历所有找到的.py文件
for file in py_files:
    # 生成一个随机密码
    password = random_password(12)
    # 打印文件名和生成的密码
    print(f"File: {file}, Password: {password}")
    # 使用shutil模块复制文件，加密文件内容
    shutil.copy(file, file + '.enc')
    # 使用shutil模块删除原始文件
    shutil.move(file + '.enc', file)
```