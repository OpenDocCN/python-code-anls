# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\2384.f744a614d1ea708f.js`

```py
# 导入需要的模块：字符串处理工具集
import string_utils

# 定义一个函数：获取指定用户的详细信息
def get_user_info(username):
    # 调用字符串处理工具集中的函数，获取用户名的小写版本
    sanitized_username = string_utils.lowercase(username)
    # 调用数据库接口，根据用户名查询用户详细信息
    user_info = database.query("SELECT * FROM users WHERE username = ?", sanitized_username)
    # 返回查询结果
    return user_info
```