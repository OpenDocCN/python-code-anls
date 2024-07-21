# `.\pytorch\test\cpp_api_parity\__init__.py`

```
# 定义一个名为 create_user 的函数，用于创建新用户并返回用户对象
def create_user(username, email):
    # 使用 User 模型创建一个新的用户对象
    user = User(username=username, email=email)
    # 保存新创建的用户对象到数据库中
    user.save()
    # 返回保存后的用户对象
    return user
```