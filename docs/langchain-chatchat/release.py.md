# `.\Langchain-Chatchat\release.py`

```
# 导入必要的模块
import os
import subprocess
import re

# 获取最新的 Git 标签
def get_latest_tag():
    # 执行 git tag 命令获取标签信息
    output = subprocess.check_output(['git', 'tag'])
    # 将输出解码为字符串并按换行符分割，去除最后一个空字符串
    tags = output.decode('utf-8').split('\n')[:-1]
    # 根据版本号排序标签，取最新的标签
    latest_tag = sorted(tags, key=lambda t: tuple(map(int, re.match(r'v(\d+)\.(\d+)\.(\d+)', t).groups())))[-1]
    return latest_tag

# 更新版本号
def update_version_number(latest_tag, increment):
    # 从最新标签中提取主版本号、次版本号和修订号
    major, minor, patch = map(int, re.match(r'v(\d+)\.(\d+)\.(\d+)', latest_tag).groups())
    # 根据用户选择的递增部分更新版本号
    if increment == 'X':
        major += 1
        minor, patch = 0, 0
    elif increment == 'Y':
        minor += 1
        patch = 0
    elif increment == 'Z':
        patch += 1
    # 组装新的版本号字符串
    new_version = f"v{major}.{minor}.{patch}"
    return new_version

# 主函数
def main():
    print("当前最近的Git标签：")
    latest_tag = get_latest_tag()
    print(latest_tag)

    print("请选择要递增的版本号部分（X, Y, Z）：")
    increment = input().upper()

    # 确保用户输入正确的递增部分
    while increment not in ['X', 'Y', 'Z']:
        print("输入错误，请输入X, Y或Z：")
        increment = input().upper()

    # 更新版本号
    new_version = update_version_number(latest_tag, increment)
    print(f"新的版本号为：{new_version}")

    print("确认更新版本号并推送到远程仓库？（y/n）")
    confirmation = input().lower()

    # 根据用户确认更新版本号并推送到远程仓库
    if confirmation == 'y':
        subprocess.run(['git', 'tag', new_version])
        subprocess.run(['git', 'push', 'origin', new_version])
        print("新版本号已创建并推送到远程仓库。")
    else:
        print("操作已取消。")

# 程序入口
if __name__ == '__main__':
    main()
```