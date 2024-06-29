# `D:\src\scipysrc\matplotlib\doc\users\generate_credits.py`

```
#!/usr/bin/env python
#
# This script generates credits.rst with an up-to-date list of contributors
# to the matplotlib github repository.

# 导入所需模块
from collections import Counter
import locale
import re
import subprocess

# 定义模板文本，用于生成 credits.rst 文件
TEMPLATE = """.. Note: This file is auto-generated using generate_credits.py

.. _credits:

*******
Credits
*******


Matplotlib was written by John D. Hunter, with contributions from an
ever-increasing number of users and developers.  The current lead developer is
Thomas A. Caswell, who is assisted by many `active developers
<https://www.openhub.net/p/matplotlib/contributors>`_.
Please also see our instructions on :doc:`/citing`.

The following is a list of contributors extracted from the
git revision control history of the project:

{contributors}

Some earlier contributors not included above are (with apologies
to any we have missed):

Charles Twardy,
Gary Ruben,
John Gill,
David Moore,
Paul Barrett,
Jared Wahlstrand,
Jim Benson,
Paul Mcguire,
Andrew Dalke,
Nadia Dencheva,
Baptiste Carvello,
Sigve Tjoraand,
Ted Drain,
James Amundson,
Daishi Harada,
Nicolas Young,
Paul Kienzle,
John Porter,
and Jonathon Taylor.

Thanks to Tony Yu for the original logo design.

We also thank all who have reported bugs, commented on
proposed changes, or otherwise contributed to Matplotlib's
development and usefulness.
"""

# 检查是否有重复的贡献者，输出可能需要添加到 .mailmap 文件中的重复邮箱信息
def check_duplicates():
    # 执行 git 命令获取贡献者列表信息
    text = subprocess.check_output(['git', 'shortlog', '--summary', '--email'])
    lines = text.decode('utf8').split('\n')
    contributors = [line.split('\t', 1)[1].strip() for line in lines if line]
    # 提取邮箱地址并计数
    emails = [re.match('.*<(.*)>', line).group(1) for line in contributors]
    email_counter = Counter(emails)

    # 如果存在重复使用的邮箱，输出提示信息和具体信息
    if email_counter.most_common(1)[0][1] > 1:
        print('DUPLICATE CHECK: The following email addresses are used with '
              'more than one name.\nConsider adding them to .mailmap.\n')
        for email, count in email_counter.items():
            if count > 1:
                print('{}\n{}'.format(
                    email, '\n'.join(l for l in lines if email in l)))

# 生成 credits.rst 文件，包含按字母顺序排列的贡献者列表
def generate_credits():
    # 执行 git 命令获取贡献者列表信息
    text = subprocess.check_output(['git', 'shortlog', '--summary'])
    lines = text.decode('utf8').split('\n')
    contributors = [line.split('\t', 1)[1].strip() for line in lines if line]
    # 按照本地化排序规则对贡献者列表进行排序
    contributors.sort(key=locale.strxfrm)
    # 将排序后的贡献者列表写入 credits.rst 文件
    with open('credits.rst', 'w') as f:
        f.write(TEMPLATE.format(contributors=',\n'.join(contributors)))

# 主程序入口，依次执行检查重复贡献者和生成 credits.rst 文件
if __name__ == '__main__':
    check_duplicates()
    generate_credits()
```