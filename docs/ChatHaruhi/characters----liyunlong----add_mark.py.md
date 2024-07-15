# `.\Chat-Haruhi-Suzumiya\characters\liyunlong\add_mark.py`

```py
# 导入操作系统模块
import os

# 定义函数，用于给指定文件夹中的文本文件添加特定标记
def add_mark(folder):
    # 遍历文件夹中的每个文件
    for file in os.listdir(folder):
        # 打开当前文件和目标文件，使用UTF-8编码
        with open(os.path.join(folder, file), 'r', encoding='utf-8') as f1, open('texts/'+file, 'w+', encoding='utf-8') as f2:
            # 逐行读取文件内容
            for line in f1.readlines():
                # 打印当前行内容（用于调试）
                print(line)
                # 替换行中的中文冒号为「，处理行末没有换行符的情况
                line = line.replace('：', ':「')
                if '\n' not in line:
                    line = line + ('」')
                else:
                    line = line.replace('\n', '」')
                # 打印修改后的行内容（用于调试）
                print(line)
                # 将修改后的行写入目标文件
                f2.write(line)
                # 写入换行符
                f2.write('\n')

# 调用函数，处理名为 'texts_source' 的文件夹中的文件
add_mark('texts_source')

# 下面是注释掉的部分，使用循环创建文件并打印数字 1，已被注释掉，不会执行
# for i in range(1, 15):
#     with open(f'texts_source/{i}.txt', 'w+') as f:
#         print(1)
```