# `.\Chat-Haruhi-Suzumiya\kyon_generator\story2chat.py`

```py
# 导入必要的库
import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统功能模块
import glob  # 导入文件名匹配模块
import json  # 导入处理JSON数据的模块

# 处理对话的函数，从输入的文件中提取非主角的对话，并写入输出文件
def process_dialogue(input_files, output_file, role, other_names):
    result = []  # 初始化结果列表
    output_dir = os.path.abspath(os.path.dirname(output_file))  # 获取输出文件的目录路径
    if not os.path.exists(output_dir):  # 如果输出目录不存在，则创建之
        os.makedirs(output_dir)
    cnt = 0  # 计数器，用于统计处理的文件数目
    for file in input_files:  # 遍历输入文件列表
        cnt += 1
        f_read = open(file, 'r', encoding='utf-8')  # 打开文件进行读取
        lines = f_read.readlines()  # 读取文件的所有行
        last_content = ""  # 初始化上一个对话内容为空字符串
        for line in lines:  # 遍历文件的每一行
            if ":" in line:  # 如果行中包含冒号
                current_role = line.split(":")[0]  # 以冒号分割，获取当前角色名
            elif '：' in line:  # 如果行中包含中文冒号
                current_role = line.split("：")[0]  # 以中文冒号分割，获取当前角色名
            else:
                current_role = ""  # 否则当前角色名为空字符串
            
            # 判断当前角色名是否为主角或其他名字列表中的角色
            if current_role in other_names + [role]:
                if not last_content == "":
                    result.append(last_content)  # 将上一个对话内容加入结果列表
                last_content = ""  # 重置上一个对话内容为空字符串
            else:
                last_content = line  # 更新上一个对话内容为当前行
    return generage_jsonl(result, output_file)  # 调用生成JSONL文件的函数，传入结果列表和输出文件路径

# 生成JSONL文件的函数，将结果列表中的对话写入指定的输出文件
def generage_jsonl(result, output_file):
    fw = open(output_file, 'w+', encoding='utf-8')  # 以写入模式打开输出文件
    """
    {"role": "阿虚", "text": "「奇特的社团和运动社团有什么不同？」", "source": "synthesized "}
    """
    # 去除结果列表中的重复元素
    seen = set()
    new_result = []
    for item in result:
        if item not in seen:
            seen.add(item)
            new_result.append(item)
            
    for content in new_result:  # 遍历去重后的结果列表
        content = content.strip()  # 去除内容两侧的空白字符
        if content:
            if ":" in content:  # 如果内容中包含冒号
                res = content.split(':')  # 以冒号分割内容，获取角色名和对话文本
            elif '：' in content:  # 如果内容中包含中文冒号
                res = content.split('：')  # 以中文冒号分割内容，获取角色名和对话文本
            if res[1] != '':  # 如果对话文本部分不为空
                text = res[1]  # 获取对话文本
                if text[0] == "「":  # 如果文本以「开头
                    text = text[1:]  # 去除开头的「
                if text[-1] == "」":  # 如果文本以」结尾
                    text = text[:-1]  # 去除结尾的」
                json.dump({"role": res[0], "text": text , "source": "story"}, fw, ensure_ascii=False)  # 将角色名、文本和来源写入JSONL文件
                fw.write("\n")  # 写入换行符结束本条JSON行
    fw.close()  # 关闭输出文件

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Process story data for Chat Haruhi Suzumiya project',  # 程序描述信息
        epilog='author:LengYue(https://github.com/zealot52099)'  # 程序结尾信息
    )

    parser.add_argument('-story_folder', required=True, help='Story folder path')  # 添加必需的命令行参数，故事文件夹路径
    parser.add_argument('-output', required=True, help='Output file path')  # 添加必需的命令行参数，输出文件路径
    parser.add_argument('-role', default='春日', help='Main role name (default: 春日)')  # 添加默认值的命令行参数，主角名字
    # 添加命令行参数 -other_names，可接受零个或多个参数，默认为空列表，用于指定其他角色名称
    parser.add_argument('-other_names', nargs='*', default=[], help='Other role names (default: None)')
    # 解析命令行参数，并将结果存储在args变量中
    args = parser.parse_args()

    # 从args中获取story_folder参数，表示故事文件夹的路径
    story_folder = args.story_folder
    # 从args中获取output参数，表示输出文件的路径
    output_file = args.output
    # 从args中获取role参数，表示角色名称
    role = args.role
    # 从args中获取other_names参数，表示其他角色名称的列表
    other_names = args.other_names

    # 在story_folder路径下查找所有的txt文件
    txt_files = glob.glob(os.path.join(story_folder, '*.txt'))
    # 如果找不到任何txt文件，则输出提示信息并退出程序（状态码1表示异常退出）
    if not txt_files:
        print(f"No txt files found in {story_folder}")
        exit(1)

    # 调用process_dialogue函数，传入找到的txt文件列表、输出文件路径、角色名称和其他角色名称列表进行处理
    process_dialogue(txt_files, output_file, role, other_names)
# 如果当前脚本被直接执行（而不是被导入为模块），则执行以下代码块
if __name__ == '__main__':
    # 调用主函数进行程序的主要逻辑处理
    main()


# 使用命令行工具来执行脚本 `story2chat.py`，并传入以下参数：
# -story_folder "/characters/haruhi/texts"：指定故事文件夹路径
# -output ./output/chat_from_story.json：指定输出文件路径和文件名
# -role "春日"：指定角色名称为 "春日"
# -other_names 凉宫 凉宫春日：指定其他可能用到的角色名称，包括 "凉宫" 和 "凉宫春日"
```