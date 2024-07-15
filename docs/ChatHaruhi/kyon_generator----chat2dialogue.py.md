# `.\Chat-Haruhi-Suzumiya\kyon_generator\chat2dialogue.py`

```py
# Chat凉宫春日 project (https://github.com/LC1332/Chat-Haruhi-Suzumiya)
# Chat凉宫春日 is a language model project that mimics characters from the Haruhi Suzumiya series
# of anime, aiming to chat in a similar style, personality, and storyline. It was developed by
# Li Lulu, Leng Ziaong, Yan Chenxi, Feng Xiaoyang, scixing, Shen Junyi, Aria Fei, Wei Mi, Wu Pingyu,
# Jia Yaokai, and others.

# 这个程序用来将一句话的对话转化为连续的一段对话
# This program converts single-sentence chat into continuous dialogues.

# 用法：python chat2dialogue.py -input_chat <input_chat> -output_dialogue <output_dialogue> -config <config_file> -role_name <role_name> -other_names <other_names>
# Usage: python chat2dialogue.py -input_chat <input_chat> -output_dialogue <output_dialogue> -config <config_file> -role_name <role_name> -other_names <other_names>
# 其中，input_chat是聊天记录文件，output_dialogue是输出文件，如果不指定，则默认为input_chat_to_dialogue.jsonl
# Where input_chat is the chat record file, output_dialogue is the output file (default is input_chat_to_dialogue.jsonl).
# config_file是配置文件，如果不指定，则默认为config.ini
# config_file is the configuration file (default is config.ini).
# role_name是主角的名字，如果不指定，则默认为春日
# role_name is the name of the protagonist (default is "春日" which means "Haruhi").
# other_names是主角的其他名字（对于凉宫春日来说，有凉宫春日，凉宫），如果不指定，则默认为空
# other_names are other names of the protagonist (e.g., for Haruhi Suzumiya, includes "凉宫春日", "凉宫"; default is empty).

# 现在ChatGPT类中间的divide_story函数还没有实现，应该实现之后就基本能跑了
# The divide_story function in the ChatGPT class is not yet implemented. Once implemented, the program should be runnable.
# 关于config.ini的配置，请咨询冷子昂和闫晨曦
# For configurations in config.ini, please consult Leng Ziaong and Yan Chenxi.

# -*- coding: utf-8 -*-
# @Time    : 2023/8/1 18:58
# @Author  : scixing, chenxi
# @FileName: chat2dialogue.py
# @Software: PyCharm
# @github  ：https://github.com/LC1332/Chat-Haruhi-Suzumiya

import json
import argparse
import configparser
import os

from ChatGPT_for_generation import ChatGPT  # Importing the ChatGPT class for text generation

from tqdm import tqdm  # Importing tqdm for progress bars


def load_chat(filename):
    # Load chat records from a JSONL file into a list of dictionaries
    with open(filename, 'r') as f:
        chats = [json.loads(line) for line in f]
    return chats


def save_dialogue(filename, dialogue):
    # Save dialogue to a JSONL file
    with open(filename, 'w+', encoding='utf-8') as f:
        for message in dialogue:
            f.write(json.dumps(message, ensure_ascii=False) + '\n')


def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Chat to Dialogue Conversion, output_dialogue and input_chat should be in the same directory')
    parser.add_argument('-input_chat', type=str, required=True, help='input chat file (jsonl)')
    parser.add_argument('-output_dialogue', type=str, default=None, help='output dialogue file (jsonl)')
    parser.add_argument('-role_name', type=str, required=True, help='role name')
    parser.add_argument('-other_names', nargs="+", default="", type=str, help='other names')

    # arugments that figure out temporary saving folder
    parser.add_argument('-temp_save_folder', default=None, type=str, help='temporary saving file path')

    return parser.parse_args()


def merge_dialogue(user_message, dialogue_text):
    # Merge user messages with existing dialogue text
    dialogue_list = dialogue_text.split('\n')  # Split dialogue into lines
    dialogue = [user_message]
    current_role = ""
    current_text = ""

    for line in dialogue_list:
        if line:
            ch = ":" if ":" in line else "："  # Determine the role delimiter
            if ch in line:
                parts = line.split(ch)
                role = parts[0].strip().replace("凉宫春日", "春日")  # Normalize role name
                text = parts[1].strip()
            else:
                role = ""
                text = line

            if role == current_role:
                current_text = current_text[:-1]
                current_text += text[1:]  # Concatenate text if the same role continues
            else:
                if current_role != "":
                    dialogue.append({"role": current_role, "text": current_text})
                current_role = role
                current_text = text

    dialogue.append({"role": current_role, "text": current_text})  # Append the last dialogue
    return {"dialogue": dialogue, "source": "synthesized"}
# 定义一个函数，用于处理主程序的逻辑
def main(input_chat, output_dialogue, role_name, other_names, temp_save_folder):
    # 创建一个 ConfigParser 对象，用于解析配置文件
    config = configparser.ConfigParser()
    # 读取指定路径下的配置文件，使用 UTF-8 编码解析
    config.read("../src_reform/config.ini", encoding='utf-8')
    # 检查指定的角色名是否在配置文件的节（sections）中
    if role_name not in config.sections():
        # 如果角色名不在配置文件中，输出提示信息
        print(f"{role_name} 角色未创建，请创建角色后再使用，或是与config.ini 中角色一致")
    else:
        # 加载聊天数据
        chat_data = load_chat(input_chat)

        # 加载配置
        configuration = {}

        # 输出配置项
        print(config.items)
        # 获取特定角色的配置项
        items = config.items(role_name)
        for key, value in items:
            configuration[key] = value

        # 初始化 ChatGPT
        chatgpt = ChatGPT(configuration)
        # 预加载模型
        chatgpt.preload()
        # 设置角色训练
        chatgpt.set_training(role_name, other_names)
        dialogue = []
        # 生成对话
        print("Generating dialogue...")

        # 如果临时保存文件夹不存在，则创建它
        if not os.path.exists(temp_save_folder):
            os.mkdir(temp_save_folder)
            print(f"创建临时文件夹{temp_save_folder}")

        # 遍历聊天数据
        for i, chat in enumerate(tqdm(chat_data)):
            role = chat['role']
            text = chat['text']

            # 生成文件名，以索引和文本前4个字符作为基础
            file_name = f"{i}_{text[:min(4,len(text))]}.jsonl"

            # 替换文件名中的无效字符
            file_name = file_name.replace("/", "_")

            # 如果目标文件已经存在于临时保存文件夹中，则跳过
            if os.path.exists(os.path.join(temp_save_folder, file_name)):
                continue

            # 构造用户消息
            user_message = f'{role}:「{text}」'

            # 获取 ChatGPT 的回复
            response = chatgpt.get_response(user_message, [])
            # 将对话合并为临时对话列表
            temp_dialogue = [merge_dialogue(user_message, response)]
            # 保存对话到临时文件
            save_dialogue(os.path.join(temp_save_folder, file_name), temp_dialogue)

        # 保存对话到输出文件

        # 构建输出文件名，如果未指定则使用输入文件名加后缀 "_to_dialogue.jsonl"
        output_dialogue = f'{input_chat[:-4]}_to_dialogue.jsonl' if output_dialogue is None else output_dialogue

        # 打开输出文件，写入临时文件夹中的所有 .jsonl 文件内容
        with open(output_dialogue, 'w', encoding='utf-8') as outfile:
            for filename in os.listdir(temp_save_folder):
                if filename.endswith('.jsonl'):
                    filepath = os.path.join(temp_save_folder, filename)

                    # 尝试打开文件，最多重试三次以避免打开失败
                    for i in range(3):
                        try:
                            with open(filepath) as infile:
                                for line in infile:
                                    outfile.write(line)
                            break
                        except:
                            if i == 2:
                                # 打开文件失败警告
                                print(f"Warning: Failed to open file {filename} after 3 attempts, skipping...")

                    # 打开文件的备选实现（已注释）
                    # with open(filepath) as infile:
                    #     for line in infile:
                    #         outfile.write(line)
if __name__ == '__main__':
    # 检查当前脚本是否作为主程序运行

    args = parse_args()
    # 解析命令行参数，并存储到变量 args 中

    input_chat = args.input_chat
    # 从 args 中获取输入聊天数据的路径

    output_dialogue = args.output_dialogue
    # 从 args 中获取输出对话数据的路径

    role_name = args.role_name
    # 从 args 中获取角色名称

    other_names_lis = args.other_names
    # 从 args 中获取其他名称列表

    temp_save_folder = args.temp_save_folder
    # 从 args 中获取临时保存文件夹路径

    if temp_save_folder == None:
        # 如果临时保存文件夹路径为空
        # 创建一个默认的文件夹名，格式为 output_<role_name>
        temp_save_folder = f"output_{role_name}"

    main(input_chat, output_dialogue, role_name, other_names_lis, temp_save_folder)
    # 调用主函数 main，传递解析得到的参数进行处理
```