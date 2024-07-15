# `.\Chat-Haruhi-Suzumiya\kyon_generator\synthesis_chat_from_story.py`

```py
# 导入需要的模块和库
from argparse import Namespace  # 导入命名空间模块
import glob  # 导入文件匹配模块
import os  # 导入操作系统模块
import argparse  # 导入命令行参数解析模块
import random  # 导入随机数模块
import openai  # 导入OpenAI API模块
import json  # 导入JSON处理模块

"""
python3 synthesis_chat_from_story.py --role_name "乔峰" --world_name "天龙八部" --story_folder "/Users/pufferfish/Chat-Haruhi-Suzumiya/characters/qiaofeng/texts" --output "/Users/pufferfish/Chat-Haruhi-Suzumiya/kyon_generator/qiaofeng.txt"
"""

# 设置OpenAI API的基本URL
openai.api_base = "http://api.ai-gaochao.com/v1"
# 设置OpenAI API的密钥
api_key1 = "sk-RwVzjYxlKJTxUDZE89"
api_key2 = "CaA512C6164a92A683E55f60Ce953b"
openai.api_key = api_key1 + api_key2

# 定义指导语句，包含角色名称和世界名称的占位符
instruction = "You are asked to come up with a set of 10 diverse dialogues. These dialogues will be used to test a ChatBot that plays the role of {role_name} from the {world_name}. We will evaluate how well this ChatBot completes these dialogues. "

# 定义详细要求文本
requirements = """
You are asked to come up with a set of 10 diverse dialogues. These dialogues will be used to test a ChatBot that plays the role of {role_name} from the {world_name}. We will evaluate how well this ChatBot completes these dialogues.
The requirements are:

1. Try not to repeat verbs in the dialogues to maximize diversity. 

2. The language used in the dialogues also should be diverse. For example, you should combine statements and questions.

3. The types of dialogues should be diverse. It should include open-ended questions, questions about the ChatBot's identity, suggestions for activities, pushing the story forward, etc.

4. The ChatBot should be able to answer these questions. For example, do not ask the ChatBot to generate any visual or audio output. Also, do not ask the ChatBot to perform any actions.

5. The dialogues should be in Chinese. 

6. Each dialogue should be 1-2 sentences long. Statements and questions are permitted.

7. You should generate appropriate questioning input for each dialogue. The input should provide an engaging context, ideally no more than 100 words.
"""

# 定义函数：查找列表中以特定前缀开头的元素的索引列表
def find_elements_with_prefix(my_list, prefix):
    return [index for index, item in enumerate(my_list) if item.startswith(prefix)]

# 定义函数：从列表中获取所有角色名称（假设每个元素格式为"角色名称-其他信息"）
def get_all_characters(my_list):
    return list(set([item[:2] for item in my_list]))

# 定义函数：合并列表中连续相同前缀的元素
def merge_list(mylist):
    merged_list = [i for i in mylist]
    
    jumped_index = []
    cur_value = mylist[0]
    for index, value in enumerate(mylist):
        if value[:2] == cur_value[:2]:
            jumped_index.append(index)
            merged_list[index] = merged_list[index] + mylist[index]
        else:
            cur_value = value
            
    # 将被合并的元素置为空字符串
    for i, v in enumerate(merged_list):
        if i in jumped_index:
            merged_list[i] = ""
    
    # 将第一个元素重新设置为原始列表的第一个元素
    merged_list[0] = mylist[0]
    
    # 注释掉的代码段，似乎是注释掉的部分
    # lst = []
    # print(len(jumped_index))
    # 遍历 merged_list 中的每个元素及其索引
    for i,v in enumerate(merged_list):
        # 如果当前元素不为空字符串
        if (v != ''):
            # 将 mylist 中对应索引处的元素与 v 合并，并更新 merged_list
            merged_list[i] = mylist[i] + "\n" + v
        # 如果当前元素为空字符串且索引 i 不在 jumped_index 中
        if (v == '' and i not in jumped_index):
            # 将 mylist 中对应索引处的元素赋给 merged_list
            merged_list[i] = mylist[i]
            # 将索引 i 添加到 lst 列表中
            lst.append(i)
            # 输出分隔线和相关信息，用于调试
            print("________")
            print("这个是原句的i : ", i, mylist[i])
            print("这个是一个合并后的v : ", v)
            print(merged_list[i])
            print("________")
            # 增加计数器
            counter += 1
    # 打印合并后的 merged_list
    print(merged_list)
    # 打印计数器的值
    print("this is counter : ", counter)
    # 打印 jumped_index 列表的内容
    print(jumped_index)
    # 打印 lst 列表的内容
    print(lst)
    # 创建一个过滤后的列表，排除 merged_list 中的空字符串元素
    filtered_list = [item for item in merged_list if item != '']
    # 打印过滤后列表的长度
    print("这个是filtered ", len(filtered_list))
    # 重置 lst 列表为空列表
    # lst = []
    # 返回合并后的 merged_list
    return merged_list
# 生成示例列表，从给定的列表中随机抽取指定数量的元素对，作为示例
def generate_examples(my_list, role_index, number):
    # 从 role_index 中随机选择 number 个索引位置
    random_elements = random.sample(role_index, number)
    # 构建示例列表，每个示例是一个列表，包含 my_list 中两个相邻元素
    example_lst = [[my_list[i-1], my_list[i]] for i in random_elements]
    return example_lst

# 查找列表中每个标记之前的元素，并将它们收集到结果列表中
def find_elements_before_marker(lst, marker):
    result = []
    for i in range(len(lst)):
        if lst[i] == marker:
            if i > 0:  # 确保不访问列表开始之前的索引
                result.append(lst[i - 1])
    return result

# 将特定字符串保存为 JSON 格式，其中包括从文本中提取的对话信息
def save2json(role_name, mystring, output):
    pass
    lines_list = mystring.splitlines()
    r_list = find_elements_before_marker(lines_list, "ChatBot Answer:")
    for i in r_list:
        dialogue = i.split(":")
        if role_name not in dialogue[0]:
            role = dialogue[0]
            if "「" in dialogue[1]:
                text = dialogue[1].replace("「", "")
            if "」" in dialogue[1]:
                text = text.replace("」", "")
            savejson = {"role": f"{role}", "text": f"{text}", "source": "synthesized"}
            json_string = json.dumps(savejson, ensure_ascii=False)
            with open(output, "a", encoding='utf-8') as file:
                file.write(json_string + "\n")

# 从 JSON 文件中删除重复的条目，并将结果写入新的文件中
def remove_duplicates(file_path):
    pass
    seen_texts = set()
    with open(file_path, encoding='utf-8') as f_in:
        with open(file_path + "nodup", 'w', encoding='utf-8') as f_out:
            for line in f_in:
                obj = json.loads(line)
                text = obj['text']
                if text not in seen_texts:
                    f_out.write(line)
                    seen_texts.add(text)

# 综合处理函数，根据给定的参数合成文本处理结果
def synthesis(**params):
    pass
    # 根据参数提取角色名称、世界名称、故事文件夹路径和输出路径
    if params.get("role_name"):
        role_name = params["role_name"]
    if params.get("world_name"):
        world_name = params["world_name"]
    if params.get("story_folder"):
        story_folder = params["story_folder"]
    if params.get("output"):
        output = params["output"]
    
    # 获取指定文件夹中所有的 .txt 文件路径
    txt_files = glob.glob(os.path.join(story_folder, '*.txt'))
    all_stories = []
    
    # 读取每个文本文件的内容，并将其加入到 all_stories 列表中
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as file:
            content = file.readlines()
            all_stories += content

    # 合并所有故事内容为一个字符串
    merged_stories = merge_list(all_stories)

    # 根据角色名称在合并后的故事中查找相关元素列表
    role_list = find_elements_with_prefix(merged_stories, role_name)
    
    # 从合并的故事中生成随机示例列表
    random_examples = generate_examples(merged_stories, role_list, 5)
    print(random_examples)
    
    # 初始化内容字符串，用于存储生成的要求内容
    content = requirements
    
    # 遍历随机示例列表，生成内容字符串
    for i in random_examples:
        if len(i[0]) < 10 or len(i[1]) < 10:
            continue
        content += "\n"
        content += "###"
        content += "\n"
        content += "Question:"
        content += "\n"
        content += i[0]
        content += "ChatBot Answer:"
        content += "\n"
        content += i[1]
    
    content += "\n"
    content += "###"
    content += "\n"
    content += "Question:"
    # 调用 OpenAI API 中的 ChatCompletion 接口生成对话内容
    result = openai.ChatCompletion.create(
        model="gpt-4",  # 指定要使用的模型为 "gpt-4"
        messages=[
            {"role": "system", "content": instruction},  # 添加系统角色的指令消息
            {"role": "user", "content": content},  # 添加用户角色的内容消息
        ],
        temperature = 0.1  # 设置温度参数为 0.1，控制生成文本的多样性
    )
    # 调用 save2json 函数，将生成的对话内容保存到 JSON 文件中
    save2json(role_name, result['choices'][0]['message']['content'], output)
    # 调用 remove_duplicates 函数，移除输出文件中的重复内容
    remove_duplicates(output)
if __name__ == '__main__':
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    parser.add_argument('--role_name', type=str)
    # 添加名为 role_name 的命令行参数，类型为字符串

    parser.add_argument('--world_name', type=str)
    # 添加名为 world_name 的命令行参数，类型为字符串

    parser.add_argument('--story_folder', type=str)
    # 添加名为 story_folder 的命令行参数，类型为字符串

    parser.add_argument('--output', type=str)
    # 添加名为 output 的命令行参数，类型为字符串
    
    args = parser.parse_args()
    # 解析命令行参数并存储到 args 中

    params_dict = vars(args)
    # 将命令行参数转换为字典形式保存在 params_dict 中

    for j in range(50):
        # 循环50次
        try:
            synthesis(**params_dict)
            # 调用 synthesis 函数，传入命令行参数字典作为关键字参数
        except:
            # 捕获所有异常
            continue
            # 发生异常时继续下一次循环

    # 下面是被注释掉的代码段，不会被执行
    # folder_names = ['tangshiye','weixiaobao','murongfu','liyunlong',\
    #             'Luna','wangduoyu','Ron','jiumozhi',\
    #             'Snape','haruhi','Malfoy','xuzhu',\
    #             'xiaofeng','duanyu','Hermione','Dumbledore',\
    #             'wangyuyan','qiaofeng',\
    #             'yuqian','Harry','McGonagall' ,\
    #             'baizhantang','tongxiangyu','guofurong',\
    #             'wanderer','zhongli','hutao',\
    #             'Sheldon','Raj','Penny']

    # role_names = ['汤师爷','韦小宝','慕容复','李云龙',\
    #           'Luna','王多鱼','Ron','鸠摩智',\
    #           'Snape','春日','Malfoy','虚竹',\
    #           '萧峰','段誉','Hermione','Dumbledore',\
    #           '王语嫣','乔峰',\
    #           '于谦','Harry','Professor McGonagall',\
    #           '白展堂','佟湘玉','郭芙蓉',\
    #           '流浪者','钟离','胡桃',\
    #           'Sheldon','Raj','Penny']

    # world_names = ['让子弹飞','鹿鼎记','天龙八部','亮剑',\
    #           'HarryPotter','西虹市首富','HarryPotter','天龙八部',\
    #           'HarryPotter','凉宫春日','HarryPotter','天龙八部',\
    #           '天龙八部','天龙八部','HarryPotter','HarryPotter',\
    #           '天龙八部','天龙八部',\
    #           '相声','HarryPotter','HarryPotter',\
    #           '同福客栈','同福客栈','同福客栈',\
    #           '原神','原神','原神',\
    #           'BigBang', 'BigBang','BigBang']          

    # for i,v in enumerate(folder_names):
    #     folder_path = "/Users/pufferfish/Chat-Haruhi-Suzumiya/characters/"
    #     out_path = "/Users/pufferfish/Chat-Haruhi-Suzumiya/kyon_generator/"
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--role_name', type=str, default=role_names[i])
    #     parser.add_argument('--world_name', type=str, default=world_names[i])
    #     parser.add_argument('--story_folder', type=str, default=folder_path+v+"/texts")
    #     parser.add_argument('--output', type=str, default=out_path+v)
    #     args = parser.parse_args()
    #     params_dict = vars(args)
    #     for j in range(50):
    #         try:
    #             synthesis(**params_dict)
    #         except:
    #             continue
    # remove_duplicates(params_dict.get("output"))
```