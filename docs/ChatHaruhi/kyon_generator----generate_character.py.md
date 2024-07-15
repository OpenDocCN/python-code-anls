# `.\Chat-Haruhi-Suzumiya\kyon_generator\generate_character.py`

```py
import argparse  # 导入处理命令行参数的模块
import configparser  # 导入读取配置文件的模块
import json  # 导入处理 JSON 数据的模块
import random  # 导入生成随机数的模块
import sys  # 导入系统相关的模块

sys.path.append('..')  # 将上级目录添加到系统路径中，以便导入自定义模块
from src_reform import utils, checkCharacter  # 导入自定义的工具函数和角色检查模块
import os  # 导入处理操作系统相关功能的模块
import openai  # 导入 OpenAI 平台的 API

# TODO

# 在这个文件中 重新实现embedding，替换 utils.get_embedding

# 把原来的embedding函数 在这里做一个 镜像

# 这里可以有一个比如叫chinese_embedding 的函数 return utils.get_embedding(model, text)
# 这个可以

# 你要实现一个if_chinese的函数，判断一个sentence是不是英文为主

# 建立一个combine_embedding函数， 先调用if_chinese然后再调用chinese_embedding或者english_embedding

# 一定要用 openai text-embedding-ada-002

# 写完之后，在test_kyon_generator.ipynb中跑通
# on the fly 增加 Hermione和Malfoy这两个人物

# 然后测试通他们对应的jsonl

def parse_args():
    parser = argparse.ArgumentParser(description='generate character 将台本文件保存成jsonl文件，动态创建新的角色')
    parser.add_argument('--cn_role_name', type=str, required=True, help='Chinese role name')
    parser.add_argument('--en_role_name', type=str, required=True, help='English role name')
    parser.add_argument('--prompt', default=None, type=str, help='prompt file path')
    parser.add_argument('--text_folder', type=str, help='character texts folder')
    return parser.parse_args()

def generate_character(cn_role_name, en_role_name, prompt=None):
    # 在config.ini中加添角色信息
    config = configparser.ConfigParser()
    # 读取配置文件
    config.read('../src_reform/config.ini', encoding='utf-8')
    configuration = {}
    if cn_role_name in config.sections():
        print(f"已存在{cn_role_name}角色的配置文件")
    else:
        # 添加新的配置项
        config.add_section(cn_role_name)
        config[cn_role_name]['character_folder'] = f"../characters/{en_role_name}"
        config[cn_role_name][
            'image_embed_jsonl_path'] = f"../characters/{en_role_name}/jsonl/image_embed.jsonl"
        config[cn_role_name][
            'title_text_embed_jsonl_path'] = f"../characters/{en_role_name}/jsonl/title_text_embed.jsonl"
        config[cn_role_name]['images_folder'] = f"../characters/{en_role_name}/images"
        config[cn_role_name]["jsonl_folder"] = f"../characters/{en_role_name}/jsonl"
        config[cn_role_name]['texts_folder'] = f"../characters/{en_role_name}/texts"
        config[cn_role_name]['system_prompt'] = f"../characters/{en_role_name}/system_prompt.txt"
        config[cn_role_name]['dialogue_path'] = f"../characters/{en_role_name}/dialogues/"
        config[cn_role_name]['max_len_story'] = "1500"
        config[cn_role_name]['max_len_history'] = "1200"
        config[cn_role_name]['gpt'] = "True"
        config[cn_role_name]['local_tokenizer'] = "THUDM/chatglm2-6b"
        config[cn_role_name]['local_model'] = "THUDM/chatglm2-6b"
        config[cn_role_name]['local_lora'] = "Jyshen/Chat_Suzumiya_GLM2LoRA"
        # 保存修改后的配置文件
        with open('../src_reform/config.ini', 'w+', encoding='utf-8') as config_file:
            config.write(config_file)
        config.read('config.ini', encoding='utf-8')
    # 检查角色文件夹
    items = config.items(cn_role_name)
    print(f"正在加载: {cn_role_name} 角色")
    for key, value in items:
        configuration[key] = value
    # 调用checkCharacter模块的checkCharacter函数，传入configuration参数进行字符检查
    checkCharacter.checkCharacter(configuration)
    
    # 如果prompt不为None，则执行以下操作
    if prompt is not None:
        # 打开prompt文件以只读模式
        fr = open(prompt, 'r')
        
        # 在路径"../characters/{en_role_name}"下创建或打开system_prompt.txt文件，以写入和读取模式，并指定UTF-8编码
        with open(os.path.join(f"../characters/{en_role_name}", 'system_prompt.txt'), 'w+', encoding='utf-8') as f:
            # 将fr文件的内容读取并写入system_prompt.txt文件中
            f.write(fr.read())
            # 打印提示信息，表示system_prompt.txt文件已创建
            print("system_prompt.txt已创建")
        
        # 关闭fr文件
        fr.close()
    
    # 返回configuration变量
    return configuration
# 定义一个存储数据的类 StoreData
class StoreData:
    # 初始化方法，接受配置信息和文本文件夹路径作为参数
    def __init__(self, configuration, text_folder):
        # 从配置信息中获取图片嵌入数据的 JSONL 文件路径
        self.image_embed_jsonl_path = configuration['image_embed_jsonl_path']
        # 从配置信息中获取标题文本嵌入数据的 JSONL 文件路径
        self.title_text_embed_jsonl_path = configuration['title_text_embed_jsonl_path']
        # 从配置信息中获取存储图片的文件夹路径
        self.images_folder = configuration['images_folder']
        # 设置文本文件夹路径
        self.texts_folder = text_folder
        # 下载并初始化模型
        self.model = utils.download_models()

    # 预加载方法，处理标题文本和图片数据
    def preload(self):
        # 存储标题文本和其嵌入向量的列表
        title_text_embed = []
        # 存储标题文本的列表
        title_text = []
        # 遍历文本文件夹中的每个文件
        for file in os.listdir(self.texts_folder):
            # 如果文件是以 .txt 结尾
            if file.endswith('.txt'):
                # 获取标题名（去除后缀 .txt）
                title_name = file[:-4]
                # 打开文件并读取内容，将标题名和内容拼接存入列表
                with open(os.path.join(self.texts_folder, file), 'r', encoding='utf-8') as fr:
                    title_text.append(f"{title_name}｜｜｜{fr.read()}")
        
        # 使用模型获取标题文本的嵌入向量
        embeddings = utils.get_embedding(self.model, title_text)
        # 将标题文本及其嵌入向量以字典形式存入列表
        for title_text, embed in zip(title_text, embeddings):
            title_text_embed.append({title_text: embed})
        
        # 调用 store 方法，将标题文本及其嵌入向量写入到 JSONL 文件中
        self.store(self.title_text_embed_jsonl_path, title_text_embed)

        # 如果存储图片的文件夹不为空
        if len(os.listdir(self.images_folder)) != 0:
            # 存储图片及其嵌入向量的列表
            image_embed = []
            # 存储图片名称的列表
            images = []
            # 遍历存储图片的文件夹中的每个文件
            for file in os.listdir(self.images_folder):
                # 获取图片名称（去除后缀）
                images.append(file[:-4])
            
            # 使用模型获取图片的嵌入向量
            embeddings = utils.get_embedding(self.model, images)
            # 将图片及其嵌入向量以字典形式存入列表
            for image, embed in zip(images, embeddings):
                image_embed.append({image: embed})
            
            # 调用 store 方法，将图片及其嵌入向量写入到 JSONL 文件中
            self.store(self.image_embed_jsonl_path, image_embed)
        
        # 打印角色创建成功的消息
        print("角色创建成功!")

    # 存储数据到 JSONL 文件的方法
    def store(self, path, data):
        # 以写入方式打开指定路径的文件，设置编码为 utf-8
        with open(path, 'w+', encoding='utf-8') as f:
            # 遍历数据列表中的每个条目
            for item in data:
                # 将条目以 JSON 格式写入文件，禁用 ASCII 码转义
                json.dump(item, f, ensure_ascii=False)
                # 写入换行符，分隔不同的 JSON 对象
                f.write('\n')

# 主程序入口
if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 获取中文角色名和英文角色名
    cn_role_name = args.cn_role_name
    en_role_name = args.en_role_name
    # 如果存在提示信息，则获取提示信息；否则设置为 None
    prompt = args.prompt if args.prompt else None
    # 如果存在文本文件夹路径，则使用给定路径；否则默认相对路径
    text_folder = args.text_folder if args.text_folder else f"../characters/{en_role_name}/texts"

    # 生成角色配置文件
    configuration = generate_character(cn_role_name, en_role_name, prompt=prompt)

    # 创建 StoreData 类的实例对象，传入角色配置信息和文本文件夹路径
    run = StoreData(configuration, text_folder)
    # 执行预加载方法，处理数据存储
    run.preload()
```