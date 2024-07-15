# `.\Chat-Haruhi-Suzumiya\kyon_generator\ChatGPT_for_generation.py`

```py
# 导入必要的库和模块
import json
import os
import numpy as np
import openai
import tiktoken
import torch
from scipy.spatial.distance import cosine
from langchain.chat_models import ChatOpenAI
import random
import time
import collections
import pickle
from argparse import Namespace
from PIL import Image
from torch import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import sys

# 添加上级目录到系统路径，用于引入上级目录的模块
sys.path.append("..")
# 导入上级目录中的utils模块
from src_reform import utils
# 导入正则表达式模块
import re

# Chat凉宫春日 project (https://github.com/LC1332/Chat-Haruhi-Suzumiya)
# Chat凉宫春日是模仿凉宫春日等一系列动漫人物，使用近似语气、个性和剧情聊天的语言模型，
# 本项目由李鲁鲁，冷子昂，闫晨曦，封小洋，scixing，沈骏一，Aria Fei, 米唯实, 吴平宇, 贾曜恺等开发。
#
# 这个python程序是src_reform/ChatGPT.py中的一个分支，用于训练数据的生成
# 之后再由冷子昂进行合并

class ChatGPT:
    def __init__(self, configuration, in_training_generating=False):
        # 初始化ChatGPT类
        self.configuration = configuration
        # 是否处于训练生成模式
        self.in_training_generating = True
        # 图像嵌入的jsonl文件路径
        self.image_embed_jsonl_path = configuration['image_embed_jsonl_path']
        # 标题文本嵌入的jsonl文件路径
        self.title_text_embed_jsonl_path = configuration['title_text_embed_jsonl_path']
        # 图片文件夹路径
        self.images_folder = configuration['images_folder']
        # 文本文件夹路径
        self.texts_folder = configuration['texts_folder']
        # 系统提示信息的路径，从文件中读取系统提示信息
        self.system_prompt = configuration['system_prompt']
        with open(self.system_prompt, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
        # 最大故事长度
        self.max_len_story = int(configuration['max_len_story'])
        # 最大历史记录长度
        self.max_len_history = int(configuration['max_len_history'])
        # 对话路径
        self.dialogue_path = configuration['dialogue_path']
        # 获取编码方式为"cl100k_base"的编码对象
        self.enc = tiktoken.get_encoding("cl100k_base")
        # 设备选择为GPU（如果可用）或CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 预加载模型文件
        self.model = utils.download_models()
        # 图像嵌入向量
        self.image_embed = None
        # 文本嵌入向量
        self.text_embed = None
        # 标题到文本的映射
        self.title_to_text = None
        # 标题列表
        self.titles = None

        # 是否处于训练生成模式
        self.is_train_generating = False
        # 角色名称
        self.role_name = ""
        # 其他名称列表
        self.other_names = []
    # 设置训练模式，指定角色名和其他名称列表
    def set_training(self, role_name, other_names):
        self.is_train_generating = True  # 设置为训练生成模式
        self.role_name = role_name  # 设置角色名
        self.other_names = other_names  # 设置其他名称列表

    # 预加载数据，加载图像和文本嵌入
    def preload(self):
        self.image_embed = self.load(load_image_embed=True)  # 加载图像嵌入数据
        self.text_embed, self.title_to_text, self.titles = self.load(load_title_text_embed=True)  # 加载标题文本嵌入数据

    # 根据参数加载数据，支持加载标题文本嵌入和图像嵌入
    def load(self, load_title_text_embed=False,
             load_image_embed=False):
        if load_title_text_embed:
            text_embed = {}  # 初始化文本嵌入字典
            title_text_embed = {}  # 初始化标题文本嵌入字典
            title_to_text = {}  # 初始化标题到文本映射字典
            with open(self.title_text_embed_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    title_text_embed.update(data)  # 更新标题文本嵌入字典
            for title_text, embed in title_text_embed.items():
                res = title_text.split("｜｜｜")
                title_to_text[res[0]] = res[1]  # 解析标题和文本映射关系
                text_embed[res[1]] = embed  # 存储文本嵌入数据
            return text_embed, title_to_text, list(title_to_text.keys())  # 返回文本嵌入字典、标题到文本映射字典和标题列表

        elif load_image_embed:
            image_embed = {}  # 初始化图像嵌入字典
            if os.path.exists(self.image_embed_jsonl_path):
                with open(self.image_embed_jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        image_embed.update(data)  # 更新图像嵌入字典
                return image_embed  # 返回图像嵌入字典
            else:
                return None  # 若文件路径不存在则返回 None

        else:
            print("Please specify the loading file！")  # 提示未指定加载文件类型

    # 根据文本查询最相似的图像，并返回对应的图像
    def text_to_image(self, text):
        query_similarity = self.get_cosine_similarity(text, get_image=True)  # 获取文本与图像的余弦相似度
        key_index = query_similarity.argmax(dim=0)  # 获取相似度最高的索引
        text = list(self.image_embed.keys())[key_index]  # 获取对应的文本
        image = text + '.jpg'  # 构造图像文件名
        if image in os.listdir(self.images_folder):  # 如果图像文件存在于文件夹中
            res = Image.open(self.images_folder + '/' + image)  # 打开图像文件
            # res.show()  # 显示图像（注释掉的代码）
            return res  # 返回图像对象
        else:
            print("Image doesn't exist")  # 提示图像文件不存在

    # 使用 OpenAI 接口获取对话模型的生成结果，返回结果内容
    def get_completion_from_messages(self, messages, model="gpt-3.5-turbo", temperature=0):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,  # 控制模型输出的随机程度
        )
        #  print(str(response.choices[0].message))  # 打印响应消息（注释掉的代码）
        return response.choices[0].message["content"]  # 返回响应消息的内容
    def get_cosine_similarity(self, texts, get_image=False, get_texts=False):
        """
        计算文本列表的余弦相似度，避免重复计算query_similarity。
        texts[0] = query
        """
        # 使用模型获取文本的嵌入向量，并将其转换成张量，放置在指定设备上
        query_embedding = torch.tensor(utils.get_embedding(self.model, texts)).reshape(1, -1).to(self.device)
        
        if get_image:
            # 如果需要获取图像嵌入向量，则使用预先存储的图像嵌入数据
            jsonl = self.image_embed
        elif get_texts:
            # 如果需要获取文本嵌入向量，则使用预先存储的文本嵌入数据
            jsonl = self.text_embed
        # else:
        #     # 否则，计算query的嵌入向量
        #     jsonl = {}
        #     embeddings = utils.get_embedding(self.model, texts[1:]).reshape(-1, 1536)
        #     for text, embed in zip(texts, embeddings):
        #         jsonl[text] = embed
        
        # 从jsonl中获取所有嵌入向量，并转换成numpy数组
        texts_embeddings = np.array([value for value in jsonl.values()])
        
        # 返回查询向量与文本嵌入向量之间的余弦相似度
        return cosine_similarity(query_embedding, torch.from_numpy(texts_embeddings).to(self.device))

    def retrieve_title(self, query_text, k):
        """
        根据查询文本和指定的k值，检索最相似的标题列表。
        """
        # 将查询文本放入列表中
        texts = [query_text]
        
        # 获取标题到嵌入向量的映射
        embed_to_title = self.titles
        
        # 计算查询文本与所有文本的余弦相似度，并转换为numpy数组
        cosine_similarities = self.get_cosine_similarity(texts, get_texts=True).cpu().numpy().tolist()
        
        # 对余弦相似度进行排序
        sorted_cosine_similarities = sorted(cosine_similarities, reverse=True)
        
        # 初始化存储最相似标题和对应索引的列表
        top_k_index = []
        top_k_title = []
        
        # 遍历排序后的余弦相似度
        for i in range(len(sorted_cosine_similarities)):
            # 找到当前相似度对应的标题
            current_title = embed_to_title[cosine_similarities.index(sorted_cosine_similarities[i])]
            
            # 如果当前标题不在已存储的top_k_title中，则加入
            if current_title not in top_k_title:
                top_k_title.append(current_title)
                top_k_index.append(cosine_similarities.index(sorted_cosine_similarities[i]))
            
            # 如果已经找到了指定数量的标题，则停止遍历
            if len(top_k_title) == k:
                break
        
        # 返回最相似的标题列表
        return top_k_title

    def organize_stories_with_maxlen_for_training(self, selected_sample):
        """
        根据最大长度限制，组织所选样本的故事列表以供训练使用。
        """
        stories = []
        
        count = 0
        
        # 遍历选定的样本主题
        for sample_topic in selected_sample:
            # 根据主题从字典中获取对应的故事文本
            sample_story = self.title_to_text[sample_topic]

            # 计算该故事文本的长度
            sample_len = len(self.enc.encode(sample_story))
            
            # 如果加上当前故事的长度超过了最大长度限制，则停止添加故事
            if sample_len + count > self.max_len_story:
                break
            
            # 将故事文本添加到列表中
            stories.append(sample_story)

            # 更新已添加故事的总长度
            count += sample_len
        
        # 返回组织好的故事列表
        return stories

    def organize_story_with_maxlen(self, selected_sample):
        """
        根据最大长度限制，组织所选样本的故事文本。
        """
        # 初始化故事文本
        story = "\n"
        
        # 初始化计数器
        count = 0
        
        # 初始化最终选择的样本列表
        final_selected = []
        
        # 遍历选定的样本主题
        for sample_topic in selected_sample:
            # 根据主题从字典中获取对应的故事文本
            sample_story = self.title_to_text[sample_topic]

            # 计算该故事文本的长度
            sample_len = len(self.enc.encode(sample_story))
            
            # 如果加上当前故事的长度超过了最大长度限制，则停止添加故事
            if sample_len + count > self.max_len_story:
                break
            
            # 将故事文本添加到总故事文本中
            story += sample_story
            story += '\n'

            # 更新已添加故事的总长度
            count += sample_len
            
            # 将当前主题添加到最终选择的样本列表中
            final_selected.append(sample_topic)
        
        # 返回组织好的总故事文本和最终选择的样本列表
        return story, final_selected
    # 组织消息的方法，接收故事、聊天历史、回复历史和新的查询作为输入
    def organize_message(self, story, history_chat, history_response, new_query):
        # 初始化消息列表，包含系统提示和用户提供的故事内容
        messages = [{'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': story}]

        # 获取历史聊天记录的长度
        n = len(history_chat)
        # 检查历史回复记录长度与历史聊天记录长度是否匹配
        if n != len(history_response):
            # 输出警告信息，提示历史记录长度不匹配，需要清理并重新开始新的对话
            print('warning, unmatched history_char length, clean and start new chat')
            # 清空历史聊天记录和历史回复记录
            history_chat = []
            history_response = []
            n = 0  # 重置长度为零

        # 将历史聊天记录和历史回复记录添加到消息列表中
        for i in range(n):
            messages.append({'role': 'user', 'content': history_chat[i]})
            messages.append({'role': 'user', 'content': history_response[i]})

        # 将新查询添加到消息列表中
        messages.append({'role': 'user', 'content': new_query})

        # 返回组织好的消息列表
        return messages

    # 保留历史聊天和回复记录的尾部内容，使其总长度不超过最大历史记录长度
    def keep_tail(self, history_chat, history_response):
        # 获取历史聊天记录的长度
        n = len(history_chat)
        # 如果历史聊天记录长度为零，则直接返回空列表
        if n == 0:
            return [], []

        # 检查历史回复记录长度是否与历史聊天记录长度匹配
        if n != len(history_response):
            # 输出警告信息，提示历史记录长度不匹配，需要清理并重新开始新的对话
            print('warning, unmatched history_char length, clean and start new chat')
            return [], []

        # 计算每个历史对话和回复的令牌长度之和
        token_len = []
        for i in range(n):
            chat_len = len(self.enc.encode(history_chat[i]))
            res_len = len(self.enc.encode(history_response[i]))
            token_len.append(chat_len + res_len)

        # 初始化保留的历史记录数量和计数器
        keep_k = 1
        count = token_len[n - 1]

        # 逐步增加保留的历史记录数量，直到超过最大历史记录长度为止
        for i in range(1, n):
            count += token_len[n - 1 - i]
            if count > self.max_len_history:
                break
            keep_k += 1

        # 返回保留的历史聊天记录和历史回复记录
        return history_chat[-keep_k:], history_response[-keep_k:]

    # 将故事分割成不同的部分，并根据特定的角色名或其他名称进行分割
    def divide_story(self, story):
        # 根据空行分割故事
        storys = re.split(r'\n{2,}', story.strip())
        for s in storys:
            lines = s.split('\n')
            for i in range(len(lines)):
                # 检查是否以角色名或其他名称开头
                if lines[i].startswith(self.role_name) or any([lines[i].startswith(name) for name in self.other_names]):
                    # 将故事分割成两部分，返回分割后的结果
                    res = '\n'.join(lines[:i]), '\n'.join(lines[i:])
                    # 直接返回结果并结束函数
                    return res
                    break
        # 如果没有找到符合条件的分割位置，返回空字符串
        return "", ""
    def organize_message_langchain_for_training(self, storys, history_chat, history_response, new_query):
        # 初始化消息列表，包含系统提示消息
        messages = [
            SystemMessage(content=self.system_prompt)
        ]

        # 遍历输入的故事列表
        for story in storys:
            # 调用函数将故事分割为 AI 和人类消息，并添加到消息列表中
            ai_message, human_message = self.divide_story(story)
            messages.append(AIMessage(content=ai_message))
            messages.append(HumanMessage(content=human_message))

        # 检查历史聊天记录和回复是否长度匹配
        n = len(history_chat)
        if n != len(history_response):
            print('warning, unmatched history_char length, clean and start new chat')
            # 如果长度不匹配，清空历史聊天记录和回复，并重置计数器
            history_chat = []
            history_response = []
            n = 0

        # 遍历历史聊天记录和回复，将其添加到消息列表中
        for i in range(n):
            messages.append(HumanMessage(content=history_chat[i]))
            messages.append(AIMessage(content=history_response[i]))

        # 将新查询添加到消息列表中作为用户消息
        messages.append(HumanMessage(content=new_query))
        # 返回构建好的消息列表
        return messages

    def organize_message_for_generator(self, story, history_chat, history_response, new_query):
        # 调用函数将故事分割为原始消息列表
        raw_messages = self.divide_story(story)

        # 初始化消息列表，包含系统提示消息
        messages = [
            SystemMessage(content=self.system_prompt),
        ]

        # 遍历原始消息列表，将其分别作为 AI 和人类消息添加到消息列表中
        for raw_message in raw_messages:
            messages.append(AIMessage(content=raw_message[0]))
            messages.append(HumanMessage(content=raw_message[1]))

        # 检查历史聊天记录和回复是否长度匹配
        n = len(history_chat)
        if n != len(history_response):
            print('warning, unmatched history_char length, clean and start new chat')
            # 如果长度不匹配，清空历史聊天记录和回复，并重置计数器
            history_chat = []
            history_response = []
            n = 0

        # 遍历历史聊天记录和回复，将其添加到消息列表中
        for i in range(n):
            messages.append(HumanMessage(content=history_chat[i]))
            messages.append(AIMessage(content=history_response[i]))

        # 将新查询添加到消息列表中作为用户消息
        messages.append(HumanMessage(content=new_query))
        # 返回构建好的消息列表
        return messages

    def organize_message_langchain(self, story, history_chat, history_response, new_query):
        # 初始化消息列表，包含系统提示消息和输入的故事作为人类消息
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=story)
        ]

        # 检查历史聊天记录和回复是否长度匹配
        n = len(history_chat)
        if n != len(history_response):
            print('warning, unmatched history_char length, clean and start new chat')
            # 如果长度不匹配，清空历史聊天记录和回复，并重置计数器
            history_chat = []
            history_response = []
            n = 0

        # 遍历历史聊天记录和回复，将其添加到消息列表中
        for i in range(n):
            messages.append(HumanMessage(content=history_chat[i]))
            messages.append(AIMessage(content=history_response[i]))

        # 将新查询添加到消息列表中作为用户消息
        messages.append(HumanMessage(content=new_query))
        # 返回构建好的消息列表
        return messages
    # 定义一个方法，用于获取对用户消息的响应
    def get_response(self, user_message, chat_history_tuple):
        
        # 初始化历史对话列表和历史响应列表
        history_chat = []
        history_response = []
        
        # 如果输入的对话历史元组不为空
        if len(chat_history_tuple) > 0:
            # 遍历对话历史元组，将对话和响应分别添加到对应的列表中
            for cha, res in chat_history_tuple:
                history_chat.append(cha)
                history_response.append(res)
        
        # 调用方法，保留列表尾部的元素，使其长度不超过预定值
        history_chat, history_response = self.keep_tail(history_chat, history_response)
        
        # 新的用户查询消息
        new_query = user_message
        
        # 如果不处于训练生成状态，或者角色名为空
        if (self.is_train_generating == False) or (self.role_name == ""):
            # 根据用户消息检索标题，最多13个字符
            selected_sample = self.retrieve_title(new_query, 13)
            # 整理故事，返回故事文本和经过整理的选定样本
            story, selected_sample = self.organize_story_with_maxlen(selected_sample)
            
            ## TODO: 后续可视化选定的样本
            # 组织消息链，包括故事文本、历史对话和历史响应，以及新的用户查询消息
            messages = self.organize_message_langchain(story, history_chat, history_response, new_query)
        else:
            # 根据用户消息检索标题，最多13个字符
            selected_sample = self.retrieve_title(new_query, 13)
            # 整理故事，返回多个故事文本
            stories = self.organize_stories_with_maxlen_for_training(selected_sample)
            
            # 组织消息链，包括多个故事文本、历史对话和历史响应，以及新的用户查询消息，用于训练
            messages = self.organize_message_langchain_for_training(stories, history_chat, history_response, new_query)
        
        # 创建一个开放AI聊天对象，温度设为0
        chat = ChatOpenAI(temperature=0)
        # 调用聊天对象，传入消息链，获取返回的消息
        return_msg = chat(messages)
        
        # 提取返回消息的内容
        response = return_msg.content
        
        # 返回最终的响应消息
        return response
```