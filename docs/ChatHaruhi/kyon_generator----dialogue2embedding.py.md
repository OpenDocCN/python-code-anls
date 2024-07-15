# `.\Chat-Haruhi-Suzumiya\kyon_generator\dialogue2embedding.py`

```py
# -*- coding: utf-8 -*-

'''
FILE_NAME: dialogue2embedding.py
Edited by 冷子昂, Aug 8 2023
'''

import json
import os
# import torch
# import torch.nn as nn
import argparse
from argparse import Namespace
# from transformers import AutoTokenizer, AutoModel
from utils import download_models, get_embedding

# 下载模型并返回
model = download_models()

# 过滤连续的序列数字
def filter_continuous_sequence(numbers):
    if not numbers:
        return []
    
    result = [numbers[0]]
    for i in range(1, len(numbers)):
        if numbers[i] != numbers[i - 1] + 1:
            result.append(numbers[i])
    
    return result

# 整理 JSON 文件，将对话重新组织并写入新的文件
def organize_json(file_path, out_path):
    # 不知道为什么我们的json格式非常奇怪，只能重新梳理，真迷
    # jsonl_file_path = '/Users/pufferfish/Downloads/story_synthesis_data/tangshiye_test_output_dialogue.jsonl'
    filename = file_path.split("/")[-1]
    out_path = out_path + filename
    print(out_path)
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            json_data = json.loads(line)
            # print(json_data["dialogue"][1])
            parts = json_data["dialogue"][0].split(':')
            # print(parts)
            new_dict = {"role":parts[0], "text":parts[1]}
            # new_json = json.dump(new_dict,ensure_ascii=False)
            json_data["dialogue"][0] = new_dict
            with open(out_path, 'a') as json_file:
                json.dump(json_data, json_file, ensure_ascii=False)
                json_file.write('\n')

# 检查对话中是否包含特定角色名
def contain_role(dialogue, role_name):
    role_name_length = len(role_name)
    for i,v in enumerate(dialogue):
        if v["role"][:role_name_length] == role_name:
            return True
    return False

# 获取角色名在对话列表中的索引列表
def get_role_name_index(dialogue, role_name):
    index_list = []
    role_name_length = len(role_name)
    for i,v in enumerate(dialogue):
        if v["role"][:role_name_length] == role_name:
            index_list.append(i)
    return index_list

# 处理 JSON 文件，将特定角色的对话集中起来并写入新文件
def train_json_file(dialogue, index_list, source, out_path):
    
    # 获取消息的字符串表示，包括角色和文本
    def get_message(sentence_json):
        return sentence_json["role"]+":"+sentence_json["text"]

    # 将查询、聊天历史、回答、嵌入和来源写入 JSON 文件
    def write_json(query, chat_history, answer, embedding, out_path):
        json_line = {
                "query": query,
                "answer": answer,
                "chat_history": chat_history,
                "embedding": embedding,
                "source": "story-synthesised"
            }
        # print(json_line)
        with open(out_path, 'a', encoding='utf-8') as json_file:
            json.dump(json_line, json_file, ensure_ascii=False)
        return
    
    # 获取对话列表的历史记录（消息的字符串表示列表）
    def get_history(dialogue):
        history = []
        for i in dialogue:
            history.append(get_message(i))
        return history

    chat_history = []  # 聊天历史记录列表
    query = ""          # 查询字符串
    embedding = []      # 嵌入列表
    answer = ""         # 回答字符串

    index_list = filter_continuous_sequence(index_list)  # 过滤连续的索引序列
    # 对于给定的索引列表中的每个索引进行迭代处理
    for index in index_list:
        # 如果索引为0，将对话历史中第一个对话消息添加到聊天历史中
        if index == 0:
            chat_history.append(get_message(dialogue[0]))
        
        # 如果索引为1，处理当前对话消息作为查询，重置聊天历史，生成答案和嵌入向量，并写入 JSON
        if index == 1:
            query = get_message(dialogue[index-1])
            chat_history = []  # 重置聊天历史为空列表
            answer = get_message(dialogue[index])
            embedding = get_embedding(model, query)  # 获取查询的嵌入向量，此处需要填充具体实现
            write_json(query, chat_history, answer, embedding, out_path)
        
        # 对于其他索引情况，处理当前对话消息作为查询，获取聊天历史，生成答案和嵌入向量，并写入 JSON
        else:
            query = get_message(dialogue[index-1])
            chat_history = get_history(dialogue[:index-2])  # 获取索引前的所有对话历史
            answer = get_message(dialogue[index])
            embedding = get_embedding(model, query)  # 获取查询的嵌入向量，此处需要填充具体实现
            write_json(query, chat_history, answer, embedding, out_path)
            # 打印调试信息，如索引、查询和聊天历史
            # print(index)
            # print(query)
            # print(chat_history)
# 定义一个函数，用于处理对话文件，将特定角色的对话转换为嵌入向量并保存到指定路径
def dialogue2embed(file_path, role_names, out_path):
    # 从文件路径中提取文件名
    filename = file_path.split("/")[-1]
    # 拼接输出路径和文件名
    out_path = out_path + filename
    # 如果角色名不是列表，转换为包含一个元素的列表
    if type(role_names) != list:
        role_names = [role_names]
    # 打开文件，按行读取
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            # 解析每行 JSON 数据，获取对话和来源
            dialogue = json.loads(line)["dialogue"]
            source = json.loads(line)["source"]
            # 遍历所有角色名
            for role_name in role_names:
                # 如果对话中包含当前角色名
                if contain_role(dialogue, role_name):
                    # 获取当前角色名在对话中的索引列表
                    index_list = get_role_name_index(dialogue, role_name)
                    # 将对话数据、角色索引列表和来源写入指定路径的训练数据文件
                    train_json_file(dialogue, index_list, source, out_path)
            # 打印调试信息（如果需要）
            # print(json.loads(line)["dialogue"])
```