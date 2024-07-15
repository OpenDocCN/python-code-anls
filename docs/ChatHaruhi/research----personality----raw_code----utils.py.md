# `.\Chat-Haruhi-Suzumiya\research\personality\raw_code\utils.py`

```py
# 导入pdb模块，用于调试程序
import pdb
# 导入os模块，提供了许多与操作系统交互的功能
import os
# 导入re模块，用于处理正则表达式
import re
# 导入random模块，提供生成随机数的功能
import random
# 导入openai模块，用于与OpenAI的API交互
import openai
# 导入json模块，用于处理JSON格式的数据
import json
# 导入logging模块，用于记录日志
import logging
# 导入time模块，提供时间相关的功能
import time
# 导入jsonlines模块，用于处理JSON lines格式的数据
import jsonlines
# 导入nltk模块，自然语言工具包
import nltk
# 从nltk.corpus中导入wordnet，用于处理英语词汇的语义
from nltk.corpus import wordnet
# 导入requests模块，用于发送HTTP请求
import requests
# 导入io模块，提供了Python的核心API与各种类型的I/O流
import io
# 导入pickle模块，用于序列化和反序列化Python对象
import pickle

# 设置日志记录器的名称
logger = logging.getLogger(__name__)

# 创建一个文件处理器，将日志写入到log.log文件中，编码为utf-8
file_handler = logging.FileHandler('log.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)  # 设置文件处理器的日志级别为INFO

# 创建一个控制台处理器，将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 设置控制台处理器的日志级别为INFO

# 设置日志记录格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将文件处理器和控制台处理器添加到日志记录器中
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)  # 设置日志记录器的日志级别为INFO

# 创建另一个控制台处理器，用于输出带有文件名和行号的详细日志信息
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
console_handler.setFormatter(formatter)

# 设置缓存开关为True
cache_sign = True

# 打开config.json文件，读取其中的配置信息
with open('config.json', 'r') as f:
    config = json.load(f)

# 如果配置中存在'proxy'项，则将openai.proxy设置为对应的值
if config.get('proxy', None):
    openai.proxy = config['proxy']

# 如果配置中存在'openai_apibase'项，则将openai.api_base设置为对应的值
if config.get('openai_apibase', None):
    openai.api_base = config['openai_apibase']

cache = None  # 初始化缓存为None

# 定义一个装饰器函数cached，用于缓存函数的执行结果
def cached(func):
    def wrapper(*args, **kwargs):
        global cache
        cache_path = './cache.pkl'  # 缓存文件的路径

        # 如果cache为None，则尝试从缓存文件中加载缓存数据
        if cache == None:
            if not os.path.exists(cache_path):
                cache = {}
            else:
                cache = pickle.load(open(cache_path, 'rb'))

        key = (func.__name__, str(args), str(kwargs.items()))  # 生成缓存键值

        # 如果cache_sign为True，并且缓存中存在当前键值，并且其值不为None或'[TOKEN LIMIT]'，则直接返回缓存结果
        if cache_sign and key in cache and cache[key] not in [None, '[TOKEN LIMIT]']:
            return cache[key]
        else:
            # 否则，执行函数，并将结果缓存起来
            result = func(*args, **kwargs)
            if result != 'busy' and result != None:
                cache[key] = result
                pickle.dump(cache, open(cache_path, 'wb'))
            return result

    return wrapper

# 定义函数get_response，根据指定模型获取对话回复
def get_response(sys_prompt, inputs, model='gpt4'):
    model = model.lower().replace(' ', '')

    # 根据不同的模型名称，调用对应的获取回复函数
    if model in ['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k']:
        return get_response_gpt(sys_prompt, inputs, model)
    elif model == 'llama2chat':
        return get_response_llama2chat(sys_prompt, inputs)

# 使用cached装饰器，定义函数get_response_ada，用于调用OpenAI的文本嵌入API获取回复
@cached
def get_response_ada(inputs):
    try:
        response = openai.Embedding.create(
            api_key=config['openai_apikey'],
            input=inputs,
            model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']
        return embeddings
    except Exception as e:
        # 捕获并记录异常信息
        logger.exception(e)
        return None

# 使用cached装饰器，定义函数get_response_llama2chat，用于调用特定模型的对话生成功能
@cached
def get_response_llama2chat(sys_prompt, inputs, retry_count=0):
    query = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': inputs}]
    try:
        # 尝试向 LLAMA 2 发送 POST 请求，并从返回的 JSON 中提取回答
        response = requests.post(f'{config["llama_port"]}/chat', json={'query': query}).json()['ans']
        # 记录 LLAMA 2 输入内容
        logger.info('LLAMA 2 Input:  ' + inputs)
        # 记录 LLAMA 2 输出内容
        logger.info('LLAMA 2 Output: ' + response)
        # 返回 LLAMA 2 的回答
        return response

    except Exception as e:
        # 处理未知异常情况，记录异常信息
        logger.exception(e)

        # 如果重试次数小于 2，则尝试重试
        if retry_count < 2:
            # 等待 5 秒后重试
            time.sleep(5)
            # 记录重试次数
            logger.warn("[LLAMA 2] RateLimit exceed, 第{}次重试".format(retry_count+1))
            # 递归调用获取 LLAMA 2 的响应
            return get_response_llama2(sys_prompt, inputs, retry_count+1) 

        # 打印获取响应失败的消息，包含重试次数
        print(f'Fail to get response after {retry_count} retry')
# 使用装饰器 @cached 来缓存函数的结果，提升性能
@cached 
def get_response_gpt(sys_prompt, inputs, model='gpt-4', retry_count=0):
    # 构建查询消息列表，包括系统提示和用户输入
    query = [{'role': 'system', 'content': sys_prompt}]
    # 如果用户输入不为空，将其添加到查询消息列表中
    if len(inputs) > 0:
        query.append({'role': 'user', 'content': inputs})
    
    try:
        # 记录系统提示和用户输入到日志中（最多100个字符）
        logger.info('ChatGPT SysPrompt:  ' + sys_prompt[:100])
        logger.info('ChatGPT Input:  ' + inputs[:100])
        # 调用 OpenAI API 进行聊天模型的完成
        response = openai.ChatCompletion.create(
            api_key=config['openai_apikey'],  # 使用配置文件中的 API 密钥
            model=model,  # 指定使用的对话模型名称
            messages=query,  # 发送构建好的消息列表
            temperature=0,  # 控制生成文本的多样性，值在[0,1]之间，0表示确定性最高
            top_p=1,  # 生成文本的概率阈值，1表示无阈值
            frequency_penalty=0.0,  # 控制生成文本的多样性，值在[-2,2]之间，0表示无惩罚
            presence_penalty=0.0,  # 控制生成文本的多样性，值在[-2,2]之间，0表示无惩罚
            request_timeout=300  # API 请求超时时间设定为300秒
        )
        # 记录生成的 GPT 输出到日志中（最多100个字符）
        logger.info('GPT Output: ' + response.choices[0]['message']['content'][:100])
        # 返回生成的文本内容
        return response.choices[0]['message']['content']

    except openai.error.InvalidRequestError as e:
        # 记录 OpenAI API 请求错误到日志中
        logger.exception(e)
        # 返回特定错误提示
        return '[TOKEN LIMIT]'

    except Exception as e:
        # 记录未知异常到日志中
        logger.exception(e)
        # 如果重试次数小于2，等待5秒后进行重试
        if retry_count < 2:
            time.sleep(5)
            logger.warn("[OPEN_AI] RateLimit exceed, 第{}次重试".format(retry_count+1))
            # 递归调用自身进行重试
            return get_response_gpt(sys_prompt, inputs, model, retry_count+1) 
        # 输出重试次数后的失败消息
        print(f'Fail to get response after {retry_count} retry')


        
if __name__ == '__main__':
    # 打印使用指定参数调用 get_response_gpt 函数的结果
    print(get_response('请帮我翻译成英文', '加速度呢~？', 'gpt-3.5-turbo'))
    '''
    import requests
    try:
        # 尝试发送 HTTP GET 请求到谷歌主页，并设定超时时间为5秒
        response = requests.get('https://www.google.com', timeout=5)
        # 如果返回状态码为200，则打印可以访问谷歌的信息
        if response.status_code == 200:
            print("Can access Google, might be over the wall!")
        else:
            print("Cannot access Google.")
    except requests.RequestException:
        # 发生 requests 异常时打印无法访问谷歌的信息
        print("Cannot access Google.")
    '''
```