# `.\Chat-Haruhi-Suzumiya\research\personality\code\utils.py`

```py
# 导入pdb模块，用于调试和分析程序
import pdb
# 导入os模块，提供与操作系统交互的功能
import os
# 导入re模块，用于支持正则表达式操作
import re
# 导入random模块，用于生成随机数和随机选择操作
import random
# 导入openai模块，用于与OpenAI API进行交互
import openai
# 导入json模块，用于处理JSON数据
import json
# 导入logging模块，用于生成日志信息
import logging
# 导入time模块，用于时间相关操作
import time
# 导入jsonlines模块，用于处理JSON Lines格式数据
import jsonlines
# 导入requests模块，用于发送HTTP请求
import requests
# 导入io模块，提供核心的I/O功能
import io
# 导入pickle模块，用于序列化和反序列化Python对象
import pickle

# 配置日志记录器
logger = logging.getLogger(__name__)

# 创建文件处理器，将日志写入文件 'log.log'，使用UTF-8编码
file_handler = logging.FileHandler('log.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)  # 设置文件处理器的日志级别为INFO

# 创建控制台处理器，将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 设置控制台处理器的日志级别为INFO

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将文件处理器和控制台处理器添加到日志记录器中
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 设置日志记录器的全局日志级别为INFO
logger.setLevel(logging.INFO)

# 创建新的控制台处理器和格式化器，用于特定配置
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
console_handler.setFormatter(formatter)

# 设置缓存标志为True
cache_sign = True

# 从配置文件 'config.json' 中读取配置信息
with open('config.json', 'r') as f:
    config = json.load(f)

# 如果配置中有代理设置，则将OpenAI的代理设置为配置中的代理
if config.get('proxy', None):
    openai.proxy = config['proxy']

# 如果配置中有OpenAI的API基础地址设置，则将OpenAI的API基础地址设置为配置中的地址
if config.get('openai_apibase', None):
    openai.api_base = config['openai_apibase']

# 初始化缓存为None
cache = None

# 缓存装饰器函数，用于缓存函数调用结果
def cached(func):
    def wrapper(*args, **kwargs):
        global cache
        cache_path = 'cache.pkl'
        # 如果缓存为空，则尝试从文件 'cache.pkl' 中加载缓存数据
        if cache == None:
            if not os.path.exists(cache_path):
                cache = {}
            else:
                cache = pickle.load(open(cache_path, 'rb'))

        # 构建函数调用的唯一键
        key = (func.__name__, str(args), str(kwargs.items()))

        # 如果缓存标志为True，并且键在缓存中存在且值不为None或'[TOKEN LIMIT]'，则返回缓存结果
        if cache_sign and key in cache and cache[key] not in [None, '[TOKEN LIMIT]']:
            return cache[key]
        else:
            # 否则执行函数调用，并更新缓存
            result = func(*args, **kwargs)
            if result != 'busy' and result != None:
                cache[key] = result
                pickle.dump(cache, open(cache_path, 'wb'))
            return result

    return wrapper

# 获取响应函数，用于向模型发送请求并获取响应
def get_response(sys_prompt, inputs, model='gpt4'):
    model = model.lower().replace(' ', '')

    # 如果模型名称是 'gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k' 中的一个，则调用相应的获取响应函数
    if model in ['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k']:
        return get_response_gpt(sys_prompt, inputs, model)

# 缓存装饰器函数，用于缓存与OpenAI GPT模型的交互结果
@cached
def get_response_gpt(sys_prompt, inputs, model='gpt-4', retry_count=0):
    # 构建请求数据，包括系统提示和用户输入
    query = [{'role': 'system', 'content': sys_prompt}]
    if len(inputs) > 0:
        query.append({'role': 'user', 'content': inputs})

    try:
        # 记录日志：系统提示和用户输入
        logger.info('ChatGPT SysPrompt:  ' + sys_prompt[:100])
        logger.info('ChatGPT Input:  ' + inputs[:100])

        # 调用OpenAI API，获取聊天模型的响应
        response = openai.ChatCompletion.create(
            api_key=config['openai_apikey'],
            model=model,  # 对话模型的名称
            messages=query,
            temperature=0,  # 值在[0,1]之间，越大表示回复越具有不确定性
            top_p=1,
            frequency_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            presence_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            request_timeout=100
        )

        # 记录日志：OpenAI GPT的输出内容
        logger.info('GPT Output: ' + response.choices[0]['message']['content'][:100])

        # 返回OpenAI GPT的响应内容
        return response.choices[0]['message']['content']

    # 捕获异常，记录错误日志
    except Exception as e:
        logger.error(f"Error in get_response_gpt: {str(e)}")
        return None
    # 处理 openai.error.InvalidRequestError 异常
    except openai.error.InvalidRequestError as e:
        # 记录异常信息到日志
        logger.exception(e)
        
        # 返回指定的错误信息
        return '[TOKEN LIMIT]'

    # 处理其他未知异常
    except Exception as e:
        # 记录异常信息到日志
        logger.exception(e)

        # 如果重试次数小于2，等待5秒后进行重试，并记录重试次数到日志
        if retry_count < 2:
            time.sleep(5)
            logger.warn("[OPEN_AI] RateLimit exceed, 第{}次重试".format(retry_count+1))
            # 递归调用函数进行重试
            return get_response_gpt(sys_prompt, inputs, model, retry_count+1) 

        # 输出重试次数后仍未成功获取响应的提示信息
        print(f'Fail to get response after {retry_count} retry')
# 如果这个脚本作为主程序运行（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == '__main__':
    # 调用名为 get_response 的函数，传入三个参数：任务描述字符串、输入问题字符串、和模型名称字符串
    print(get_response('Act as a calculator', '123+456=?', 'gpt-3.5-turbo'))
```