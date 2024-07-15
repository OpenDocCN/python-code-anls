# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\BaiChuanAPIGPT.py`

```py
import os  # 导入操作系统相关的模块
import json  # 导入处理 JSON 数据的模块
import time  # 导入时间处理模块
import hashlib  # 导入哈希算法相关的模块
import requests  # 导入发送 HTTP 请求的模块
import copy  # 导入复制相关的模块

from .BaseLLM import BaseLLM  # 从当前目录下的 BaseLLM 模块导入 BaseLLM 类

BAICHUAN_API_AK = os.getenv("BAICHUAN_API_AK")  # 从环境变量中获取百川 API 的 Access Key
BAICHUAN_API_SK = os.getenv("BAICHUAN_API_SK")  # 从环境变量中获取百川 API 的 Secret Key

def sign(secret_key, data):
    json_data = json.dumps(data)  # 将数据转换为 JSON 格式的字符串
    time_stamp = int(time.time())  # 获取当前时间戳
    input_string = secret_key + json_data + str(time_stamp)  # 构造待加密的字符串
    md5 = hashlib.md5()  # 创建 MD5 哈希对象
    md5.update(input_string.encode('utf-8'))  # 更新哈希对象的内容
    encrypted = md5.hexdigest()  # 获取加密后的结果（16进制表示）
    return encrypted  # 返回加密结果

def do_request(messages, api_key, secret_key):
    url = "https://api.baichuan-ai.com/v1/chat"  # API 请求的 URL

    data = {
        "model": "Baichuan2-53B",  # API 请求的模型名称
        "messages": messages  # 发送的消息内容
    }

    signature = sign(secret_key, data)  # 生成签名

    headers = {
        "Content-Type": "application/json",  # 请求的内容类型为 JSON
        "Authorization": "Bearer " + api_key,  # 授权信息，使用 Bearer Token 形式
        "X-BC-Request-Id": "your requestId",  # 请求的唯一标识 ID
        "X-BC-Timestamp": str(int(time.time())),  # 请求的时间戳
        "X-BC-Signature": signature,  # 请求的签名
        "X-BC-Sign-Algo": "MD5",  # 签名算法为 MD5
    }

    response = requests.post(url, data=json.dumps(data), headers=headers)  # 发送 POST 请求
    if response.status_code == 200:  # 如果响应状态码为 200 OK
        return response.json()  # 解析 JSON 格式的响应数据并返回
    else:
        return None  # 如果响应状态码不为 200，则返回空值

class BaiChuanAPIGPT(BaseLLM):
    def __init__(self, model="baichuan-api", api_key=None, secret_key=None, verbose=False, if_trick=True):
        self.if_trick = if_trick  # 初始化是否使用技巧的标志
        super(BaiChuanAPIGPT, self).__init__()  # 调用父类 BaseLLM 的初始化方法
        self.api_key = api_key or BAICHUAN_API_AK  # 设置 API 的 Access Key，如果未提供则使用环境变量中的值
        self.secret_key = secret_key or BAICHUAN_API_SK  # 设置 API 的 Secret Key，如果未提供则使用环境变量中的值
        self.verbose = verbose  # 是否输出详细信息的标志
        self.model_name = model  # 设置模型名称
        self.messages = []  # 初始化消息列表为空
        if self.verbose:
            print('model name, ', self.model_name)  # 如果 verbose 为 True，则打印模型名称
            if self.api_key is None or self.secret_key is None:
                print('Please set BAICHUAN_API_AK and BAICHUAN_API_SK')  # 如果 API 的 Access Key 或 Secret Key 未设置，则打印警告信息

    def initialize_message(self):
        self.messages = []  # 初始化消息列表为空

    def ai_message(self, payload):
        if len(self.messages) == 0:
            self.user_message("请根据我的要求进行角色扮演:")  # 如果消息列表为空，则添加提示用户进行角色扮演的消息
        elif len(self.messages) % 2 == 1:
            self.messages.append({"role": "assistant", "content": payload})  # 如果消息列表长度为奇数，则添加助手角色的消息
        elif len(self.messages) % 2 == 0:
            self.messages[-1]["content"] += "\n" + payload  # 如果消息列表长度为偶数，则将上一条消息内容和当前内容拼接

    def system_message(self, payload):
        self.messages.append({"role": "user", "content": payload})  # 添加系统消息，角色为用户

    def user_message(self, payload):
        if len(self.messages) % 2 == 0:
            self.messages.append({"role": "user", "content": payload})  # 如果消息列表长度为偶数，则添加用户消息
        elif len(self.messages) % 2 == 1:
            self.messages[-1]["content"] += "\n" + payload  # 如果消息列表长度为奇数，则在上一条消息后添加当前内容
    # 定义一个方法用于获取响应
    def get_response(self):
        # 最大尝试次数设为5
        max_try = 5
        # 等待间隔设为3秒
        sleep_interval = 3
        
        # 深拷贝消息列表，以防修改原始数据
        chat_messages = copy.deepcopy(self.messages)
        
        # 如果需要进行特殊处理
        if self.if_trick == True:
            # 取出最后一条消息的内容并按换行符分割成列表
            lines = chat_messages[-1]["content"].split('\n')
            # 在倒数第二行之前插入一条特定提示
            lines.insert(-1, '请请模仿上述经典桥段进行回复\n')
            # 将修改后的内容重新组合成字符串，并更新到最后一条消息中
            chat_messages[-1]["content"] = '\n'.join(lines)

        # 尝试获取响应，最多重试max_try次
        for i in range(max_try):
            # 调用外部函数执行请求，传入消息列表、API密钥和密钥
            response = do_request(chat_messages, self.api_key, self.secret_key)
            # 如果成功获取到响应
            if response is not None:
                # 如果开启了详细输出模式，打印成功获取响应的消息
                if self.verbose:
                    print('Get Baichuan API response success')
                # 从响应中取出数据中的消息列表
                messages = response['data']['messages']
                # 如果消息列表不为空，返回最后一条消息的内容（去除双引号和单引号）
                if len(messages) > 0:
                    return messages[-1]['content'].strip("\"'")
            else:
                # 如果开启了详细输出模式，打印获取响应失败并进行重试
                if self.verbose:
                    print('Get Baichuan API response failed, retrying...')
                # 休眠一段时间后再重试
                time.sleep(sleep_interval)
            
    # 定义一个方法用于打印提示信息
    def print_prompt(self):
        # 遍历消息列表，打印每条消息的角色和内容
        for message in self.messages:
            print(f"{message['role']}: {message['content']}")
```