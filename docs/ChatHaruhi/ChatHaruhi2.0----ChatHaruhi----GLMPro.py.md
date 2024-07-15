# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\GLMPro.py`

```py
from .BaseLLM import BaseLLM
import os

# 从环境变量中获取 ZHIPU_API 的值
zhipu_api = os.environ['ZHIPU_API']

import zhipuai
import time

class GLMPro( BaseLLM ):
    def __init__(self, model="chatglm_pro", verbose = False ):
        # 调用父类的初始化方法
        super(GLMPro,self).__init__()

        # 设置 zhipuai 模块的 API key
        zhipuai.api_key = zhipu_api

        # 设置是否显示详细信息
        self.verbose = verbose

        # 设置模型名称
        self.model_name = model

        # 初始化对话提示列表
        self.prompts = []

        # 如果 verbose 为 True，则输出模型名称和部分 API key
        if self.verbose == True:
            print('model name, ', self.model_name )
            if len( zhipu_api ) > 8:
                print( 'found apikey ', zhipu_api[:4], '****', zhipu_api[-4:] )
            else:
                print( 'found apikey but too short, ' )
        

    def initialize_message(self):
        # 初始化对话提示列表为空
        self.prompts = []

    def ai_message(self, payload):
        # 向对话提示列表添加助手角色和内容
        self.prompts.append({"role":"assistant","content":payload})

    def system_message(self, payload):
        # 向对话提示列表添加用户角色和内容
        self.prompts.append({"role":"user","content":payload})

    def user_message(self, payload):
        # 向对话提示列表添加用户角色和内容
        self.prompts.append({"role":"user","content":payload})

    def get_response(self):
        # 设置 zhipuai 模块的 API key
        zhipuai.api_key = zhipu_api

        # 最大尝试次数和等待间隔
        max_test_name = 5
        sleep_interval = 3

        # 初始化请求 ID 为 None
        request_id = None

        # 尝试异步提交请求直到成功
        for test_time in range( max_test_name ):
            response = zhipuai.model_api.async_invoke(
                model = self.model_name,
                prompt = self.prompts,
                temperature = 0)
            if response['success'] == True:
                # 获取任务 ID
                request_id = response['data']['task_id']

                # 如果 verbose 为 True，则输出提交请求的信息
                if self.verbose == True:
                    print('submit request, id = ', request_id )
                break
            else:
                print
```