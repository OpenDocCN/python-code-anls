# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\GLMAPI.py`

```py
import os  # 导入操作系统模块

from .BaseLLM import BaseLLM  # 导入当前目录下的BaseLLM模块

zhipu_api = os.environ['ZHIPU_API']  # 从环境变量中获取ZHIPU_API的值

from zhipuai import ZhipuAI  # 导入zhipuai模块中的ZhipuAI类
import time  # 导入时间模块

FAIL = "FAIL"  # 定义常量FAIL，表示失败状态
SUCCESS = "SUCCESS"  # 定义常量SUCCESS，表示成功状态
PROCESSING = "PROCESSING"  # 定义常量PROCESSING，表示处理中状态

class GLMAPI(BaseLLM):
    def __init__(self, model="glm-3-turbo", verbose=False):
        super(GLMAPI, self).__init__()  # 调用父类BaseLLM的构造函数

        self.client = ZhipuAI(api_key=zhipu_api)  # 初始化ZhipuAI客户端对象，使用给定的API密钥

        self.verbose = verbose  # 设定对象的verbose属性，用于控制详细输出

        self.model_name = model  # 设定对象的model_name属性，设置模型名称

        self.prompts = []  # 初始化一个空列表prompts，用于存储消息

        if self.verbose == True:  # 如果verbose属性为True
            print('model name, ', self.model_name)  # 输出模型名称信息
            if len(zhipu_api) > 8:  # 如果zhipu_api的长度大于8
                print('found apikey ', zhipu_api[:4], '****', zhipu_api[-4:])  # 输出部分API密钥信息
            else:
                print('found apikey but too short, ')  # 输出API密钥过短的提示信息

    def initialize_message(self):
        self.prompts = []  # 将prompts列表重置为空列表，用于初始化消息

    def ai_message(self, payload):
        self.prompts.append({"role": "assistant", "content": payload})  # 向prompts列表中添加助手角色的消息内容

    def system_message(self, payload):
        self.prompts.append({"role": "user", "content": payload})  # 向prompts列表中添加用户角色的消息内容

    def user_message(self, payload):
        self.prompts.append({"role": "user", "content": payload})  # 向prompts列表中添加用户角色的消息内容
    def get_response(self):
        max_test_name = 5  # 最大尝试次数
        sleep_interval = 3  # 重试间隔时间

        response_id = None  # 响应 ID 初始化为 None

        # 尝试异步提交请求，直到成功
        for test_time in range(max_test_name):
            response = self.client.chat.asyncCompletions.create(
                model=self.model_name,  # 填写需要调用的模型名称
                messages=self.prompts,  # 提交的消息列表
            )
            if response.task_status != FAIL:
                response_id = response.id

                if self.verbose:
                    print("model name : ", response.model)
                    print('submit request, id = ', response_id)
                break
            else:
                print('submit GLM request failed, retrying...')
                time.sleep(sleep_interval)

        if response_id:
            # 尝试获取响应，直到成功
            for test_time in range(2 * max_test_name):
                result = self.client.chat.asyncCompletions.retrieve_completion_result(id=response_id)

                if result.task_status == FAIL:
                    if self.verbose:
                        print('response id : ', response_id, "task is fail")
                    break

                if result.task_status == PROCESSING:
                    if self.verbose:
                        print('response id : ', response_id, "task is processing")
                    time.sleep(sleep_interval)
                    continue

                # 成功获取响应
                if self.verbose:
                    print(
                        f"prompt tokens:{result.usage.prompt_tokens} completion tokens:{result.usage.completion_tokens}")
                    print(f"choices:{result.choices}")
                    print(f"result:{result}")

                return result.choices[-1].message.content

        # 如果未能成功获取响应，打印错误信息
        print('submit GLM request failed, please check your api key and model name')
        return ''

    def print_prompt(self):
        for message in self.prompts:
            print(f"{message['role']}: {message['content']}")


这段代码主要包含了两个方法：`get_response` 和 `print_prompt`。`get_response` 方法尝试异步提交请求并获取响应，通过重试机制确保任务完成。`print_prompt` 方法用于打印提示信息列表中的内容。
```