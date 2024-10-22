# `.\chatglm4-finetune\basic_demo\openai_api_server.py`

```
# 导入操作系统相关的模块
import os
# 导入用于运行子进程的模块
import subprocess

# 设置模型路径，优先从环境变量获取，如果没有则使用默认值 'THUDM/glm-4-9b-chat'
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4-9b-chat')

# 注释掉的代码行，用于另一个模型的路径
# MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4v-9b')


# 检查模型路径是否包含 '4v'（不区分大小写）
if '4v' in MODEL_PATH.lower():
    # 如果包含 '4v'，则运行 glm4v_server.py 脚本，传入模型路径
    subprocess.run(["python", "glm4v_server.py", MODEL_PATH])
else:
    # 如果不包含 '4v'，则运行 glm_server.py 脚本，传入模型路径
    subprocess.run(["python", "glm_server.py", MODEL_PATH])
```