# `.\VisualGLM-6B\api.py`

```
# 导入所需的库
import os
import json
import uvicorn
from fastapi import FastAPI, Request
from model import is_chinese, get_infer_setting, generate_input, chat
import datetime
import torch

# 设置 GPU 编号
gpu_number = 0
# 获取推理设置中的模型和分词器
model, tokenizer = get_infer_setting(gpu_device=gpu_number)

# 创建 FastAPI 应用
app = FastAPI()

# 定义 POST 请求处理函数
@app.post('/')
async def visual_glm(request: Request):
    # 从请求中获取 JSON 数据
    json_post_raw = await request.json()
    print("Start to process request")

    # 将 JSON 数据转换为字符串
    json_post = json.dumps(json_post_raw)
    # 将 JSON 字符串转换为字典
    request_data = json.loads(json_post)
    # 从请求数据中获取文本、图像编码和历史记录
    input_text, input_image_encoded, history = request_data['text'], request_data['image'], request_data['history']
    
    # 设置输入参数
    input_para = {
        "max_length": 2048,
        "min_length": 50,
        "temperature": 0.8,
        "top_p": 0.4,
        "top_k": 100,
        "repetition_penalty": 1.2
    }
    # 更新输入参数
    input_para.update(request_data)

    # 判断输入文本是否为中文
    is_zh = is_chinese(input_text)
    # 生成输入数据
    input_data = generate_input(input_text, input_image_encoded, history, input_para)
    input_image, gen_kwargs =  input_data['input_image'], input_data['gen_kwargs']
    
    # 使用模型进行对话生成
    with torch.no_grad():
        answer, history, _ = chat(None, model, tokenizer, input_text, history=history, image=input_image, \
                            max_length=gen_kwargs['max_length'], top_p=gen_kwargs['top_p'], \
                            top_k = gen_kwargs['top_k'], temperature=gen_kwargs['temperature'], english=not is_zh)
        
    # 获取当前时间
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    # 构建响应数据
    response = {
        "result": answer,
        "history": history,
        "status": 200,
        "time": time
    }
    # 返回响应数据
    return response

# 运行 FastAPI 应用
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)
```