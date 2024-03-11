# `.\VisualGLM-6B\api_hf.py`

```
# 导入所需的库
import os
import json
from transformers import AutoTokenizer, AutoModel
import uvicorn
from fastapi import FastAPI, Request
import datetime
from model import process_image
import torch

# 从预训练模型中加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).half().cuda()

# 创建 FastAPI 应用
app = FastAPI()

# 定义 POST 请求处理函数
@app.post('/')
async def visual_glm(request: Request):
    # 从请求中获取 JSON 数据
    json_post_raw = await request.json()
    print("Start to process request")

    # 将 JSON 数据转换为字符串再转换为字典
    json_post = json.dumps(json_post_raw)
    request_data = json.loads(json_post)

    # 从请求数据中获取历史记录、编码后的图像、查询文本
    history = request_data.get("history")
    image_encoded = request_data.get("image")
    query = request_data.get("text")
    
    # 处理编码后的图像，返回图像路径
    image_path = process_image(image_encoded)

    # 使用模型进行推理
    with torch.no_grad():    
        result = model.stream_chat(tokenizer, image_path, query, history=history)
    
    # 获取最后一个结果
    last_result = None
    for value in result:
        last_result = value
    answer = last_result[0]

    # 如果图像路径存在，则删除图像文件
    if os.path.isfile(image_path):
        os.remove(image_path)
    
    # 获取当前时间并格式化
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
if __name__ == "__main__":
   uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)
```