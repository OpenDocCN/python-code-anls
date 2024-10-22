# `.\cogvideo-finetune\tools\caption\video_caption.py`

```py
# 导入必要的库
import io

import argparse  # 用于解析命令行参数
import numpy as np  # 用于数值计算
import torch  # PyTorch深度学习库
from decord import cpu, VideoReader, bridge  # 视频处理库
from transformers import AutoModelForCausalLM, AutoTokenizer  # 变换器模型库

MODEL_PATH = "THUDM/cogvlm2-llama3-caption"  # 模型路径

# 判断是否使用GPU，若可用则使用CUDA
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 根据设备能力设置Torch数据类型
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")
# 添加量化参数的选项，选择4位或8位精度
parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
# 解析命令行参数，返回参数对象
args = parser.parse_args([])


def load_video(video_data, strategy='chat'):
    # 设置Decord的桥接为PyTorch
    bridge.set_bridge('torch')
    mp4_stream = video_data  # 将输入的视频数据存储在mp4_stream中
    num_frames = 24  # 设定要提取的帧数
    # 从字节流创建视频读取器
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None  # 初始化帧ID列表
    total_frames = len(decord_vr)  # 获取视频总帧数
    # 根据选择的策略决定帧提取方式
    if strategy == 'base':
        clip_end_sec = 60  # 设置视频片段结束时间
        clip_start_sec = 0  # 设置视频片段开始时间
        # 计算开始帧和结束帧
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        # 生成等间隔的帧ID列表
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        # 获取每帧的时间戳
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]  # 提取时间戳的第一个元素
        max_second = round(max(timestamps)) + 1  # 计算最大秒数
        frame_id_list = []  # 初始化帧ID列表
        # 遍历每秒，找到最接近的帧
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))  # 找到最接近的时间戳
            index = timestamps.index(closest_num)  # 获取对应帧的索引
            frame_id_list.append(index)  # 将索引添加到帧ID列表
            if len(frame_id_list) >= num_frames:  # 如果达到所需帧数，则停止
                break

    # 根据帧ID列表获取视频帧
    video_data = decord_vr.get_batch(frame_id_list)
    # 调整视频数据的维度顺序
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data  # 返回提取的视频数据


# 从预训练模型加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

# 从预训练模型加载语言模型并设置设备类型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True
).eval().to(DEVICE)  # 设置为评估模式并转移到指定设备


def predict(prompt, video_data, temperature):
    strategy = 'chat'  # 设定策略为聊天模式

    # 加载视频数据
    video = load_video(video_data, strategy=strategy)

    history = []  # 初始化对话历史
    query = prompt  # 设置查询内容
    # 构建模型输入
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version=strategy
    )
    # 准备输入字典，将数据转移到CUDA设备
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
    }
    # 设置生成参数
    gen_kwargs = {
        "max_new_tokens": 2048,  # 设置最大生成的标记数
        "pad_token_id": 128002,  # 设置填充标记ID
        "top_k": 1,  # 设置Top-k采样
        "do_sample": False,  # 是否进行采样
        "top_p": 0.1,  # 设置Top-p采样
        "temperature": temperature,  # 设置温度
    }
    # 在不计算梯度的上下文中执行代码，以节省内存和加速计算
        with torch.no_grad():
            # 使用模型生成输出，输入参数包含输入数据和生成时的额外参数
            outputs = model.generate(**inputs, **gen_kwargs)
            # 截取生成输出，从第二个维度开始，去除输入部分
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            # 将输出张量解码为可读字符串，跳过特殊标记
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 返回解码后的响应
            return response
# 定义一个测试函数
def test():
    # 设置提示语，用于描述视频的请求
    prompt = "Please describe this video in detail."
    # 设置温度值，用于控制生成文本的随机性
    temperature = 0.1
    # 以二进制模式打开视频文件，并读取其内容
    video_data = open('test.mp4', 'rb').read()
    # 调用预测函数，传入提示语、视频数据和温度值，获取响应
    response = predict(prompt, video_data, temperature)
    # 打印响应结果
    print(response)


# 判断是否为主程序执行
if __name__ == '__main__':
    # 调用测试函数
    test()
```