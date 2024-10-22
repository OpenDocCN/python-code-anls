# CogVideo & CogVideoX 微调代码源码解析（十三）



# Video Caption

Typically, most video data does not come with corresponding descriptive text, so it is necessary to convert the video
data into textual descriptions to provide the essential training data for text-to-video models.

## Update and News
- 🔥🔥 **News**: ```py/9/19```: The caption model used in the CogVideoX training process to convert video data into text
  descriptions, [CogVLM2-Caption](https://huggingface.co/THUDM/cogvlm2-llama3-caption), is now open-source. Feel
  free to download and use it.


## Video Caption via CogVLM2-Caption

🤗 [Hugging Face](https://huggingface.co/THUDM/cogvlm2-llama3-caption) | 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-caption/) 

CogVLM2-Caption is a video captioning model used to generate training data for the CogVideoX model.

### Install
```py
pip install -r requirements.txt
```

### Usage

```py
python video_caption.py
```

Example:
<div align="center">
    <img width="600px" height="auto" src="./assests/CogVLM2-Caption-example.png">
</div>

## Video Caption via CogVLM2-Video

[Code](https://github.com/THUDM/CogVLM2/tree/main/video_demo) | 🤗 [Hugging Face](https://huggingface.co/THUDM/cogvlm2-video-llama3-chat) | 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/cogvlm2-video-llama3-chat) | 📑 [Blog](https://cogvlm2-video.github.io/) ｜ [💬 Online Demo](http://cogvlm2-online.cogviewai.cn:7868/)

CogVLM2-Video is a versatile video understanding model equipped with timestamp-based question answering capabilities.
Users can input prompts such as `Please describe this video in detail.` to the model to obtain a detailed video caption:
<div align="center">
    <a href="https://cogvlm2-video.github.io/"><img width="600px" height="auto" src="./assests/cogvlm2-video-example.png"></a>
</div>

Users can use the provided [code](https://github.com/THUDM/CogVLM2/tree/main/video_demo) to load the model or configure a RESTful API to generate video captions.

## Citation

🌟 If you find our work helpful, please leave us a star and cite our paper.

CogVLM2-Caption:
```py
@article{yang2024cogvideox,
  title={CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer},
  author={Yang, Zhuoyi and Teng, Jiayan and Zheng, Wendi and Ding, Ming and Huang, Shiyu and Xu, Jiazheng and Yang, Yuanming and Hong, Wenyi and Zhang, Xiaohan and Feng, Guanyu and others},
  journal={arXiv preprint arXiv:2408.06072},
  year={2024}
}
```
CogVLM2-Video:
```py
@article{hong2024cogvlm2,
  title={CogVLM2: Visual Language Models for Image and Video Understanding},
  author={Hong, Wenyi and Wang, Weihan and Ding, Ming and Yu, Wenmeng and Lv, Qingsong and Wang, Yan and Cheng, Yean and Huang, Shiyu and Ji, Junhui and Xue, Zhao and others},
  journal={arXiv preprint arXiv:2408.16500},
  year={2024}
}
```

# ビデオキャプション

通常、ほとんどのビデオデータには対応する説明文が付いていないため、ビデオデータをテキストの説明に変換して、テキストからビデオへのモデルに必要なトレーニングデータを提供する必要があります。

## 更新とニュース
- 🔥🔥 **ニュース**: ```py/9/19```：CogVideoX
  のトレーニングプロセスで、ビデオデータをテキストに変換するためのキャプションモデル [CogVLM2-Caption](https://huggingface.co/THUDM/cogvlm2-llama3-caption)
  がオープンソース化されました。ぜひダウンロードしてご利用ください。
## CogVLM2-Captionによるビデオキャプション

🤗 [Hugging Face](https://huggingface.co/THUDM/cogvlm2-llama3-caption) | 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-caption/) 

CogVLM2-Captionは、CogVideoXモデルのトレーニングデータを生成するために使用されるビデオキャプションモデルです。

### インストール
```py
pip install -r requirements.txt
```

### 使用方法
```py
python video_caption.py
```

例:
<div align="center">
    <img width="600px" height="auto" src="./assests/CogVLM2-Caption-example.png">
</div>



## CogVLM2-Video を使用したビデオキャプション

[Code](https://github.com/THUDM/CogVLM2/tree/main/video_demo) | 🤗 [Hugging Face](https://huggingface.co/THUDM/cogvlm2-video-llama3-chat) | 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/cogvlm2-video-llama3-chat) | 📑 [Blog](https://cogvlm2-video.github.io/) ｜ [💬 Online Demo](http://cogvlm2-online.cogviewai.cn:7868/)


CogVLM2-Video は、タイムスタンプベースの質問応答機能を備えた多機能なビデオ理解モデルです。ユーザーは `このビデオを詳細に説明してください。` などのプロンプトをモデルに入力して、詳細なビデオキャプションを取得できます：
<div align="center">
    <a href="https://cogvlm2-video.github.io/"><img width="600px" height="auto" src="./assests/cogvlm2-video-example.png"></a>
</div>

ユーザーは提供された[コード](https://github.com/THUDM/CogVLM2/tree/main/video_demo)を使用してモデルをロードするか、RESTful API を構成してビデオキャプションを生成できます。

## Citation

🌟 If you find our work helpful, please leave us a star and cite our paper.

CogVLM2-Caption:
```py
@article{yang2024cogvideox,
  title={CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer},
  author={Yang, Zhuoyi and Teng, Jiayan and Zheng, Wendi and Ding, Ming and Huang, Shiyu and Xu, Jiazheng and Yang, Yuanming and Hong, Wenyi and Zhang, Xiaohan and Feng, Guanyu and others},
  journal={arXiv preprint arXiv:2408.06072},
  year={2024}
}
```
CogVLM2-Video:
```py
@article{hong2024cogvlm2,
  title={CogVLM2: Visual Language Models for Image and Video Understanding},
  author={Hong, Wenyi and Wang, Weihan and Ding, Ming and Yu, Wenmeng and Lv, Qingsong and Wang, Yan and Cheng, Yean and Huang, Shiyu and Ji, Junhui and Xue, Zhao and others},
  journal={arXiv preprint arXiv:2408.16500},
  year={2024}
}
```


# 视频Caption

通常，大多数视频数据不带有相应的描述性文本，因此需要将视频数据转换为文本描述，以提供必要的训练数据用于文本到视频模型。

## 项目更新
- 🔥🔥 **News**: ```py/9/19```: CogVideoX 训练过程中用于将视频数据转换为文本描述的 Caption
  模型 [CogVLM2-Caption](https://huggingface.co/THUDM/cogvlm2-llama3-caption)
  已经开源。欢迎前往下载并使用。

## 通过 CogVLM2-Caption 模型生成视频Caption

🤗 [Hugging Face](https://huggingface.co/THUDM/cogvlm2-llama3-caption) | 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-caption/) 

CogVLM2-Caption是用于生成CogVideoX模型训练数据的视频caption模型。

### 安装依赖
```py
pip install -r requirements.txt
```

### 运行caption模型

```py
python video_caption.py
```

示例：
<div align="center">
    <img width="600px" height="auto" src="./assests/CogVLM2-Caption-example.png">
</div>

## 通过 CogVLM2-Video 模型生成视频Caption

[Code](https://github.com/THUDM/CogVLM2/tree/main/video_demo) | 🤗 [Hugging Face](https://huggingface.co/THUDM/cogvlm2-video-llama3-chat) | 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/cogvlm2-video-llama3-chat) | 📑 [Blog](https://cogvlm2-video.github.io/) ｜ [💬 Online Demo](http://cogvlm2-online.cogviewai.cn:7868/)

CogVLM2-Video 是一个多功能的视频理解模型，具备基于时间戳的问题回答能力。用户可以输入诸如 `Describe this video in detail.` 的提示语给模型，以获得详细的视频Caption：


<div align="center">
    <a href="https://cogvlm2-video.github.io/"><img width="600px" height="auto" src="./assests/cogvlm2-video-example.png"></a>
</div>

用户可以使用提供的[代码](https://github.com/THUDM/CogVLM2/tree/main/video_demo)加载模型或配置 RESTful API 来生成视频Caption。


## Citation

🌟 If you find our work helpful, please leave us a star and cite our paper.

CogVLM2-Caption:
```py
@article{yang2024cogvideox,
  title={CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer},
  author={Yang, Zhuoyi and Teng, Jiayan and Zheng, Wendi and Ding, Ming and Huang, Shiyu and Xu, Jiazheng and Yang, Yuanming and Hong, Wenyi and Zhang, Xiaohan and Feng, Guanyu and others},
  journal={arXiv preprint arXiv:2408.06072},
  year={2024}
}
```
CogVLM2-Video:
```py
@article{hong2024cogvlm2,
  title={CogVLM2: Visual Language Models for Image and Video Understanding},
  author={Hong, Wenyi and Wang, Weihan and Ding, Ming and Yu, Wenmeng and Lv, Qingsong and Wang, Yan and Cheng, Yean and Huang, Shiyu and Ji, Junhui and Xue, Zhao and others},
  journal={arXiv preprint arXiv:2408.16500},
  year={2024}
}
```

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

# `.\cogvideo-finetune\tools\convert_weight_sat2hf.py`

```
"""
# 此脚本演示如何从文本提示转换和生成视频
# 使用 CogVideoX 和 🤗Huggingface Diffusers Pipeline。
# 此脚本需要安装 `diffusers>=0.30.2` 库。

# 函数列表：
#     - reassign_query_key_value_inplace: 就地重新分配查询、键和值的权重。
#     - reassign_query_key_layernorm_inplace: 就地重新分配查询和键的层归一化。
#     - reassign_adaln_norm_inplace: 就地重新分配自适应层归一化。
#     - remove_keys_inplace: 就地移除状态字典中指定的键。
#     - replace_up_keys_inplace: 就地替换“up”块中的键。
#     - get_state_dict: 从保存的检查点中提取状态字典。
#     - update_state_dict_inplace: 就地更新状态字典以进行新的键分配。
#     - convert_transformer: 将变换器检查点转换为 CogVideoX 格式。
#     - convert_vae: 将 VAE 检查点转换为 CogVideoX 格式。
#     - get_args: 解析脚本的命令行参数。
#     - generate_video: 使用 CogVideoX 管道从文本提示生成视频。
"""

# 导入 argparse 模块用于解析命令行参数
import argparse
# 从 typing 导入 Any 和 Dict 类型
from typing import Any, Dict

# 导入 PyTorch 库
import torch
# 从 transformers 库导入 T5EncoderModel 和 T5Tokenizer
from transformers import T5EncoderModel, T5Tokenizer

# 从 diffusers 库导入多个类
from diffusers import (
    AutoencoderKLCogVideoX,  # 自动编码器类
    CogVideoXDDIMScheduler,   # 调度器类
    CogVideoXImageToVideoPipeline,  # 图像到视频的管道类
    CogVideoXPipeline,        # 主管道类
    CogVideoXTransformer3DModel,  # 3D 变换器模型类
)

# 函数：就地重新分配查询、键和值的权重
def reassign_query_key_value_inplace(key: str, state_dict: Dict[str, Any]):
    # 根据原始键生成新的键，替换查询键值
    to_q_key = key.replace("query_key_value", "to_q")
    to_k_key = key.replace("query_key_value", "to_k")
    to_v_key = key.replace("query_key_value", "to_v")
    # 将状态字典中该键的值分割成三部分（查询、键和值）
    to_q, to_k, to_v = torch.chunk(state_dict[key], chunks=3, dim=0)
    # 将分割后的查询、键和值添加到状态字典中
    state_dict[to_q_key] = to_q
    state_dict[to_k_key] = to_k
    state_dict[to_v_key] = to_v
    # 从状态字典中移除原始键
    state_dict.pop(key)

# 函数：就地重新分配查询和键的层归一化
def reassign_query_key_layernorm_inplace(key: str, state_dict: Dict[str, Any]):
    # 从键中提取层 ID 和权重或偏差类型
    layer_id, weight_or_bias = key.split(".")[-2:]

    # 根据键名确定新键名
    if "query" in key:
        new_key = f"transformer_blocks.{layer_id}.attn1.norm_q.{weight_or_bias}"
    elif "key" in key:
        new_key = f"transformer_blocks.{layer_id}.attn1.norm_k.{weight_or_bias}"

    # 将状态字典中原键的值移到新键中
    state_dict[new_key] = state_dict.pop(key)

# 函数：就地重新分配自适应层归一化
def reassign_adaln_norm_inplace(key: str, state_dict: Dict[str, Any]):
    # 从键中提取层 ID 和权重或偏差类型
    layer_id, _, weight_or_bias = key.split(".")[-3:]

    # 将状态字典中该键的值分割为 12 部分
    weights_or_biases = state_dict[key].chunk(12, dim=0)
    # 合并特定部分形成新的权重或偏差
    norm1_weights_or_biases = torch.cat(weights_or_biases[0:3] + weights_or_biases[6:9])
    norm2_weights_or_biases = torch.cat(weights_or_biases[3:6] + weights_or_biases[9:12])

    # 构建新键名并更新状态字典
    norm1_key = f"transformer_blocks.{layer_id}.norm1.linear.{weight_or_bias}"
    state_dict[norm1_key] = norm1_weights_or_biases

    norm2_key = f"transformer_blocks.{layer_id}.norm2.linear.{weight_or_bias}"
    state_dict[norm2_key] = norm2_weights_or_biases

    # 从状态字典中移除原始键
    state_dict.pop(key)

# 函数：就地移除状态字典中的指定键
def remove_keys_inplace(key: str, state_dict: Dict[str, Any]):
    # 从状态字典中移除指定的键
    state_dict.pop(key)
# 定义一个函数，替换状态字典中的特定键，直接在字典中修改
def replace_up_keys_inplace(key: str, state_dict: Dict[str, Any]):
    # 将键字符串按点分割成列表
    key_split = key.split(".")
    # 获取指定层的索引，假设索引在第三个位置
    layer_index = int(key_split[2])
    # 计算替换后的层索引
    replace_layer_index = 4 - 1 - layer_index

    # 将分割后的键更新为 "up_blocks" 作为新的第二层
    key_split[1] = "up_blocks"
    # 更新层索引为计算后的新索引
    key_split[2] = str(replace_layer_index)
    # 将分割的键重新拼接为字符串
    new_key = ".".join(key_split)

    # 在状态字典中用新键替换旧键对应的值
    state_dict[new_key] = state_dict.pop(key)


# 定义一个字典，用于重命名 Transformer 模型的键
TRANSFORMER_KEYS_RENAME_DICT = {
    # 重命名 final_layernorm 键为 norm_final
    "transformer.final_layernorm": "norm_final",
    # 将 transformer 键重命名为 transformer_blocks
    "transformer": "transformer_blocks",
    # 重命名注意力层的键
    "attention": "attn1",
    # 重命名 MLP 层的键
    "mlp": "ff.net",
    # 重命名密集层的键
    "dense_h_to_4h": "0.proj",
    "dense_4h_to_h": "2",
    # 处理 layers 键的重命名
    ".layers": "",
    # 将 dense 键重命名为 to_out.0
    "dense": "to_out.0",
    # 处理输入层归一化的重命名
    "input_layernorm": "norm1.norm",
    # 处理后注意力层归一化的重命名
    "post_attn1_layernorm": "norm2.norm",
    # 重命名时间嵌入的层
    "time_embed.0": "time_embedding.linear_1",
    "time_embed.2": "time_embedding.linear_2",
    # 处理 Patch 嵌入的重命名
    "mixins.patch_embed": "patch_embed",
    # 处理最终层的重命名
    "mixins.final_layer.norm_final": "norm_out.norm",
    "mixins.final_layer.linear": "proj_out",
    # 处理 ADA LN 调制层的重命名
    "mixins.final_layer.adaLN_modulation.1": "norm_out.linear",
    # 处理特定于 CogVideoX-5b-I2V 的重命名
    "mixins.pos_embed.pos_embedding": "patch_embed.pos_embedding",  # Specific to CogVideoX-5b-I2V
}

# 定义一个字典，用于特殊键的重映射
TRANSFORMER_SPECIAL_KEYS_REMAP = {
    # 映射特定的查询键值处理函数
    "query_key_value": reassign_query_key_value_inplace,
    # 映射查询层归一化列表的处理函数
    "query_layernorm_list": reassign_query_key_layernorm_inplace,
    # 映射键层归一化列表的处理函数
    "key_layernorm_list": reassign_query_key_layernorm_inplace,
    # 映射 ADA LN 调制层的处理函数
    "adaln_layer.adaLN_modulations": reassign_adaln_norm_inplace,
    # 映射嵌入令牌的处理函数
    "embed_tokens": remove_keys_inplace,
    # 映射频率正弦的处理函数
    "freqs_sin": remove_keys_inplace,
    # 映射频率余弦的处理函数
    "freqs_cos": remove_keys_inplace,
    # 映射位置嵌入的处理函数
    "position_embedding": remove_keys_inplace,
}

# 定义一个字典，用于重命名 VAE 模型的键
VAE_KEYS_RENAME_DICT = {
    # 将块的键重命名为 resnets. 
    "block.": "resnets.",
    # 将 down 的键重命名为 down_blocks.
    "down.": "down_blocks.",
    # 将 downsample 的键重命名为 downsamplers.0
    "downsample": "downsamplers.0",
    # 将 upsample 的键重命名为 upsamplers.0
    "upsample": "upsamplers.0",
    # 将 nin_shortcut 的键重命名为 conv_shortcut
    "nin_shortcut": "conv_shortcut",
    # 将编码器的块重命名
    "encoder.mid.block_1": "encoder.mid_block.resnets.0",
    "encoder.mid.block_2": "encoder.mid_block.resnets.1",
    # 将解码器的块重命名
    "decoder.mid.block_1": "decoder.mid_block.resnets.0",
    "decoder.mid.block_2": "decoder.mid_block.resnets.1",
}

# 定义一个字典，用于特殊键的重映射，适用于 VAE
VAE_SPECIAL_KEYS_REMAP = {
    # 映射损失的处理函数
    "loss": remove_keys_inplace,
    # 映射 up 的处理函数
    "up.": replace_up_keys_inplace,
}

# 定义一个常量，表示标记器的最大长度
TOKENIZER_MAX_LENGTH = 226


# 定义一个函数，从保存的字典中获取状态字典
def get_state_dict(saved_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 默认状态字典为保存的字典
    state_dict = saved_dict
    # 如果保存的字典中包含 "model" 键，则提取模型部分
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    # 如果保存的字典中包含 "module" 键，则提取模块部分
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    # 如果保存的字典中包含 "state_dict" 键，则提取状态字典
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    # 返回最终提取的状态字典
    return state_dict


# 定义一个函数，直接在状态字典中更新键
def update_state_dict_inplace(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    # 用新键替换旧键在字典中的值
    state_dict[new_key] = state_dict.pop(old_key)


# 定义一个函数，用于转换 Transformer 模型
def convert_transformer(
    ckpt_path: str,
    num_layers: int,
    num_attention_heads: int,
    use_rotary_positional_embeddings: bool,
    i2v: bool,
    dtype: torch.dtype,
):
    # 定义一个前缀键，表示模型的前缀部分
    PREFIX_KEY = "model.diffusion_model."

    # 从指定路径加载原始状态字典，设置 map_location 为 "cpu" 和 mmap 为 True
    original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", mmap=True))
    # 创建一个 CogVideoXTransformer3DModel 实例，设置输入通道、层数、注意力头数等参数
    transformer = CogVideoXTransformer3DModel(
        # 根据 i2v 的值决定输入通道数
        in_channels=32 if i2v else 16,
        # 设置模型的层数
        num_layers=num_layers,
        # 设置注意力头的数量
        num_attention_heads=num_attention_heads,
        # 是否使用旋转位置嵌入
        use_rotary_positional_embeddings=use_rotary_positional_embeddings,
        # 是否使用学习到的位置嵌入
        use_learned_positional_embeddings=i2v,
    ).to(dtype=dtype)  # 将模型转换为指定的数据类型

    # 遍历原始状态字典的键列表
    for key in list(original_state_dict.keys()):
        # 从键中去掉前缀，以获得新的键名
        new_key = key[len(PREFIX_KEY) :]
        # 遍历重命名字典，替换键名中的特定部分
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        # 更新原始状态字典中的键值对
        update_state_dict_inplace(original_state_dict, key, new_key)

    # 再次遍历原始状态字典的键列表
    for key in list(original_state_dict.keys()):
        # 遍历特殊键的映射字典
        for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
            # 如果特殊键不在当前键中，则继续下一个键
            if special_key not in key:
                continue
            # 调用处理函数以更新状态字典
            handler_fn_inplace(key, original_state_dict)
    
    # 加载更新后的状态字典到 transformer 中，严格匹配键
    transformer.load_state_dict(original_state_dict, strict=True)
    # 返回 transformer 实例
    return transformer
# 定义一个函数，将 VAE 模型从检查点路径转换
def convert_vae(ckpt_path: str, scaling_factor: float, dtype: torch.dtype):
    # 从指定路径加载原始状态字典，使用 CPU 映射
    original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", mmap=True))
    # 创建一个新的 VAE 对象，并将其数据类型设置为指定的 dtype
    vae = AutoencoderKLCogVideoX(scaling_factor=scaling_factor).to(dtype=dtype)

    # 遍历原始状态字典的所有键
    for key in list(original_state_dict.keys()):
        # 复制当前键以便修改
        new_key = key[:]
        # 遍历重命名字典，将旧键替换为新键
        for replace_key, rename_key in VAE_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        # 更新原始状态字典中的键
        update_state_dict_inplace(original_state_dict, key, new_key)

    # 再次遍历原始状态字典的所有键
    for key in list(original_state_dict.keys()):
        # 遍历特殊键映射字典
        for special_key, handler_fn_inplace in VAE_SPECIAL_KEYS_REMAP.items():
            # 如果特殊键不在当前键中，则跳过
            if special_key not in key:
                continue
            # 使用处理函数处理原始状态字典
            handler_fn_inplace(key, original_state_dict)

    # 加载更新后的状态字典到 VAE 模型中，严格匹配
    vae.load_state_dict(original_state_dict, strict=True)
    # 返回转换后的 VAE 对象
    return vae


# 定义获取命令行参数的函数
def get_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加原始变换器检查点路径参数
    parser.add_argument(
        "--transformer_ckpt_path", type=str, default=None, help="Path to original transformer checkpoint")
    # 添加原始 VAE 检查点路径参数
    parser.add_argument("--vae_ckpt_path", type=str, default=None, help="Path to original vae checkpoint")
    # 添加输出路径参数，作为必需参数
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    # 添加是否以 fp16 格式保存模型权重的布尔参数
    parser.add_argument("--fp16", action="store_true", default=False, help="Whether to save the model weights in fp16")
    # 添加是否以 bf16 格式保存模型权重的布尔参数
    parser.add_argument("--bf16", action="store_true", default=False, help="Whether to save the model weights in bf16")
    # 添加是否在保存后推送到 HF Hub 的布尔参数
    parser.add_argument(
        "--push_to_hub", action="store_true", default=False, help="Whether to push to HF Hub after saving"
    )
    # 添加文本编码器缓存目录路径参数
    parser.add_argument(
        "--text_encoder_cache_dir", type=str, default=None, help="Path to text encoder cache directory"
    )
    # 添加变换器块数量参数，默认值为 30
    parser.add_argument("--num_layers", type=int, default=30, help="Number of transformer blocks")
    # 添加注意力头数量参数，默认值为 30
    parser.add_argument("--num_attention_heads", type=int, default=30, help="Number of attention heads")
    # 添加是否使用旋转位置嵌入的布尔参数
    parser.add_argument(
        "--use_rotary_positional_embeddings", action="store_true", default=False, help="Whether to use RoPE or not"
    )
    # 添加 VAE 的缩放因子参数，默认值为 1.15258426
    parser.add_argument("--scaling_factor", type=float, default=1.15258426, help="Scaling factor in the VAE")
    # 添加 SNR 偏移比例参数，默认值为 3.0
    parser.add_argument("--snr_shift_scale", type=float, default=3.0, help="Scaling factor in the VAE")
    # 添加是否以 fp16 格式保存模型权重的布尔参数
    parser.add_argument("--i2v", action="store_true", default=False, help="Whether to save the model weights in fp16")
    # 解析命令行参数并返回
    return parser.parse_args()


# 如果脚本作为主程序执行
if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()

    # 初始化 transformer 和 vae 为 None
    transformer = None
    vae = None
    # 检查是否同时传递了 --fp16 和 --bf16 参数
    if args.fp16 and args.bf16:
        # 如果同时存在则抛出值错误
        raise ValueError("You cannot pass both --fp16 and --bf16 at the same time.")

    # 根据输入参数选择数据类型
    dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32

    # 如果提供了变换器检查点路径，则转换变换器
    if args.transformer_ckpt_path is not None:
        transformer = convert_transformer(
            # 传递变换器检查点路径及相关参数
            args.transformer_ckpt_path,
            args.num_layers,
            args.num_attention_heads,
            args.use_rotary_positional_embeddings,
            args.i2v,
            dtype,
        )
    # 如果提供了 VAE 检查点路径，则转换 VAE
    if args.vae_ckpt_path is not None:
        vae = convert_vae(args.vae_ckpt_path, args.scaling_factor, dtype)

    # 设置文本编码器的模型 ID
    text_encoder_id = "google/t5-v1_1-xxl"
    # 从预训练模型中加载分词器
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_id, model_max_length=TOKENIZER_MAX_LENGTH)
    # 从预训练模型中加载文本编码器
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_id, cache_dir=args.text_encoder_cache_dir)
    # 处理参数以确保数据连续性
    for param in text_encoder.parameters():
        # 使参数数据连续
        param.data = param.data.contiguous()

    # 从配置中创建调度器
    scheduler = CogVideoXDDIMScheduler.from_config(
        {
            # 设置调度器的超参数
            "snr_shift_scale": args.snr_shift_scale,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "v_prediction",
            "rescale_betas_zero_snr": True,
            "set_alpha_to_one": True,
            "timestep_spacing": "trailing",
        }
    )
    # 根据 i2v 参数选择管道类
    if args.i2v:
        pipeline_cls = CogVideoXImageToVideoPipeline
    else:
        pipeline_cls = CogVideoXPipeline

    # 实例化管道
    pipe = pipeline_cls(
        # 传递所需的组件
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )

    # 如果选择 fp16 则将管道转为 fp16
    if args.fp16:
        pipe = pipe.to(dtype=torch.float16)
    # 如果选择 bf16 则将管道转为 bf16
    if args.bf16:
        pipe = pipe.to(dtype=torch.bfloat16)

    # 保存预训练的管道到指定路径
    pipe.save_pretrained(args.output_path, safe_serialization=True, push_to_hub=args.push_to_hub)
```

# `.\cogvideo-finetune\tools\export_sat_lora_weight.py`

```py
# 导入所需的类型和库
from typing import Any, Dict
import torch 
import argparse 
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.models.modeling_utils import load_state_dict

# 定义函数，获取状态字典，输入为一个字典，输出为一个字典
def get_state_dict(saved_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 初始化状态字典为输入字典
    state_dict = saved_dict
    # 如果字典中包含"model"键，更新状态字典为"model"对应的值
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    # 如果字典中包含"module"键，更新状态字典为"module"对应的值
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    # 如果字典中包含"state_dict"键，更新状态字典为"state_dict"对应的值
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    # 返回最终的状态字典
    return state_dict

# 定义LORA键重命名的字典
LORA_KEYS_RENAME = {
    'attention.query_key_value.matrix_A.0': 'attn1.to_q.lora_A.weight',
    'attention.query_key_value.matrix_A.1': 'attn1.to_k.lora_A.weight',
    'attention.query_key_value.matrix_A.2': 'attn1.to_v.lora_A.weight',
    'attention.query_key_value.matrix_B.0': 'attn1.to_q.lora_B.weight',
    'attention.query_key_value.matrix_B.1': 'attn1.to_k.lora_B.weight',
    'attention.query_key_value.matrix_B.2': 'attn1.to_v.lora_B.weight',
    'attention.dense.matrix_A.0': 'attn1.to_out.0.lora_A.weight',
    'attention.dense.matrix_B.0': 'attn1.to_out.0.lora_B.weight'
}

# 定义前缀键和相关常量
PREFIX_KEY = "model.diffusion_model."
SAT_UNIT_KEY = "layers"
LORA_PREFIX_KEY = "transformer_blocks"

# 导出LORA权重的函数，输入为检查点路径和保存目录
def export_lora_weight(ckpt_path,lora_save_directory):
    # 加载检查点并获取合并后的状态字典
    merge_original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", mmap=True))

    # 初始化LORA状态字典
    lora_state_dict = {}
    # 遍历合并后的状态字典的所有键
    for key in list(merge_original_state_dict.keys()):
        # 获取新键，去掉前缀
        new_key = key[len(PREFIX_KEY) :]
        # 遍历LORA键重命名字典
        for special_key, lora_keys in LORA_KEYS_RENAME.items():
            # 如果新键以特殊键结尾，则进行替换
            if new_key.endswith(special_key):
                new_key = new_key.replace(special_key, lora_keys)
                new_key = new_key.replace(SAT_UNIT_KEY, LORA_PREFIX_KEY)
                # 将替换后的键及其对应值添加到LORA状态字典
                lora_state_dict[new_key] = merge_original_state_dict[key]

    # 检查LORA状态字典的长度是否为240
    if len(lora_state_dict) != 240:
        raise ValueError("lora_state_dict length is not 240")

    # 获取LORA状态字典的所有键
    lora_state_dict.keys()

    # 调用LoraBaseMixin的写入LORA层函数，保存权重
    LoraBaseMixin.write_lora_layers(
        state_dict=lora_state_dict,
        save_directory=lora_save_directory,
        is_main_process=True,
        weight_name=None,
        save_function=None,
        safe_serialization=True
    )

# 定义获取命令行参数的函数
def get_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加原始检查点路径参数
    parser.add_argument(
        "--sat_pt_path", type=str, required=True, help="Path to original sat transformer checkpoint"
    )
    # 添加LORA保存目录参数
    parser.add_argument("--lora_save_directory", type=str, required=True, help="Path where converted lora should be saved") 
    # 返回解析后的参数
    return parser.parse_args()

# 主程序入口
if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    # 调用导出LORA权重的函数
    export_lora_weight(args.sat_pt_path, args.lora_save_directory)
```

# `.\cogvideo-finetune\tools\llm_flux_cogvideox\gradio_page.py`

```
# 导入操作系统模块
import os
# 导入 Gradio 库，用于构建用户界面
import gradio as gr
# 导入垃圾回收模块
import gc
# 导入随机数生成模块
import random
# 导入 PyTorch 库
import torch
# 导入 NumPy 库
import numpy as np
# 导入图像处理库
from PIL import Image
# 导入 Transformers 库
import transformers
# 从 Diffusers 库导入视频生成相关的类
from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler, DiffusionPipeline
# 从 Diffusers 库导入导出视频的工具
from diffusers.utils import export_to_video
# 从 Transformers 库导入自动分词器
from transformers import AutoTokenizer
# 导入日期和时间处理模块
from datetime import datetime, timedelta
# 导入多线程模块
import threading
# 导入时间模块
import time
# 导入 MoviePy 库进行视频编辑
import moviepy.editor as mp

# 设置浮点数矩阵乘法的精度为高
torch.set_float32_matmul_precision("high")

# 设置默认值
caption_generator_model_id = "/share/home/zyx/Models/Meta-Llama-3.1-8B-Instruct"  # 生成视频描述的模型路径
image_generator_model_id = "/share/home/zyx/Models/FLUX.1-dev"  # 生成图像的模型路径
video_generator_model_id = "/share/official_pretrains/hf_home/CogVideoX-5b-I2V"  # 生成视频的模型路径
seed = 1337  # 随机数种子

# 创建输出目录，若已存在则不报错
os.makedirs("./output", exist_ok=True)
# 创建临时目录，用于 Gradio
os.makedirs("./gradio_tmp", exist_ok=True)

# 从指定模型加载自动分词器
tokenizer = AutoTokenizer.from_pretrained(caption_generator_model_id, trust_remote_code=True)
# 创建文本生成管道，用于生成视频描述
caption_generator = transformers.pipeline(
    "text-generation",  # 指定任务为文本生成
    model=caption_generator_model_id,  # 指定模型
    device_map="balanced",  # 设置设备映射为平衡模式
    model_kwargs={  # 模型参数
        "local_files_only": True,  # 仅使用本地文件
        "torch_dtype": torch.bfloat16,  # 设置张量数据类型
    },
    trust_remote_code=True,  # 允许使用远程代码
    tokenizer=tokenizer  # 使用加载的分词器
)

# 从指定模型加载图像生成管道
image_generator = DiffusionPipeline.from_pretrained(
    image_generator_model_id,  # 指定图像生成模型
    torch_dtype=torch.bfloat16,  # 设置张量数据类型
    device_map="balanced"  # 设置设备映射为平衡模式
)
# image_generator.to("cuda")  # 可选择将生成器移动到 GPU（被注释掉）

# 从指定模型加载视频生成管道
video_generator = CogVideoXImageToVideoPipeline.from_pretrained(
    video_generator_model_id,  # 指定视频生成模型
    torch_dtype=torch.bfloat16,  # 设置张量数据类型
    device_map="balanced"  # 设置设备映射为平衡模式
)

# 启用视频生成器的 VAE 切片功能
video_generator.vae.enable_slicing()
# 启用视频生成器的 VAE 平铺功能
video_generator.vae.enable_tiling()

# 设置视频生成器的调度器，使用自定义配置
video_generator.scheduler = CogVideoXDPMScheduler.from_config(
    video_generator.scheduler.config, timestep_spacing="trailing"  # 设置时间步长为后续模式
)

# 定义系统提示
SYSTEM_PROMPT = """
# 系统提示内容，说明视频生成任务和规则
You are part of a team of people that create videos using generative models. You use a video-generation model that can generate a video about anything you describe.

For example, if you respond with "A beautiful morning in the woods with the sun peaking through the trees", the video generation model will create a video of exactly as described. Your task is to summarize the descriptions of videos provided by users and create detailed prompts to feed into the generative model.

There are a few rules to follow:
- You will only ever output a single video description per request.
- If the user mentions to summarize the prompt in [X] words, make sure not to exceed the limit.

Your responses should just be the video generation prompt. Here are examples:
# 定义包含详细描述的字符串，描述玩具船在蓝色地毯上的场景
- "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting."
# 定义包含街头艺术家的字符串，描述其在城市墙壁上喷涂的情景
- "A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart of the city, holding a can of spray paint, spray-painting a colorful bird on a mottled wall."
# 去除多余的空格并保存为用户提示
""".strip()

# 定义用户提示模板，用于生成视频生成模型的提示
USER_PROMPT = """
Could you generate a prompt for a video generation model? Please limit the prompt to [{0}] words.
""".strip()

# 定义生成字幕的函数，接受一个提示参数
def generate_caption(prompt):
    # 随机选择字数（25、50、75或100）以限制生成的字幕长度
    num_words = random.choice([25, 50, 75, 100])
    # 格式化用户提示，将随机字数插入提示模板中
    user_prompt = USER_PROMPT.format(num_words)

    # 创建消息列表，包含系统角色和用户角色的内容
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt + "\n" + user_prompt},
    ]

    # 调用字幕生成器生成字幕，指定最大新令牌数和是否返回完整文本
    response = caption_generator(
        messages,
        max_new_tokens=226,
        return_full_text=False
    )
    # 获取生成的字幕文本
    caption = response[0]["generated_text"]
    # 如果字幕以双引号开头和结尾，去掉这两个引号
    if caption.startswith("\"") and caption.endswith("\""):
        caption = caption[1:-1]
    # 返回生成的字幕
    return caption

# 定义生成图像的函数，接受字幕和进度参数
def generate_image(caption, progress=gr.Progress(track_tqdm=True)):
    # 调用图像生成器生成图像，指定相关参数
    image = image_generator(
        prompt=caption,
        height=480,
        width=720,
        num_inference_steps=30,
        guidance_scale=3.5,
    ).images[0]
    # 返回生成的图像，重复一次以便于后续处理
    return image, image  # One for output One for State

# 定义生成视频的函数，接受字幕、图像和进度参数
def generate_video(
        caption,
        image,
        progress=gr.Progress(track_tqdm=True)
):
    # 创建一个随机种子生成器
    generator = torch.Generator().manual_seed(seed)
    # 调用视频生成器生成视频帧，指定相关参数
    video_frames = video_generator(
        image=image,
        prompt=caption,
        height=480,
        width=720,
        num_frames=49,
        num_inference_steps=50,
        guidance_scale=6,
        use_dynamic_cfg=True,
        generator=generator,
    ).frames[0]
    # 保存生成的视频并获取视频路径
    video_path = save_video(video_frames)
    # 将视频转换为 GIF 并获取 GIF 路径
    gif_path = convert_to_gif(video_path)
    # 返回视频路径和 GIF 路径
    return video_path, gif_path

# 定义保存视频的函数，接受张量作为参数
def save_video(tensor):
    # 获取当前时间戳以命名视频文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建视频文件路径
    video_path = f"./output/{timestamp}.mp4"
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # 导出张量为视频文件，指定帧率
    export_to_video(tensor, video_path, fps=8)
    # 返回视频文件路径
    return video_path

# 定义将视频转换为 GIF 的函数，接受视频路径作为参数
def convert_to_gif(video_path):
    # 加载视频文件
    clip = mp.VideoFileClip(video_path)
    # 设置视频的帧率
    clip = clip.set_fps(8)
    # 调整视频的高度以进行 GIF 输出
    clip = clip.resize(height=240)
    # 创建 GIF 文件路径
    gif_path = video_path.replace(".mp4", ".gif")
    # 将视频写入 GIF 文件，指定帧率
    clip.write_gif(gif_path, fps=8)
    # 返回 GIF 文件路径
    return gif_path

# 定义删除旧文件的函数，功能尚未实现
def delete_old_files():
    # 无限循环，持续执行文件清理操作
        while True:
            # 获取当前日期和时间
            now = datetime.now()
            # 计算截止时间，当前时间减去10分钟
            cutoff = now - timedelta(minutes=10)
            # 定义要清理的目录列表
            directories = ["./output", "./gradio_tmp"]
    
            # 遍历目录列表
            for directory in directories:
                # 遍历当前目录中的文件名
                for filename in os.listdir(directory):
                    # 构造文件的完整路径
                    file_path = os.path.join(directory, filename)
                    # 检查路径是否为文件
                    if os.path.isfile(file_path):
                        # 获取文件的最后修改时间
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        # 如果文件的修改时间早于截止时间，删除该文件
                        if file_mtime < cutoff:
                            os.remove(file_path)
            # 暂停600秒（10分钟），然后继续循环
            time.sleep(600)
# 启动一个新线程来删除旧文件，设置为守护线程以便主程序退出时自动结束
threading.Thread(target=delete_old_files, daemon=True).start()

# 创建一个 Gradio 应用程序的界面
with gr.Blocks() as demo:
    # 添加一个 Markdown 组件，显示标题
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               LLM + FLUX + CogVideoX-I2V Space 🤗
            </div>
    """)
    # 创建一个行布局以排列组件
    with gr.Row():
        # 创建第一列布局
        with gr.Column():
            # 创建一个文本框用于输入提示
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=5)
            # 创建一个按钮用于生成字幕
            generate_caption_button = gr.Button("Generate Caption")
            # 创建一个文本框用于显示生成的字幕
            caption = gr.Textbox(label="Caption", placeholder="Caption will appear here", lines=5)
            # 创建一个按钮用于生成图像
            generate_image_button = gr.Button("Generate Image")
            # 创建一个图像组件用于显示生成的图像
            image_output = gr.Image(label="Generated Image")
            # 创建一个状态组件，用于保存图像状态
            state_image = gr.State()
            # 设置生成字幕按钮的点击事件，调用生成字幕函数
            generate_caption_button.click(fn=generate_caption, inputs=prompt, outputs=caption)
            # 设置生成图像按钮的点击事件，调用生成图像函数
            generate_image_button.click(fn=generate_image, inputs=caption, outputs=[image_output, state_image])
        # 创建第二列布局
        with gr.Column():
            # 创建一个视频组件用于显示生成的视频
            video_output = gr.Video(label="Generated Video", width=720, height=480)
            # 创建一个文件组件用于下载视频，初始设置为不可见
            download_video_button = gr.File(label="📥 Download Video", visible=False)
            # 创建一个文件组件用于下载 GIF，初始设置为不可见
            download_gif_button = gr.File(label="📥 Download GIF", visible=False)
            # 创建一个按钮用于从图像生成视频
            generate_video_button = gr.Button("Generate Video from Image")
            # 设置生成视频按钮的点击事件，调用生成视频函数
            generate_video_button.click(fn=generate_video, inputs=[caption, state_image],
                                        outputs=[video_output, download_gif_button])

# 如果当前模块是主程序，则启动 Gradio 应用程序
if __name__ == "__main__":
    demo.launch()
```

# `.\cogvideo-finetune\tools\llm_flux_cogvideox\llm_flux_cogvideox.py`

```py
# 文档字符串，提供项目原始实验代码的链接和使用说明
"""
The original experimental code for this project can be found at:

https://gist.github.com/a-r-r-o-w/d070cce059ab4ceab3a9f289ff83c69c

By using this code, description prompts will be generated through a local large language model, and images will be
generated using the black-forest-labs/FLUX.1-dev model, followed by video generation via CogVideoX.
The entire process utilizes open-source solutions, without the need for any API keys.

You can use the generate.sh file in the same folder to automate running this code
for batch generation of videos and images.

bash generate.sh

"""

# 导入命令行参数解析库
import argparse
# 导入垃圾回收库
import gc
# 导入JSON处理库
import json
# 导入操作系统功能库
import os
# 导入路径操作库
import pathlib
# 导入随机数生成库
import random
# 导入类型提示功能
from typing import Any, Dict

# 从transformers库导入自动标记器
from transformers import AutoTokenizer

# 设置环境变量，指定TORCH_LOGS的日志内容
os.environ["TORCH_LOGS"] = "+dynamo,recompiles,graph_breaks"
# 设置环境变量，开启TORCHDYNAMO的详细输出
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# 导入numpy库
import numpy as np
# 导入PyTorch库
import torch
# 导入transformers库
import transformers
# 从diffusers库导入视频生成相关的管道和调度器
from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler, DiffusionPipeline
# 从diffusers库导入日志记录工具
from diffusers.utils.logging import get_logger
# 从diffusers库导入视频导出工具
from diffusers.utils import export_to_video

# 设置PyTorch的浮点数乘法精度为高
torch.set_float32_matmul_precision("high")

# 获取日志记录器实例
logger = get_logger(__name__)

# 定义系统提示字符串，指导生成视频描述的任务
SYSTEM_PROMPT = """
You are part of a team of people that create videos using generative models. You use a video-generation model that can generate a video about anything you describe.

For example, if you respond with "A beautiful morning in the woods with the sun peaking through the trees", the video generation model will create a video of exactly as described. You task is to summarize the descriptions of videos provided to by users, and create details prompts to feed into the generative model.

There are a few rules to follow:
- You will only ever output a single video description per request.
- If the user mentions to summarize the prompt in [X] words, make sure to not exceed the limit.

You responses should just be the video generation prompt. Here are examples:
- “A lone figure stands on a city rooftop at night, gazing up at the full moon. The moon glows brightly, casting a gentle light over the quiet cityscape. Below, the windows of countless homes shine with warm lights, creating a contrast between the bustling life below and the peaceful solitude above. The scene captures the essence of the Mid-Autumn Festival, where despite the distance, the figure feels connected to loved ones through the shared beauty of the moonlit sky.”
- "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting."
# 包含一个描述场景的字符串，描述了一位街头艺术家和他的创作
- "A street artist, clad in a worn-out denim jacket and a colorful banana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall"
""".strip()

# 定义用户提示的模板，要求生成视频生成模型的提示，限制字数
USER_PROMPT = """
Could you generate a prompt for a video generation model? 
Please limit the prompt to [{0}] words.
""".strip()


# 定义一个获取命令行参数的函数
def get_args():
    # 创建命令行解析器实例
    parser = argparse.ArgumentParser()
    # 添加命令行参数：视频数量，类型为整数，默认值为5
    parser.add_argument(
        "--num_videos",
        type=int,
        default=5,
        help="Number of unique videos you would like to generate."
    )
    # 添加命令行参数：模型路径，类型为字符串，默认值为指定的模型路径
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX-5B",
        help="The path of Image2Video CogVideoX-5B",
    )
    # 添加命令行参数：标题生成模型ID，类型为字符串，默认值为指定的模型ID
    parser.add_argument(
        "--caption_generator_model_id",
        type=str,
        default="THUDM/glm-4-9b-chat",
        help="Caption generation model. default GLM-4-9B",
    )
    # 添加命令行参数：标题生成模型缓存目录，类型为字符串，默认值为None
    parser.add_argument(
        "--caption_generator_cache_dir",
        type=str,
        default=None,
        help="Cache directory for caption generation model."
    )
    # 添加命令行参数：图像生成模型ID，类型为字符串，默认值为指定的模型ID
    parser.add_argument(
        "--image_generator_model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Image generation model."
    )
    # 添加命令行参数：图像生成模型缓存目录，类型为字符串，默认值为None
    parser.add_argument(
        "--image_generator_cache_dir",
        type=str,
        default=None,
        help="Cache directory for image generation model."
    )
    # 添加命令行参数：图像生成推理步骤数量，类型为整数，默认值为50
    parser.add_argument(
        "--image_generator_num_inference_steps",
        type=int,
        default=50,
        help="Caption generation model."
    )
    # 添加命令行参数：引导比例，类型为浮点数，默认值为7
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7,
        help="Guidance scale to be use for generation."
    )
    # 添加命令行参数：是否使用动态CFG，动作类型为布尔值，默认值为False
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        help="Whether or not to use cosine dynamic guidance for generation [Recommended].",
    )
    # 添加命令行参数：输出目录，类型为字符串，默认值为"outputs/"
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="Location where generated images and videos should be stored.",
    )
    # 添加命令行参数：是否编译转换器，动作类型为布尔值，默认值为False
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether or not to compile the transformer of image and video generators."
    )
    # 添加命令行参数：是否启用VAE平铺，动作类型为布尔值，默认值为False
    parser.add_argument(
        "--enable_vae_tiling",
        action="store_true",
        help="Whether or not to use VAE tiling when encoding/decoding."
    )
    # 添加命令行参数：随机种子，类型为整数，默认值为42
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility."
    )
    # 解析命令行参数并返回结果
    return parser.parse_args()


# 定义一个重置内存的函数
def reset_memory():
    # 垃圾回收器收集所有未使用的对象
    gc.collect()
    # 清空CUDA的缓存
    torch.cuda.empty_cache()
    # 重置CUDA的峰值内存统计信息
    torch.cuda.reset_peak_memory_stats()
    # 重置CUDA的累积内存统计信息
    torch.cuda.reset_accumulated_memory_stats()


# 使用无梯度计算的上下文定义主函数
@torch.no_grad()
def main(args: Dict[str, Any]) -> None:
    # 将输出目录转换为路径对象
    output_dir = pathlib.Path(args.output_dir)
    # 如果输出目录不存在，则创建该目录
    os.makedirs(output_dir.as_posix(), exist_ok=True)

    # 设置随机种子以保证结果可重现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 为所有 GPU 设备设置随机种子，以确保结果可重现
    torch.cuda.manual_seed_all(args.seed)

    # 重置内存，以清理之前的计算图和变量
    reset_memory()
    # 从预训练模型中加载分词器，允许信任远程代码
    tokenizer = AutoTokenizer.from_pretrained(args.caption_generator_model_id, trust_remote_code=True)
    # 创建文本生成管道，使用指定的模型，并配置相关参数
    caption_generator = transformers.pipeline(
        "text-generation",  # 设置任务为文本生成
        model=args.caption_generator_model_id,  # 使用指定的模型ID
        device_map="auto",  # 自动分配设备（CPU/GPU）
        model_kwargs={  # 模型的其他参数配置
            "local_files_only": True,  # 仅使用本地文件
            "cache_dir": args.caption_generator_cache_dir,  # 设置缓存目录
            "torch_dtype": torch.bfloat16,  # 设置张量的数据类型为 bfloat16
        },
        trust_remote_code=True,  # 允许信任远程代码
        tokenizer=tokenizer  # 使用加载的分词器
    )

    # 初始化用于存储生成的标题的列表
    captions = []
    # 遍历指定数量的视频
    for i in range(args.num_videos):
        # 随机选择生成标题的字数
        num_words = random.choice([50, 75, 100])
        # 格式化用户提示，以包含字数信息
        user_prompt = USER_PROMPT.format(num_words)

        # 创建包含系统和用户消息的列表
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},  # 系统消息
            {"role": "user", "content": user_prompt},  # 用户消息
        ]

        # 生成标题，限制新生成的标记数
        outputs = caption_generator(messages, max_new_tokens=226)
        # 提取生成的文本内容
        caption = outputs[0]["generated_text"][-1]["content"]
        # 如果标题以引号开始和结束，去除引号
        if caption.startswith("\"") and caption.endswith("\""):
            caption = caption[1:-1]
        # 将生成的标题添加到列表中
        captions.append(caption)
        # 记录生成的标题
        logger.info(f"Generated caption: {caption}")

    # 将生成的标题保存到 JSON 文件中
    with open(output_dir / "captions.json", "w") as file:
        json.dump(captions, file)  # 将标题列表写入 JSON 文件

    # 删除标题生成器以释放内存
    del caption_generator
    # 重置内存
    reset_memory()

    # 从预训练模型加载图像生成器
    image_generator = DiffusionPipeline.from_pretrained(
        args.image_generator_model_id,  # 使用指定的图像生成模型ID
        cache_dir=args.image_generator_cache_dir,  # 设置缓存目录
        torch_dtype=torch.bfloat16  # 设置张量的数据类型为 bfloat16
    )
    # 将图像生成器移动到 GPU
    image_generator.to("cuda")

    # 如果编译选项被启用，则编译图像生成器的转换器
    if args.compile:
        image_generator.transformer = torch.compile(image_generator.transformer, mode="max-autotune", fullgraph=True)

    # 如果启用 VAE 瓦片功能，则允许图像生成器的 VAE 使用瓦片
    if args.enable_vae_tiling:
        image_generator.vae.enable_tiling()

    # 初始化用于存储生成的图像的列表
    images = []
    # 遍历生成的标题并生成对应的图像
    for index, caption in enumerate(captions):
        # 使用图像生成器生成图像，指定相关参数
        image = image_generator(
            prompt=caption,  # 使用标题作为提示
            height=480,  # 设置生成图像的高度
            width=720,  # 设置生成图像的宽度
            num_inference_steps=args.image_generator_num_inference_steps,  # 设置推理步骤数量
            guidance_scale=3.5,  # 设置指导比例
        ).images[0]  # 获取生成的图像

        # 处理标题以创建合法的文件名
        filename = caption[:25].replace(".", "_").replace("'", "_").replace('"', "_").replace(",", "_")
        # 保存生成的图像到指定目录
        image.save(output_dir / f"{index}_{filename}.png")
        # 将生成的图像添加到列表中
        images.append(image)

    # 删除图像生成器以释放内存
    del image_generator
    # 重置内存
    reset_memory()

    # 从预训练模型加载视频生成器
    video_generator = CogVideoXImageToVideoPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16).to("cuda")  # 移动到 GPU

    # 设置视频生成器的调度器
    video_generator.scheduler = CogVideoXDPMScheduler.from_config(
        video_generator.scheduler.config,  # 使用当前调度器的配置
        timestep_spacing="trailing"  # 设置时间步间隔为 trailing
    )

    # 如果编译选项被启用，则编译视频生成器的转换器
    if args.compile:
        video_generator.transformer = torch.compile(video_generator.transformer, mode="max-autotune", fullgraph=True)

    # 如果启用 VAE 瓦片功能，则允许视频生成器的 VAE 使用瓦片
    if args.enable_vae_tiling:
        video_generator.vae.enable_tiling()

    # 创建随机数生成器并设置种子
    generator = torch.Generator().manual_seed(args.seed)  # 确保随机结果可重现
    # 遍历 captions 和 images 的组合，获取索引及对应的描述和图像
        for index, (caption, image) in enumerate(zip(captions, images)):
            # 调用视频生成器，生成视频帧
            video = video_generator(
                # 设置生成视频的图像和描述
                image=image,
                prompt=caption,
                # 指定视频的高度和宽度
                height=480,
                width=720,
                # 设置生成的帧数和推理步骤
                num_frames=49,
                num_inference_steps=50,
                # 设置引导比例和动态配置选项
                guidance_scale=args.guidance_scale,
                use_dynamic_cfg=args.use_dynamic_cfg,
                # 提供随机数生成器
                generator=generator,
            ).frames[0]  # 获取生成的视频的第一帧
            # 格式化文件名，限制为前25个字符并替换特殊字符
            filename = caption[:25].replace(".", "_").replace("'", "_").replace('"', "_").replace(",", "_")
            # 导出生成的视频到指定目录，命名为索引加文件名
            export_to_video(video, output_dir / f"{index}_{filename}.mp4", fps=8)  # 设置每秒帧数为8
# 判断当前模块是否是主程序入口
if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    # 调用主函数，并传入获取的参数
    main(args)
```

# `.\cogvideo-finetune\tools\load_cogvideox_lora.py`

```py
# 版权声明，表明版权所有者及其保留权利
# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# 根据 Apache License, Version 2.0 进行授权
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 你可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件以“原样”基础分发，不提供任何形式的保证或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以获取有关权限和限制的具体信息
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入数学模块以进行数学计算
import math 
# 导入随机模块以生成随机数
import random 
# 导入时间模块以进行时间相关操作
import time
# 从 diffusers.utils 导入导出视频的功能
from diffusers.utils import export_to_video
# 从 diffusers.image_processor 导入 VAE 图像处理器
from diffusers.image_processor import VaeImageProcessor
# 导入日期和时间处理的模块
from datetime import datetime, timedelta
# 从 diffusers 导入多个类以供后续使用
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
# 导入操作系统模块以进行系统级操作
import os
# 导入 PyTorch 库以进行深度学习
import torch
# 导入参数解析模块以处理命令行参数
import argparse

# 根据是否有可用的 GPU 设定设备类型
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义获取命令行参数的函数
def get_args():
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加预训练模型路径参数
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # 添加 LoRA 权重路径参数
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        default=None,
        required=True,
        help="Path to lora weights.",
    )
    # 添加 LoRA 权重的秩参数
    parser.add_argument(
        "--lora_r",
        type=int,
        default=128,
        help="""LoRA weights have a rank parameter, with the default for 2B trans set at 128 and 5B trans set at 256. 
        This part is used to calculate the value for lora_scale, which is by default divided by the alpha value, 
        used for stable learning and to prevent underflow. In the SAT training framework,
        alpha is set to 1 by default. The higher the rank, the better the expressive capability,
        but it requires more memory and training time. Increasing this number blindly isn't always better.
        The formula for lora_scale is: lora_r / alpha.
        """,
    )
    # 添加 LoRA 权重的 alpha 参数
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=1,
        help="""LoRA weights have a rank parameter, with the default for 2B trans set at 128 and 5B trans set at 256. 
        This part is used to calculate the value for lora_scale, which is by default divided by the alpha value, 
        used for stable learning and to prevent underflow. In the SAT training framework,
        alpha is set to 1 by default. The higher the rank, the better the expressive capability,
        but it requires more memory and training time. Increasing this number blindly isn't always better.
        The formula for lora_scale is: lora_r / alpha.
        """,
    )
    # 添加用于生成内容的提示参数
    parser.add_argument(
        "--prompt",
        type=str,
        help="prompt",
    )
    # 向解析器添加一个名为 output_dir 的参数
        parser.add_argument(
            "--output_dir",  # 参数的名称
            type=str,  # 参数类型为字符串
            default="output",  # 默认值为 "output"
            help="The output directory where the model predictions and checkpoints will be written.",  # 参数的帮助说明
        )
    # 解析命令行参数并返回结果
        return parser.parse_args()
# 如果该脚本是主程序，则执行以下代码
if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    # 从预训练模型创建视频处理管道，并将其移动到指定设备
    pipe = CogVideoXPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    # 加载 LoRA 权重，指定权重文件名和适配器名称
    pipe.load_lora_weights(args.lora_weights_path,  weight_name="pytorch_lora_weights.safetensors", adapter_name="cogvideox-lora")
    # pipe.fuse_lora(lora_scale=args.lora_alpha/args.lora_r, ['transformer'])  # 注释掉的代码，用于融合 LoRA 权重
    # 计算 LoRA 缩放因子
    lora_scaling=args.lora_alpha/args.lora_r
    # 设置适配器及其对应的缩放因子
    pipe.set_adapters(["cogvideox-lora"], [lora_scaling])

    # 根据调度器配置创建视频调度器
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 创建输出目录，如果不存在则自动创建
    os.makedirs(args.output_dir, exist_ok=True)

    # 生成视频帧，设置相关参数
    latents = pipe(
        prompt=args.prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        use_dynamic_cfg=True,
        output_type="pt",
        guidance_scale=3.0,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).frames
    # 获取生成的帧的批量大小
    batch_size = latents.shape[0]
    # 初始化一个列表，用于存储视频帧
    batch_video_frames = []
    # 遍历每一帧，处理并转换为 PIL 图像
    for batch_idx in range(batch_size):
        pt_image = latents[batch_idx]
        # 将当前帧的各个通道堆叠成一个张量
        pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

        # 将 PyTorch 图像转换为 NumPy 格式
        image_np = VaeImageProcessor.pt_to_numpy(pt_image)
        # 将 NumPy 图像转换为 PIL 图像
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        # 将处理后的 PIL 图像添加到帧列表中
        batch_video_frames.append(image_pil)

    # 获取当前时间戳，用于视频文件命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 构造视频文件路径
    video_path = f"{args.output_dir}/{timestamp}.mp4"
    # 创建视频文件目录（如果不存在）
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # 选择第一组帧作为视频内容
    tensor = batch_video_frames[0]
    # 计算帧率，假设每 6 帧为 1 秒
    fps=math.ceil((len(batch_video_frames[0]) - 1) / 6)

    # 将处理后的帧导出为视频文件
    export_to_video(tensor, video_path, fps=fps)
```

# `.\cogvideo-finetune\tools\parallel_inference\parallel_inference_xdit.py`

```py
# 这是一个用于 CogVideo 的并行推理脚本，原始脚本来源于 xDiT 项目
"""
This is a parallel inference script for CogVideo. The original script
can be found from the xDiT project at

https://github.com/xdit-project/xDiT/blob/main/examples/cogvideox_example.py

By using this code, the inference process is parallelized on multiple GPUs,
and thus speeded up.

Usage:
1. pip install xfuser
2. mkdir results
3. run the following command to generate video
torchrun --nproc_per_node=4 parallel_inference_xdit.py \
    --model <cogvideox-model-path> --ulysses_degree 1 --ring_degree 2 \
    --use_cfg_parallel --height 480 --width 720 --num_frames 9 \
    --prompt 'A small dog.'

You can also use the run.sh file in the same folder to automate running this
code for batch generation of videos, by running:

sh ./run.sh

"""

# 导入必要的库
import time
import torch
import torch.distributed
from diffusers import AutoencoderKLTemporalDecoder
from xfuser import xFuserCogVideoXPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
)
from diffusers.utils import export_to_video

# 主函数
def main():
    # 创建参数解析器并描述用途
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    # 添加命令行参数并解析
    args = xFuserArgs.add_cli_args(parser).parse_args()
    # 从命令行参数创建引擎配置
    engine_args = xFuserArgs.from_cli_args(args)

    # 检查 ulysses_degree 是否有效
    num_heads = 30
    # 如果 ulysses_degree 大于 0 且不是 num_heads 的因子，则引发错误
    if engine_args.ulysses_degree > 0 and num_heads % engine_args.ulysses_degree != 0:
        raise ValueError(
            f"ulysses_degree ({engine_args.ulysses_degree}) must be a divisor of the number of heads ({num_heads})"
        )

    # 创建引擎和输入配置
    engine_config, input_config = engine_args.create_config()
    # 获取本地进程的排名
    local_rank = get_world_group().local_rank

    # 从预训练模型加载管道
    pipe = xFuserCogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )
    # 如果启用 CPU 离线，进行相应设置
    if args.enable_sequential_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        pipe.vae.enable_tiling()
    else:
        # 将管道移动到指定的 GPU 设备
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)

    # 重置 GPU 的峰值内存统计
    torch.cuda.reset_peak_memory_stats()
    # 记录开始时间
    start_time = time.time()

    # 执行推理，生成视频帧
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        guidance_scale=6,
    ).frames[0]

    # 记录结束时间
    end_time = time.time()
    # 计算推理耗时
    elapsed_time = end_time - start_time
    # 获取当前设备的峰值内存使用量
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    # 构建包含各种并行配置参数的字符串，用于输出文件名
        parallel_info = (
            f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
            f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
            f"tp{engine_args.tensor_parallel_degree}_"
            f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
        )
        # 检查是否为数据并行的最后一组
        if is_dp_last_group():
            # 获取数据并行的全局大小
            world_size = get_data_parallel_world_size()
            # 根据输入配置构建分辨率字符串
            resolution = f"{input_config.width}x{input_config.height}"
            # 生成输出文件名，包含并行信息和分辨率
            output_filename = f"results/cogvideox_{parallel_info}_{resolution}.mp4"
            # 将输出内容导出为视频文件
            export_to_video(output, output_filename, fps=8)
            # 打印保存的输出文件名
            print(f"output saved to {output_filename}")
    
        # 检查当前进程是否为最后一个进程
        if get_world_group().rank == get_world_group().world_size - 1:
            # 打印当前周期的耗时和内存使用情况
            print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
        # 销毁分布式环境的运行时状态
        get_runtime_state().destory_distributed_env()
# 判断当前脚本是否为主程序
if __name__ == "__main__":
    # 调用主函数
    main()
```

# `.\cogvideo-finetune\tools\replicate\predict_i2v.py`

```py
# Cog 的预测接口 ⚙️
# https://cog.run/python

# 导入必要的库
import os  # 用于操作系统功能
import subprocess  # 用于执行子进程命令
import time  # 用于时间相关操作
import torch  # 用于深度学习库
from diffusers import CogVideoXImageToVideoPipeline  # 导入视频生成管道
from diffusers.utils import export_to_video, load_image  # 导入工具函数
from cog import BasePredictor, Input, Path  # 导入 Cog 的基础预测器和输入处理

# 定义模型缓存目录
MODEL_CACHE = "model_cache_i2v"
# 定义模型下载的 URL
MODEL_URL = (
    f"https://weights.replicate.delivery/default/THUDM/CogVideo/{MODEL_CACHE}.tar"
)
# 设置环境变量以离线模式运行
os.environ["HF_DATASETS_OFFLINE"] = "1"  # 禁用数据集在线下载
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 禁用变换器在线下载
os.environ["HF_HOME"] = MODEL_CACHE  # 设置 Hugging Face 的缓存目录
os.environ["TORCH_HOME"] = MODEL_CACHE  # 设置 PyTorch 的缓存目录
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE  # 设置数据集缓存目录
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE  # 设置变换器缓存目录
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE  # 设置 Hugging Face Hub 缓存目录

# 定义下载模型权重的函数
def download_weights(url, dest):
    start = time.time()  # 记录开始时间
    print("downloading url: ", url)  # 输出下载 URL
    print("downloading to: ", dest)  # 输出下载目标路径
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)  # 调用 pget 下载模型权重
    print("downloading took: ", time.time() - start)  # 输出下载所用时间

# 定义预测类
class Predictor(BasePredictor):
    def setup(self) -> None:
        """将模型加载到内存中以提高多个预测的效率"""

        # 如果模型缓存目录不存在，则下载模型权重
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # 使用预训练模型初始化管道
        # model_id: THUDM/CogVideoX-5b-I2V
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            MODEL_CACHE, torch_dtype=torch.bfloat16  # 使用 bfloat16 数据类型
        ).to("cuda")  # 将模型转移到 GPU

        self.pipe.enable_model_cpu_offload()  # 启用模型 CPU 离线处理
        self.pipe.vae.enable_tiling()  # 启用 VAE 的平铺处理

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt", default="Starry sky slowly rotating."
        ),  # 输入提示的默认值
        image: Path = Input(description="Input image"),  # 输入图像路径
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),  # 去噪步骤数量的输入
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=6
        ),  # 分类无关引导的比例输入
        num_frames: int = Input(
            description="Number of frames for the output video", default=49
        ),  # 输出视频的帧数输入
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),  # 随机种子的输入
    ) -> Path:
        """对模型进行单次预测"""

        # 如果没有提供种子，则生成一个随机种子
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")  # 生成随机种子
        print(f"Using seed: {seed}")  # 输出使用的种子

        img = load_image(image=str(image))  # 加载输入图像

        # 调用管道进行视频生成
        video = self.pipe(
            prompt=prompt,  # 输入提示
            image=img,  # 输入图像
            num_videos_per_prompt=1,  # 每个提示生成一个视频
            num_inference_steps=num_inference_steps,  # 去噪步骤数量
            num_frames=num_frames,  # 输出视频帧数
            guidance_scale=guidance_scale,  # 分类无关引导比例
            generator=torch.Generator(device="cuda").manual_seed(seed),  # 随机数生成器
        ).frames[0]  # 获取生成的视频帧

        out_path = "/tmp/out.mp4"  # 设置输出视频的路径

        export_to_video(video, out_path, fps=8)  # 导出视频到指定路径
        return Path(out_path)  # 返回输出视频路径
```

# `.\cogvideo-finetune\tools\replicate\predict_t2v.py`

```py
# Cog的预测接口 ⚙️
# https://cog.run/python

# 导入必要的库和模块
import os  # 用于与操作系统交互
import subprocess  # 用于执行子进程
import time  # 用于时间管理
import torch  # 深度学习框架
from diffusers import CogVideoXPipeline  # 导入CogVideoXPipeline类
from diffusers.utils import export_to_video  # 导入视频导出工具
from cog import BasePredictor, Input, Path  # 导入Cognition框架的基础类和输入类

MODEL_CACHE = "model_cache"  # 定义模型缓存目录
MODEL_URL = (  # 定义模型权重下载URL
    f"https://weights.replicate.delivery/default/THUDM/CogVideo/{MODEL_CACHE}.tar"
)
# 设置环境变量，强制使用离线模式以避免下载模型
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE  # 设置Hugging Face的根目录
os.environ["TORCH_HOME"] = MODEL_CACHE  # 设置PyTorch的根目录
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE  # 设置数据集缓存目录
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE  # 设置变换器缓存目录
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE  # 设置Hugging Face Hub缓存目录

# 定义下载权重的函数
def download_weights(url, dest):
    start = time.time()  # 记录开始时间
    print("downloading url: ", url)  # 输出下载URL
    print("downloading to: ", dest)  # 输出下载目标路径
    # 使用子进程命令下载文件
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)  # 输出下载所需时间

# 定义预测器类，继承自BasePredictor
class Predictor(BasePredictor):
    def setup(self) -> None:
        """将模型加载到内存中，以提高多次预测的效率"""

        # 检查模型缓存目录是否存在
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)  # 如果不存在，下载模型权重

        # 加载指定的模型
        # model_id: THUDM/CogVideoX-5b
        self.pipe = CogVideoXPipeline.from_pretrained(
            MODEL_CACHE,  # 使用缓存的模型
            torch_dtype=torch.bfloat16,  # 设置模型的数据类型
        ).to("cuda")  # 将模型移动到GPU

        self.pipe.enable_model_cpu_offload()  # 启用CPU卸载以优化内存使用
        self.pipe.vae.enable_tiling()  # 启用VAE的分块处理

    # 定义预测方法
    def predict(
        self,
        prompt: str = Input(  # 输入提示，描述生成内容
            description="Input prompt",
            default="A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance.",
        ),
        num_inference_steps: int = Input(  # 输入去噪步骤数量
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(  # 输入无分类指导的比例
            description="Scale for classifier-free guidance", ge=1, le=20, default=6
        ),
        num_frames: int = Input(  # 输入输出视频的帧数
            description="Number of frames for the output video", default=49
        ),
        seed: int = Input(  # 输入随机种子，留空以随机化
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    # 预测模型的单次运行，返回生成的视频路径
    ) -> Path:
        # 文档字符串，说明函数的功能
        """Run a single prediction on the model"""
    
        # 如果没有提供种子，则随机生成一个种子
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        # 输出当前使用的种子
        print(f"Using seed: {seed}")
    
        # 调用模型管道生成视频，使用提供的参数
        video = self.pipe(
            # 传入的提示文本
            prompt=prompt,
            # 每个提示生成一个视频
            num_videos_per_prompt=1,
            # 推理步骤的数量
            num_inference_steps=num_inference_steps,
            # 视频帧数
            num_frames=num_frames,
            # 指导比例
            guidance_scale=guidance_scale,
            # 设定随机数生成器，使用指定的种子
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]  # 取得生成的视频的第一帧
    
        # 设置视频输出路径
        out_path = "/tmp/out.mp4"
    
        # 将生成的视频导出为 MP4 文件，帧率为8
        export_to_video(video, out_path, fps=8)
        # 返回输出路径的 Path 对象
        return Path(out_path)
```

# Enhance CogVideoX Generated Videos with VEnhancer

This tutorial will guide you through using the VEnhancer tool to enhance videos generated by CogVideoX, including
achieving higher frame rates and higher resolutions.

## Model Introduction

VEnhancer implements spatial super-resolution, temporal super-resolution (frame interpolation), and video refinement in
a unified framework. It can flexibly adapt to different upsampling factors (e.g., 1x~8x) for spatial or temporal
super-resolution. Additionally, it provides flexible control to modify the refinement strength, enabling it to handle
diverse video artifacts.

VEnhancer follows the design of ControlNet, copying the architecture and weights of the multi-frame encoder and middle
block from a pre-trained video diffusion model to build a trainable conditional network. This video ControlNet accepts
low-resolution keyframes and noisy full-frame latents as inputs. In addition to the time step t and prompt, our proposed
video-aware conditioning also includes noise augmentation level σ and downscaling factor s as additional network
conditioning inputs.

## Hardware Requirements

+ Operating System: Linux (requires xformers dependency)
+ Hardware: NVIDIA GPU with at least 60GB of VRAM per card. Machines such as H100, A100 are recommended.

## Quick Start

1. Clone the repository and install dependencies as per the official instructions:

```py
git clone https://github.com/Vchitect/VEnhancer.git
cd VEnhancer
## Torch and other dependencies can use those from CogVideoX. If you need to create a new environment, use the following commands:
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

## Install required dependencies
pip install -r requirements.txt
```

Where:

- `input_path` is the path to the input video
- `prompt` is the description of the video content. The prompt used by this tool should be shorter, not exceeding 77
  words. You may need to simplify the prompt used for generating the CogVideoX video.
- `target_fps` is the target frame rate for the video. Typically, 16 fps is already smooth, with 24 fps as the default
  value.
- `up_scale` is recommend to be set to 2,3,4. The target resolution is limited to be around 2k and below.
- `noise_aug` value depends on the input video quality. Lower quality needs higher noise levels, which corresponds to
  stronger refinement. 250~300 is for very low-quality videos. good videos: <= 200.
- `steps`  if you want fewer steps, please change solver_mode to "normal" first, then decline the number of steps. "
  fast" solver_mode has fixed steps (15).
  The code will automatically download the required models from Hugging Face during execution.

Typical runtime logs are as follows:

```py
/share/home/zyx/.conda/envs/cogvideox/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_fwd")
/share/home/zyx/.conda/envs/cogvideox/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_bwd")
2024-08-20 13:25:17,553 - video_to_video - INFO - checkpoint_path: ./ckpts/venhancer_paper.pt
/share/home/zyx/.conda/envs/cogvideox/lib/python3.10/site-packages/open_clip/factory.py:88: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path, map_location=map_location)
2024-08-20 13:25:37,486 - video_to_video - INFO - Build encoder with FrozenOpenCLIPEmbedder
/share/home/zyx/Code/VEnhancer/video_to_video/video_to_video_model.py:35: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  load_dict = torch.load(cfg.model_path, map_location='cpu')
2024-08-20 13:25:55,391 - video_to_video - INFO - Load model path ./ckpts/venhancer_paper.pt, with local status <All keys matched successfully>
2024-08-20 13:25:55,392 - video_to_video - INFO - Build diffusion with GaussianDiffusion
2024-08-20 13:26:16,092 - video_to_video - INFO - input video path: inputs/000000.mp4
2024-08-20 13:26:16,093 - video_to_video - INFO - text: Wide-angle aerial shot at dawn,soft morning light casting long shadows,an elderly man walking his dog through a quiet,foggy park,trees and benches in the background,peaceful and serene atmosphere
2024-08-20 13:26:16,156 - video_to_video - INFO - input frames length: 49
2024-08-20 13:26:16,156 - video_to_video - INFO - input fps: 8.0
2024-08-20 13:26:16,156 - video_to_video - INFO - target_fps: 24.0
2024-08-20 13:26:16,311 - video_to_video - INFO - input resolution: (480, 720)
2024-08-20 13:26:16,312 - video_to_video - INFO - target resolution: (1320, 1982)
2024-08-20 13:26:16,312 - video_to_video - INFO - noise augmentation: 250
2024-08-20 13:26:16,312 - video_to_video - INFO - scale s is set to: 8
2024-08-20 13:26:16,399 - video_to_video - INFO - video_data shape: torch.Size([145, 3, 1320, 1982])
/share/home/zyx/Code/VEnhancer/video_to_video/video_to_video_model.py:108: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with amp.autocast(enabled=True):
2024-08-20 13:27:19,605 - video_to_video - INFO - step: 0
2024-08-20 13:30:12,020 - video_to_video - INFO - step: 1
2024-08-20 13:33:04,956 - video_to_video - INFO - step: 2
2024-08-20 13:35:58,691 - video_to_video - INFO - step: 3
2024-08-20 13:38:51,254 - video_to_video - INFO - step: 4
2024-08-20 13:41:44,150 - video_to_video - INFO - step: 5
2024-08-20 13:44:37,017 - video_to_video - INFO - step: 6
2024-08-20 13:47:30,037 - video_to_video - INFO - step: 7
2024-08-20 13:50:22,838 - video_to_video - INFO - step: 8
2024-08-20 13:53:15,844 - video_to_video - INFO - step: 9
2024-08-20 13:56:08,657 - video_to_video - INFO - step: 10
2024-08-20 13:59:01,648 - video_to_video - INFO - step: 11
2024-08-20 14:01:54,541 - video_to_video - INFO - step: 12
2024-08-20 14:04:47,488 - video_to_video - INFO - step: 13
2024-08-20 14:10:13,637 - video_to_video - INFO - sampling, finished.

```

Running on a single A100 GPU, enhancing each 6-second CogVideoX generated video with default settings will consume 60GB
of VRAM and take 40-50 minutes.


# VEnhancer で CogVideoX によって生成されたビデオを強化する

このチュートリアルでは、VEnhancer ツールを使用して、CogVideoX で生成されたビデオを強化し、より高いフレームレートと高い解像度を実現する方法を説明します。

## モデルの紹介

VEnhancer は、空間超解像、時間超解像（フレーム補間）、およびビデオのリファインメントを統一されたフレームワークで実現します。空間または時間の超解像のために、さまざまなアップサンプリング係数（例：1x〜8x）に柔軟に対応できます。さらに、多様なビデオアーティファクトを処理するために、リファインメント強度を変更する柔軟な制御を提供します。

VEnhancer は ControlNet の設計に従い、事前訓練されたビデオ拡散モデルのマルチフレームエンコーダーとミドルブロックのアーキテクチャとウェイトをコピーして、トレーニング可能な条件ネットワークを構築します。このビデオ ControlNet は、低解像度のキーフレームとノイズを含む完全なフレームを入力として受け取ります。さらに、タイムステップ t とプロンプトに加えて、提案されたビデオ対応条件により、ノイズ増幅レベル σ およびダウンスケーリングファクター s が追加のネットワーク条件として使用されます。

## ハードウェア要件

+ オペレーティングシステム: Linux (xformers 依存関係が必要)
+ ハードウェア: 単一カードあたり少なくとも 60GB の VRAM を持つ NVIDIA GPU。H100、A100 などのマシンを推奨します。

## クイックスタート

1. 公式の指示に従ってリポジトリをクローンし、依存関係をインストールします。

```py
git clone https://github.com/Vchitect/VEnhancer.git
cd VEnhancer
## Torch などの依存関係は CogVideoX の依存関係を使用できます。新しい環境を作成する必要がある場合は、以下のコマンドを使用してください。
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

## 必須の依存関係をインストールします。
pip install -r requirements.txt
```

2. コードを実行します。

```py
python enhance_a_video.py --up_scale 4 --target_fps 24 --noise_aug 250 --solver_mode 'fast' --steps 15 --input_path inputs/000000.mp4 --prompt 'Wide-angle aerial shot at dawn, soft morning light casting long shadows, an elderly man walking his dog through a quiet, foggy park, trees and benches in the background, peaceful and serene atmosphere' --save_dir 'results/'
```

次の設定を行います：

- `input_path` 是输入视频的路径
- `prompt` 是视频内容的描述。此工具使用的提示词应更短，不超过77个字。您可能需要简化用于生成CogVideoX视频的提示词。
- `target_fps` 是视频的目标帧率。通常，16 fps已经很流畅，默认值为24 fps。
- `up_scale` 推荐设置为2、3或4。目标分辨率限制在2k左右及以下。
- `noise_aug` 的值取决于输入视频的质量。质量较低的视频需要更高的噪声级别，这对应于更强的优化。250~300适用于非常低质量的视频。对于高质量视频，设置为≤200。
- `steps` 如果想减少步数，请先将solver_mode改为“normal”，然后减少步数。“fast”模式的步数是固定的（15步）。
  代码在执行过程中会自动从Hugging Face下载所需的模型。

コードの実行中に、必要なモデルは Hugging Face から自動的にダウンロードされます。

```py
/share/home/zyx/.conda/envs/cogvideox/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_fwd")
/share/home/zyx/.conda/envs/cogvideox/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_bwd")
2024-08-20 13:25:17,553 - video_to_video - INFO - checkpoint_path: ./ckpts/venhancer_paper.pt
/share/home/zyx/.conda/envs/cogvideox/lib/python3.10/site-packages/open_clip/factory.py:88: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path, map_location=map_location)
2024-08-20 13:25:37,486 - video_to_video - INFO - Build encoder with FrozenOpenCLIPEmbedder
/share/home/zyx/Code/VEnhancer/video_to_video/video_to_video_model.py:35: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  load_dict = torch.load(cfg.model_path, map_location='cpu')
2024-08-20 13:25:55,391 - video_to_video - INFO - Load model path ./ckpts/venhancer_paper.pt, with local status <All keys matched successfully>
2024-08-20 13:25:55,392 - video_to_video - INFO - Build diffusion with GaussianDiffusion
2024-08-20 13:26:16,092 - video_to_video - INFO - input video path: inputs/000000.mp4
2024-08-20 13:26:16,093 - video_to_video - INFO - text: Wide-angle aerial shot at dawn,soft morning light casting long shadows,an elderly man walking his dog through a quiet,foggy park,trees and benches in the background,peaceful and serene atmosphere
2024-08-20 13:26:16,156 - video_to_video - INFO - input frames length: 49
2024-08-20 13:26:16,156 - video_to_video - INFO - input fps: 8.0
2024-08-20 13:26:16,156 - video_to_video - INFO - target_fps: 24.0
2024-08-20 13:26:16,311 - video_to_video - INFO - input resolution: (480, 720)
2024-08-20 13:26:16,312 - video_to_video - INFO - target resolution: (1320, 1982)
2024-08-20 13:26:16,312 - video_to_video - INFO - noise augmentation: 250
2024-08-20 13:26:16,312 - video_to_video - INFO - scale s is set to: 8
2024-08-20 13:26:16,399 - video_to_video - INFO - video_data shape: torch.Size([145, 3, 1320, 1982])
/share/home/zyx/Code/VEnhancer/video_to_video/video_to_video_model.py:108: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with amp.autocast(enabled=True):
2024-08-20 13:27:19,605 - video_to_video - INFO - step: 0
2024-08-20 13:30:12,020 - video_to_video - INFO - step: 1
2024-08-20 13:33:04,956 - video_to_video - INFO - step: 2
2024-08-20 13:35:58,691 - video_to_video - INFO - step: 3
2024-08-20 13:38:51,254 - video_to_video - INFO - step: 4
2024-08-20 13:41:44,150 - video_to_video - INFO - step: 5
2024-08-20 13:44:37,017 - video_to_video - INFO - step: 6
2024-08-20 13:47:30,037 - video_to_video - INFO - step: 7
2024-08-20 13:50:22,838 - video_to_video - INFO - step: 8
2024-08-20 13:53:15,844 - video_to_video - INFO - step: 9
2024-08-20 13:56:08,657 - video_to_video - INFO - step: 10
2024-08-20 13:59:01,648 - video_to_video - INFO - step: 11
2024-08-20 14:01:54,541 - video_to_video - INFO - step: 12
2024-08-20 14:04:47,488 - video_to_video - INFO - step: 13
2024-08-20 14:10:13,637 - video_to_video - INFO - sampling, finished.

```

A100 GPU を単一で使用している場合、CogVideoX によって生成された 6 秒間のビデオを強化するには、デフォルト設定で 60GB の VRAM を消費し、40〜50 分かかります。


# 使用 VEnhancer 对 CogVdieoX 生成视频进行增强

本教程将要使用 VEnhancer 工具 对 CogVdieoX 生成视频进行增强, 包括更高的帧率和更高的分辨率

## 模型介绍

VEnhancer 在一个统一的框架中实现了空间超分辨率、时间超分辨率（帧插值）和视频优化。它可以灵活地适应不同的上采样因子（例如，1x~
8x）用于空间或时间超分辨率。此外，它提供了灵活的控制，以修改优化强度，从而处理多样化的视频伪影。

VEnhancer 遵循 ControlNet 的设计，复制了预训练的视频扩散模型的多帧编码器和中间块的架构和权重，构建了一个可训练的条件网络。这个视频
ControlNet 接受低分辨率关键帧和包含噪声的完整帧作为输入。此外，除了时间步 t 和提示词外，我们提出的视频感知条件还将噪声增强的噪声级别
σ 和降尺度因子 s 作为附加的网络条件输入。

## 硬件需求

+ 操作系统: Linux (需要依赖xformers)
+ 硬件: NVIDIA GPU 并至少保证单卡显存超过60G，推荐使用 H100，A100等机器。

## 快速上手

1. 按照官方指引克隆仓库并安装依赖

```py
git clone https://github.com/Vchitect/VEnhancer.git
cd VEnhancer
## torch等依赖可以使用CogVideoX的依赖，如果你需要创建一个新的环境，可以使用以下命令
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

## 安装必须的依赖
pip install -r requirements.txt
```

2. 运行代码

```py
python enhance_a_video.py \
--up_scale 4 --target_fps 24 --noise_aug 250 \
--solver_mode 'fast' --steps 15 \
--input_path inputs/000000.mp4 \
--prompt 'Wide-angle aerial shot at dawn,soft morning light casting long shadows,an elderly man walking his dog through a quiet,foggy park,trees and benches in the background,peaceful and serene atmosphere' \
--save_dir 'results/' 
```

其中:

- `input_path` 是输入视频的路径
- `prompt` 是视频内容的描述。此工具使用的提示词应更短，不超过77个字。您可能需要简化用于生成CogVideoX视频的提示词。
- `target_fps` 是视频的目标帧率。通常，16 fps已经很流畅，默认值为24 fps。
- `up_scale` 推荐设置为2、3或4。目标分辨率限制在2k左右及以下。
- `noise_aug` 的值取决于输入视频的质量。质量较低的视频需要更高的噪声级别，这对应于更强的优化。250~300适用于非常低质量的视频。对于高质量视频，设置为≤200。
- `steps` 如果想减少步数，请先将solver_mode改为“normal”，然后减少步数。“fast”模式的步数是固定的（15步）。
  代码在执行过程中会自动从Hugging Face下载所需的模型。

代码运行过程中，会自动从Huggingface拉取需要的模型

运行日志通常如下:

```py
/share/home/zyx/.conda/envs/cogvideox/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_fwd")
/share/home/zyx/.conda/envs/cogvideox/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_bwd")
2024-08-20 13:25:17,553 - video_to_video - INFO - checkpoint_path: ./ckpts/venhancer_paper.pt
/share/home/zyx/.conda/envs/cogvideox/lib/python3.10/site-packages/open_clip/factory.py:88: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path, map_location=map_location)
2024-08-20 13:25:37,486 - video_to_video - INFO - Build encoder with FrozenOpenCLIPEmbedder
/share/home/zyx/Code/VEnhancer/video_to_video/video_to_video_model.py:35: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  load_dict = torch.load(cfg.model_path, map_location='cpu')
2024-08-20 13:25:55,391 - video_to_video - INFO - Load model path ./ckpts/venhancer_paper.pt, with local status <All keys matched successfully>
2024-08-20 13:25:55,392 - video_to_video - INFO - Build diffusion with GaussianDiffusion
2024-08-20 13:26:16,092 - video_to_video - INFO - input video path: inputs/000000.mp4
2024-08-20 13:26:16,093 - video_to_video - INFO - text: Wide-angle aerial shot at dawn,soft morning light casting long shadows,an elderly man walking his dog through a quiet,foggy park,trees and benches in the background,peaceful and serene atmosphere
2024-08-20 13:26:16,156 - video_to_video - INFO - input frames length: 49
2024-08-20 13:26:16,156 - video_to_video - INFO - input fps: 8.0
2024-08-20 13:26:16,156 - video_to_video - INFO - target_fps: 24.0
2024-08-20 13:26:16,311 - video_to_video - INFO - input resolution: (480, 720)
2024-08-20 13:26:16,312 - video_to_video - INFO - target resolution: (1320, 1982)
2024-08-20 13:26:16,312 - video_to_video - INFO - noise augmentation: 250
2024-08-20 13:26:16,312 - video_to_video - INFO - scale s is set to: 8
2024-08-20 13:26:16,399 - video_to_video - INFO - video_data shape: torch.Size([145, 3, 1320, 1982])
/share/home/zyx/Code/VEnhancer/video_to_video/video_to_video_model.py:108: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with amp.autocast(enabled=True):
2024-08-20 13:27:19,605 - video_to_video - INFO - step: 0
2024-08-20 13:30:12,020 - video_to_video - INFO - step: 1
2024-08-20 13:33:04,956 - video_to_video - INFO - step: 2
2024-08-20 13:35:58,691 - video_to_video - INFO - step: 3
2024-08-20 13:38:51,254 - video_to_video - INFO - step: 4
2024-08-20 13:41:44,150 - video_to_video - INFO - step: 5
2024-08-20 13:44:37,017 - video_to_video - INFO - step: 6
2024-08-20 13:47:30,037 - video_to_video - INFO - step: 7
2024-08-20 13:50:22,838 - video_to_video - INFO - step: 8
2024-08-20 13:53:15,844 - video_to_video - INFO - step: 9
2024-08-20 13:56:08,657 - video_to_video - INFO - step: 10
2024-08-20 13:59:01,648 - video_to_video - INFO - step: 11
2024-08-20 14:01:54,541 - video_to_video - INFO - step: 12
2024-08-20 14:04:47,488 - video_to_video - INFO - step: 13
2024-08-20 14:10:13,637 - video_to_video - INFO - sampling, finished.

```

使用A100单卡运行，对于每个CogVideoX生产的6秒视频，按照默认配置，会消耗60G显存，并用时40-50分钟。