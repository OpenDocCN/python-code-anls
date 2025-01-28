# `.\minimind-v\3-eval_chat.py`

```
# 导入所需的库
import os  # 用于操作系统相关功能（如文件路径、目录管理等）
import random  # 用于生成随机数
import numpy as np  # 用于科学计算，尤其是数组处理
import torch  # 用于深度学习和张量计算
import warnings  # 用于发出警告信息
from PIL import Image  # 用于处理图像文件
from transformers import AutoTokenizer, AutoModelForCausalLM  # 用于加载和处理预训练的自然语言处理模型
from model.model import Transformer  # 导入自定义的Transformer模型
from model.LMConfig import LMConfig  # 导入自定义的语言模型配置
from model.vision_utils import get_vision_model, get_img_process, get_img_embedding  # 导入图像处理工具

# 忽略所有警告信息
warnings.filterwarnings('ignore')


# 计算模型中可训练参数的数量
def count_parameters(model):
    # 遍历模型中的所有参数，计算其总数量（仅计算需要梯度的参数）
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 初始化模型，包括加载预训练模型、设置设备等
def init_model(lm_config, device, multi):
    # 加载用于模型的分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model_from = 1  # 1表示从权重文件加载模型，2表示使用transformers库的模型

    # 如果选择从权重文件加载模型
    if model_from == 1:
        # 如果使用MOE模块，根据配置选择加载路径
        moe_path = '_moe' if lm_config.use_moe else ''
        # 根据multi值选择不同的模型权重文件
        if multi:
            ckp = f'./out/{lm_config.dim}{moe_path}_vlm_sft_multi.pth'
        else:
            ckp = f'./out/{lm_config.dim}{moe_path}_vlm_sft.pth'
        # 初始化自定义的Transformer模型
        model = Transformer(lm_config)
        # 加载模型权重
        state_dict = torch.load(ckp, map_location=device)

        # 处理不需要的前缀（移除权重字典中的不需要的前缀部分）
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        # 将权重加载到模型中（不严格匹配，允许某些参数缺失）
        model.load_state_dict(state_dict, strict=False)
    else:
        # 如果选择使用transformers库的预训练模型
        model = AutoModelForCausalLM.from_pretrained('minimind-v-v1-small', trust_remote_code=True)

    # 将模型移至指定设备（GPU或CPU）
    model = model.to(device)
    # 输出模型的参数量（以百万和十亿为单位）
    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')

    # 获取视觉模型和预处理函数
    vision_model, preprocess = get_vision_model(encoder_type="clip")
    # 将视觉模型移至指定设备
    vision_model = vision_model.to(device)
    # 返回初始化后的模型、分词器、视觉模型和图像预处理函数
    return model, tokenizer, vision_model, preprocess


# 设置随机种子，确保实验可重复
def setup_seed(seed):
    # 设置Python随机数生成器的种子
    random.seed(seed)
    # 设置NumPy的随机数生成器种子
    np.random.seed(seed)
    # 设置PyTorch的随机数生成器种子
    torch.manual_seed(seed)
    # 设置CUDA的随机数生成器种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置PyTorch使用的cuDNN库为确定性模式，避免不确定性
    torch.backends.cudnn.deterministic = True
    # 设置PyTorch不使用cuDNN加速模式
    torch.backends.cudnn.benchmark = False


# 主程序入口
if __name__ == "__main__":
    # -------------------------- 基本参数设置 -----------------------------------
    multi = False  # 设置multi参数，控制单图或多图推理
    out_dir = 'out'  # 设置输出目录
    temperature = 0.5  # 设置生成文本时的温度参数
    top_k = 8  # 设置生成时的top-k采样数
    setup_seed(1337)  # 设置随机种子为1337，确保实验可重复
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 设置设备为GPU或CPU
    dtype = 'bfloat16'  # 设置数据类型为bfloat16
    max_seq_len = 1024  # 设置最大序列长度
    encoder_type = "clip"  # 设置编码器类型为clip
    # 根据选择的编码器类型配置语言模型的配置
    if encoder_type == "clip":
        lm_config = LMConfig()  # 使用默认的LMConfig配置
    else:
        lm_config = LMConfig(image_special_token='<' * 98 + '>' * 98, image_ids=[30] * 98 + [32] * 98)
    lm_config.max_seq_len = max_seq_len  # 设置最大序列长度
    # 初始化模型、分词器、视觉模型和预处理函数
    model, tokenizer, vision_model, preprocess = init_model(lm_config, device, multi)
    model.eval()  # 将模型设置为评估模式

    # -------------------------- 问题和目录设置 -----------------------------------
    if multi:
        # 设置图像目录（如果使用多图推理）
        image_dir = './dataset/eval_multi_images/bird/'
        # 设置提示语句
        prompt = f"{lm_config.image_special_token}\n{lm_config.image_special_token}\nName all the differences between these two birds."
    else:
        # 如果不是多图推理，则设置图片目录和提示语句
        image_dir = './dataset/eval_images/'
        prompt = lm_config.image_special_token + '\n这个图片描述的是什么内容？'

    # 获取图片目录下的所有文件，并按文件名排序
    image_files = sorted(os.listdir(image_dir))

    # -------------------------- 推理逻辑 -----------------------------------
    if multi:
        # 多图推理：所有图像编码一次性推理
        image_encoders = []
        for image_file in image_files:
            # 获取图片路径
            image_path = os.path.join(image_dir, image_file)
            # 打开并转换图片为RGB格式
            image = Image.open(image_path).convert('RGB')
            # 对图片进行预处理和编码
            image_process = get_img_process(image, preprocess).to(vision_model.device)
            image_encoder = get_img_embedding(image_process, vision_model).unsqueeze(0)
            image_encoders.append(image_encoder)
            print(f'[Image]: {image_file}')
        # 将所有图片编码拼接成一个张量
        image_encoders = torch.cat(image_encoders, dim=0).unsqueeze(0)

        # 构建对话历史和生成的新提示
        messages = [{"role": "user", "content": prompt}]
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[
                     -(max_seq_len - 1):]
        x = tokenizer(new_prompt).data['input_ids']
        x = torch.tensor(x, dtype=torch.long, device=device)[None, ...]

        # 使用模型进行生成
        with torch.no_grad():
            res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_seq_len, temperature=temperature,
                                   top_k=top_k, stream=True, image_encoders=image_encoders)
            print('[A]: ', end='')
            history_idx = 0
            for y in res_y:
                answer = tokenizer.decode(y[0].tolist())
                if answer and answer[-1] == '�':
                    y = next(res_y)
                    continue
                print(answer[history_idx:], end='', flush=True)
                history_idx = len(answer)
            print('\n')
    else:
        # 单图推理：对每张图像单独推理
        for image_file in image_files:
            # 获取图像文件的完整路径
            image_path = os.path.join(image_dir, image_file)
            # 打开图像文件并将其转换为 RGB 格式
            image = Image.open(image_path).convert('RGB')
            # 对图像进行处理，准备输入到模型，并将其移到合适的设备（如 GPU）
            image_process = get_img_process(image, preprocess).to(vision_model.device)
            # 获取图像的特征向量（嵌入表示），并增加一个批次维度
            image_encoder = get_img_embedding(image_process, vision_model).unsqueeze(0)

            # 创建一个包含用户输入的消息字典
            messages = [{"role": "user", "content": prompt}]
            # 应用聊天模板，生成新的 prompt，并根据最大序列长度截断
            new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[
                         -(max_seq_len - 1):]
            # 使用 tokenizer 将新的 prompt 转换为 token ID
            x = tokenizer(new_prompt).data['input_ids']
            # 将 token ID 转换为张量，并指定数据类型和设备
            x = torch.tensor(x, dtype=torch.long, device=device)[None, ...]

            # 打印当前正在处理的图像文件名
            print(f'[Image]: {image_file}')
            # 在不计算梯度的情况下进行推理
            with torch.no_grad():
                # 使用模型进行生成，返回生成的结果
                res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_seq_len, temperature=temperature,
                                       top_k=top_k, stream=True, image_encoders=image_encoder)
                # 打印模型输出的每个生成的回答
                print('[A]: ', end='')
                history_idx = 0
                # 遍历生成的每个 token
                for y in res_y:
                    # 解码每个 token，生成相应的回答
                    answer = tokenizer.decode(y[0].tolist())
                    # 如果解码结果以无效字符（如乱码）结尾，则跳过当前生成的 token
                    if answer and answer[-1] == '�':
                        y = next(res_y)
                        continue
                    # 打印回答的剩余部分
                    print(answer[history_idx:], end='', flush=True)
                    # 更新历史索引，以便下次输出时从上次输出的位置继续
                    history_idx = len(answer)
                # 换行以结束本轮输出
                print('\n')
```