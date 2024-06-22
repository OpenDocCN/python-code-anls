# `.\transformers\models\blip\convert_blip_original_pytorch_to_hf.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，禁止未经许可使用此文件
# 可以在以下链接获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
import argparse  # 用于解析命令行参数
import re  # 用于正则表达式操作
import requests  # 用于发送 HTTP 请求
import torch  # PyTorch 深度学习库

# 从 BLIP 项目中导入模型
from models.blip import blip_decoder
from models.blip_itm import blip_itm
from models.blip_vqa import blip_vqa
from PIL import Image  # Python Imaging Library，用于图像处理
from torchvision import transforms  # PyTorch 图像转换模块
from torchvision.transforms.functional import InterpolationMode  # 图像插值模式

# 从 Transformers 库中导入相关模块
from transformers import (
    BertTokenizer,
    BlipConfig,
    BlipForConditionalGeneration,
    BlipForImageTextRetrieval,
    BlipForQuestionAnswering,
)

# 加载演示图像
def load_demo_image(image_size, device):
    # 图像 URL
    img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    # 从 URL 获取图像并转换为 RGB 格式
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    # 图像转换操作
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    # 对图像进行转换并添加维度，然后移动到指定设备
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

# 重命名键名
def rename_key(key):
    if "visual_encoder" in key:
        key = re.sub("visual_encoder*", "vision_model.encoder", key)
    if "blocks" in key:
        key = re.sub(r"blocks", "layers", key)
    if "attn" in key:
        key = re.sub(r"attn", "self_attn", key)
    if "norm1" in key:
        key = re.sub(r"norm1", "layer_norm1", key)
    if "norm2" in key:
        key = re.sub(r"norm2", "layer_norm2", key)
    if "encoder.norm" in key:
        key = re.sub(r"encoder.norm", "post_layernorm", key)
    if "encoder.patch_embed.proj" in key:
        key = re.sub(r"encoder.patch_embed.proj", "embeddings.patch_embedding", key)

    if "encoder.pos_embed" in key:
        key = re.sub(r"encoder.pos_embed", "embeddings.position_embedding", key)
    if "encoder.cls_token" in key:
        key = re.sub(r"encoder.cls_token", "embeddings.class_embedding", key)

    if "self_attn" in key:
        key = re.sub(r"self_attn.proj", "self_attn.projection", key)

    return key

# 转换 BLIP 检查点
@torch.no_grad()
def convert_blip_checkpoint(pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = BlipConfig.from_pretrained(config_path)
    # 如果条件不成立，创建一个具有默认参数的 BlipConfig 对象
    else:
        config = BlipConfig(projection_dim=512, text_config={}, vision_config={})

    # 创建一个用于生成条件文本的 BlipForConditionalGeneration 模型，并设置为评估模式
    hf_model = BlipForConditionalGeneration(config).eval()

    # 指定预训练模型的 URL
    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"

    # 使用指定的预训练模型 URL、图像大小和 ViT 类型创建模型
    pt_model = blip_decoder(pretrained=model_url, image_size=384, vit="base")
    pt_model = pt_model.eval()

    # 复制修改后的模型参数字典
    modified_state_dict = pt_model.state_dict()
    # 遍历并重命名模型参数字典的键
    for key in modified_state_dict.copy():
        value = modified_state_dict.pop(key)
        renamed_key = rename_key(key)
        modified_state_dict[renamed_key] = value

    # 加载修改后的模型参数到 hf_model
    hf_model.load_state_dict(modified_state_dict)

    # 加载演示图像
    image_size = 384
    image = load_demo_image(image_size=image_size, device="cpu")
    # 创建 BertTokenizer 对象
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # 对输入文本进行编码
    input_ids = tokenizer(["a picture of"]).input_ids

    # 生成文本输出
    out = hf_model.generate(image, input_ids)

    # 断言输出结果是否符合预期
    assert out[0].tolist() == [30522, 1037, 3861, 1997, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102]

    # 生成文本输出
    out = hf_model.generate(image)

    # 断言输出结果是否符合预期
    assert out[0].tolist() == [30522, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102]

    # 如果指定了 pytorch_dump_folder_path，则保存模型参数
    if pytorch_dump_folder_path is not None:
        hf_model.save_pretrained(pytorch_dump_folder_path)

    # 指定 VQA 模型的预训练模型 URL
    model_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
    )

    # 使用指定的预训练模型 URL、图像大小和 ViT 类型创建 VQA 模型
    vqa_model = blip_vqa(pretrained=model_url, image_size=image_size, vit="base")
    vqa_model.eval()

    # 复制修改后的模型参数字典
    modified_state_dict = vqa_model.state_dict()
    # 遍历并重命名模型参数字典的键
    for key in modified_state_dict.copy():
        value = modified_state_dict.pop(key)
        renamed_key = rename_key(key)
        modified_state_dict[renamed_key] = value

    # 创建 BlipForQuestionAnswering 模型
    hf_vqa_model = BlipForQuestionAnswering(config)

    # 加载修改后的模型参数到 hf_vqa_model
    hf_vqa_model.load_state_dict(modified_state_dict)

    # 创建问题文本
    question = ["How many dogs are in this image?"]
    # 对问题文本进行编码
    question_input_ids = tokenizer(question, return_tensors="pt").input_ids

    # 生成问题回答
    answer = hf_vqa_model.generate(question_input_ids, image)
    # 打印解码后的回答
    print(tokenizer.decode(answer[0]))

    # 断言解码后的回答是否符合预期
    assert tokenizer.decode(answer[0]) == "[UNK] 1 [SEP]"
    
    # 如果指定了 pytorch_dump_folder_path，则保存模型参数
    if pytorch_dump_folder_path is not None:
        hf_vqa_model.save_pretrained(pytorch_dump_folder_path + "_vqa")

    # 指定检索模型的预训练模型 URL
    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth"

    # 使用指定的预训练模型 URL、图像大小和 ViT 类型创建检索模型
    itm_model = blip_itm(pretrained=model_url, image_size=image_size, vit="base")
    itm_model.eval()

    # 复制修改后的模型参数字典
    modified_state_dict = itm_model.state_dict()
    # 遍历并重命名模型参数字典的键
    for key in modified_state_dict.copy():
        value = modified_state_dict.pop(key)
        renamed_key = rename_key(key)
        modified_state_dict[renamed_key] = value

    # 创建 BlipForImageTextRetrieval 模型
    hf_itm_model = BlipForImageTextRetrieval(config)

    # 创建问题文本
    question = ["A picture of a woman with a dog sitting in a beach"]
    # 使用分词器对问题进行编码，返回 PyTorch 张量，进行了最大长度填充和截断处理，最大长度为 35
    question_input_ids = tokenizer(
        question,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=35,
    ).input_ids
    
    # 载入修改后的状态字典到 Hugging Face 模型中
    hf_itm_model.load_state_dict(modified_state_dict)
    # 将模型设置为评估模式
    hf_itm_model.eval()
    
    # 使用 Hugging Face 模型对输入问题和图像进行推断，使用 ITM（Image Text Matching）头部
    out_itm = hf_itm_model(question_input_ids, image, use_itm_head=True)
    # 使用 Hugging Face 模型对输入问题和图像进行推断，不使用 ITM 头部
    out = hf_itm_model(question_input_ids, image, use_itm_head=False)
    
    # 断言模型输出的第一个元素是否等于给定值（用于检验模型是否正确）
    assert out[0].item() == 0.2110687494277954
    # 断言使用 softmax 函数处理后的 ITM 输出中第二列的元素是否等于给定值（用于检验模型是否正确）
    assert torch.nn.functional.softmax(out_itm[0], dim=1)[:, 1].item() == 0.45698845386505127
    
    # 如果 PyTorch 模型保存文件夹路径不为空，则保存 ITM 头部模型
    if pytorch_dump_folder_path is not None:
        hf_itm_model.save_pretrained(pytorch_dump_folder_path + "_itm")
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个参数，用于指定输出的 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加一个参数，用于指定要转换的模型的配置文件路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数 convert_blip_checkpoint，将参数传递给该函数
    convert_blip_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
```