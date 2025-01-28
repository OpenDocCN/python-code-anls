# `.\minimind-v\model\vision_utils.py`

```
# 导入必要的库
import warnings
# 从 transformers 库中导入 CLIP 处理器和模型
from transformers import CLIPProcessor, CLIPModel, SiglipProcessor, SiglipModel
# 导入图像处理相关库
from PIL import Image
# 导入网络请求库
import requests
# 导入 PyTorch 库
import torch
# 导入 transformers 库本身
import transformers

# 忽略所有警告信息
warnings.filterwarnings('ignore')


# 定义获取视觉模型的函数，根据编码器类型加载不同的模型和处理器
def get_vision_model(encoder_type):
    # 判断是使用 CLIP 模型还是 Siglip 模型
    if encoder_type == "clip":
        # 设置 CLIP 模型路径
        model_path = "./model/clip_model/clip-vit-base-patch32"
        # 加载预训练的 CLIP 模型
        model = CLIPModel.from_pretrained(model_path)
        # 加载对应的 CLIP 处理器
        processor = CLIPProcessor.from_pretrained(model_path)
    else:
        # 设置 Siglip 模型路径
        model_path = "./model/siglip_model/siglip-vit-base-patch16"
        # 加载预训练的 Siglip 模型
        model = SiglipModel.from_pretrained(model_path)
        # 加载对应的 Siglip 处理器
        processor = SiglipProcessor.from_pretrained(model_path)
    # 返回加载的模型和处理器
    return (model, processor)


# 定义图像处理函数，将图像调整为合适的格式
def get_img_process(image, processor):
    # 调整图像大小为 224x224
    image = image.resize((224, 224))
    # 如果图像有透明通道（RGBA 或 LA），则将其转换为 RGB 模式
    if image.mode in ['RGBA', 'LA']:  # 处理有透明通道的图像
        image = image.convert('RGB')
    # 使用处理器处理图像并返回张量
    # inputs = processor(images=image, return_tensors="pt", clean_up_tokenization_spaces=False)
    inputs = processor(images=image, return_tensors="pt")
    # 返回处理后的图像输入
    return inputs


# 定义获取图像嵌入的函数
def get_img_embedding(batch_encoding, vision_model):
    # 初始化一个空列表来存储图像特征
    embeddings = []

    # 定义 hook 函数，用于提取目标层的输出
    def hook_fn(module, input, output):
        # 将目标层的特征添加到 embeddings 列表中
        embeddings.append(output.last_hidden_state)

    # 判断 batch_encoding 是否是 BatchEncoding 类型或 BatchFeature 类型
    if (isinstance(batch_encoding, transformers.tokenization_utils_base.BatchEncoding)
            or isinstance(batch_encoding, transformers.feature_extraction_utils.BatchFeature)):
        # 从 batch_encoding 中提取图像张量（像素值）
        image_tensor = batch_encoding['pixel_values']
    else:
        # 如果 batch_encoding 不是上述类型，则直接使用 batch_encoding 作为图像张量
        image_tensor = batch_encoding  # torch.Size([32, 4, 3, 224, 224])

    # 如果图像张量是 4 维，则添加一个额外的维度来表示批次
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度

    # 获取批次大小
    batch_size = image_tensor.size(0)

    # 禁用梯度计算，减少内存消耗
    with torch.no_grad():
        # 注册 hook 到模型的目标层（例如 vision_model 的编码器层）
        layer = vision_model.vision_model.encoder
        hook = layer.register_forward_hook(hook_fn)

        # 对每个图像进行特征提取
        for i in range(batch_size):
            # 获取当前批次中的单个图像
            single_image = image_tensor[i]  # 添加批次维度
            # 使用模型提取图像特征
            _ = vision_model.get_image_features(single_image)
        # 移除 hook
        hook.remove()

    # 将所有提取的特征向量拼接成一个张量
    all_embeddings = torch.stack(embeddings, dim=0).squeeze()
    # 返回拼接后的特征张量
    return all_embeddings
```