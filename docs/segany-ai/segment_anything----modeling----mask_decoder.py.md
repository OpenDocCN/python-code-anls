# `.\segment-anything\segment_anything\modeling\mask_decoder.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.nn 模块中导入 functional 模块并重命名为 F
from torch.nn import functional as F

# 从 typing 模块中导入 List, Tuple, Type 类型
from typing import List, Tuple, Type

# 从当前目录下的 common 模块中导入 LayerNorm2d 类
from .common import LayerNorm2d

# 定义 MaskDecoder 类，继承自 nn.Module 类
class MaskDecoder(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        预测给定图像和提示嵌入的蒙版，使用transformer架构。

        参数:
          transformer_dim (int): transformer的通道维度
          transformer (nn.Module): 用于预测蒙版的transformer
          num_multimask_outputs (int): 在区分蒙版时要预测的蒙版数量
          activation (nn.Module): 在放大蒙版时要使用的激活类型
          iou_head_depth (int): 用于预测蒙版质量的MLP的深度
          iou_head_hidden_dim (int): 用于预测蒙版质量的MLP的隐藏维度
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
    # 定义一个方法，用于根据图像和提示嵌入预测掩码
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        # 调用 predict_masks 方法，获取预测的掩码和 IOU 预测
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # 根据 multimask_output 决定输出单个掩码还是多个掩码
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # 准备输出，返回预测的掩码和 IOU 预测
        return masks, iou_pred

    # 定义一个方法，用于预测掩码
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测掩模。详细信息请参阅 'forward'。"""
        # 拼接输出的标记
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 在批处理方向上扩展每个图像数据以适应每个掩模
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # 运行变换器
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # 放大掩模嵌入并使用掩模标记预测掩模
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # 生成掩模质量预测
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
# 从给定链接中轻微调整而来的代码，定义了一个多层感知机（MLP）神经网络模型
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # 输入维度
        hidden_dim: int,  # 隐藏层维度
        output_dim: int,  # 输出维度
        num_layers: int,  # 神经网络层数
        sigmoid_output: bool = False,  # 是否对输出进行 sigmoid 激活
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)  # 隐藏层维度列表
        # 创建多个线性层组成的神经网络，输入维度为 input_dim，输出维度为 output_dim
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        # 遍历神经网络的每一层
        for i, layer in enumerate(self.layers):
            # 对隐藏层使用 ReLU 激活函数，最后一层不使用激活函数
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        # 如果需要对输出进行 sigmoid 激活，则应用 sigmoid 函数
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
```