# SO-VITS-SVC源码解析 15

# `vencoder/dphubert/utils/__init__.py`

很抱歉，我需要看到代码才能为您解释其作用。请提供相关代码，我会尽力帮助您。


```py

```

# `vencoder/hubert/hubert_model.py`

This is a PyTorch implementation of a masked language modeling model. It consists of a feature extractor, a feature projection function, a positional encoding, a dropout layer, a self-attention mechanism, a logit layer, and a linear layer. The model takes as input a fixed-length sequence of text represented as a tensor of dimension (batch\_size, sequence\_length, feature\_dim). It outputs a tuple of two tensors, the logits and the mask, for each input sequence.


```py
import copy
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as t_func
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


class Hubert(nn.Module):
    def __init__(self, num_label_embeddings: int = 100, mask: bool = True):
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection()
        self.positional_embedding = PositionalConvEmbedding()
        self.norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(
            nn.TransformerEncoderLayer(
                768, 12, 3072, activation="gelu", batch_first=True
            ),
            12,
        )
        self.proj = nn.Linear(768, 256)

        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)

    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask

    def encode(
            self, x: torch.Tensor, layer: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_extractor(x)
        x = self.feature_projection(x.transpose(1, 2))
        x, mask = self.mask(x)
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x, output_layer=layer)
        return x, mask

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.cosine_similarity(
            x.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        return logits / 0.1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.encode(x)
        x = self.proj(x)
        logits = self.logits(x)
        return logits, mask


```

It looks like the code you provided is a simple implementation of a convolutional neural network (CNN) for image classification. The `FeatureExtractor` class appears to be responsible for extracting features from the input image, while the `nn.Module` interface suggests that it should be used as a part of a larger network model.

The `FeatureExtractor` class has a forward method that takes an input tensor `x`, and returns a tensor of the same size. This forward method performs a series of convolutional neural network (CNN) modules, which are行程记 (x, y) where x is the input tensor and y is the output tensor.

The `nn.Module` interface provides a number of methods and attributes that can be used to customize the behavior of the CNN model. For example, the `conv0`, `conv1`, `conv2`, `conv3`, `conv4`, and `conv5` methods define the convolutional layers, while the `conv6` method defines the pooling layer. Additionally, the `norm0` and `norm1` attributes can be used to normalize the input and output tensors, respectively.

Overall, it looks like this implementation is just one part of a larger image classification network, and may be used to extract features from the input images, rather than perform other tasks such as predicting the input images.


```py
class HubertSoft(Hubert):
    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def units(self, wav: torch.Tensor) -> torch.Tensor:
        wav = t_func.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav)
        return self.proj(x)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 512, 10, 5, bias=False)
        self.norm0 = nn.GroupNorm(512, 512)
        self.conv1 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv2 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv3 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv4 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv5 = nn.Conv1d(512, 512, 2, 2, bias=False)
        self.conv6 = nn.Conv1d(512, 512, 2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = t_func.gelu(self.norm0(self.conv0(x)))
        x = t_func.gelu(self.conv1(x))
        x = t_func.gelu(self.conv2(x))
        x = t_func.gelu(self.conv3(x))
        x = t_func.gelu(self.conv4(x))
        x = t_func.gelu(self.conv5(x))
        x = t_func.gelu(self.conv6(x))
        return x


```

这段代码定义了一个名为 FeatureProjection 的类，继承自 PyTorch 的 nn.Module 类。这个类用于实现一个前馈神经网络，以对输入数据进行特征提取和降维。

FeatureProjection 类包含了一个 init() 方法，用于设置网络的结构，包括一个 normalization 层（通过从输入数据中计算梯度并加权平均来达到对数据的归一化）、一个 linear 层（具有 512 个节点，输出是输入数据的线性变换）和一个 dropout 层（具有 0.1 的 dropout 概率）。

接下来，该类实现了一个 forward() 方法，该方法接受一个输入张量，首先通过 apply 函数将其转换为第一维，然后将输入张量与 normalization 层中的计算结果、线性层中的计算结果以及 dropout 层中的计算结果相结合，并返回结果。

另外，该类还实现了一个名为 PositionalConvEmbedding 的子类，该子类实现了一个位置编码卷积嵌入层。该子类的实现与该类中常规卷积层的实现类似，但是对每个输入张量执行一次额外的 normalization 操作，以便将输入数据与通道维度对齐。


```py
class FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalConvEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            768,
            768,
            kernel_size=128,
            padding=128 // 2,
            groups=16,
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        x = t_func.gelu(x[:, :, :-1])
        return x.transpose(1, 2)


```

这段代码定义了一个名为 "TransformerEncoder" 的类，继承自 PyTorch 的 nn.Module 类。这个类的实现了一个 Transformer Encoder，可以对输入序列中的每个位置进行编码。

在类的初始化函数中，首先调用父类的初始化函数，然后创建一个包含 num_layers 层的列表，每个层都是对输入层和输出层的复本。这样就创建了一个包含多个自定义的层，可以按照自己的方式对输入数据进行处理。

在 forward 函数中，首先将输入数据 src 传递给第一个层，然后根据设置的 mask 变量对输入数据进行遮罩，如果沒有设置该参数，则使用默认的 ones 作为遮罩。接着将结果传递给第二个层，并将第二个层的输出存储在一个变量 output 中。然后重复这个过程，直到达到设置好的 output_layer。

通过 layer 函数，实现了对输入层的保护和子层之间的数据可传性。


```py
class TransformerEncoder(nn.Module):
    def __init__(
            self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
            self,
            src: torch.Tensor,
            mask: torch.Tensor = None,
            src_key_padding_mask: torch.Tensor = None,
            output_layer: Optional[int] = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers[:output_layer]:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        return output


```

This code appears to be a masked data distribution that generates spans of a fixed length as input and returns a binary mask indicating which spans are masked. The input spans are expected to be ordered and have a length between 1 and the specified `sequence_length`, while the masked spans are expected to have the same length as the input spans and be included in the same order.

The code first sets the `mask_prob` parameter to a random value between 0 and 1, and then sets `sequence_length` to a random value within the specified range. It then computes the number of masked spans in the batch and sets `num_masked_spans` to the calculated value.

The code then creates a binary mask with a length of `sequence_length` and a probability of `mask_prob`. The mask is created by first creating a uniform distribution that samples from a range of `mask_prob` values, and then randomly selecting `num_masked_spans` of these values.

Finally, the code creates a scatter index that indicates which indices belong to the masked spans, and scatters the masked indices along with the corresponding mask values.

Note that if `sequence_length` is too short or `mask_prob` is too high, the output of the code may be incorrect or contain errors.


```py
def _compute_mask(
        shape: Tuple[int, int],
        mask_prob: float,
        mask_length: int,
        device: torch.device,
        min_masks: int = 0,
) -> torch.Tensor:
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=device
    )

    # get random indices to mask
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    mask_idxs = mask_indices + offsets

    # scatter indices to mask
    mask = mask.scatter(1, mask_idxs, True)

    return mask


```

这段代码定义了一个名为 `hubert_soft` 的函数，它接受一个字符串参数 `path`，表示一个预训练的模型。这个函数返回一个名为 `HubertSoft` 的类实例。

具体来说，函数内部首先创建一个名为 `HubertSoft` 的类实例，然后从传入的 `path` 参数中加载预训练模型。接着，检查点（checkpoint）中是否包含 `module`  key，如果是，则只加载 `module` 的部分。然后，加载预训练模型的 `state_dict`，并确保模型处于 `eval` 状态。最后，返回创建的 `HubertSoft` 实例。


```py
def hubert_soft(
        path: str,
) -> HubertSoft:
    r"""HuBERT-Soft from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        path (str): path of a pretrained model
    """
    hubert = HubertSoft()
    checkpoint = torch.load(path)
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")
    hubert.load_state_dict(checkpoint)
    hubert.eval()
    return hubert

```

# `vencoder/hubert/hubert_model_onnx.py`

This is a PyTorch implementation of a masked language modeling model called "MaskedSpec2". This model is a pre-trained model参加了不大于10%的训练，用以提升他在需要标记数据语料的大规模语料上的表现。

It implements the following functionalities:

1. `feature_extractor(x)`: This function generates a fixed-length context representation of input `x`.
2. `feature_projection(x)`: This function applies a projection to the context representation, which is a 256-dimensional tensor.
3. `mask(x)`: This function generates a binary mask for each label in the input vocabulary.
4. `encoder(x, output_layer=None)`: This function applies the masked language modeling encoder to the input tensor `x`.
5. `norm(x)`: This function normalizes the input tensor `x`.
6. `dropout(x)`: This function applies the given percentage of dropout to the input tensor `x`.
7. `logits(x)`: This function generates the output logits for the masked language modeling model.


```py
import copy
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as t_func
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


class Hubert(nn.Module):
    def __init__(self, num_label_embeddings: int = 100, mask: bool = True):
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection()
        self.positional_embedding = PositionalConvEmbedding()
        self.norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(
            nn.TransformerEncoderLayer(
                768, 12, 3072, activation="gelu", batch_first=True
            ),
            12,
        )
        self.proj = nn.Linear(768, 256)

        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)

    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask

    def encode(
            self, x: torch.Tensor, layer: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_extractor(x)
        x = self.feature_projection(x.transpose(1, 2))
        x, mask = self.mask(x)
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x, output_layer=layer)
        return x, mask

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.cosine_similarity(
            x.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        return logits / 0.1


```

This is a PyTorch implementation of a pre-trained convolutional neural network (CNN) model for image classification tasks. The model is based on the Inception V3 architecture and has been fine-tuned on the Kinetics部门的iliqivi courts dataset.

The `FeatureExtractor` class is a custom implementation for extracting features from input images. It contains a sequence of 6 convolutional layers with a maximum pooling layer. The convolutional layers use the `t_func` library for translation-invariant convolution.

The `__init__` method initializes the CNN by setting the first convolutional layer's parameters to the initial values. The `forward` method defines the forward pass of the network.

The `forward` method applies a series of convolutional, activation, and pooling operations to the input image. The output image is returned from the last convolutional layer.


```py
class HubertSoft(Hubert):
    def __init__(self):
        super().__init__()

    def units(self, wav: torch.Tensor) -> torch.Tensor:
        wav = t_func.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav)
        return self.proj(x)

    def forward(self, x):
        return self.units(x)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 512, 10, 5, bias=False)
        self.norm0 = nn.GroupNorm(512, 512)
        self.conv1 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv2 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv3 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv4 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv5 = nn.Conv1d(512, 512, 2, 2, bias=False)
        self.conv6 = nn.Conv1d(512, 512, 2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = t_func.gelu(self.norm0(self.conv0(x)))
        x = t_func.gelu(self.conv1(x))
        x = t_func.gelu(self.conv2(x))
        x = t_func.gelu(self.conv3(x))
        x = t_func.gelu(self.conv4(x))
        x = t_func.gelu(self.conv5(x))
        x = t_func.gelu(self.conv6(x))
        return x


```

这段代码定义了一个名为 FeatureProjection 的类，继承自 PyTorch 的 nn.Module 类。这个类用于实现一个前馈神经网络，用于在图像分类任务中进行特征提取和数据降维操作。

FeatureProjection 类包含三个成员变量：norm、projection 和 dropout，分别代表层 norm、目标特征图和 dropout。norm 是一个 LayerNorm 层，用于实现对输入数据的归一化处理；projection 是一个线性变换层，用于从 512 个特征图降维到 768 个；dropout 是一个 dropout 层，用于防止过拟合。

在 forward 方法中，首先调用父类的 forward 方法，然后执行对输入数据的归一化处理，接着是目标特征图的生成，最后对结果进行 dropout。

另外一个定义的类名为 PositionalConvEmbedding，也是一个继承自 nn.Module 的类。这个类包含一个名为 forward 的方法，用于实现一个位置卷积嵌入层。这个层的主要作用是在图像上进行特征提取，通过在图像的不同位置应用不同大小的卷积来获取不同的特征。


```py
class FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalConvEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            768,
            768,
            kernel_size=128,
            padding=128 // 2,
            groups=16,
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        x = t_func.gelu(x[:, :, :-1])
        return x.transpose(1, 2)


```

这段代码定义了一个名为 `TransformerEncoder` 的类，继承自 PyTorch 的 `nn.Module` 类。这个类的实现了一个 Transformer Encoder，可以对输入的 src 数据进行编码处理，将编码后的结果存储在输出变量 output 中。

在类的初始化函数 `__init__` 中，定义了两个参数：`encoder_layer` 和 `num_layers`。这两个参数分别表示 Transformer Encoder 的层数和每个 encoder layer 中的子模块数量。通过这两个参数，可以创建一个由多个子模块组成的 Transformer Encoder。

在 `forward` 函数中，通过嵌套循环对每个子模块进行前向传播，获取编码后的结果，并将结果存储在 output 变量中。在每次前向传播的过程中，使用了多个源 key 和对应的 mask，以及一个占位输出 layer 的值，来对输入的 src 数据进行编码处理。


```py
class TransformerEncoder(nn.Module):
    def __init__(
            self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
            self,
            src: torch.Tensor,
            mask: torch.Tensor = None,
            src_key_padding_mask: torch.Tensor = None,
            output_layer: Optional[int] = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers[:output_layer]:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        return output


```

This code appears to be a masked data preprocessing function that applies SpecAugment to images. It takes a batch of images and a batch size, and applies SpecAugment to each image in the batch.

SpecAugment is a data augmentation technique that generates random著


```py
def _compute_mask(
        shape: Tuple[int, int],
        mask_prob: float,
        mask_length: int,
        device: torch.device,
        min_masks: int = 0,
) -> torch.Tensor:
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=device
    )

    # get random indices to mask
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    mask_idxs = mask_indices + offsets

    # scatter indices to mask
    mask = mask.scatter(1, mask_idxs, True)

    return mask


```

这段代码定义了一个名为 `hubert_soft` 的函数，它接受一个字符串参数 `path`，表示一个预训练的模型。这个函数返回一个名为 `HubertSoft` 的类实例。

具体来说，函数首先创建一个名为 `HubertSoft` 的类实例，然后从传入的模型路径中加载预训练模型。如果模型路径中存在名为 "module." 的模块，则不会从模型的 `__init__` 函数中加载该模块。

接着，函数将加载的预训练模型加载到其类实例中，并将其设为 `eval` 状态。这样，函数的函数体部分就可以直接使用加载的预训练模型来进行语音转换等任务了。

最后，函数返回生成的 `HubertSoft` 类实例。


```py
def hubert_soft(
        path: str,
) -> HubertSoft:
    r"""HuBERT-Soft from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        path (str): path of a pretrained model
    """
    hubert = HubertSoft()
    checkpoint = torch.load(path)
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")
    hubert.load_state_dict(checkpoint)
    hubert.eval()
    return hubert

```

# `vencoder/hubert/__init__.py`

我需要您提供具体的代码内容，才能帮助您解释代码的作用。


```py

```

# `vencoder/wavlm/modules.py`

这段代码是一个名为"WavLM"的PyTorch实现，用于进行大规模的学生自监督预训练，以进行完整的语音处理。以下是该代码的一些要害功能：

1. 将训练数据分为训练集和验证集。
2. 加载预训练的预训练模型，该模型在训练集上进行微调，然后在验证集上进行评估。
3. 将特征的名称转换为小写。
4. 创建一个字典，其中包含一些数据增强的技巧，例如：

  ```py
  # 转义
  l = torch.linspace(0, 5000, 1000)[0]
  r = torch.linspace(0, 5000, 500)[0]

  a = torch.tensor([l, r], dtype=torch.float32)
  b = torch.tensor([0, 0], dtype=torch.float32)
  ```

  这些技巧可能会对训练数据产生影响，并增强模型的性能。

5. 在训练时，使用掩码对输入数据进行二进制编码。
6. 使用PyTorch中的`torch.autograd`机制，确保在导出模型的权重时使用最新的计算狮。
7. 最后，训练指定的模型。


```py
# --------------------------------------------------------
# WavLM: Large-Scale Self-Supervised  Pre-training  for Full Stack Speech Processing (https://arxiv.org/abs/2110.13900.pdf)
# Github source: https://github.com/microsoft/unilm/tree/master/wavlm
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import math
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
```

这段代码定义了一个名为 `TransposeLast` 的类，继承自 `nn.Module`。这个类的成员函数 `forward()` 中，传入一个输入张量 `x`，并返回一个经过 transpose 操作后的张量。

具体来说，这段代码实现了一个 `TransposeLast` 模型，这个模型的主要作用是改变输入张量 `x` 的顺序，即将张量的最后一层从后往前进行翻转。这个翻转可以通过 `self.deconstruct_idx` 参数来指定，如果 `self.deconstruct_idx` 为 `None`，则表示不进行翻转操作。

在 `forward()` 函数中，首先判断 `self.deconstruct_idx` 是否为 `None`，如果是，则直接返回输入张量 `x`，不做任何操作。否则，使用 `x[:, self.deconstruct_idx]` 将张量的最后一层从后往前翻转，并返回翻转后的张量。


```py
from torch import Tensor, nn
from torch.nn import Parameter


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


```

这段代码定义了一个名为 Fp32LayerNorm 的类，继承自 PyTorch 中的nn.LayerNorm类。这个类的目标是实现一个在输入数据上执行对256个参数（例如，对一个含有32个通道的张量执行归一化操作）的 Layer。

在 Fp32LayerNorm 的__init__方法中，调用父类的初始化方法并传入参数。然后，在 forward 方法中，执行对输入数据的归一化操作。具体来说，它会执行以下操作：

1. 使用 F.layer_norm 函数实现一个基于输入数据的 Layer。这个 Layer 包括一个前向传播部分和一个后向传播部分。前向传播部分将输入数据与注意力权重（如果有的话）相乘，然后将这些乘积归一化。后向传播部分执行一个数值矩阵转置操作。
2. 设置 Layer 的超参数（options）。这些参数允许你控制层的行为。在这里，我们使用了 `input.float()` 和 `self.normalized_shape` 作为输入数据和归一化后的形状，`self.weight.float()` 和 `self.bias.float()` 作为层参数。我们还使用了 `self.eps` 参数，它是用于在输出数据中添加随机噪声以防止“爆炸”的常数。

由于 Fp32LayerNorm 类继承自层 norms，因此它包含一个名为 `batch_norm` 的方法。这个方法在 forward 方法中作为 `output.float()` 的后继。通过执行这个后继，你会根据输入数据的形状将注意力权重添加到输出数据中。


```py
class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


```

这段代码定义了一个名为 Fp32GroupNorm 的类，继承自 PyTorch 中的nn.GroupNorm类。这个类的目标是实现一个在输入数据上执行对输入数据的归一化的操作。

具体来说，这段代码的作用是创建一个 Fp32GroupNorm 实例，并在其 forward 方法中执行输入数据的归一化操作。这个归一化操作在计算输入数据的每个元素时，会对输入数据执行以下操作：

1. 对输入数据中的每个元素执行浮点数除法运算，将结果存储在输入数据的每个位置。
2. 如果定义了权重参数 self.weight，则使用该权重对每个归一化后的元素进行乘法操作，并将结果存储在输入数据的每个位置。
3. 如果定义了 bias 参数 self.bias，则将该 bias 值对输入数据中的每个元素执行归一化操作，并将结果存储在输入数据的每个位置。
4. 对输入数据中的每个元素执行浮点数取整操作，将结果存储在输入数据的每个位置。
5. 如果定义了 eps 参数，则对输入数据中的每个元素执行除以 eps 的操作，并将结果存储在输入数据的每个位置。

最后，F.group_norm函数是 PyTorch中的一个名为 group_norm 的函数，它是归一化操作的实际实现。


```py
class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


```



这段代码定义了一个名为 "GradMultiply" 的类，用于在 PyTorch 的 autograd 计算图上进行矩阵乘法操作。该类继承自 PyTorch 的 "Function" 类，因此可以像普通函数一样使用，但具有了更多的功能，例如可以带有参数 `ctx` 和 `grad`。"

具体来说，该类的 "forward" 方法实现了一个将输入张量 `x` 和参数 `scale` 乘以 `scale` 并返回的结果张量 `res`。该方法的实现中使用了 PyTorch 的 autograd 特性，可以自动计算梯度和反向梯度。"backward" 方法则返回了梯度和梯度的反向梯度。"

该类还定义了一个名为 "SamePad" 的类，用于实现一个具有相同尺寸的 "padding" 层。该类中的 "__init__" 方法会在创建对象时对传入的 "kernel_size" 参数进行检查，如果该参数不是奇数，则会将该奇数减去 1，否则则不变。该类中的 "forward" 方法会在输入张量上执行一次矩阵乘法操作，并将结果保留相同的大小。


```py
class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


```

This is a custom implementation of the GLU (Generalized LU) linear layer. The layer has a single hidden layer with a ReLU activation function. The input dimension is 16, and the output dimension is 32.

The MultiHeadedAttention class is a custom implementation of the multi-head self-attention mechanism. It can be used to process input data of various shapes, such as images, text, etc.


```py
class Swish(nn.Module):
    """Swish function
    """

    def __init__(self):
        """Construct an MultiHeadedAttention object."""
        super(Swish, self).__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.act(x)


class GLU_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, glu_type="sigmoid", bias_in_glu=True):
        super(GLU_Linear, self).__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim

        if glu_type == "sigmoid":
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = Swish()
        elif glu_type == "relu":
            self.glu_act = torch.nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = torch.nn.GELU()

        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x):
        # to be consistent with GLU_Linear, we assume the input always has the #channel (#dim) in the last dimension of the tensor, so need to switch the dimension first for 1D-Conv case
        x = self.linear(x)

        if self.glu_type == "bilinear":
            x = (x[:, :, 0:self.output_dim] * x[:, :, self.output_dim:self.output_dim * 2])
        else:
            x = (x[:, :, 0:self.output_dim] * self.glu_act(x[:, :, self.output_dim:self.output_dim * 2]))

        return x


```



这段代码定义了一个名为 `gelu_accurate` 的函数，它接受一个输入张量 `x`，并返回一个张量。函数的作用是计算输入张量 `x` 和一个称为 ` gelu_accurate` 的函数的值的乘积，其中 ` gelu_accurate` 函数是一个自定义的激活函数，它的实现基于深度学习中的 ` gelu` 函数，但是通过 `math.sqrt(2 / math.pi)` 来替换了 ` gelu` 函数中的 `sqrt(2 / math.pi)`。

具体来说，函数 ` gelu_accurate` 的实现如下：

```pypython
def gelu_accurate(x):
   if not hasattr(gelu_accurate, "_a"):
       gelu_accurate._a = math.sqrt(2 / math.pi)
   return (
       0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
   )
```

首先，函数检查 ` gelu_accurate` 是否定义了 `_a` 属性，如果没有，就定义一个新的 `_a` 属性，其值为 `math.sqrt(2 / math.pi)`。接着，函数返回一个乘积，其中包含输入张量 `x` 和一个计算出来的值 `output`，该值的计算基于 ` gelu` 函数，但是使用 `math.sqrt(2 / math.pi)` 来代替 `sqrt(2 / math.pi)`。最后，函数将结果张量化，使其与输入张量 `x` 具有相同的形状。

函数 `get_activation_fn` 接受一个激活函数名称作为参数，并返回相应的激活函数。它可以通过深度学习中的 `torch.nn.functional.gelu` 函数来实现，但是为了使用 ` gelu_accurate` 自定义激活函数，该函数需要进行一些修改。具体来说，需要将 `gelu` 函数中的 `sqrt(2 / math.pi)` 替换为 `math.sqrt(2 / math.pi)`。


```py
def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        warnings.warn(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "glu":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


```

这段代码定义了一个名为“init_bert_params”的函数，用于初始化BERT模型的参数。它实现了三个条件：1）如果`normal_init_linear_weights`被设置为`True`，则线性层的权重将使用均值0和标准差0.02初始化，并将偏移量设置为指定的值；2）如果`normal_init_embed_weights`被设置为`True`，则嵌入层的权重将使用均值0初始化；3）如果`normal_init_proj_weights`被设置为`True`，则对于多头注意力的in_project_weight，它的权重将使用均值0初始化（此处的初始化值将被验证）。

在这段注释中，开发人员还解释了代码的功能，以及如何根据给定的参数进行默认初始化。如果需要设置其他参数，例如对齐线性、嵌入层或者多头注意力，可以对其进行修改以实现所需的初始化。


```py
def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


```

This appears to be a PyTorch implementation of a noise function that can be used for unsupervised classification. The `noise` function takes in a model and an integer `kernel_size` and returns a binary noise distribution that can be used to randomly replace the input data with corrupted data during training.

The `noise` function first gathers the weight and sizes of the input tensor and then splits the weight matrix into blocks. For a convolutional neural network (CNN), it randomly drops selected blocks to create corrupted data. For a fully connected neural network, it randomly drops the input data to create corrupted data. The drop rate is determined by the `p` parameter, which is a probability of the drop.

The `mask` is created by appending a binary mask to the input tensor, where the value in the masked data is `1` and the value outside the mask is `0`. This is done to ensure that the corrupted data affects the input data only.

Finally, the `mask` is scaled and applied to the input data, which is already passed through the model. This allows the model to use the corrupted data during training as intended.


```py
def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


```

This is a class that implements the揭开（peeling） operation for neural networks. It is based on the key-value （key-value） attention model.

The揭开 operation takes an input of key-value pairs (attention weights), a target length (tgt\_len), a source length (src\_len), and a batch size (batch\_size). It returns the updated attention weights for the target length.

This implementation assumes that you have the input data, the target data, and the model.

This class can be used to apply the揭开 operation to the weights of a neural network. You can use this class in the key-value-based attention module of a neural network to calculate the attention weights for a given target.


```py
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            q_noise=0.0,
            qn_block_size=8,
            has_relative_attention_bias=False,
            num_buckets=32,
            max_distance=128,
            gru_rel_pos=False,
            rescale_init=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        k_bias = True
        if rescale_init:
            k_bias = False

        k_embed_dim = embed_dim
        q_embed_dim = embed_dim

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, k_embed_dim, bias=k_bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, q_embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
                torch.log(relative_positions.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(
            relative_position,
            bidirectional=True
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
            position_bias: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, src_len)

        if (
                not is_tpu  # don't use PyTorch version on TPUs
                and incremental_state is None
                and not static_kv
                # A workaround for quantization to work. Otherwise JIT compilation
                # treats bias in linear module as method.
                and not torch.jit.is_scripting()
                and self.q_head_dim == self.head_dim
        ):
            assert key is not None and value is not None
            assert attn_mask is None

            attn_mask_rel_pos = None
            if position_bias is not None:
                attn_mask_rel_pos = position_bias
                if self.gru_rel_pos:
                    query_layer = query.transpose(0, 1)
                    new_x_shape = query_layer.size()[:-1] + (self.num_heads, -1)
                    query_layer = query_layer.view(*new_x_shape)
                    query_layer = query_layer.permute(0, 2, 1, 3)
                    _B, _H, _L, __ = query_layer.size()

                    gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(
                        _B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
                    gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                    attn_mask_rel_pos = gate_a_1.view(bsz * self.num_heads, -1, 1) * position_bias

                attn_mask_rel_pos = attn_mask_rel_pos.view((-1, tgt_len, tgt_len))
            k_proj_bias = self.k_proj.bias
            if k_proj_bias is None:
                k_proj_bias = torch.zeros_like(self.q_proj.bias)

            x, attn = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                # self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask_rel_pos,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )
            return x, attn, position_bias

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.q_head_dim)
                .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                    .view(-1, bsz * self.num_heads, self.k_head_dim)
                    .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v, position_bias

        if position_bias is not None:
            if self.gru_rel_pos == 1:
                query_layer = q.view(bsz, self.num_heads, tgt_len, self.q_head_dim)
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(
                    _B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                position_bias = gate_a_1.view(bsz * self.num_heads, -1, 1) * position_bias

            position_bias = position_bias.view(attn_weights.size())

            attn_weights = attn_weights + position_bias

        attn_weights_float = F.softmax(
            attn_weights, dim=-1
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights, position_bias

    @staticmethod
    def _append_prev_key_padding_mask(
            key_padding_mask: Optional[Tensor],
            prev_key_padding_mask: Optional[Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(
            self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

```