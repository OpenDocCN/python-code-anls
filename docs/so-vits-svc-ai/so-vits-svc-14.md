# SO-VITS-SVC源码解析 14

# `vencoder/dphubert/hardconcrete.py`

This is a class that wraps a binary mask with log-alpha value and softmax function, which is used for spatial sparsity regularization in deep neural networks. The L0 norm of the mask is calculated as the sum of the log-alpha values.

The forward method for this class samples a hard concrete mask by randomly initializing the log-alpha values, sampling the mask dynamically, and applying the mask to the binary mask. The sampled binary mask is stored in the variable `self.compiled_mask`, which is cached for subsequent inference. If the mask is not cached, it is compiled dynamically by sampling the log-alpha values and setting the mask to zeros for small values, using the number of expected zero elements in the input as an approximation of the L0 norm.

The extra\_repr method returns the mask as a string, where each element represents the mask, with zeros for non-existent values.


```py
"""Implementation of the hard Concrete distribution.

Originally from:
https://github.com/asappresearch/flop/blob/master/flop/hardconcrete.py

"""

import math

import torch
import torch.nn as nn


class HardConcrete(nn.Module):
    """A HarcConcrete module.
    Use this module to create a mask of size N, which you can
    then use to perform L0 regularization.

    To obtain a mask, simply run a forward pass through the module
    with no input data. The mask is sampled in training mode, and
    fixed during evaluation mode, e.g.:

    >>> module = HardConcrete(n_in=100)
    >>> mask = module()
    >>> norm = module.l0_norm()
    """

    def __init__(
        self,
        n_in: int,
        init_mean: float = 0.5,
        init_std: float = 0.01,
        temperature: float = 2/3,     # from CoFi
        stretch: float = 0.1,
        eps: float = 1e-6
    ) -> None:
        """Initialize the HardConcrete module.
        Parameters
        ----------
        n_in : int
            The number of hard concrete variables in this mask.
        init_mean : float, optional
            Initial drop rate for hard concrete parameter,
            by default 0.5.,
        init_std: float, optional
            Used to initialize the hard concrete parameters,
            by default 0.01.
        temperature : float, optional
            Temperature used to control the sharpness of the
            distribution, by default 1.0
        stretch : float, optional
            Stretch the sampled value from [0, 1] to the interval
            [-stretch, 1 + stretch], by default 0.1.
        """
        super().__init__()

        self.n_in = n_in
        self.limit_l = -stretch
        self.limit_r = 1.0 + stretch
        self.log_alpha = nn.Parameter(torch.zeros(n_in))
        self.beta = temperature
        self.init_mean = init_mean
        self.init_std = init_std
        self.bias = -self.beta * math.log(-self.limit_l / self.limit_r)

        self.eps = eps
        self.compiled_mask = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of this module."""
        self.compiled_mask = None
        mean = math.log(1 - self.init_mean) - math.log(self.init_mean)
        self.log_alpha.data.normal_(mean, self.init_std)

    def l0_norm(self) -> torch.Tensor:
        """Compute the expected L0 norm of this mask.
        Returns
        -------
        torch.Tensor
            The expected L0 norm.
        """
        return (self.log_alpha + self.bias).sigmoid().sum()

    def forward(self) -> torch.Tensor:
        """Sample a hard concrete mask.
        Returns
        -------
        torch.Tensor
            The sampled binary mask
        """
        if self.training:
            # Reset the compiled mask
            self.compiled_mask = None
            # Sample mask dynamically
            u = self.log_alpha.new(self.n_in).uniform_(self.eps, 1 - self.eps)
            s = torch.sigmoid((torch.log(u / (1 - u)) + self.log_alpha) / self.beta)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
            mask = s.clamp(min=0., max=1.)

        else:
            # Compile new mask if not cached
            if self.compiled_mask is None:
                # Get expected sparsity
                expected_num_zeros = self.n_in - self.l0_norm().item()
                num_zeros = round(expected_num_zeros)
                # Approximate expected value of each mask variable z;
                # We use an empirically validated magic number 0.8
                soft_mask = torch.sigmoid(self.log_alpha / self.beta * 0.8)
                # Prune small values to set to 0
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
                self.compiled_mask = soft_mask
            mask = self.compiled_mask

        return mask

    def extra_repr(self) -> str:
        return str(self.n_in)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())

```

# `vencoder/dphubert/model.py`

这段代码定义了一个名为 "Speech SSL models supporting pruning" 的类别，其中包括一些支持剪枝的语音模型。具体来说，这些模型都是基于 PyTorch 框架实现的，其中包含了两个子类：PyTorch Audio models 和 TensorFlow Audio models。

PyTorch Audio models 是一个基于 PyTorch 实现的类，它继承自 PyTorch 的 Audio 模块，并且实现了模型的两个音频端口。这个类的实现中，通过将一个音频信号转换为第二个音频信号，使得模型可以在训练和推理过程中同时处理两个音频流。

TensorFlow Audio models 是一个基于 TensorFlow 实现的类，它继承自 TensorFlow 的 Audio 模块，并且实现了模型的两个音频端口。这个类的实现中，同样通过将一个音频信号转换为第二个音频信号，使得模型可以在训练和推理过程中同时处理两个音频流。

两个子类的实现中，都有两个 forward 方法，其中第一个方法返回了原始的音频信号，而第二个方法则返回了经过模型的处理后的音频信号。不过，这两个方法中都有个参数 spectrogram_庆噪export，这个参数是用来控制模型是否输出 spectrogram 信息。如果这个参数的值为 False，那么模型就不会输出 spectrogram 信息，而如果为 True，那么模型就会输出 spectrogram 信息。


```py
"""Speech SSL models supporting pruning.

Originally from:
https://github.com/pytorch/audio/blob/main/torchaudio/models/wav2vec2/model.py

"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

```

This appears to be a class definition for an audio model that uses a convolutional neural network (CNN) followed by a recurrent neural network (RNN) and a self-attention mechanism. The `CNN` is used for preprocessing the input audio waveforms, while the RNN processes the input in a way that is similar to how it is stored in the ` encoder ` module. The self-attention mechanism allows the model to focus on different parts of the input audio.

The class has two public methods, ` forward ` and ` additional_linear_diag `. The ` forward ` method takes an input tensor of shape `(batch, frames)` and an optional tensor `lengths` indicating the valid length of each audio in the batch. It returns a tuple of two tensors, the first is the output tensor of the model, and the second is a tensor indicating the valid length of the output tensor. The ` additional_linear_diag` method is a simple way to add a linear transformation to the input tensor, which is applied to the output of the self-attention mechanism.

In addition to the forward and additional_linear_diag methods, the class has several class variables, including ` conv_config`, ` use_attention`, ` use_feed_forward`, ` num_heads`, ` remaining_heads`, and ` ff_interm_features`, which are used within the model to configure the specific architecture.


```py
from . import components


class Wav2Vec2Model(Module):
    """Acoustic model used in *wav2vec 2.0* :cite:`baevski2020wav2vec`.

    Note:
        To build the model, please use one of the factory functions.
        :py:func:`wav2vec2_model`, :py:func:`wav2vec2_base`, :py:func:`wav2vec2_large`,
        :py:func:`wav2vec2_large_lv60k`, :py:func:`hubert_base`, :py:func:`hubert_large`,
        and :py:func:`hubert_xlarge`.

    See Also:
        * :class:`torchaudio.pipelines.Wav2Vec2Bundle`: Pretrained models (without fine-tuning)
        * :class:`torchaudio.pipelines.Wav2Vec2ASRBundle`: ASR pipelines with pretrained models.

    Args:
        feature_extractor (torch.nn.Module):
            Feature extractor that extracts feature vectors from raw audio Tensor.

        encoder (torch.nn.Module):
            Encoder that converts the audio features into the sequence of probability
            distribution (in negative log-likelihood) over labels.

        aux (torch.nn.Module or None, optional):
            Auxiliary module. If provided, the output from encoder is passed to this module.
    """  # noqa: E501

    def __init__(
        self,
        normalize_waveform: bool,
        feature_extractor: Module,
        encoder: Module,
        aux: Optional[Module] = None,
    ):
        super().__init__()
        self.normalize_waveform = normalize_waveform
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.aux = aux

    @torch.jit.export
    def extract_features(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> Tuple[List[Tensor], Optional[Tensor]]:
        """Extract feature vectors from raw waveforms

        This returns the list of outputs from the intermediate layers of
        transformer block in encoder.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that the entire audio waveform
                length is valid.
            num_layers (int or None, optional):
                If given, limit the number of intermediate layers to go through.
                Providing `1` will stop the computation after going through one
                intermediate layers. If not given, the outputs from all the
                intermediate layers are returned.

        Returns:
            (List[Tensor], Optional[Tensor]):
            List of Tensors
                Features from requested layers.
                Each Tensor is of shape: `(batch, time frame, feature dimension)`
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of each feature Tensor.
        """
        if self.normalize_waveform:
            if lengths is not None:
                waveforms = [
                    F.layer_norm(wave[:length], (length,)) for wave, length in zip(waveforms, lengths)
                ]
                waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
            else:
                waveforms = F.layer_norm(waveforms, waveforms.shape[-1:])

        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder.extract_features(x, lengths, num_layers)   # (num_layers+1,), including the input
        return x, lengths
    
    def get_num_params(self):
        """Calculate the current size."""
        feature_extractor_size, encoder_in_features = self.feature_extractor.get_num_params_and_final_out_channels()
        encoder_size = self.encoder.get_num_params(encoder_in_features)
        return feature_extractor_size + encoder_size
    
    def prune(self):
        self.eval()     # must be in eval mode
        conv_config, conv_out_index = self.feature_extractor.prune()    # [(output_channel, kernel_size, stride), ...]
        transformer_config = self.encoder.prune(conv_out_index)     # NOTE: this is a defaultdict(list)
        use_attention = transformer_config["use_attention"]
        use_feed_forward = transformer_config["use_feed_forward"]
        num_heads = transformer_config["num_heads"]     # can be []
        remaining_heads = transformer_config["remaining_heads"]     # can be []
        ff_interm_features = transformer_config["ff_interm_features"]

        return conv_config, use_attention, use_feed_forward, num_heads, remaining_heads, ff_interm_features

    def forward(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute the sequence of probability distribution over labels.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that all the audio in ``waveforms``
                have valid length. Default: ``None``.

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor
                The sequences of probability distribution (in logit) over labels.
                Shape: `(batch, frames, num labels)`.
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of the output Tensor.
        """
        if self.normalize_waveform:
            if lengths is not None:
                waveforms = [
                    F.layer_norm(wave[:length], (length,)) for wave, length in zip(waveforms, lengths)
                ]
                waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
            else:
                waveforms = F.layer_norm(waveforms, waveforms.shape[-1:])

        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder(x, lengths)
        if self.aux is not None:
            x = self.aux(x)
        return x, lengths


```

This is a Python implementation of a `Wav2Vec2Model` class that takes in a few parameters and returns a model for processing and generating waveforms.

The model has an encoder and a decoder. The encoder takes in a tensor of features and the decoder takes in a tensor of waveforms and produces a tensor of spectrograms. The model uses the `components` object to get the configuration settings of the components used in the model, such as the number of layers, the encoder and decoder convolutional layers, and the attention mechanism.

The feature extractor is extracted from the encoder. The feature extractor is responsible for extracting the most relevant features from the input waveform and returning them as a tensor. The feature extractor has a specific configuration that is passed to the `components._get_feature_extractor` method to get the actual feature extractor.

The model also has an attention mechanism that is implemented using the `torch.nn.Attention` module. The attention mechanism allows the decoder to focus on certain parts of the input waveform to generate the output spectrogram.

The `Wav2Vec2Model` class has a forward method that defines how the input waveform is passed through the model. The forward method returns the output of the model.

In summary, this model is designed to generate output spectrograms from input waveforms using a combination of the encoder and decoder. It uses the `components` object to get the configuration settings of the components used in the model and the `torch.nn.Attention` module to implement the attention mechanism.


```py
def wav2vec2_model(**configs) -> Wav2Vec2Model:
    """Wraps the original wav2vec2_model and wavlm_model."""

    if "encoder_remaining_heads" in configs:
        return wavlm_model(**configs)
    
    return wav2vec2_model_original(**configs)


def wav2vec2_model_original(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_use_attention: List[bool],
    encoder_use_feed_forward: List[bool],
    encoder_num_heads: List[int],
    encoder_head_dim: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: List[int],
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: Optional[int],
    normalize_waveform: bool,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds custom :class:`~torchaudio.models.Wav2Vec2Model`.

    Note:
        The "feature extractor" below corresponds to
        `ConvFeatureExtractionModel <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L736>`__
        in the original ``fairseq`` implementation.
        This is referred as "(convolutional) feature encoder" in the *wav2vec 2.0*
        :cite:`baevski2020wav2vec` paper.

        The "encoder" below corresponds to `TransformerEncoder <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L817>`__,
        and this is referred as "Transformer" in the paper.

    Args:
        extractor_mode (str): Operation mode of feature extractor.
            Valid values are ``"group_norm"`` or ``"layer_norm"``.
            If ``"group_norm"``, then a single normalization is applied
            in the first convolution block. Otherwise, all the convolution
            blocks will have layer normalization.

            This option corresponds to ``extractor_mode`` from ``fairseq``.
        extractor_conv_layer_config (list of integer tuples or None):
            Configuration of convolution layers in feature extractor.
            List of convolution configuration,
            i.e. ``[(output_channel, kernel_size, stride), ...]``

            If ``None`` is provided, then the following default value is used.

            .. code-block:: python

               [
                 (512, 10, 5),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 2, 2),
                 (512, 2, 2),
               ]

            This option corresponds to ``conv_feature_layers`` from ``fairseq``.

        extractor_conv_bias (bool):
            Whether to include bias term to each convolution operation.

            This option corresponds to ``conv_bias`` from ``fairseq``.

        encoder_embed_dim (int):
            The dimension of embedding in encoder.

            This option corresponds to ``encoder_embed_dim`` from ``fairseq``.

        encoder_projection_dropout (float):
            The dropout probability applied after the input feature is projected
            to ``encoder_embed_dim``.

            This option corresponds to ``dropout_input`` from ``fairseq``.

        encoder_pos_conv_kernel (int):
            The kernel size of convolutional positional embeddings.

            This option corresponds to ``conv_pos`` from ``fairseq``.

        encoder_pos_conv_groups (int):
            The number of groups of convolutional positional embeddings.

            This option corresponds to ``conv_pos_groups`` from ``fairseq``.

        encoder_num_layers (int):
            The number of self attention layers in transformer block.

            This option corresponds to ``encoder_layers`` from ``fairseq``.

        encoder_num_heads (int):
            The number of heads in self attention layers.

            This option corresponds to ``encoder_attention_heads`` from ``fairseq``.

        encoder_attention_dropout (float):
            The dropout probability applied after softmax in self-attention layer.

            This option corresponds to ``attention_dropout`` from ``fairseq``.

        encoder_ff_interm_features (int):
            The dimension of hidden features in feed forward layer.

            This option corresponds to ``encoder_ffn_embed_dim`` from ``fairseq``.

        encoder_ff_interm_dropout (float):
            The dropout probability applied in feedforward layer.

            This option correspinds to ``activation_dropout`` from ``fairseq``.

        encoder_dropout (float):
            The dropout probability applied at the end of feed forward layer.

            This option corresponds to ``dropout`` from ``fairseq``.

        encoder_layer_norm_first (bool):
            Control the order of layer norm in transformer layer and each encoder layer.
            If True, in transformer layer, layer norm is applied before features are fed
            to encoder layers. In encoder layer, two layer norms are applied before and after
            self attention.
            If False, in transformer layer, layer norm is applied after features are fed
            to encoder layers. In encoder layer, two layer norms are applied after self
            attention, before and after feed forward.

            This option corresponds to ``layer_norm_first`` from ``fairseq``.

        encoder_layer_drop (float):
            Probability to drop each encoder layer during training.

            This option corresponds to ``layerdrop`` from ``fairseq``.

        aux_num_out (int or None):
            When provided, attach an extra linear layer on top of encoder, which can be
            used for fine-tuning.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    feature_extractor = components._get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias, 
        prune_conv_channels=extractor_prune_conv_channels,
    )
    encoder = components._get_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        use_attention=encoder_use_attention,
        use_feed_forward=encoder_use_feed_forward,
        num_heads=encoder_num_heads,
        head_dim=encoder_head_dim,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
        prune_attention_heads=encoder_prune_attention_heads,
        prune_attention_layer=encoder_prune_attention_layer,
        prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_embed_dim, out_features=aux_num_out)
    return Wav2Vec2Model(normalize_waveform, feature_extractor, encoder, aux)


```

This is a model architecture for a speech encoder. It appears to be a variation of the Wav2Vec2 model proposed by the Mozilla Research team, where the input static and dynamic sections are combined into a single dynamic section. The input to the model is a 2D tensor of shape `(batch_size, sequence_length, feature_dim)` where `batch_size` is the number of input sequences, `sequence_length` is the length of each sequence, and `feature_dim` is the number of features in each input sequence. The model has a single exported feature, which is a one-hot encoded representation of the input audio signal. The model has an encoder with 12 layers and a dropout rate of 0.15. It also has a decoder with 6 layers and a dropout rate of 0.15.


```py
def wav2vec2_base(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds "base" :class:`~torchaudio.models.Wav2Vec2Model` from *wav2vec 2.0* :cite:`baevski2020wav2vec`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=3072,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )


```



Wav2Vec2Model类是训练中的Wav2Vec2模型，它使用来自Speech Model Transformer（SMT）的AdaTり有很大的进展。该模型的作用是为特定任务从原始语音数据中提取出文本。它包含三个主要类：extractor、encoder 和 decoder。extractor类负责提取输入的语音特征，encoder类负责将提取到的语音特征进行编码，而decoder类则将这些编码后的语音数据返回。该模型支持给定的训练设置，可以根据需要进行调整。经过训练的模型可以用于许多文本相关任务，如语音识别，语音合成等。


```py
def wav2vec2_large(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds "large" :class:`~torchaudio.models.Wav2Vec2Model` from *wav2vec 2.0* :cite:`baevski2020wav2vec`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1024,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )


```




```py
def wav2vec2_large_lv60k(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.0,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.0,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds "large lv-60k" :class:`~torchaudio.models.Wav2Vec2Model` from *wav2vec 2.0* :cite:`baevski2020wav2vec`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )


```

This is a function that returns a Wav2Vec2Model object.

The function takes in several parameters:

- `extractor_mode`: The mode for the extractor, which can be either "group_norm" or "稀疏"。
- `extractor_conv_layer_config`: The configuration for the extractor's convolutional layers, including the number of channels, the kernel size, and the padding strategy.
- `extractor_conv_bias`: Whether to learn the bias for the convolutional layers.
- `encoder_embed_dim`: The dimensionality of the encoder embeddings.
- `encoder_projection_dropout`: The dropout rate for the projection of the encoder embeddings.
- `encoder_pos_conv_kernel`: The kernel size for the position convolution.
- `encoder_pos_conv_groups`: The number of groups for the position convolution.
- `encoder_num_layers`: The number of encoder layers.
- `encoder_use_attention`: A boolean indicating whether to use attention in the encoder.
- `encoder_use_feed_forward`: A boolean indicating whether to use feed-forward in the encoder.
- `encoder_num_heads`: The number of attention heads for the encoder.
- `encoder_attention_dropout`: The dropout rate for the attention in the encoder.
- `encoder_ff_interm_features`: The number of features from the feed-forward encoder interpolated to the attention computation.
- `encoder_ff_interm_dropout`: The dropout rate for the feed-forward encoder interpolated to the attention computation.
- `encoder_dropout`: The dropout rate for the encoder.
- `aux_num_out`: The number of auxiliary output features.
- `extractor_prune_conv_channels`: The number of channels to be pruned from the extractor's convolutional layers.
- `encoder_prune_attention_heads`: The number of attention heads to be pruned from the encoder.
- `encoder_prune_attention_layer`: The number of layers to be pruned from the encoder.
- `encoder_prune_feed_forward_intermediate`: The number of intermediate features to be pruned from the encoder.
- `encoder_prune_feed_forward_layer`: The number of layers to be pruned from the encoder.

The function returns an object of the Wav2Vec2Model class.


```py
def hubert_base(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.05,
    aux_num_out: Optional[int] = None,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds "base" :class:`HuBERT <torchaudio.models.Wav2Vec2Model>` from *HuBERT* :cite:`hsu2021hubert`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_use_attention=[True] * 12,
        encoder_use_feed_forward=[True] * 12,
        encoder_num_heads=[12] * 12,
        encoder_head_dim=64,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=[3072] * 12,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )


```

This is a model architecture for a Wav2Vec2 model. It takes an input of shape `(batch_size, audio_signal_length, audio_signal_samples)` and outputs a tensor of shape `(batch_size, audio_signal_length, audio_signal_samples, n_classes)`. The model has 24 layers in total, and each layer contains an encoder block and a decoder block.

The encoder block has a dropout rate of `0.35` and uses the `nn.Dropout` layer. The encoder block has a linear projection with a dropout rate of `0.45` and uses the `nn.Linear` layer. The decoder block has a dropout rate of `0.35` and uses the `nn.Dropout` layer.

The model also has an attention mechanism, where the attention scores are computed based on the input features and the output of the encoder block. The attention scores have a dropout rate of `0.4`.

This model can be trained using the `wav2vec2_model` function from the `transformers` library.


```py
def hubert_large(
    encoder_projection_dropout: float = 0.0,
    encoder_attention_dropout: float = 0.0,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.0,
    encoder_layer_drop: float = 0.0,
    aux_num_out: Optional[int] = None,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds "large" :class:`HuBERT <torchaudio.models.Wav2Vec2Model>` from *HuBERT* :cite:`hsu2021hubert`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1024,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )


```



This is a PyTorch implementation of a Wav2Vec2 model. It includes an encoder and a decoder, both of which are based on the Layer-Norm except for the position-wise convolutional layers which use a different technique to handle positional information.

The encoder has 48 layers and uses a combination of position-wise convolutional layers, position-wise average Pooling, and linear layers. The decoder has 48 layers and uses the same average pooling layer as the encoder.

The model also includes some hyperparameters such as the dropout rate, number of attention heads, and the number of layers in the encoder and decoder.

It's worth noting that this model is just one possible implementation and it may not be the only best solution for a given task.


```py
def hubert_xlarge(
    encoder_projection_dropout: float = 0.0,
    encoder_attention_dropout: float = 0.0,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.0,
    encoder_layer_drop: float = 0.0,
    aux_num_out: Optional[int] = None,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds "extra large" :class:`HuBERT <torchaudio.models.Wav2Vec2Model>` from *HuBERT* :cite:`hsu2021hubert`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1280,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=48,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=5120,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )


```

这段代码定义了一个名为 `_init_hubert_pretrain_model` 的函数，它接受一个模块（module）作为参数。这个函数的作用是在创建该模块时进行初始化。

函数首先根据模块的类型来执行相应的初始化操作。如果模块是 `components.LayerNorm`，则使用 `torch.nn.init.kaiming_normal_` 函数对模块的卷积层权重进行初始化。如果模块是 `components.ConvolutionalPositionalEmbedding`，则执行以下操作：

1. 将模型的标量嵌入（embedding）的维度设置为 4.0，并计算其平方根为 2.0。
2. 使用 `torch.nn.init.normal_` 函数对模型的卷积层权重、偏移（bias）和静息电荷（memory）进行初始化。
3. 对模块中的自注意力（Self-Attention）层，使用 `torch.nn.init.xavier_uniform_` 函数对模型的键（key）和值（value）进行初始化。
4. 如果模块是 `components.Transformer`，则调用 `components._init_transformer_params` 函数对模型的参数进行初始化。

如果模块的类型不是上面列举的类型，函数将不会执行相应的初始化操作，而是直接返回。


```py
def _init_hubert_pretrain_model(module):
    if isinstance(module, components.LayerNorm):
        torch.nn.init.kaiming_normal_(module.conv.weight)
    elif isinstance(module, components.ConvolutionalPositionalEmbedding):
        # normalize the weight to normal distribution.
        std = math.sqrt(4.0 / (module.embed_dim * module.kernel_size))
        torch.nn.init.normal_(module.conv.weight, mean=0.0, std=std)
        torch.nn.init.constant_(module.conv.bias, 0.0)
    elif isinstance(module, components.SelfAttention):
        # normalize the query, key, value, and out_proj parameters in self attention module.
        torch.nn.init.xavier_uniform_(module.k_proj.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(module.v_proj.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(module.q_proj.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(module.out_proj.weight)
        torch.nn.init.constant_(module.out_proj.bias, 0.0)
    elif isinstance(module, components.Transformer):
        module.apply(components._init_transformer_params)
    else:
        pass


```

This is a PyTorch implementation of a Wav2Vec2 model. It includes an encoder and a decoder.

The encoder uses the encoder_conv_layer_configs to extract features from the input audio data and passes them through a convolutional neural network with learn_speech=True. The extracted features are then passed through a few position-wise convolutional neural networks with varying dilation factors to learn spatial information.

The decoder uses the decoder_conv_layer_configs to重构 the input features. It first adds a learn_decoder=True layer with a learned parameters (num_layers, dropout rate, etc.) and then passes through a convolutional neural network with learn_decoder=True.

The Wav2Vec2 model also includes a variable called aux_num_out which is the output of the last convolutional neural network of the encoder. If this variable is not set, the model will have a single output layer.

This implementation实现了一个Wav2Vec2，可以在Wav，Vec2，及其变体上进行微调。


```py
def wavlm_model(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_use_attention: List[bool],
    encoder_use_feed_forward: List[bool],
    encoder_total_num_heads: List[int],
    encoder_remaining_heads: List[List[int]],
    encoder_num_buckets: int,
    encoder_max_distance: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: List[int],
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: Optional[int],
    normalize_waveform: bool,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds custom WaveLM model :cite:`chen2022wavlm`. The architecture is compatible
    with Wav2Vec2 model :cite:`baevski2020wav2vec`, and so the output object is
    :class:`~torchaudio.models.Wav2Vec2Model`. Most of the arguments have the same meaning
    as in :py:func:`wav2vec2_model` so please refer there for documentation.

    Args:
        extractor_mode (str): Operation mode of feature extractor.
            See :py:func:`wav2vec2_model`.

        extractor_conv_layer_config (list of integer tuples or None):
            See :py:func:`wav2vec2_model`.

        extractor_conv_bias (bool):
            See :py:func:`wav2vec2_model`.

        encoder_embed_dim (int):
            See :py:func:`wav2vec2_model`.

        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.

        encoder_pos_conv_kernel (int):
            See :py:func:`wav2vec2_model`.

        encoder_pos_conv_groups (int):
            See :py:func:`wav2vec2_model`.

        encoder_num_layers (int):
            See :py:func:`wav2vec2_model`.

        encoder_num_heads (int):
            See :py:func:`wav2vec2_model`.

        encoder_num_buckets (int):
            Number of buckets for relative position embedding.
        encoder_max_distance (int):
            Maximum distance for relative position embedding.

        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.

        encoder_ff_interm_features (int):
            See :py:func:`wav2vec2_model`.

        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.

        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.

        encoder_layer_norm_first (bool):
            See :py:func:`wav2vec2_model`.

        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.

        aux_num_out (int or None):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    feature_extractor = components._get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias,
        prune_conv_channels=extractor_prune_conv_channels,
    )
    encoder = components._get_wavlm_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        use_attention=encoder_use_attention,
        use_feed_forward=encoder_use_feed_forward,
        total_num_heads=encoder_total_num_heads,
        remaining_heads=encoder_remaining_heads,
        num_buckets=encoder_num_buckets,
        max_distance=encoder_max_distance,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
        prune_attention_heads=encoder_prune_attention_heads,
        prune_attention_layer=encoder_prune_attention_layer,
        prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_embed_dim, out_features=aux_num_out)
    return Wav2Vec2Model(normalize_waveform, feature_extractor, encoder, aux)


```

This is a class called `Wav2Vec2Model` which is a subclass of `TtaudioAudioModel` and implements the model for training async-speech-dependency-invariance (ASDDI) and synchronize the audio and text features. This model is based on the TensorFlow Probability (TF-P) library and uses the PyTorch implementation.

This model takes audio and text features as input and outputs a model that can be used for training in an async way. It has 12 layers and uses attention mechanism to highlight important regions of the audio and text features. It also has a variable number of attention heads and a variable number of output vectors depending on the encoder and decoder.


```py
def wavlm_base(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    """Builds "base" WaveLM model :cite:`chen2022wavlm`. The architecture is compatible
    with Wav2Vec2 model :cite:`baevski2020wav2vec`, and so the output class is
    :class:`~torchaudio.models.Wav2Vec2Model`.

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """
    return wavlm_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_num_buckets=320,
        encoder_max_distance=800,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=3072,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
    )


```

It seems that you are looking for a model to convert wav audio data to vectors, and this model seems to be an implementation of the Wav2Vec2Model class from the TensorFlow Audio library, which is based on the famous pre-trained model https://github.com/VT-vision/cat饮℃。Wav2Vec2Model是通过对输入的音频信号进行预处理，提取特征，然后将其转换为向量表示。具体来说，这个模型使用一个Wav2Vec2层，该层包括一个编码器和一个解码器。编码器将输入信号的每个时间步转换为一个连续的音频波形，解码器将编码器的输出转换为从0到采样率完整音频数据中的采样点。在训练期间，编码器通过反向传播算法从解码器中学习参数，并在测试期间使用它们重构音频信号以获得重构的音频波形。


```py
def wavlm_large(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    """Builds "large" WaveLM model :cite:`chen2022wavlm`. The architecture is compatible
    with Wav2Vec2 model :cite:`baevski2020wav2vec`, and so the output class is
    :class:`~torchaudio.models.Wav2Vec2Model`.

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """
    return wavlm_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1024,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_num_buckets=320,
        encoder_max_distance=800,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
    )

```

# `vencoder/dphubert/pruning_utils.py`

这段代码定义了一些用于剪枝的辅助函数。这些函数主要针对线性层进行操作。

具体来说，这段代码实现了以下功能：

1. 定义了一个名为 `prune_linear_layer` 的函数，它接受一个线性层（`nn.Linear` 类型）、一个维度（` Union[str, int]` 类型）和一个索引（` torch.LongTensor` 类型）。这个函数会在原地对传入的线性层进行处理，主要针对输入的维度进行设置。

2. `prune_linear_layer` 函数接受两个参数：线性层的权重（`nn.Parameter` 类型）和是否移除维度0的标量（` Union[str, int]` 类型）。

3. 对传入的线性层的权重进行处理，主要是移除维度0的标量。

4. 如果维度0存在，则对它的值进行处理。

5. 返回处理后的线性层。


```py
"""Utility functions for pruning."""

from typing import Union

import torch
import torch.nn as nn


def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: str):
    "Prune linear layer in place."
    # NOTE: weight: (out_features, in_features), bias: (out_features,)
    if dim == "input":
        dim = 1
        layer.in_features = len(index)
    elif dim == "output":
        dim = 0
        layer.out_features = len(index)
    else:
        raise ValueError

    layer.weight = nn.Parameter(layer.weight.index_select(dim, index).clone().detach())
    if layer.bias is not None and dim == 0:
        layer.bias = nn.Parameter(layer.bias.index_select(0, index).clone().detach())


```

这段代码定义了一个名为 `prune_conv1d_layer` 的函数，它接受一个 `nn.Conv1d` 层、一个 `torch.LongTensor` 类型的索引 `index` 和一个维度 `dim`。这个函数的作用是在原地对传入的 `nn.Conv1d` 层进行处理，具体来说，根据 `dim` 的值来改变输入通道（如果 `dim` 是 `"input"`，则输入通道将变为输出通道的数量；如果 `dim` 是 `"output"`，则输出通道将变为输入通道的数量）。然后，它提取了 `nn.Conv1d` 层的权重和 bias，并将权重的指数选择了 `dim` 对应的索引，同时将 bias 也选择了相应的索引。最后，函数返回经过 pruned 的 `nn.Conv1d` 层实例。


```py
def prune_conv1d_layer(layer: nn.Conv1d, index: torch.LongTensor, dim: str):
    """Prune conv1d in place."""
    # NOTE: weight: (out_channels, in_channels, kernel_size), bias: (out_channels,)
    if dim == "input":
        dim = 1
        layer.in_channels = len(index)
    elif dim == "output":
        dim = 0
        layer.out_channels = len(index)
    else:
        raise ValueError
    
    layer.weight = nn.Parameter(layer.weight.index_select(dim, index).clone().detach())
    if layer.bias is not None and dim == 0:
        layer.bias = nn.Parameter(layer.bias.index_select(0, index).clone().detach())


```

这段代码定义了一个名为 `prune_layer_norm` 的函数，它接受一个名为 `layernorm` 的层归一（Layer Normalization）对象和一个名为 `index` 的二维张量，并对其进行操作以实现层归一或组归一。

具体来说，这段代码的主要作用是：

1. 如果 `layernorm` 是一个层归一对象，那么将其参数设置为 `(layernorm.weight.index_select(0, index).clone().detach()`，`(layernorm.bias.index_select(0, index).clone().detach())`，并且将 `layernorm` 的 `normalized_shape` 设置为 `(len(index),)`。

2. 如果 `layernorm` 是一个组归一对象，那么将其参数设置为 `(layernorm.num_groups = len(index), layernorm.num_channels = len(index))`。

这里，我们使用 `pyTorch` 中的 `nn.Parameter` 和 `nn.GroupNorm` 类来实现层归一和组归一。`nn.Parameter` 用于创建一个不可变（non-volatile）的参数，而 `nn.GroupNorm` 则用于实现层归一。


```py
def prune_layer_norm(layernorm: Union[nn.LayerNorm, nn.GroupNorm], index: torch.LongTensor):
    """Prune layer norm or group norm in place."""
    layernorm.weight = nn.Parameter(layernorm.weight.index_select(0, index).clone().detach())
    layernorm.bias = nn.Parameter(layernorm.bias.index_select(0, index).clone().detach())
    if isinstance(layernorm, nn.LayerNorm):
        layernorm.normalized_shape = (len(index),)
    elif isinstance(layernorm, nn.GroupNorm):
        layernorm.num_groups = len(index)
        layernorm.num_channels = len(index)

```

# `vencoder/dphubert/__init__.py`

很抱歉，我需要更多的上下文来回答您的问题。如果能提供更多上下文，我会尽力解释代码的作用。


```py

```

# `vencoder/dphubert/utils/import_huggingface_wavlm.py`

这段代码的作用是将从Hugging Face Model ark中的预训练权重加载到PyTorch的torchaudio格式中。具体来说，它将加载一个名为"wav2vec2.0"的模型，并将该模型的权重存储为dict类型的数据，键为"wav2vec2.0"。这个 dict 数据将存储为文件，以torchaudio的格式打开。

此外，它还从torch中导入了一个名为"Module"的类型，它可能是用于在训练和推理过程中定义自己的组件。


```py
"""Import Hugging Face transformers's wav2vec2.0 pretrained weights to torchaudios's format.

Originally from:
https://github.com/pytorch/audio/blob/main/torchaudio/models/wav2vec2/utils/import_huggingface.py

"""

import logging
from typing import Any, Dict

from torch.nn import Module

from ..model import Wav2Vec2Model, wav2vec2_model, wavlm_model

_LG = logging.getLogger(__name__)


```

这段代码定义了一个名为 `_get_config` 的函数，它接受一个配置字典 `cfg` 作为参数。函数的返回值是一个配置字典，其中包含了提取器模式、提取器卷积层配置、编码器嵌入维度等参数。

具体来说，函数首先根据 `cfg` 中的特征提取模式，定义了 `extractor_mode`、`extractor_conv_layer_config`、`extractor_conv_bias` 等参数，然后将这些参数添加到配置字典中。

接着，函数定义了编码器的嵌入维度、位置编码卷积层的参数，以及预处理的信息，如 `encoder_embed_dim`、`encoder_projection_dropout` 等，并将它们添加到配置字典中。

最后，函数根据 `cfg` 中的配置项，对配置字典进行进一步的调整，如 `encoder_num_layers`、`encoder_num_heads` 等参数，并添加到配置字典中。

整个函数的作用是返回一个配置完整的字典，包含了提取器模式、编码器参数、位置编码卷积层参数等，这些参数可以帮助程序根据传入的配置字典来正确地配置计算图和计算参数。


```py
def _get_config(cfg):
    config = {
        "extractor_mode": f"{cfg.feat_extract_norm}_norm",
        "extractor_conv_layer_config": list(zip(cfg.conv_dim, cfg.conv_kernel, cfg.conv_stride)),
        "extractor_conv_bias": cfg.conv_bias,
        "encoder_embed_dim": cfg.hidden_size,
        "encoder_projection_dropout": cfg.feat_proj_dropout,
        "encoder_pos_conv_kernel": cfg.num_conv_pos_embeddings,
        "encoder_pos_conv_groups": cfg.num_conv_pos_embedding_groups,
        "encoder_num_layers": cfg.num_hidden_layers,
        "encoder_num_heads": cfg.num_attention_heads,
        "encoder_attention_dropout": cfg.attention_dropout,
        "encoder_ff_interm_features": cfg.intermediate_size,
        "encoder_ff_interm_dropout": cfg.activation_dropout,
        "encoder_dropout": cfg.hidden_dropout,
        "encoder_layer_norm_first": cfg.do_stable_layer_norm,
        "encoder_layer_drop": cfg.layerdrop,
    }
    return config


```

This is a Python module that defines a configuration object for an neural network model. The model is an extractor model that uses convolutional neural networks to extract features from input data, and the configuration object includes various settings and parameters for the model.

The configuration object has the following fields:

* `cfg.conv_dim`: The number of convolutional layers in the model.
* `cfg.conv_kernel`: The size of the convolutional kernel.
* `cfg.conv_stride`: The step size of the convolutional stride.
* `cfg.hidden_size`: The number of hidden layers in the model.
* `cfg.feat_proj_dropout`: The probability of dropout in the projection of the convolutional features.
* `cfg.num_conv_pos_embeddings`: The number of positional embeddings in the convolutional features.
* `cfg.num_conv_pos_embedding_groups`: The number of groups of positional embeddings in the convolutional features.
* `cfg.num_layers`: The number of layers in the model.
* `cfg.use_attention`: A list of boolean values indicating whether to enable attention mechanisms in the model.
* `cfg.use_feed_forward`: A list of boolean values indicating whether to enable feedforward networks in the model.
* `cfg.num_attention_heads`: The number of attention heads in the attention module.
* `cfg.encoder_embed_dim`: The number of input features in the encoder.
* `cfg.encoder_projection_dropout`: The probability of dropout in the projection of the encoder embeddings.
* `cfg.hidden_dropout`: The probability of dropout in the hidden layer.
* `cfg.layerdrop`: The dropout probability in the layers of the model.
* `cfg.norm_func`: The normalization function to use in the model.
* `cfg.use_pooling`: A list of boolean values indicating whether to enable pooling in the model.
* `cfg.obj_pooling_aggr`: The averagepooling function to use in the object pooling module.
* `cfg.obj_pooling_战术`: The strategy to use in the object pooling module.
* `cfg.do_stable_layer_norm`: A boolean value indicating whether to enable static layer normalization in the model.
* `cfg.stable_layer_norm_init`: A list of initialization functions for the static layer normalization.
* `cfg.use_梵高每隔注册的域数`: A boolean value indicating whether to enable the梵高每隔注册的域数技巧。
* `cfg.attention_dropout`: The probability of dropout in the attention module.
* `cfg.ff_interm_features`: A list of the intermediate features used in the feedforward network。
* `cfg.ff_interm_dropout`: The probability of dropout in the feedforward network。
* `cfg.activation_dropout`: The probability of dropout in the activation module。
* `cfg.num_buckets`: The number of dilated bucket in the buckets。
* `cfg.max_bucket_distance`: The maximum distance between the dilated桶。
* `cfg.num_layers`: The number of layers in the model.
* `cfg.use_decoder`: A list of boolean values indicating whether to enable the decoder module in the model.
* `cfg.num_pos_conv_kernel`: The number of positional convolutional kernels in the encoder.
* `cfg.num_pos_conv_groups`: The number of groups of positional convolutional kernels in the encoder。
* `cfg.num_decoder_layers`: The number of layers in the decoder。
* `cfg.normalize_waveform`: A boolean value indicating whether to enable normalization in the model.


```py
def _get_config_wavlm(cfg):
    config = {
        "extractor_mode": f"{cfg.feat_extract_norm}_norm",
        "extractor_conv_layer_config": list(zip(cfg.conv_dim, cfg.conv_kernel, cfg.conv_stride)),
        "extractor_conv_bias": cfg.conv_bias,
        "encoder_embed_dim": cfg.hidden_size,
        "encoder_projection_dropout": cfg.feat_proj_dropout,
        "encoder_pos_conv_kernel": cfg.num_conv_pos_embeddings,
        "encoder_pos_conv_groups": cfg.num_conv_pos_embedding_groups,
        "encoder_num_layers": cfg.num_hidden_layers,
        "encoder_use_attention": [True] * cfg.num_hidden_layers,
        "encoder_use_feed_forward": [True] * cfg.num_hidden_layers,
        "encoder_total_num_heads": [cfg.num_attention_heads for _ in range(cfg.num_hidden_layers)],
        "encoder_remaining_heads": [list(range(cfg.num_attention_heads)) for _ in range(cfg.num_hidden_layers)],
        "encoder_num_buckets": cfg.num_buckets,
        "encoder_max_distance": cfg.max_bucket_distance,
        "encoder_attention_dropout": cfg.attention_dropout,
        "encoder_ff_interm_features": [cfg.intermediate_size for _ in range(cfg.num_hidden_layers)],
        "encoder_ff_interm_dropout": cfg.activation_dropout,
        "encoder_dropout": cfg.hidden_dropout,
        "encoder_layer_norm_first": cfg.do_stable_layer_norm,
        "encoder_layer_drop": cfg.layerdrop,
        "normalize_waveform": cfg.feat_extract_norm == "layer",
    }
    return config


```

这段代码定义了一个名为 `_build` 的函数，它接受两个参数 `config` 和 `original`。函数的作用是构建一个特定的神经网络模型，并返回该模型的实例。

函数首先判断 `original` 是否属于 `Wav2Vec2ForCTC` 或 `WavLMForCTC` 类，如果是，则执行下一步操作，否则给出警告。

接着判断 `is_wavlm`，如果是，则尝试使用 `wavlm_model` 函数，如果不是，则使用 `wav2vec2_model` 函数。

接下来，如果 `is_wavlm` 为 `True`，则使用 `wavlm_model` 函数时需要传递一个字典 `config`，其中需要包含一个名为 `aux_num_out` 的键，表示模型中使用的额外词汇数。如果 `is_wavlm` 为 `False`，则使用 `wav2vec2_model` 函数。

然后，如果 `is_wavlm` 为 `True`，则使用 `wavlm_model` 函数的 `feature_extractor` 成员函数加载预处理后的输入数据，并打印 `feature_extractor` 的 `state_dict`；如果 `is_wavlm` 为 `False`，则使用 `wav2vec2_model` 函数的 `feature_projection` 成员函数加载预处理后的输入数据，并打印 `feature_projection` 的 `state_dict`。

接下来，如果 `is_wavlm` 为 `True`，则将 `transform_wavlm_encoder_state` 函数应用于模型的编码器状态字典，以适应 `wavlm_model`。

然后，如果 `is_wavlm` 为 `False`，则使用 `encoder.transformer.load_state_dict` 函数加载输入数据的编码器状态字典，并应用于 `wav2vec2_model`。

接着，如果 `is_for_ctc` 为 `True`，则将 `original.lm_head` 的 `state_dict` 加载到 `imported.aux` 中。

最后，函数返回 `imported` 对象，它是一个经过构建的神经网络模型实例。


```py
def _build(config, original):
    is_for_ctc = original.__class__.__name__ in ["Wav2Vec2ForCTC", "WavLMForCTC"]
    if is_for_ctc:
        aux_num_out = original.config.vocab_size
        wav2vec2 = original.wav2vec2
    else:
        _LG.warning(
            "The model is not an instance of Wav2Vec2ForCTC or WavLMForCTC. " '"lm_head" module is not imported.'
        )
        aux_num_out = None
        wav2vec2 = original
    is_wavlm = original.__class__.__name__ in ["WavLMModel", "WavLMForCTC"]
    if is_wavlm:
        imported = wavlm_model(**config, aux_num_out=aux_num_out)
    else:
        imported = wav2vec2_model(**config, aux_num_out=aux_num_out)
    print(imported.feature_extractor.load_state_dict(wav2vec2.feature_extractor.state_dict(), strict=False))
    print(imported.encoder.feature_projection.load_state_dict(wav2vec2.feature_projection.state_dict(), strict=False))
    encoder_state_dict = wav2vec2.encoder.state_dict()
    if is_wavlm:  # Rename paramaters of linear transformations for compatibility with the HF model
        transform_wavlm_encoder_state(encoder_state_dict, config["encoder_num_layers"])
    print(imported.encoder.transformer.load_state_dict(encoder_state_dict, strict=False))
    if is_for_ctc:
        imported.aux.load_state_dict(original.lm_head.state_dict())
    return imported


```



这段代码定义了一个名为 `transform_wavlm_encoder_state` 的函数，它接受一个名为 `state` 的字典，其中包含任何类型，以及一个名为 `encoder_num_layers` 的整数。它的作用是将 WavLM 编码器的状态从 HuggingFace 的格式中转换出来，具体方法是将线性投影权重和偏置合并，然后将它们与 WavLM 编码器的结构对齐。

接下来定义了一个名为 `import_huggingface_model` 的函数，它接收一个名为 `original` 的模型实例，然后返回一个名为 `Wav2Vec2Model` 的类实例。这个函数通过从 Transformers 库中加载对应的模型配置，并使用预训练的 Wav2Vec2 模型，将加载的 Wav 音频文件转换成文本形式。

最后，代码还定义了一个名为 `build_wav2vec2_model` 的函数，它接收一个名为 `original` 的模型实例和一个字符串 `"WavLMModel"`，然后使用给定的参数构建一个 Wav2Vec2 模型。如果模型是 WavLM 类型，则使用 WavLMForCTC 的配置；否则使用默认的配置。


```py
def transform_wavlm_encoder_state(state: Dict[str, Any], encoder_num_layers: int):
    """Converts WavLM encoder state from HuggingFace format. In particular, concatenates linear projection weights and
    biases to align with the structure of ``torch.nn.MultiheadAttention``.
    """
    pass
    

def import_huggingface_model(original: Module) -> Wav2Vec2Model:
    """Builds :class:`Wav2Vec2Model` from the corresponding model object of
    `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        original (torch.nn.Module): An instance of ``Wav2Vec2ForCTC`` from ``transformers``.

    Returns:
        Wav2Vec2Model: Imported model.

    Example
        >>> from torchaudio.models.wav2vec2.utils import import_huggingface_model
        >>>
        >>> original = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        >>> model = import_huggingface_model(original)
        >>>
        >>> waveforms, _ = torchaudio.load("audio.wav")
        >>> logits, _ = model(waveforms)
    """
    _LG.info("Importing model.")
    _LG.info("Loading model configuration.")
    is_wavlm = original.__class__.__name__ in ["WavLMModel", "WavLMForCTC"]
    if is_wavlm:
        config = _get_config_wavlm(original.config)
    else:
        config = _get_config(original.config)
    _LG.debug("  - config: %s", config)
    _LG.info("Building model.")
    imported = _build(config, original)
    return imported

```