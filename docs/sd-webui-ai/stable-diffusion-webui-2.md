# SDWebUI源码解析 2

# `modules/sd_hijack.py`

这段代码是一个PyTorch脚本，它实现了LDM（长距离依赖建模）模型的训练和测试。LDM模型是一种基于注意力机制的神经网络模型，主要用于自然语言处理任务，如机器翻译、问答系统等。

具体来说，这段代码包括以下几个部分：

1. 导入需要用到的库，如os、sys、traceback、torch、numpy等。

2. 从名为"modules.shared"的共享模块中导入了几个函数，这些函数可以被用来在LDM模型中使用。

3. 从名为"options"的函数中导入了opts变量，这些变量可以在LDM模型的配置文件中进行设置。

4. 从名为"device"的函数中导入了device变量，这个变量用于在多个GPU设备中选择一个可用的设备。

5. 从名为"cmd_opts"的函数中导入了cmd_opts变量，这个变量可以用来执行命令行操作。

6. 从名为"torch"的函数中导入了torch变量，这个变量用于与PyTorch对象进行交互。

7. 从名为"einsum"的函数中导入了einsum变量，这个函数可以用于计算矩阵的乘积。

8. 从名为"torch.autograd"的函数中导入了auto\_grad变量，这个函数可以用于计算梯度。

9. 从名为"numpy"的函数中导入了numpy变量，这个函数可以用于NumPy数组操作。

10. 从名为"torch.Tensor"的函数中导入了Tensor变量，这个函数可以用于实现向量和矩阵运算。

11. 从名为"torch.nn.functional"的函数中导入了functional变量，这个函数可以用于实现激活函数等。

12. 从名为"models.loader"的函数中导入了models.loader变量，这个函数可以用于加载数据集、语料库等。

13. 从名为"data.utils"的函数中导入了data.utils变量，这个函数可以用于数据处理和清洗等。

14. 从名为"transformers"的函数中导入了transformers变量，这个函数可以用于实现Transformer架构。

15. 从名为"auto.只会输出不会进栈"的函数中导入了output变量，这个函数可以用于在训练过程中输出一些信息，以便于调试等。

这段代码的作用是实现一个基于Attention机制的LDM模型，可以用于自然语言处理任务，如机器翻译、问答系统等。


```py
import os
import sys
import traceback
import torch
import numpy as np
from torch import einsum

from modules.shared import opts, device, cmd_opts

from ldm.util import default
from einops import rearrange
import ldm.modules.attention


# see https://github.com/basujindal/stable-diffusion/pull/117 for discussion
```

这段代码是一个名为`split_cross_attention_forward`的函数，属于一个名为`Attention`的类中。它的作用是实现交叉注意力的前馈传播。

具体来说，这段代码接收一个输入序列`x`，以及一个 optional 的`context` 参数，用于控制输出的大小。函数首先将输入序列`x`转换为张量，并将其存储在变量`context`中或者直接设置为输入序列`x`。然后，它使用一些数学操作将输入序列`x`转换为另一个张量`context`，并且训练了输出序列`v`。接下来，函数实现了交叉注意力的前馈传播，将输入序列`x`与输出序列`v`相结合，使得每个时间步的结果是输入序列`x`和输出序列`v`的对应位置的加权乘积。最后，函数返回了经过前馈传播得到的输出序列`v`。


```py
def split_cross_attention_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)
    del context, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)
    for i in range(0, q.shape[0], 2):
        end = i + 2
        s1 = einsum('b i d, b j d -> b i j', q[i:end], k[i:end])
        s1 *= self.scale

        s2 = s1.softmax(dim=-1)
        del s1

        r1[i:end] = einsum('b i j, b j d -> b i d', s2, v[i:end])
        del s2

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)


```

This is a Python implementation of the Inversion淀粉搜索模型 (iClip), which is based on the transformers architecture. It appears to be used for text classification tasks.

The main function is `iClip.create_model()`, which takes a language model (e.g., pre-trained BERT) and performs fine-tuning on it. The function returns a model instance that can be used for training and prediction.

The model consists of several components: a text encoder, a converter encoder, and a predictor. The text encoder is initialized with the language model and fine-tuned on the target task. The converter encoder is used to convert the input text to a fixed token space, and the predictor uses the encoder outputs to make predictions.

The model also includes an attention mechanism to help the predictor focus on relevant parts of the input text.

The function `iClip.create_model()` takes an optional `model_type` parameter, which specifies the type of the model to use. Other options include `freeze_out_责任的mid_module_and_weights` to freeze the middleware layers and `freeze_scaled_weights` to disable normalization.

Finally, the function `iClip.create_model()` returns the model instance, which can be used for training and prediction by calling the `train()` and `predict()` methods.


```py
class StableDiffusionModelHijack:
    ids_lookup = {}
    word_embeddings = {}
    word_embeddings_checksums = {}
    fixes = None
    comments = []
    dir_mtime = None
    layers = None
    circular_enabled = False

    def load_textual_inversion_embeddings(self, dirname, model):
        mt = os.path.getmtime(dirname)
        if self.dir_mtime is not None and mt <= self.dir_mtime:
            return

        self.dir_mtime = mt
        self.ids_lookup.clear()
        self.word_embeddings.clear()

        tokenizer = model.cond_stage_model.tokenizer

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        def process_file(path, filename):
            name = os.path.splitext(filename)[0]

            data = torch.load(path)

            # textual inversion embeddings
            if 'string_to_param' in data:
                param_dict = data['string_to_param']
                if hasattr(param_dict, '_parameters'):
                    param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
                assert len(param_dict) == 1, 'embedding file has multiple terms in it'
                emb = next(iter(param_dict.items()))[1]
            elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
                assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

                emb = next(iter(data.values()))
                if len(emb.shape) == 1:
                    emb = emb.unsqueeze(0)

            self.word_embeddings[name] = emb.detach()
            self.word_embeddings_checksums[name] = f'{const_hash(emb.reshape(-1))&0xffff:04x}'

            ids = tokenizer([name], add_special_tokens=False)['input_ids'][0]

            first_id = ids[0]
            if first_id not in self.ids_lookup:
                self.ids_lookup[first_id] = []
            self.ids_lookup[first_id].append((ids, name))

        for fn in os.listdir(dirname):
            try:
                process_file(os.path.join(dirname, fn), fn)
            except Exception:
                print(f"Error loading emedding {fn}:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue

        print(f"Loaded a total of {len(self.word_embeddings)} text inversion embeddings.")

    def hijack(self, m):
        model_embeddings = m.cond_stage_model.transformer.text_model.embeddings

        model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
        m.cond_stage_model = FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        if cmd_opts.opt_split_attention:
            ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward

        def flatten(el):
            flattened = [flatten(children) for children in el.children()]
            res = [el]
            for c in flattened:
                res += c
            return res

        self.layers = flatten(m)

    def apply_circular(self, enable):
        if self.circular_enabled == enable:
            return

        self.circular_enabled = enable

        for layer in [layer for layer in self.layers if type(layer) == torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'


```

This is a function that uses a combination of custom tokenizers, model training, and error handling to improve the translation quality of a pre-trained language model. The function takes in an already translated batch of tokens, and adds custom comments with any fix suggestions and multiple word input tokens, truncated input tokens, and used custom terms. It also applies multi-head self-attention and newtoken mechanisms to improve the model's capabilities.

The function returns the output of the model, which is the sanitized, translated, and attention-weighted vector.


```py
class FrozenCLIPEmbedderWithCustomWords(torch.nn.Module):
    def __init__(self, wrapped, hijack):
        super().__init__()
        self.wrapped = wrapped
        self.hijack = hijack
        self.tokenizer = wrapped.tokenizer
        self.max_length = wrapped.max_length
        self.token_mults = {}

        tokens_with_parens = [(k, v) for k, v in self.tokenizer.get_vocab().items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

    def forward(self, text):
        self.hijack.fixes = []
        self.hijack.comments = []
        remade_batch_tokens = []
        id_start = self.wrapped.tokenizer.bos_token_id
        id_end = self.wrapped.tokenizer.eos_token_id
        maxlen = self.wrapped.max_length - 2
        used_custom_terms = []

        cache = {}
        batch_tokens = self.wrapped.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
        batch_multipliers = []
        for tokens in batch_tokens:
            tuple_tokens = tuple(tokens)

            if tuple_tokens in cache:
                remade_tokens, fixes, multipliers = cache[tuple_tokens]
            else:
                fixes = []
                remade_tokens = []
                multipliers = []
                mult = 1.0

                i = 0
                while i < len(tokens):
                    token = tokens[i]

                    possible_matches = self.hijack.ids_lookup.get(token, None)

                    mult_change = self.token_mults.get(token) if opts.enable_emphasis else None
                    if mult_change is not None:
                        mult *= mult_change
                    elif possible_matches is None:
                        remade_tokens.append(token)
                        multipliers.append(mult)
                    else:
                        found = False
                        for ids, word in possible_matches:
                            if tokens[i:i+len(ids)] == ids:
                                emb_len = int(self.hijack.word_embeddings[word].shape[0])
                                fixes.append((len(remade_tokens), word))
                                remade_tokens += [0] * emb_len
                                multipliers += [mult] * emb_len
                                i += len(ids) - 1
                                found = True
                                used_custom_terms.append((word, self.hijack.word_embeddings_checksums[word]))
                                break

                        if not found:
                            remade_tokens.append(token)
                            multipliers.append(mult)

                    i += 1

                if len(remade_tokens) > maxlen - 2:
                    vocab = {v: k for k, v in self.wrapped.tokenizer.get_vocab().items()}
                    ovf = remade_tokens[maxlen - 2:]
                    overflowing_words = [vocab.get(int(x), "") for x in ovf]
                    overflowing_text = self.wrapped.tokenizer.convert_tokens_to_string(''.join(overflowing_words))

                    self.hijack.comments.append(f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")

                remade_tokens = remade_tokens + [id_end] * (maxlen - 2 - len(remade_tokens))
                remade_tokens = [id_start] + remade_tokens[0:maxlen-2] + [id_end]
                cache[tuple_tokens] = (remade_tokens, fixes, multipliers)

            multipliers = multipliers + [1.0] * (maxlen - 2 - len(multipliers))
            multipliers = [1.0] + multipliers[0:maxlen - 2] + [1.0]

            remade_batch_tokens.append(remade_tokens)
            self.hijack.fixes.append(fixes)
            batch_multipliers.append(multipliers)

        if len(used_custom_terms) > 0:
            self.hijack.comments.append("Used custom terms: " + ", ".join([f'{word} [{checksum}]' for word, checksum in used_custom_terms]))

        tokens = torch.asarray(remade_batch_tokens).to(device)
        outputs = self.wrapped.transformer(input_ids=tokens)
        z = outputs.last_hidden_state

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers = torch.asarray(batch_multipliers).to(device)
        original_mean = z.mean()
        z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= original_mean / new_mean

        return z


```



这段代码定义了一个名为EmbeddingsWithFixes的类，该类继承自PyTorch中的nn.Module类。

该类有两个静态方法，分别用于实例化该类以及前向传播函数。在实例化该类时，它将传入一个已经定义好的wrapped实例和一个embeddings实例。

在前向传播函数中，输入层是输入到该类的第一个隐藏层，该函数会对输入层中的每个单词进行处理，包括去除batch_fixes中的偏移量和替换嵌入值。

具体来说，该函数首先将输入层中的每个单词的嵌入值获取出来，然后使用wrapped实例将嵌入值进行转换，接着使用一些特殊的权重对嵌入值进行归一化处理。接下来，该函数使用np.ndarray切片的方式来获取batch_fixes中的单词列表，然后对每个单词的嵌入值进行相应的处理。

最后，该函数返回经过处理的输入层嵌入值。


```py
class EmbeddingsWithFixes(torch.nn.Module):
    def __init__(self, wrapped, embeddings):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is not None:
            for fixes, tensor in zip(batch_fixes, inputs_embeds):
                for offset, word in fixes:
                    emb = self.embeddings.word_embeddings[word]
                    emb_len = min(tensor.shape[0]-offset, emb.shape[0])
                    tensor[offset:offset+emb_len] = self.embeddings.word_embeddings[word][0:emb_len]

        return inputs_embeds


```

这段代码定义了一个名为 "add\_circular\_option\_to\_conv\_2d" 的函数，它将 PyTorch 中的 `torch.nn.Conv2d` 类的初始化函数 `__init__` 进行修改，使得该函数在创建新的 `torch.nn.Conv2d` 实例时，可以通过传递 `padding_mode='circular'` 的参数，来创建一个具有圆形padding的 `Conv2d` 实例。

具体来说，该代码将 `torch.nn.Conv2d.__init__` 函数中的 `padding_mode` 参数由默认值 `'default'` 修改为 `'circular'`，从而实现了在创建新的 `Conv2d` 实例时，可以设置圆形padding。

另外，该代码还将 `add_circular_option_to_conv_2d` 函数中的代码直接输出，但是不要在函数内部使用该函数，以免在函数内部也创建了一个新的 `StableDiffusionModelHijack` 实例。


```py
def add_circular_option_to_conv_2d():
    conv2d_constructor = torch.nn.Conv2d.__init__

    def conv2d_constructor_circular(self, *args, **kwargs):
        return conv2d_constructor(self, *args, padding_mode='circular', **kwargs)

    torch.nn.Conv2d.__init__ = conv2d_constructor_circular


model_hijack = StableDiffusionModelHijack()

```

# `modules/sd_samplers.py`

这段代码的作用是定义了一个名为 SamplerData 的命名元组类型，该类型包含一个名为 name 的属性，用于指定数据源；一个名为 constructor 的属性，用于指定数据源的构造函数；以及一个名为 aliases 的属性，用于指定别名。

具体来说， SamplerData 类型定义了以下结构体：
python
from collections import namedtuple

SamplerData = namedtuple('SamplerData', ['name', 'constructor', 'aliases'])

SamplerData 中包含三个属性：

* `name`：一个字符串，用于指定数据源。
* `constructor`：一个函数，用于指定数据源的构造函数。
* `aliases`：一个列表，用于指定别名。

这段代码还引入了以下几个模块：

from PIL import Image
from k_diffusion.sampling import Sampling
from ldm.models.diffusion.ddim import DiffusionDIM
from ldm.models.diffusion.plms import PLM

这些模块的作用是在后续的代码中进行图像处理、数据采样、模拟扩散等操作。


```py
from collections import namedtuple
import numpy as np
import torch
import tqdm
from PIL import Image

import k_diffusion.sampling
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms

from modules.shared import opts, cmd_opts, state
import modules.shared as shared


SamplerData = namedtuple('SamplerData', ['name', 'constructor', 'aliases'])

```

这段代码的作用是创建一个差分扩散（Diffusion Sampler）的列表，给定一个数据集（diffusion data），然后为每个数据点实例化一个差分扩散采样器（SamplerData实例），每个采样器实例都会在采样时使用给定的函数名称（function name）对数据进行差分扩散采样，并将生成的采样数据存储在采样器数据中。

具体来说，这段代码的实现可以分为以下几个步骤：

1. 从“samplers_k_diffusion”列表中复制每个采样器的配置信息，包括采样器的函数名称（function name）和参数列表（parameter list）。
2. 从“samplers_data_k_diffusion”列表中循环每个数据点实例化一个“SamplerData”类实例，将复制来的采样器配置信息添加到实例的“funcname”参数中，并在采样时使用给定的函数名称对数据进行差分扩散采样。
3. 循环遍历“samplers_k_diffusion”列表中的每个采样器实例，将其创建的“SamplerData”实例添加到“samplers_data_k_diffusion”列表中。

这段代码的主要目的是创建一个差分扩散数据集中每个数据点的采样器实例，以便对数据进行采样。通过使用给定的函数名称对数据进行采样，可以实现数据的随机化，这在机器学习领域中非常常见。


```py
samplers_k_diffusion = [
    ('Euler a', 'sample_euler_ancestral', ['k_euler_a']),
    ('Euler', 'sample_euler', ['k_euler']),
    ('LMS', 'sample_lms', ['k_lms']),
    ('Heun', 'sample_heun', ['k_heun']),
    ('DPM2', 'sample_dpm_2', ['k_dpm_2']),
    ('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a']),
]

samplers_data_k_diffusion = [
    SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases)
    for label, funcname, aliases in samplers_k_diffusion
    if hasattr(k_diffusion.sampling, funcname)
]

```

这段代码包括以下几个主要部分：

1. `samplers` 列表包含了两个采样器，分别是 `samplers_data_k_diffusion` 和 `PLMS`。这两个采样器都被定义为来源于 `ldm.models.diffusion` 和 `ldm.models.diffusion.plms` 两个模型。

2. `samplers_for_img2img` 列表包含了所有不属于 `PLMS` 的采样器。

3. 函数 `sample_to_image` 从 `samplers` 列表中选取一个采样器，然后对选中的采样器返回的第一个样本进行处理。具体地，它将首先将样本从模型的第一阶段中解码，然后将其归一化到模型的第一阶段中。接着，它将将采样率乘以 255，并将像素值从 npy 数组中移动到模型的第一阶段中的通道维度。最后，它返回一个 `Image` 对象，将像素值存储在numpy数组中。

总结起来，这段代码定义了一个函数 `sample_to_image`，该函数对给定的采样器返回的第一个样本进行处理，并返回一个 `Image` 对象。这个函数将被用于样本到图像的转换中。


```py
samplers = [
    *samplers_data_k_diffusion,
    SamplerData('DDIM', lambda model: VanillaStableDiffusionSampler(ldm.models.diffusion.ddim.DDIMSampler, model), []),
    SamplerData('PLMS', lambda model: VanillaStableDiffusionSampler(ldm.models.diffusion.plms.PLMSSampler, model), []),
]
samplers_for_img2img = [x for x in samplers if x.name != 'PLMS']


def sample_to_image(samples):
    x_sample = shared.sd_model.decode_first_stage(samples[0:1].type(shared.sd_model.dtype))[0]
    x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
    x_sample = x_sample.astype(np.uint8)
    return Image.fromarray(x_sample)


```

这段代码定义了一个名为 `store_latent` 的函数，它接受一个已经解码的图像 `decoded`，将其存储到状态变量 `state.current_latent` 中。

接着，它检查一个名为 `opts.show_progress_every_n_steps` 的选项是否大于0。如果是，并且在每次处理图像时，如果不是允许使用并行处理，就会从 `sample_to_image` 函数中采样一张图像，并将其存储到 `shared.state.current_image` 中。

以下是 `p_sample_ddim_hook` 函数的代码，它是一个 hook，用于在采样 DDI（动态图像映射）模型的同时进行动态图像映射（DIM）处理的 Sampler 对象上进行处理。它接收一个 Sampler 对象 `sampler_wrapper`，以及一个或多个参数 `x_dec`、`cond` 和 `ts`，并对其进行处理。如果 `sampler_wrapper.mask` 存在，就可以从 Sampler 的初始化 latent 值中计算出采样图像，否则就将当前图像 `x_dec` 和图像掩码 `x_dec` 相加，并将其存储到 `shared.state.current_image` 中。最后，返回原始的 Sampler 对象。


```py
def store_latent(decoded):
    state.current_latent = decoded

    if opts.show_progress_every_n_steps > 0 and shared.state.sampling_step % opts.show_progress_every_n_steps == 0:
        if not shared.parallel_processing_allowed:
            shared.state.current_image = sample_to_image(decoded)


def p_sample_ddim_hook(sampler_wrapper, x_dec, cond, ts, *args, **kwargs):
    if sampler_wrapper.mask is not None:
        img_orig = sampler_wrapper.sampler.model.q_sample(sampler_wrapper.init_latent, ts)
        x_dec = img_orig * sampler_wrapper.mask + sampler_wrapper.nmask * x_dec

        store_latent(sampler_wrapper.init_latent * sampler_wrapper.mask + sampler_wrapper.nmask * x_dec)

    else:
        store_latent(x_dec)

    return sampler_wrapper.orig_p_sample_ddim(x_dec, cond, ts, *args, **kwargs)


```



这段代码是一个函数 `extended_tdqm`，它使用了 PyTorch 中的 `tqdm` 跟踪器来跟踪数据进度。函数接受一个序列 `sequence`，以及一个或多个参数 `args` 和 `desc`。函数内部使用了 PyTorch 的状态变量 `state`，包括一个状态 `sampling_steps` 和一个状态 `sampling_step`。

函数内部首先将 `sampling_steps` 设置为序列的长度 `len(sequence)`，然后将 `sampling_step` 设置为状态初始值 `0`.接下来，函数内部使用 `tqdm.tqdm()` 函数来迭代序列中的每个元素 `x`，并跟踪当前的 `sampling_step` 和 `shared.total_tqdm.update()`。如果状态变量 `state.interrupted` 为 `True`，函数就会退出迭代。

函数内部还定义了一个函数 `ddim.tqdm`，这个函数接收一个或多个参数，与 `extended_tdqm` 函数的参数相同，但是返回的参数是一个函数，接受一个或多个参数，返回的是一个迭代器 `yield`。这个函数将 `extended_tdqm` 函数作为参数传入，返回的迭代器与 `extended_tdqm` 函数返回的相同。


```py
def extended_tdqm(sequence, *args, desc=None, **kwargs):
    state.sampling_steps = len(sequence)
    state.sampling_step = 0

    for x in tqdm.tqdm(sequence, *args, desc=state.job, file=shared.progress_print_out, **kwargs):
        if state.interrupted:
            break

        yield x

        state.sampling_step += 1
        shared.total_tqdm.update()


ldm.models.diffusion.ddim.tqdm = lambda *args, desc=None, **kwargs: extended_tdqm(*args, desc=desc, **kwargs)
```

It looks like the code is trying to define a function `sample` that uses a `Sampler` object to sample from a dataset based on a given probability distribution `p`. The function takes in a few arguments:

* `p`: The probability distribution to sample from
* `x`: The input to the function, which should be a tensor of shape `(batch_size, sequence_length, embedding_dim)`
* `conditioning`: A condition to be applied to the input (not used in this case)
* `unconditional_conditioning`: An optional condition to be applied to the input (not used in this case)
* `samples_ddim`: The desired batch size for the sample, which should be an integer that is larger than or equal to the batch size of the input (not used in this case)
* `noise`: A noise distribution to be used when drawing samples from the `p` distribution (not used in this case)

It looks like the `Sampler` object has a method called `make_schedule` that takes a batch size `p.steps` and an optional verbose flag `verbose` and returns a schedule that specifies the number of steps to take at each timestep. This method is called inside the body of the function definition.

If the `make_schedule` method raises an exception, the function falls back to using the `sample` method of the `Sampler` object instead, which should handle the case where `p` is not a probability distribution. This method takes the same arguments as the `sample` method, and returns the same output.

It is not clear from the code provided how the `Sampler` object is defined or what it does. It is possible that it is a custom implementation of a sample-from-dataset QNaA like node.


```py
ldm.models.diffusion.plms.tqdm = lambda *args, desc=None, **kwargs: extended_tdqm(*args, desc=desc, **kwargs)


class VanillaStableDiffusionSampler:
    def __init__(self, constructor, sd_model):
        self.sampler = constructor(sd_model)
        self.orig_p_sample_ddim = self.sampler.p_sample_ddim if hasattr(self.sampler, 'p_sample_ddim') else self.sampler.p_sample_plms
        self.mask = None
        self.nmask = None
        self.init_latent = None

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning):
        t_enc = int(min(p.denoising_strength, 0.999) * p.steps)

        # existing code fails with cetin step counts, like 9
        try:
            self.sampler.make_schedule(ddim_num_steps=p.steps, verbose=False)
        except Exception:
            self.sampler.make_schedule(ddim_num_steps=p.steps+1, verbose=False)

        x1 = self.sampler.stochastic_encode(x, torch.tensor([t_enc] * int(x.shape[0])).to(shared.device), noise=noise)

        self.sampler.p_sample_ddim = lambda x_dec, cond, ts, *args, **kwargs: p_sample_ddim_hook(self, x_dec, cond, ts, *args, **kwargs)
        self.mask = p.mask
        self.nmask = p.nmask
        self.init_latent = p.init_latent

        samples = self.sampler.decode(x1, conditioning, t_enc, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning)

        return samples

    def sample(self, p, x, conditioning, unconditional_conditioning):
        for fieldname in ['p_sample_ddim', 'p_sample_plms']:
            if hasattr(self.sampler, fieldname):
                setattr(self.sampler, fieldname, lambda x_dec, cond, ts, *args, **kwargs: p_sample_ddim_hook(self, x_dec, cond, ts, *args, **kwargs))
        self.mask = None
        self.nmask = None
        self.init_latent = None

        samples_ddim, _ = self.sampler.sample(S=p.steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning, x_T=x)
        return samples_ddim


```



这段代码定义了一个名为CFGDenoiser的类，继承自PyTorch中的nn.Module类。这个类用于对输入数据进行预处理，特别是在需要使用Batch一起计算的情况下。

在__init__方法中，首先调用父类的init__方法，然后创建一个self.inner_model实例，一个self.mask和一个self.nmask实例。还定义了一个self.init_latent实例，这里没有给出更多的信息，无法知道它的具体作用。

在forward方法中，对输入的x,sigma和uncond进行处理。其中，sigma是条件噪声的强度，uncond是条件概率分布，cond是真实的条件概率。通过调用self.inner_model函数，对输入数据进行处理，并使用sigma和uncond计算出denoised值。然后，根据是否使用了Batch一起计算的条件，对denoised值进行进一步处理。如果使用了Batch一起计算，则需要对输入数据进行复制，然后进行处理。最后，返回处理后的denoised值。

CFGDenoiser类的实例可以被存储为同一类中的许多个，例如使用GFGDenoiser类可以将多个训练数据集的denoised值存储为同一张GPU内存上，以便在多个batch计算时进行共享。


```py
class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.mask = None
        self.nmask = None
        self.init_latent = None

    def forward(self, x, sigma, uncond, cond, cond_scale):
        if shared.batch_cond_uncond:
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            denoised = uncond + (cond - uncond) * cond_scale
        else:
            uncond = self.inner_model(x, sigma, cond=uncond)
            cond = self.inner_model(x, sigma, cond=cond)
            denoised = uncond + (cond - uncond) * cond_scale

        if self.mask is not None:
            denoised = self.init_latent * self.mask + self.nmask * denoised

        return denoised


```

This appears to be a Python implementation of a method for generating images from low-quality images using a diffusion model and a series of transformations. The `样本图像2图像` function takes an image, a denoising strength, a conditioning strength, and a conditional scaling factor as input and returns the denoised image. The `样本`函数 takes the same input and returns the denoised image.

The `样本图像2图像` function uses the `k_diffusion` library to sample from the diffusion model. The `k_diffusion.sampling.trange` function generates an aliased range for the sampling.

The `样本` function uses the `self.func` function to generate the denoised image. This function appears to take the denoised image, the diffusion model configuration, and a series of transformation parameters as input and returns the denoised image. The transformation parameters include the denoising strength, the conditioning strength, and the conditional scaling factor.


```py
def extended_trange(count, *args, **kwargs):
    state.sampling_steps = count
    state.sampling_step = 0

    for x in tqdm.trange(count, *args, desc=state.job, file=shared.progress_print_out, **kwargs):
        if state.interrupted:
            break

        yield x

        state.sampling_step += 1
        shared.total_tqdm.update()


class KDiffusionSampler:
    def __init__(self, funcname, sd_model):
        self.model_wrap = k_diffusion.external.CompVisDenoiser(sd_model)
        self.funcname = funcname
        self.func = getattr(k_diffusion.sampling, self.funcname)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

    def callback_state(self, d):
        store_latent(d["denoised"])

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning):
        t_enc = int(min(p.denoising_strength, 0.999) * p.steps)
        sigmas = self.model_wrap.get_sigmas(p.steps)

        noise = noise * sigmas[p.steps - t_enc - 1]

        xi = x + noise

        sigma_sched = sigmas[p.steps - t_enc - 1:]

        self.model_wrap_cfg.mask = p.mask
        self.model_wrap_cfg.nmask = p.nmask
        self.model_wrap_cfg.init_latent = p.init_latent

        if hasattr(k_diffusion.sampling, 'trange'):
            k_diffusion.sampling.trange = lambda *args, **kwargs: extended_trange(*args, **kwargs)

        return self.func(self.model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': p.cfg_scale}, disable=False, callback=self.callback_state)

    def sample(self, p, x, conditioning, unconditional_conditioning):
        sigmas = self.model_wrap.get_sigmas(p.steps)
        x = x * sigmas[0]

        if hasattr(k_diffusion.sampling, 'trange'):
            k_diffusion.sampling.trange = lambda *args, **kwargs: extended_trange(*args, **kwargs)

        samples_ddim = self.func(self.model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': p.cfg_scale}, disable=False, callback=self.callback_state)
        return samples_ddim


```

# `modules/shared.py`

这段代码是一个基于PyTorch框架的机器学习项目，它的主要作用是编译和运行一个名为“my_project”的机器学习模型。这个模型使用了预训练的PyTorch模型，经过训练后可以在各种数据集上取得较好的表现。

具体来说，这段代码执行以下操作：

1. 导入需要用到的PyTorch库和相关的工具，包括argparse、json、os、gradio、torch和tqdm。

2. 定义了一个config.json文件，这个文件是配置这个机器学习模型的参数和 paths。

3. 定义了一个名为“my_project”的类，这个类包含了训练和测试数据读取、模型加载和运行等操作。

4. 在my_project类中，加载了需要用到的 PyTorch modules，包括artists、paths、codeformer_model等。

5. 导入了代码分离的上下文，以便在运行时使用gr High-level API。

6. 在运行时，启动了一个private server，监听localhost的8050端口，用于接收用户提交的数据，并返回训练过程中的状态信息。

7. 使用代码分割技术，将代码拆分成多个小文件，这样可以提高代码的读性和维护性。

8. 在script_path和sd_path类中，读取并返回训练数据和测试数据的路径，这些数据通常包含为用户提供高质量体验所需的全部信息。

9. 使用gradio创建了一个简单的用户界面，以便用户查看模型的训练进度和结果。

10. 在运行时，使用tqdm等工具来监控训练过程中的指标和事件，并输出训练过程中的状态信息。

11. 通过运行my_project类的函数，用户可以训练my_project模型的指定数据集，也可以在测试集上查看模型的性能。


```py
import sys
import argparse
import json
import os

import gradio as gr
import torch
import tqdm

import modules.artists
from modules.paths import script_path, sd_path
import modules.codeformer_model

config_filename = "config.json"

```

这段代码是一个Python脚本，用于训练一个基于Stable Diffusion模型的FPGA GAN。它包括以下主要功能：

1. 查找预训练的模型文件：通过Join方法将脚本目录和模型文件夹链接起来，如果模型文件不存在，则默认下载预训练的模型。
2. 设置训练参数：通过ArgumentParser类设置训练参数，包括模型文件、检查点文件、GPU安装目录等。
3. 允许在运行时执行自定义脚本：通过argparse类设置允许在运行时执行自定义脚本。
4. 设置训练模式：通过argparse类设置运行时是否启用训练模式，以优化FPGA GAN的性能。
5. 允许使用Stable Diffusion模型：通过argparse类设置FPGA GAN支持使用Stable Diffusion模型。
6. 设置内存限制：通过argparse类设置FPGA GAN允许的最大内存使用量。
7. 设置嵌入式目录：通过argparse类设置FPGA GAN允许的嵌入式文本逆向目录。
8. 设置GPU安装目录：通过argparse类设置FPGA GAN允许的GPU安装目录。
9. 允许使用FPGA驱动程序：通过argparse类设置FPGA GAN允许使用FPGA驱动程序。
10. 设置指标：通过argparse类设置指标，以评估FPGA GAN的性能。

总的来说，这段代码是一个用于训练FPGA GAN的Python脚本，可以允许用户在运行时执行自定义脚本，以优化FPGA GAN的性能。


```py
sd_model_file = os.path.join(script_path, 'model.ckpt')
if not os.path.exists(sd_model_file):
    sd_model_file = "models/ldm/stable-diffusion-v1/model.ckpt"

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.join(sd_path, "configs/stable-diffusion/v1-inference.yaml"), help="path to config which constructs model",)
parser.add_argument("--ckpt", type=str, default=os.path.join(sd_path, sd_model_file), help="path to checkpoint of model",)
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN'))
parser.add_argument("--gfpgan-model", type=str, help="GFPGAN model file name", default='GFPGANv1.3.pth')
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats")
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware accleration in browser)")
parser.add_argument("--max-batch-count", type=int, default=16, help="maximum batch count value for the UI")
parser.add_argument("--embeddings-dir", type=str, default='embeddings', help="embeddings directory for textual inversion (default: embeddings)")
parser.add_argument("--allow-code", action='store_true', help="allow custom script execution from webui")
parser.add_argument("--medvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a little speed for low VRM usage")
```

这段代码是使用 PyTorch 创建一个机器学习模型的命令行界面（Command Line Interface，CLI）的一部分。

具体来说，它定义了一系列参数，用于在训练过程中指定使用稳定扩散模型优化、始终批处理条件计算、卸载 GFPGAN、使用精确度选项、在 GUI 中使用 Share 选项、ESRGAN 模型的路径等。

其中，`--lowvram` 和 `--always-batch-cond-uncond` 参数用于在训练过程中指定低电压内存（VRAM）的使用量，`--unload-gfpgan` 参数用于在每次处理图像后卸载 GFPGAN，`--precision` 参数用于指定评估的精度，`--share` 和 `--esrgan-models-path` 参数用于在 Gradio 中使用 Share 选项，`--port` 参数用于指定 Gradio 的服务器端口。

此外，`--show-negative-prompt` 参数用于在训练过程中允许用户输入负采样率，从而提高模型性能。


```py
parser.add_argument("--lowvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a lot of speed for very low VRM usage")
parser.add_argument("--always-batch-cond-uncond", action='store_true', help="a workaround test; may help with speed if you use --lowvram")
parser.add_argument("--unload-gfpgan", action='store_true', help="unload GFPGAN every time after processing images. Warning: seems to cause memory leaks")
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site (doesn't work for me but you might have better luck)")
parser.add_argument("--esrgan-models-path", type=str, help="path to directory with ESRGAN models", default=os.path.join(script_path, 'ESRGAN'))
parser.add_argument("--opt-split-attention", action='store_true', help="enable optimization that reduce vram usage by a lot for about 10%% decrease in performance")
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
parser.add_argument("--port", type=int, help="launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available", default=None)
parser.add_argument("--show-negative-prompt", action='store_true', help="enable the field that lets you input negative prompt", default=False)

cmd_opts = parser.parse_args()

if torch.has_cuda:
    device = torch.device("cuda")
```

这段代码使用了PyTorch中的`torch.device()`函数来设置设备。它根据给定的选项`cmd_opts.always_batch_cond_uncond`和`cmd_opts.lowvram`以及`cmd_opts.medvram`来判断是否使用GPU或CPU。如果`torch.has_mps`为真，则使用GPU，否则使用CPU。

接着，它创建了一个名为`State`的类来记录任务的状态。该类包含了一些方法，例如`interrupt()`用于设置或取消任务的干扰，`nextjob()`用于轮询是否有新的任务需要运行，以及一些与任务相关的计数器，例如`job_count`用于记录当前正在运行的任务数。

最后，该代码将`State`实例的`interrupted`属性设置为`True`，这意味着当任务被中断时，此实例将`True`。


```py
elif torch.has_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
batch_cond_uncond = cmd_opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram and not cmd_opts.medvram


class State:
    interrupted = False
    job = ""
    job_no = 0
    job_count = 0
    sampling_step = 0
    sampling_steps = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0


    def interrupt(self):
        self.interrupted = True

    def nextjob(self):
        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0


```

这段代码的作用是创建一个名为 `state` 的 `State` 对象，并将其赋值为 `None`。

接下来，它创建了一个名为 `artist_db` 的 `ArtistsDatabase` 对象，将 `os.path.join(script_path, 'artists.csv')` 中的艺术家数据库文件加载到了内存中。

然后，它创建了一个名为 `face_restorers` 的列表，用于保存已经修复过的面部。

最后，它定义了一个名为 `find_any_font` 的函数，该函数将搜索计算机上可用的字体，并在字体存在时返回该字体。如果找不到任何字体，它将返回 "Arial.TTF"。


```py
state = State()

artist_db = modules.artists.ArtistsDatabase(os.path.join(script_path, 'artists.csv'))

face_restorers = []


def find_any_font():
    fonts = ['/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']

    for font in fonts:
        if os.path.exists(font):
            return font

    return "Arial.TTF"


```

This appears to be a Python script for controlling the face-restoration model in DeepLab. The script has several options for controlling the model, including a code-formatter weight parameter and a maximum image size.

The script also has two options for controlling the console output:

* `show_progressbar`: This option enables a progress bar to be displayed in the console when the model is being built or reconstructed.
* `show_progress_every_n_steps`: This option specifies how many times the console output should be refreshed during the model construction process. The default value is 0, which means the console will not be refreshed at all.

There is also an option to enable the use of multiple TQDM (Terminal Quotable Multiplexer Delimiter) lines to display the progress of the model construction process.

The `face_restoration_model` option allows the user to specify the name of a pre-trained face-restoration model to be used.

The `maximum` and `step` options specify the maximum size and step size for the image that will be reconstructed.


```py
class Options:
    class OptionInfo:
        def __init__(self, default=None, label="", component=None, component_args=None):
            self.default = default
            self.label = label
            self.component = component
            self.component_args = component_args

    data = None
    data_labels = {
        "outdir_samples": OptionInfo("", "Output directory for images; if empty, defaults to two directories below"),
        "outdir_txt2img_samples": OptionInfo("outputs/txt2img-images", 'Output directory for txt2img images'),
        "outdir_img2img_samples": OptionInfo("outputs/img2img-images", 'Output directory for img2img images'),
        "outdir_extras_samples": OptionInfo("outputs/extras-images", 'Output directory for images from extras tab'),
        "outdir_grids": OptionInfo("", "Output directory for grids; if empty, defaults to two directories below"),
        "outdir_txt2img_grids": OptionInfo("outputs/txt2img-grids", 'Output directory for txt2img grids'),
        "outdir_img2img_grids": OptionInfo("outputs/img2img-grids", 'Output directory for img2img grids'),
        "save_to_dirs": OptionInfo(False, "When writing images/grids, create a directory with name derived from the prompt"),
        "save_to_dirs_prompt_len": OptionInfo(10, "When using above, how many words from prompt to put into directory name", gr.Slider, {"minimum": 1, "maximum": 32, "step": 1}),
        "outdir_save": OptionInfo("log/images", "Directory for saving images using the Save button"),
        "samples_save": OptionInfo(True, "Save indiviual samples"),
        "samples_format": OptionInfo('png', 'File format for indiviual samples'),
        "grid_save": OptionInfo(True, "Save image grids"),
        "return_grid": OptionInfo(True, "Show grid in results for web"),
        "grid_format": OptionInfo('png', 'File format for grids'),
        "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
        "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
        "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
        "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
        "export_for_4chan": OptionInfo(True, "If PNG image is larger than 4MB or any dimension is larger than 4000, downscale and save copy as JPG"),
        "enable_pnginfo": OptionInfo(True, "Save text information about generation parameters as chunks to png files"),
        "font": OptionInfo(find_any_font(), "Font for image grids  that have text"),
        "enable_emphasis": OptionInfo(True, "Use (text) to make model pay more attention to text text and [text] to make it pay less attention"),
        "save_txt": OptionInfo(False, "Create a text file next to every image with generation parameters."),
        "ESRGAN_tile": OptionInfo(192, "Tile size for upscaling. 0 = no tiling.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
        "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap, in pixels for upscaling. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
        "random_artist_categories": OptionInfo([], "Allowed categories for random artists selection when using the Roll button", gr.CheckboxGroup, {"choices": artist_db.categories()}),
        "upscale_at_full_resolution_padding": OptionInfo(16, "Inpainting at full resolution: padding, in pixels, for the masked region.", gr.Slider, {"minimum": 0, "maximum": 128, "step": 4}),
        "show_progressbar": OptionInfo(True, "Show progressbar"),
        "show_progress_every_n_steps": OptionInfo(0, "Show show image creation progress every N sampling steps. Set 0 to disable.", gr.Slider, {"minimum": 0, "maximum": 32, "step": 1}),
        "multiple_tqdm": OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job. Broken in PyCharm console."),
        "face_restoration_model": OptionInfo(None, "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
        "code_former_weight": OptionInfo(0.5, "CodeFormer weight parameter; 0 = maximum effect; 1 = minimum effect", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    }

    def __init__(self):
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __setattr__(self, key, value):
        if self.data is not None:
            if key in self.data:
                self.data[key] = value

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.data_labels:
            return self.data_labels[item].default

        return super(Options, self).__getattribute__(item)

    def save(self, filename):
        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file)

    def load(self, filename):
        with open(filename, "r", encoding="utf8") as file:
            self.data = json.load(file)


```

这段代码的作用是设置一个名为 `TotalTQDM` 的类，用于在训练深度学习模型时使用 `tqdm` 库实现进度的监控和更新。

具体来说，代码首先定义了几个变量，包括 `opts`、`sd_upscalers`、`sd_model`、`progress_print_out`。其中，`opts` 是一个 `Options` 对象，用于存储选项，例如最大迭代次数、批处理大小等。`sd_upscalers` 是一个列表，用于存储动态评估学习率增长的算子。`sd_model` 是一个用于保存模型参数的对象。`progress_print_out` 是 `sys.stdout`，用于在模型训练过程中输出进度信息。

接着，代码定义了一个 `TotalTQDM` 类。这个类包含以下方法：

- `__init__`：初始化 `tqdm` 对象，用于在训练过程中输出进度信息。
- `reset`：重置 `tqdm` 对象的进度信息。
- `update`：更新 `tqdm` 对象的进度信息。
- `clear`：关闭 `tqdm` 对象，并清空 `sd_upscalers` 列表和 `sd_model` 对象。

最后，代码创建了一个 `TotalTQDM` 实例，并在主循环中一直运行该实例，直到模型训练完成。


```py
opts = Options()
if os.path.exists(config_filename):
    opts.load(config_filename)

sd_upscalers = []

sd_model = None

progress_print_out = sys.stdout


class TotalTQDM:
    def __init__(self):
        self._tqdm = None

    def reset(self):
        self._tqdm = tqdm.tqdm(
            desc="Total progress",
            total=state.job_count * state.sampling_steps,
            position=1,
            file=progress_print_out
        )

    def update(self):
        if not opts.multiple_tqdm:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.update()

    def clear(self):
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None


```

这段代码使用了Python中的pandas库和numpy库来实现一个名为“TotalTQDM”的函数，该函数的作用是获取一个数据框中所有数据的总和。

具体来说，这段代码创建了一个名为“total_tqdm”的新对象，该对象使用了total参数，该参数在函数中作为参数传递给了TotalTQDM函数，从而获取了数据框中所有数据的总和。

如果这个数据框没有被传递给TotalTQDM函数，那么这段代码将无法运行，因为无法获取数据。


```py
total_tqdm = TotalTQDM()

```

# `modules/txt2img.py`

这段代码是一个Python脚本，主要作用是实现将文本描述转换为图像的函数txt2img。txt2img接受一个或多个参数，包括一个提示文本、一个负提示文本、一个采样器指数、一个恢复面部外观的布尔值、一个图像处理层 sampling 的步长、一个图像处理层的配置缩放因子、一个种子号和一个图像的大小。函数使用StableDiffusionProcessingTxt2Img类将文本描述转换为图像，如果遇到错误，将返回一个Processed对象，否则将返回一个images, js 和 plaintext_to_html组成的元组。


```py
import modules.scripts
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html


def txt2img(prompt: str, negative_prompt: str, steps: int, sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, height: int, width: int, *args):
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        sampler_index=sampler_index,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
    )

    print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)
    processed = modules.scripts.scripts_txt2img.run(p, *args)

    if processed is not None:
        pass
    else:
        processed = process_images(p)

    shared.total_tqdm.clear()

    return processed.images, processed.js(), plaintext_to_html(processed.info)


```

# `modules/ui.py`

这段代码的作用是：

1. 从各个网站（如：百度、微博、新闻网站等）自动收集新闻数据。
2. 解析和提取新闻文章中的关键词、主题等信息。
3. 将新闻文章中的关键词和主题进行标准化，以便进行后续的文本处理。
4. 使用PyTorch训练一个自然语言处理（NLP）模型，对收集到的新闻文章进行人工标注。
5. 将标注后的新闻文章进行排序，以提高网站的推荐准确性。

具体来说，这段代码包括以下步骤：

1. 通过`base64`包将收集到的新闻文章中的图片数据编码为可以识别的格式。
2. 通过`html`包解析新闻文章的HTML代码，提取出需要提取的信息。
3. 通过`json`包将提取的信息进行JSON格式化。
4. 使用`mimetypes`包将新闻文章中的关键词转换为常用的网络用语，以方便后续处理。
5. 通过`os`包获取收集的新闻文章的目录，并将图片文件存储到对应的文件夹中。
6. 使用`torch`包中的` Image`类对图片文件进行预处理，以便后续训练模型。
7. 使用`numpy`和`torch`包中的`nn.Module`类，创建一个神经网络模型，用于对新闻文章进行标注。
8. 使用`torch.optim`包中的`SGD`类，设置优化参数，对模型进行训练。
9. 通过循环，将新闻文章中的图片随机读取，并使用预处理后的图片训练模型。
10. 计算出模型预测的新闻文章的关键词，并按照新闻文章中关键词出现的次数进行排序。

总之，这段代码将使用Python等编程语言，实现一个自动从新闻网站中自动抓取数据、进行标注和排序的过程，以提高网站的新闻推荐准确性。


```py
import base64
import html
import io
import json
import mimetypes
import os
import random
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image

```

这段代码使用了Gradio库来创建一个交互式图形界面，用于展示各种机器学习模型的训练和评估结果。具体来说，它实现了以下功能：

1. 导入必要的模块和函数：从项目中导入路径模块、shared模块、Samplers模块、RealsRGAN模块、scripts模块、GGFGAN模块、以及从codeformer模块导入的几个模型类。

2. 创建一个gr.FileVar，用于存储需要评估的模型及文件名。然后，创建一个gr.TkMaster，用于显示图形界面。

3. 创建一个名为"Training"的函数，该函数调用了Gradio中的 Training 模式。在这个函数中，使用了 Samplers 和 RealESRGAN 模型来处理图像数据。然后，使用 cmd_opts类将训练选项和指标进行了设置，并使用 opts类将训练过程中的参数进行了设置。

4. 创建一个名为 "Evaluate"的函数，该函数调用了Gradio中的 Evaluate 模式。在这个函数中，使用 RealESRGAN 和 samples.txt文件来评估模型的性能。然后，使用 Samplers 和 RealESRGAN 模型来处理图像数据。

5. 在脚本中，使用 gradio.utils.翰墨行写了几个函数，实现了翰墨行的一些常用功能，如 format_image,put_text,text_field,file_选择等等。

6. 导入mimetypes模块，以便为不同类型的文件提供适当的媒体类型。

这段代码的作用是创建一个交互式图形界面，用于训练和评估不同的机器学习模型。通过调用不同的函数，用户可以选择不同的模型、训练或评估不同的指标，以及查看训练过程中的进展。


```py
import gradio as gr
import gradio.utils
import gradio.routes

from modules.paths import script_path
from modules.shared import opts, cmd_opts
import modules.shared as shared
from modules.sd_samplers import samplers, samplers_for_img2img
import modules.realesrgan_model as realesrgan
import modules.scripts
import modules.gfpgan_model
import modules.codeformer_model

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the bowser will not show any UI
mimetypes.init()
```

这段代码是一个用于添加一个名为"application/javascript"的MIME类型，该类型适用于".js"文件的后缀。

第一个条件语句检查是否启用了命令行选项中的"share"和"listen"，如果不启用了它们，则执行以下操作：

- 通过调用gradio.utils.version_check，将"None"作为参数传递，确保在运行该代码时，gradio版本检查为空。
- 通过调用gradio.utils.get_local_ip_address，设置为"127.0.0.1"，使得 gradio 在本地计算机上获取 IP 地址。

然后，定义了一个名为"gr_show"的函数，用于在 gradio 中显示图片。

接下来，定义了一个名为"sample_img2img"的变量，使用 if 语句从 gradio 的资产目录中加载图片，如果图片路径存在，则加载图片，否则将其设置为 None。

最后，通过 gr_show 函数，可以将定义的参数"visible=True"传递给 gradio，并在显示时执行该函数，从而更新可见性。


```py
mimetypes.add_type('application/javascript', '.js')


if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

```

这段代码是一个 CSS 样式，它的作用是隐藏某些元素的 progressbar。

具体来说，这段代码包含三个部分：

1. `.wrap .m-12 svg { display:none!important; }`：这个部分定义了一个类的 CSS 属性，它的值为 `display:none!important;`。这个属性 targets 带有 `.m-12` 类的 SVG 元素，如果这个元素被显示了，它会设置 `display` 属性为 `block`，否则会设置为 `none`。
2. `.wrap .m-12::before { content:"Loading..." }`：这个部分定义了一个条件的 CSS 属性，它的值为 `content:"Loading..."`。这个属性 targets 带有 `.m-12` 类的元素，如果这个元素被显示了，它会设置内容的文本为 "Loading..."，否则不会改变。
3. `.meta-text { display:none!important; }`：这个部分定义了一个条件的 CSS 属性，它的值为 `display:none!important;`。这个属性 targets 所有的元素，如果这个元素被显示了，它会设置 `display` 属性为 `block`，否则会设置为 `none`。


```py
css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
"""

def plaintext_to_html(text):
    text = "".join([f"<p>{html.escape(x)}</p>\n" for x in text.split('\n')])
    return text


def image_from_url_text(filedata):
    if type(filedata) == list:
        if len(filedata) == 0:
            return None

        filedata = filedata[0]

    if filedata.startswith("data:image/png;base64,"):
        filedata = filedata[len("data:image/png;base64,"):]

    filedata = base64.decodebytes(filedata.encode('utf-8'))
    image = Image.open(io.BytesIO(filedata))
    return image


```

这段代码定义了两个函数，分别是 `send_gradio_gallery_to_image` 和 `save_files`。

1. `send_gradio_gallery_to_image` 函数接收一个参数 `x`，它的作用是确保 `x` 列表不空，然后返回一个图像对象。该函数首先从 `x` 列表中提取第一个元素，然后使用 `image_from_url_text` 函数从 URL 文本中返回图像对象。如果 `x` 列表为空，该函数返回 `None`。

2. `save_files` 函数接收两个参数：`js_data` 和 `images`。它的作用是在指定的输出目录中保存 `js_data` 和 `images` 中的文件。该函数首先创建一个目录，如果目录不存在，则创建它。然后，它遍历 `images` 列表，将 `images` 中的每个文件保存到指定的目录中。对于每个文件，它使用 `filename_base` 生成一个以秒为单位的时间戳作为文件名的一部分。如果文件名中包含 "data:image/png;base64,"，则文件已经是 PNG 格式，否则它会将文件名保存到 `filenames` 列表中。最后，它将 `js_data` 和 `images` 中的所有文件名合并为一个字符串，并将该字符串作为参数传递给 `plaintext_to_html` 函数。

这两个函数一起工作，用于将 Gradio 画廊中的图像下载到本地，并保存到指定的文件夹中。


```py
def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None

    return image_from_url_text(x[0])


def save_files(js_data, images):
    import csv

    os.makedirs(opts.outdir_save, exist_ok=True)

    filenames = []

    data = json.loads(js_data)

    with open(os.path.join(opts.outdir_save, "log.csv"), "a", encoding="utf8", newline='') as file:
        at_start = file.tell() == 0
        writer = csv.writer(file)
        if at_start:
            writer.writerow(["prompt", "seed", "width", "height", "sampler", "cfgs", "steps", "filename"])

        filename_base = str(int(time.time() * 1000))
        for i, filedata in enumerate(images):
            filename = filename_base + ("" if len(images) == 1 else "-" + str(i + 1)) + ".png"
            filepath = os.path.join(opts.outdir_save, filename)

            if filedata.startswith("data:image/png;base64,"):
                filedata = filedata[len("data:image/png;base64,"):]

            with open(filepath, "wb") as imgfile:
                imgfile.write(base64.decodebytes(filedata.encode('utf-8')))

            filenames.append(filename)

        writer.writerow([data["prompt"], data["seed"], data["width"], data["height"], data["sampler"], data["cfg_scale"], data["steps"], filenames[0]])

    return '', '', plaintext_to_html(f"Saved: {filenames[0]}")


```

这段代码定义了一个名为 `wrap_gradio_call` 的函数，它接受一个函数作为参数并返回一个新的函数，这个新函数在调用原始函数时会对结果进行处理。

具体来说，这个新函数 `f` 在内部调用了原始函数 `func`，并对结果进行了一些处理。处理的主要步骤包括：

1. 记录执行时间，以便在之后的计算中使用。
2. 如果调用 `func` 时出现异常，会对异常进行输出并记录到 `shared.state.job` 和 `shared.state.job_count` 变量中。
3. 将最后一个结果(即 HTML 结果)添加到结果列表中。
4. 对结果进行排序，使得结果按照时间戳升序排列。
5. 如果没有发生异常，使用 `time.perf_counter()` 计算新的时间花费，以便在之后的计算中使用。
6. 返回原始函数的结果。

由于 `wrap_gradio_call` 函数本身没有定义如何使用它，因此它的实际作用是将一个 Gradio 老子应用程序的函数包装成一个可以输出结果的函数，这个结果可以被 Gradio 用于显示性能数据。


```py
def wrap_gradio_call(func):
    def f(*args, **kwargs):
        t = time.perf_counter()

        try:
            res = list(func(*args, **kwargs))
        except Exception as e:
            print("Error completing request", file=sys.stderr)
            print("Arguments:", args, kwargs, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

            shared.state.job = ""
            shared.state.job_count = 0

            res = [None, '', f"<div class='error'>{plaintext_to_html(type(e).__name__+': '+str(e))}</div>"]

        elapsed = time.perf_counter() - t

        # last item is always HTML
        res[-1] = res[-1] + f"<p class='performance'>Time taken: {elapsed:.2f}s</p>"

        shared.state.interrupted = False

        return tuple(res)

    return f


```

该函数的作用是检查模型进度，并输出一个带有 progressbar 的信息，如果模型正在运行，输出 progress，否则输出一个空字符串。

具体来说，函数首先检查 shared.state.job_count 是否为 0，如果是，则立即返回，并且不显示 progress。然后，函数开始计算进度，如果 shared.state.job_count 大于 0，则计算进度。接下来，函数检查 shared.state.sampling_steps 是否大于 0，如果是，则计算 1/shared.state.job_count * shared.state.sampling_step/shared.state.sampling_steps。最后，函数将计算出的进度值限制在 0 到 1 之间，并输出 progressbar 和一张预览图片。

如果 opts.show_progressbar 设置为 True，函数将输出带有 progressbar 的信息；如果 opts.show_progress_every_n_steps 大于 0，则函数将每隔 n_steps 步输出一次 progress。函数的最终输出结果是一个元组，包含当前时间戳、progressbar 和一张预览图片。


```py
def check_progress_call():

    if shared.state.job_count == 0:
        return "", gr_show(False), gr_show(False)

    progress = 0

    if shared.state.job_count > 0:
        progress += shared.state.job_no / shared.state.job_count
    if shared.state.sampling_steps > 0:
        progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

    progress = min(progress, 1)

    progressbar = ""
    if opts.show_progressbar:
        progressbar = f"""<div class='progressDiv'><div class='progress' style="width:{progress * 100}%">{str(int(progress*100))+"%" if progress > 0.01 else ""}</div></div>"""

    image = gr_show(False)
    preview_visibility = gr_show(False)

    if opts.show_progress_every_n_steps > 0:
        if shared.parallel_processing_allowed:

            if shared.state.sampling_step - shared.state.current_image_sampling_step >= opts.show_progress_every_n_steps and shared.state.current_latent is not None:
                shared.state.current_image = modules.sd_samplers.sample_to_image(shared.state.current_latent)
                shared.state.current_image_sampling_step = shared.state.sampling_step

        image = shared.state.current_image

        if image is None or progress >= 1:
            image = gr.update(value=None)
        else:
            preview_visibility = gr_show(True)

    return f"<span style='display: none'>{time.time()}</span><p>{progressbar}</p>", preview_visibility, image


```

这是一个 Python 函数，作用是获取用户输入的 prompt 参数中的艺术家名称，如果 prompt 为空，则返回默认艺术家名称。函数的实现涉及两个部分：`roll_artist` 和 `visit`。

1. `roll_artist` 函数，用于获取艺术家名称。它接收一个参数 `prompt`，首先通过 `set()` 函数过滤出所有属于 `shared.artist_db.categories()` 中的艺术家，然后通过 `if` 语句判断是否所有的 `opts.random_artist_categories` 都为空。如果是，则随机从 `shared.artist_db.artists()` 中选择一个艺术家。否则，从 `opts.random_artist_categories` 中选择一个艺术家。最后，将艺术家名称添加到返回的结果中，如果没有 `prompt` 参数，则返回默认的艺术家名称。

2. `visit` 函数，用于访问目录中的子目录。它接收三个参数：当前目录、函数要执行的函数名称和一个路径。如果当前目录下的子目录存在，则递归地调用 `visit` 函数。如果当前目录不存在子目录，则执行指定的函数并返回。

该函数的作用是在给定的艺术家库中随机获取艺术家名称，并在给定的目录中递归地访问子目录。


```py
def roll_artist(prompt):
    allowed_cats = set([x for x in shared.artist_db.categories() if len(opts.random_artist_categories)==0 or x in opts.random_artist_categories])
    artist = random.choice([x for x in shared.artist_db.artists if x.category in allowed_cats])

    return prompt + ", " + artist.name if prompt != '' else artist.name


def visit(x, func, path=""):
    if hasattr(x, 'children'):
        for c in x.children:
            visit(c, func, path)
    elif x.label is not None:
        func(path + "/" + str(x.label), x)


```

This is a Python script that appears to convert an interactive text adventure game (txt2img) to an interactive HTML game (img2img). The script uses the 'txt2img' and 'img2img' interfaces, which appear to provide text input and output for the game, respectively.

The script creates a settings file (ui-config.json) for the HTML game, which is loaded from the 'txt2img' interface. The script defines a set of CSS styles that are applied to the HTML game.

The script then defines a function loadsave, which is used to save and load the game's settings. The function reads the settings from the 'txt2img' interface and applies any changes specified by the user (e.g., changing the radio buttons).

Finally, the script uses the loadsave function to save the game's settings to the UI file and, if the file already exists, to overwrite the file.

Overall, it looks like the script is attempting to convert a simple text-based game to an interactive HTML game that can be rendered in a web browser.


```py
def create_ui(txt2img, img2img, run_extras, run_pnginfo):
    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", elem_id="txt2img_prompt", show_label=False, placeholder="Prompt", lines=1)
            negative_prompt = gr.Textbox(label="Negative prompt", elem_id="txt2img_negative_prompt", show_label=False, placeholder="Negative prompt", lines=1, visible=cmd_opts.show_negative_prompt)
            roll = gr.Button('Roll', elem_id="txt2img_roll", visible=len(shared.artist_db.artists) > 0)
            submit = gr.Button('Generate', elem_id="txt2img_generate", variant='primary')
            check_progress = gr.Button('Check progress', elem_id="check_progress", visible=False)

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=20)
                sampler_index = gr.Radio(label='Sampling method', elem_id="txt2img_sampling", choices=[x.name for x in samplers], value=samplers[0].name, type="index")

                with gr.Row():
                    restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1)
                    tiling = gr.Checkbox(label='Tiling', value=False)

                with gr.Row():
                    batch_count = gr.Slider(minimum=1, maximum=cmd_opts.max_batch_count, step=1, label='Batch count', value=1)
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1)

                cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=7.0)

                with gr.Group():
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)

                seed = gr.Number(label='Seed', value=-1)

                with gr.Group():
                    custom_inputs = modules.scripts.scripts_txt2img.setup_ui(is_img2img=False)

            with gr.Column(variant='panel'):
                with gr.Group():
                    txt2img_preview = gr.Image(elem_id='txt2img_preview', visible=False)
                    txt2img_gallery = gr.Gallery(label='Output', elem_id='txt2img_gallery').style(grid=4)


                with gr.Group():
                    with gr.Row():
                        save = gr.Button('Save')
                        send_to_img2img = gr.Button('Send to img2img')
                        send_to_inpaint = gr.Button('Send to inpaint')
                        send_to_extras = gr.Button('Send to extras')
                        interrupt = gr.Button('Interrupt')

                progressbar = gr.HTML(elem_id="progressbar")

                with gr.Group():
                    html_info = gr.HTML()
                    generation_info = gr.Textbox(visible=False)


            txt2img_args = dict(
                fn=txt2img,
                _js="submit",
                inputs=[
                    prompt,
                    negative_prompt,
                    steps,
                    sampler_index,
                    restore_faces,
                    tiling,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    seed,
                    height,
                    width,
                ] + custom_inputs,
                outputs=[
                    txt2img_gallery,
                    generation_info,
                    html_info
                ]
            )

            prompt.submit(**txt2img_args)
            submit.click(**txt2img_args)

            check_progress.click(
                fn=check_progress_call,
                show_progress=False,
                inputs=[],
                outputs=[progressbar, txt2img_preview, txt2img_preview],
            )


            interrupt.click(
                fn=lambda: shared.state.interrupt(),
                inputs=[],
                outputs=[],
            )

            save.click(
                fn=wrap_gradio_call(save_files),
                inputs=[
                    generation_info,
                    txt2img_gallery,
                ],
                outputs=[
                    html_info,
                    html_info,
                    html_info,
                ]
            )

            roll.click(
                fn=roll_artist,
                inputs=[
                    prompt,
                ],
                outputs=[
                    prompt
                ]
            )


    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", elem_id="img2img_prompt", show_label=False, placeholder="Prompt", lines=1)
            negative_prompt = gr.Textbox(label="Negative prompt", elem_id="img2img_negative_prompt", show_label=False, placeholder="Negative prompt", lines=1, visible=cmd_opts.show_negative_prompt)
            submit = gr.Button('Generate', elem_id="img2img_generate", variant='primary')
            check_progress = gr.Button('Check progress', elem_id="check_progress", visible=False)

        with gr.Row().style(equal_height=False):

            with gr.Column(variant='panel'):
                with gr.Group():
                    switch_mode = gr.Radio(label='Mode', elem_id="img2img_mode", choices=['Redraw whole image', 'Inpaint a part of image', 'Loopback', 'SD upscale'], value='Redraw whole image', type="index", show_label=False)
                    init_img = gr.Image(label="Image for img2img", source="upload", interactive=True, type="pil")
                    init_img_with_mask = gr.Image(label="Image for inpainting with mask", elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", visible=False, image_mode="RGBA")
                    resize_mode = gr.Radio(label="Resize mode", show_label=False, choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value="Just resize")

                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=20)
                sampler_index = gr.Radio(label='Sampling method', choices=[x.name for x in samplers_for_img2img], value=samplers_for_img2img[0].name, type="index")
                mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, visible=False)
                inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='fill', type="index", visible=False)

                with gr.Row():
                    inpaint_full_res = gr.Checkbox(label='Inpaint at full resolution', value=False, visible=False)
                    inpainting_mask_invert = gr.Radio(label='Masking mode', choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index", visible=False)

                with gr.Row():
                    restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1)
                    tiling = gr.Checkbox(label='Tiling', value=False)
                    sd_upscale_overlap = gr.Slider(minimum=0, maximum=256, step=16, label='Tile overlap', value=64, visible=False)

                with gr.Row():
                    sd_upscale_upscaler_name = gr.Radio(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index", visible=False)

                with gr.Row():
                    batch_count = gr.Slider(minimum=1, maximum=cmd_opts.max_batch_count, step=1, label='Batch count', value=1)
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1)

                with gr.Group():
                    cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=7.0)
                    denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.75)
                    denoising_strength_change_factor = gr.Slider(minimum=0.9, maximum=1.1, step=0.01, label='Denoising strength change factor', value=1, visible=False)

                with gr.Group():
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)

                seed = gr.Number(label='Seed', value=-1)

                with gr.Group():
                    custom_inputs = modules.scripts.scripts_img2img.setup_ui(is_img2img=True)

            with gr.Column(variant='panel'):
                with gr.Group():
                    img2img_preview = gr.Image(elem_id='img2img_preview', visible=False)
                    img2img_gallery = gr.Gallery(label='Output', elem_id='img2img_gallery').style(grid=4)

                with gr.Group():
                    with gr.Row():
                        save = gr.Button('Save')
                        img2img_send_to_img2img = gr.Button('Send to img2img')
                        img2img_send_to_inpaint = gr.Button('Send to inpaint')
                        img2img_send_to_extras = gr.Button('Send to extras')
                        interrupt = gr.Button('Interrupt')

                progressbar = gr.HTML(elem_id="progressbar")

                with gr.Group():
                    html_info = gr.HTML()
                    generation_info = gr.Textbox(visible=False)

            def apply_mode(mode):
                is_classic = mode == 0
                is_inpaint = mode == 1
                is_loopback = mode == 2
                is_upscale = mode == 3

                return {
                    init_img: gr_show(not is_inpaint),
                    init_img_with_mask: gr_show(is_inpaint),
                    mask_blur: gr_show(is_inpaint),
                    inpainting_fill: gr_show(is_inpaint),
                    batch_count: gr_show(not is_upscale),
                    batch_size: gr_show(not is_loopback),
                    sd_upscale_upscaler_name: gr_show(is_upscale),
                    sd_upscale_overlap: gr_show(is_upscale),
                    inpaint_full_res: gr_show(is_inpaint),
                    inpainting_mask_invert: gr_show(is_inpaint),
                    denoising_strength_change_factor: gr_show(is_loopback),
                }

            switch_mode.change(
                apply_mode,
                inputs=[switch_mode],
                outputs=[
                    init_img,
                    init_img_with_mask,
                    mask_blur,
                    inpainting_fill,
                    batch_count,
                    batch_size,
                    sd_upscale_upscaler_name,
                    sd_upscale_overlap,
                    inpaint_full_res,
                    inpainting_mask_invert,
                    denoising_strength_change_factor,
                ]
            )

            img2img_args = dict(
                fn=img2img,
                _js="submit",
                inputs=[
                    prompt,
                    negative_prompt,
                    init_img,
                    init_img_with_mask,
                    steps,
                    sampler_index,
                    mask_blur,
                    inpainting_fill,
                    restore_faces,
                    tiling,
                    switch_mode,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    denoising_strength,
                    denoising_strength_change_factor,
                    seed,
                    height,
                    width,
                    resize_mode,
                    sd_upscale_upscaler_name,
                    sd_upscale_overlap,
                    inpaint_full_res,
                    inpainting_mask_invert,
                ] + custom_inputs,
                outputs=[
                    img2img_gallery,
                    generation_info,
                    html_info
                ]
            )

            prompt.submit(**img2img_args)
            submit.click(**img2img_args)

            check_progress.click(
                fn=check_progress_call,
                show_progress=False,
                inputs=[],
                outputs=[progressbar, img2img_preview, img2img_preview],
            )

            interrupt.click(
                fn=lambda: shared.state.interrupt(),
                inputs=[],
                outputs=[],
            )

            save.click(
                fn=wrap_gradio_call(save_files),
                inputs=[
                    generation_info,
                    img2img_gallery,
                ],
                outputs=[
                    html_info,
                    html_info,
                    html_info,
                ]
            )

            send_to_img2img.click(
                fn=lambda x: image_from_url_text(x),
                _js="extract_image_from_gallery",
                inputs=[txt2img_gallery],
                outputs=[init_img],
            )

            send_to_inpaint.click(
                fn=lambda x: image_from_url_text(x),
                _js="extract_image_from_gallery",
                inputs=[txt2img_gallery],
                outputs=[init_img_with_mask],
            )

            img2img_send_to_img2img.click(
                fn=lambda x: image_from_url_text(x),
                _js="extract_image_from_gallery",
                inputs=[img2img_gallery],
                outputs=[init_img],
            )

            img2img_send_to_inpaint.click(
                fn=lambda x: image_from_url_text(x),
                _js="extract_image_from_gallery",
                inputs=[img2img_gallery],
                outputs=[init_img_with_mask],
            )

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Group():
                    image = gr.Image(label="Source", source="upload", interactive=True, type="pil")

                upscaling_resize = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Resize", value=2)

                with gr.Group():
                    extras_upscaler_1 = gr.Radio(label='Upscaler 1', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index")

                with gr.Group():
                    extras_upscaler_2 = gr.Radio(label='Upscaler 2', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index")
                    extras_upscaler_2_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Upscaler 2 visibility", value=1)

                with gr.Group():
                    gfpgan_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="GFPGAN visibility", value=0, interactive=modules.gfpgan_model.have_gfpgan)

                with gr.Group():
                    codeformer_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="CodeFormer visibility", value=0, interactive=modules.codeformer_model.have_codeformer)
                    codeformer_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="CodeFormer weight (0 = maximum effect, 1 = minimum effect)", value=0, interactive=modules.codeformer_model.have_codeformer)

                submit = gr.Button('Generate', elem_id="extras_generate", variant='primary')

            with gr.Column(variant='panel'):
                result_image = gr.Image(label="Result")
                html_info_x = gr.HTML()
                html_info = gr.HTML()

        extras_args = dict(
            fn=run_extras,
            inputs=[
                image,
                gfpgan_visibility,
                codeformer_visibility,
                codeformer_weight,
                upscaling_resize,
                extras_upscaler_1,
                extras_upscaler_2,
                extras_upscaler_2_visibility,
            ],
            outputs=[
                result_image,
                html_info_x,
                html_info,
            ]
        )

        submit.click(**extras_args)

        send_to_extras.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery",
            inputs=[txt2img_gallery],
            outputs=[image],
        )

        img2img_send_to_extras.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery",
            inputs=[img2img_gallery],
            outputs=[image],
        )

    pnginfo_interface = gr.Interface(
        wrap_gradio_call(run_pnginfo),
        inputs=[
            gr.Image(label="Source", source="upload", interactive=True, type="pil"),
        ],
        outputs=[
            gr.HTML(),
            gr.HTML(),
            gr.HTML(),
        ],
        allow_flagging="never",
        analytics_enabled=False,
    )

    def create_setting_component(key):
        def fun():
            return opts.data[key] if key in opts.data else opts.data_labels[key].default

        info = opts.data_labels[key]
        t = type(info.default)

        if info.component is not None:
            args = info.component_args() if callable(info.component_args) else info.component_args
            item = info.component(label=info.label, value=fun, **(args or {}))
        elif t == str:
            item = gr.Textbox(label=info.label, value=fun, lines=1)
        elif t == int:
            item = gr.Number(label=info.label, value=fun)
        elif t == bool:
            item = gr.Checkbox(label=info.label, value=fun)
        else:
            raise Exception(f'bad options item type: {str(t)} for key {key}')

        return item

    def run_settings(*args):
        up = []

        for key, value, comp in zip(opts.data_labels.keys(), args, settings_interface.input_components):
            opts.data[key] = value
            up.append(comp.update(value=value))

        opts.save(shared.config_filename)

        return 'Settings saved.', '', ''

    settings_interface = gr.Interface(
        run_settings,
        inputs=[create_setting_component(key) for key in opts.data_labels.keys()],
        outputs=[
            gr.Textbox(label='Result'),
            gr.HTML(),
            gr.HTML(),
        ],
        title=None,
        description=None,
        allow_flagging="never",
        analytics_enabled=False,
    )

    interfaces = [
        (txt2img_interface, "txt2img"),
        (img2img_interface, "img2img"),
        (extras_interface, "Extras"),
        (pnginfo_interface, "PNG Info"),
        (settings_interface, "Settings"),
    ]

    with open(os.path.join(script_path, "style.css"), "r", encoding="utf8") as file:
        css = file.read()

    if not cmd_opts.no_progressbar_hiding:
        css += css_hide_progressbar

    demo = gr.TabbedInterface(
        interface_list=[x[0] for x in interfaces],
        tab_names=[x[1] for x in interfaces],
        analytics_enabled=False,
        css=css,
    )

    ui_config_file = os.path.join(modules.paths.script_path, 'ui-config.json')
    ui_settings = {}
    settings_count = len(ui_settings)
    error_loading = False

    try:
        if os.path.exists(ui_config_file):
            with open(ui_config_file, "r", encoding="utf8") as file:
                ui_settings = json.load(file)
    except Exception:
        error_loading = True
        print("Error loading settings:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    def loadsave(path, x):
        def apply_field(obj, field, condition=None):
            key = path + "/" + field

            saved_value = ui_settings.get(key, None)
            if saved_value is None:
                ui_settings[key] = getattr(obj, field)
            elif condition is None or condition(saved_value):
                setattr(obj, field, saved_value)

        if type(x) == gr.Slider:
            apply_field(x, 'value')
            apply_field(x, 'minimum')
            apply_field(x, 'maximum')
            apply_field(x, 'step')

        if type(x) == gr.Radio:
            apply_field(x, 'value', lambda val: val in x.choices)

    visit(txt2img_interface, loadsave, "txt2img")
    visit(img2img_interface, loadsave, "img2img")

    if not error_loading and (not os.path.exists(ui_config_file) or settings_count != len(ui_settings)):
        with open(ui_config_file, "w", encoding="utf8") as file:
            json.dump(ui_settings, file, indent=4)

    return demo


```

这段代码的作用是读取一个名为"script.js"的文件，并将其内容存储在名为jsfile的文件对象中。然后，定义了一个名为template_response的函数，该函数使用gradio_routes_templates_response函数来获取模板文件的内容，并将其内容与JavaScript代码混合。最后，将生成的模板文件内容返回给调用者，并对其进行初始化。

具体来说，代码的作用是：读取一个名为"script.js"的文件，并将其内容存储在名为jsfile的文件对象中。然后，定义了一个名为template_response的函数，该函数使用gradio_routes_templates_response函数来获取模板文件的内容，并将其内容与JavaScript代码混合。最后，将生成的模板文件内容返回给调用者，并对其进行初始化。


```py
with open(os.path.join(script_path, "script.js"), "r", encoding="utf8") as jsfile:
    javascript = jsfile.read()


def template_response(*args, **kwargs):
    res = gradio_routes_templates_response(*args, **kwargs)
    res.body = res.body.replace(b'</head>', f'<script>{javascript}</script></head>'.encode("utf8"))
    res.init_headers()
    return res


gradio_routes_templates_response = gradio.routes.templates.TemplateResponse
gradio.routes.templates.TemplateResponse = template_response

```