# `stable-diffusion-webui\modules\sd_hijack_clip.py`

```
# 导入数学库
import math
# 导入命名元组
from collections import namedtuple

# 导入 PyTorch 库
import torch

# 导入自定义模块
from modules import prompt_parser, devices, sd_hijack
# 导入共享选项
from modules.shared import opts

# 定义 PromptChunk 类，用于存储提示块的标记、权重和文本反转嵌入信息
class PromptChunk:
    """
    This object contains token ids, weight (multipliers:1.4) and textual inversion embedding info for a chunk of prompt.
    If a prompt is short, it is represented by one PromptChunk, otherwise, multiple are necessary.
    Each PromptChunk contains an exact amount of tokens - 77, which includes one for start and end token,
    so just 75 tokens from prompt.
    """

    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []

# 定义 PromptChunkFix 命名元组，用于标记文本反转嵌入的向量在提示块中的偏移
PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])
"""An object of this type is a marker showing that textual inversion embedding's vectors have to placed at offset in the prompt
chunk. Thos objects are found in PromptChunk.fixes and, are placed into FrozenCLIPEmbedderWithCustomWordsBase.hijack.fixes, and finally
are applied by sd_hijack.EmbeddingsWithFixes's forward function."""

# 定义 FrozenCLIPEmbedderWithCustomWordsBase 类，用于包装 FrozenCLIPEmbedder 模块，增强其功能
class FrozenCLIPEmbedderWithCustomWordsBase(torch.nn.Module):
    """A pytorch module that is a wrapper for FrozenCLIPEmbedder module. it enhances FrozenCLIPEmbedder, making it possible to
    have unlimited prompt length and assign weights to tokens in prompt.
    """

    def __init__(self, wrapped, hijack):
        super().__init__()

        self.wrapped = wrapped
        """Original FrozenCLIPEmbedder module; can also be FrozenOpenCLIPEmbedder or xlmr.BertSeriesModelWithTransformation,
        depending on model."""

        self.hijack: sd_hijack.StableDiffusionModelHijack = hijack
        self.chunk_length = 75

        self.is_trainable = getattr(wrapped, 'is_trainable', False)
        self.input_key = getattr(wrapped, 'input_key', 'txt')
        self.legacy_ucg_val = None
    # 创建一个空的 PromptChunk 对象并返回
    def empty_chunk(self):
        chunk = PromptChunk()
        # 设置 tokens 列表，包含起始标记和多个结束标记
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        # 设置 multipliers 列表，全为 1.0
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    # 返回已知长度的提示的最大标记数，需要多一个 PromptChunk 来表示
    def get_target_prompt_token_count(self, token_count):
        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    # 将一批文本转换为一批标记 id
    def tokenize(self, texts):
        raise NotImplementedError

    # 将一批标记 id（在 Python 列表中）转换为表示这些标记的数字张量
    def encode_with_transformers(self, tokens):
        """
        converts a batch of token ids (in python lists) into a single tensor with numeric respresentation of those tokens;
        All python lists with tokens are assumed to have same length, usually 77.
        if input is a list with B elements and each element has T tokens, expected output shape is (B, T, C), where C depends on
        model - can be 768 and 1024.
        Among other things, this call will read self.hijack.fixes, apply it to its inputs, and clear it (setting it to None).
        """
        raise NotImplementedError

    # 将文本转换为具有该文本标记嵌入的张量。注意，这些是传递到 transformers 之前的嵌入
    def encode_embedding_init_text(self, init_text, nvpt):
        """
        Converts text into a tensor with this text's tokens' embeddings. Note that those are embeddings before they are passed through
        transformers. nvpt is used as a maximum length in tokens. If text produces less teokens than nvpt, only this many is returned.
        """
        raise NotImplementedError
    # 处理文本列表，对每个文本调用tokenize_line()方法，并使用缓存。返回结果列表和所有文本中最大的token数。
    def process_texts(self, texts):
        # 初始化token计数
        token_count = 0

        # 初始化缓存字典
        cache = {}
        
        # 初始化批处理结果列表
        batch_chunks = []
        
        # 遍历文本列表
        for line in texts:
            # 如果文本已经在缓存中，则直接使用缓存结果
            if line in cache:
                chunks = cache[line]
            else:
                # 否则调用tokenize_line()方法对文本进行处理
                chunks, current_token_count = self.tokenize_line(line)
                # 更新token计数为当前文本的token数和之前计数的最大值
                token_count = max(current_token_count, token_count)

                # 将处理结果存入缓存
                cache[line] = chunks

            # 将处理结果添加到批处理结果列表
            batch_chunks.append(chunks)

        # 返回批处理结果列表和最大token数
        return batch_chunks, token_count
    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        """
        处理 tokens 的方法，将一个单独的提示块发送给transformers神经网络进行编码。
        remade_batch_tokens 是一个 tokens 的批处理 - 一个列表，其中每个元素是一个 tokens 的列表；通常列表中有 77 个 tokens。
        batch_multipliers 是相同的，但是用于多重器而不是 tokens。多重器用于给予 transformers 网络的输出更多或更少的权重。每个多重器对应一个 token。
        """
        tokens = torch.asarray(remade_batch_tokens).to(devices.device)

        # 这是为了 SD2：SD1 使用相同的 token 作为填充和文本结束符，而 SD2 使用不同的。
        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index+1:tokens.shape[1]] = self.id_pad

        z = self.encode_with_transformers(tokens)

        pooled = getattr(z, 'pooled', None)

        # 恢复原始均值可能不正确，但似乎可以很好地防止出现其他情况下会发生的伪影
        batch_multipliers = torch.asarray(batch_multipliers).to(devices.device)
        original_mean = z.mean()
        z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z = z * (original_mean / new_mean)

        if pooled is not None:
            z.pooled = pooled

        return z
# 定义一个自定义类 FrozenCLIPEmbedderWithCustomWords，继承自 FrozenCLIPEmbedderWithCustomWordsBase 类
class FrozenCLIPEmbedderWithCustomWords(FrozenCLIPEmbedderWithCustomWordsBase):
    # 初始化方法，接受 wrapped 和 hijack 两个参数
    def __init__(self, wrapped, hijack):
        # 调用父类的初始化方法
        super().__init__(wrapped, hijack)
        # 获取 wrapped 对象的 tokenizer 属性
        self.tokenizer = wrapped.tokenizer

        # 获取 tokenizer 的词汇表
        vocab = self.tokenizer.get_vocab()

        # 获取逗号标记的 token
        self.comma_token = vocab.get(',</w>', None)

        # 初始化 token_mults 字典
        self.token_mults = {}
        # 获取包含括号的 token 列表
        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        # 遍历 tokens_with_parens 列表
        for text, ident in tokens_with_parens:
            # 初始化倍数为 1.0
            mult = 1.0
            # 遍历 token 的每个字符
            for c in text:
                # 根据字符类型调整倍数
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            # 如果倍数不为 1.0，则将 token 的标识和倍数存入 token_mults 字典
            if mult != 1.0:
                self.token_mults[ident] = mult

        # 获取起始标识、结束标识和填充标识
        self.id_start = self.wrapped.tokenizer.bos_token_id
        self.id_end = self.wrapped.tokenizer.eos_token_id
        self.id_pad = self.id_end

    # 定义 tokenize 方法，接受 texts 参数
    def tokenize(self, texts):
        # 使用 wrapped 对象的 tokenizer 对文本进行标记化，不截断，不添加特殊标记
        tokenized = self.wrapped.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

        # 返回标记化结果
        return tokenized

    # 定义 encode_with_transformers 方法，接受 tokens 参数
    def encode_with_transformers(self, tokens):
        # 使用 wrapped 对象的 transformer 对 tokens 进行编码，输出隐藏状态
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=-opts.CLIP_stop_at_last_layers)

        # 如果指定停止在最后几层，则获取指定层的隐藏状态并进行最终层归一化
        if opts.CLIP_stop_at_last_layers > 1:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)
        else:
            # 否则直接获取最后一层的隐藏状态
            z = outputs.last_hidden_state

        # 返回隐藏状态
        return z
    # 定义一个方法，用于将初始文本编码为嵌入向量
    def encode_embedding_init_text(self, init_text, nvpt):
        # 获取嵌入层
        embedding_layer = self.wrapped.transformer.text_model.embeddings
        # 使用分词器将初始文本转换为 token IDs
        ids = self.wrapped.tokenizer(init_text, max_length=nvpt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        # 获取 token 嵌入向量并将其转换为张量，然后在指定设备上进行操作
        embedded = embedding_layer.token_embedding.wrapped(ids.to(embedding_layer.token_embedding.wrapped.weight.device)).squeeze(0)

        # 返回嵌入向量
        return embedded
# 定义一个自定义类 FrozenCLIPEmbedderForSDXLWithCustomWords，继承自 FrozenCLIPEmbedderWithCustomWords 类
class FrozenCLIPEmbedderForSDXLWithCustomWords(FrozenCLIPEmbedderWithCustomWords):
    # 定义初始化方法，接受 wrapped 和 hijack 两个参数
    def __init__(self, wrapped, hijack):
        # 调用父类的初始化方法
        super().__init__(wrapped, hijack)

    # 定义一个方法 encode_with_transformers，接受 tokens 参数
    def encode_with_transformers(self, tokens):
        # 使用 wrapped 对象的 transformer 方法对 tokens 进行编码，返回输出结果
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=self.wrapped.layer == "hidden")

        # 如果 wrapped 对象的 layer 属性为 "last"
        if self.wrapped.layer == "last":
            # 则将最后一个隐藏层的输出赋值给 z
            z = outputs.last_hidden_state
        else:
            # 否则将指定层的隐藏状态输出赋值给 z
            z = outputs.hidden_states[self.wrapped.layer_idx]

        # 返回 z
        return z
```