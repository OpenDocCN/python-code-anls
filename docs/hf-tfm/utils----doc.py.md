# `.\utils\doc.py`

```py
# 版权声明和许可证信息
"""
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# 导入必要的模块
import functools
import re
import types

# 定义一个装饰器函数，用于添加起始文档字符串
def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn
    return docstring_decorator

# 定义一个装饰器函数，用于给模型的前向方法添加起始文档字符串
def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        # 组合文档字符串和现有文档字符串（如果存在）
        docstring = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        # 获取类名，并构建介绍信息
        class_name = f"[`{fn.__qualname__.split('.')[0]}`]"
        intro = f"   The {class_name} forward method, overrides the `__call__` special method."
        # 添加额外的提示信息
        note = r"""
    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>
"""
        # 更新函数的文档字符串
        fn.__doc__ = intro + note + docstring
        return fn
    return docstring_decorator

# 定义一个装饰器函数，用于添加结尾文档字符串
def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn
    return docstring_decorator

# PyTorch 返回值介绍模板
PT_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(torch.FloatTensor)`: A [`{full_output_type}`] or a tuple of
        `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
        elements depending on the configuration ([`{config_class}`]) and inputs.

"""

# TensorFlow 返回值介绍模板
TF_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(tf.Tensor)`: A [`{full_output_type}`] or a tuple of `tf.Tensor` (if
        `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the
        configuration ([`{config_class}`]) and inputs.

"""

# 辅助函数：获取字符串的缩进
def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search(r"^(\s*)\S", t)
    return "" if search is None else search.groups()[0]

# 辅助函数：将输出参数文档转换为适当的显示格式
def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    # 获取文本缩进
    indent = _get_indent(output_args_doc)
    blocks = []  # 初始化块列表
    current_block = ""  # 初始化当前块字符串
    # 对于输出的参数文档，按行遍历处理
    for line in output_args_doc.split("\n"):
        # 如果当前行的缩进与指定缩进相同，表示这行是一个新参数的名称
        if _get_indent(line) == indent:
            # 如果当前块已经有内容，将其添加到块列表中，并重新初始化当前块
            if len(current_block) > 0:
                blocks.append(current_block[:-1])  # 去除最后的换行符并添加到块列表
            current_block = f"{line}\n"  # 初始化当前块为当前行
        else:
            # 否则，当前行是当前参数描述的一部分
            # 需要去除两个空格的缩进，然后添加到当前块中
            current_block += f"{line[2:]}\n"
    
    # 将最后一个块添加到块列表中（因为最后一个块不会触发上述条件）
    blocks.append(current_block[:-1])  # 去除最后的换行符并添加到块列表
    
    # 对每个块进行格式化，以便进行正确的渲染
    for i in range(len(blocks)):
        blocks[i] = re.sub(r"^(\s+)(\S+)(\s+)", r"\1- **\2**\3", blocks[i])  # 使用粗体格式化参数名
        blocks[i] = re.sub(r":\s*\n\s*(\S)", r" -- \1", blocks[i])  # 格式化参数描述，添加破折号
    
    # 将所有块合并成一个字符串，并返回结果
    return "\n".join(blocks)
def _prepare_output_docstrings(output_type, config_class, min_indent=None):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    # 获取输出类型的文档字符串
    output_docstring = output_type.__doc__

    # 从文档字符串中提取参数部分
    lines = output_docstring.split("\n")
    i = 0
    # 寻找并移除文档字符串头部，保留参数列表
    while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
        i += 1
    if i < len(lines):
        params_docstring = "\n".join(lines[(i + 1) :])
        # 转换参数文档字符串的格式
        params_docstring = _convert_output_args_doc(params_docstring)
    else:
        # 如果找不到 `Args` 或 `Parameters` 部分，抛出异常
        raise ValueError(
            f"No `Args` or `Parameters` section is found in the docstring of `{output_type.__name__}`. Make sure it has "
            "docstring and contain either `Args` or `Parameters`."
        )

    # 添加返回值部分的介绍
    full_output_type = f"{output_type.__module__}.{output_type.__name__}"
    # 根据输出类型的名称选择介绍文本
    intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith("TF") else PT_RETURN_INTRODUCTION
    # 格式化介绍文本，包括输出类型和配置类
    intro = intro.format(full_output_type=full_output_type, config_class=config_class)
    # 将介绍文本与参数文档字符串合并
    result = intro + params_docstring

    # 如果指定了最小缩进量，则应用该缩进
    if min_indent is not None:
        lines = result.split("\n")
        i = 0
        # 查找第一个非空行的缩进量
        while len(lines[i]) == 0:
            i += 1
        indent = len(_get_indent(lines[i]))
        # 如果缩进量过小，则对所有非空行添加缩进
        if indent < min_indent:
            to_add = " " * (min_indent - indent)
            lines = [(f"{to_add}{line}" if len(line) > 0 else line) for line in lines]
            result = "\n".join(lines)

    return result
    # 将预测的标记类别 ID 赋值给变量 labels
    labels = predicted_token_class_ids
    # 使用模型处理输入数据和标签，计算损失值
    loss = model(**inputs, labels=labels).loss
    # 获取损失值的数值，并保留两位小数
    round(loss.item(), 2)
"""

PT_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    >>> inputs = tokenizer(question, text, return_tensors="pt")
    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    >>> answer_start_index = outputs.start_logits.argmax()
    >>> answer_end_index = outputs.end_logits.argmax()

    >>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    >>> tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    {expected_output}

    >>> # target is "nice puppet"
    >>> target_start_index = torch.tensor([{qa_target_start_index}])
    >>> target_end_index = torch.tensor([{qa_target_end_index}])

    >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
    >>> loss = outputs.loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""

PT_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example of single-label classification:

    ```
    >>> import torch
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_class_id = logits.argmax().item()
    >>> model.config.id2label[predicted_class_id]
    {expected_output}

    >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    >>> num_labels = len(model.config.id2label)
    >>> model = {model_class}.from_pretrained("{checkpoint}", num_labels=num_labels)

    >>> labels = torch.tensor([1])
    >>> loss = model(**inputs, labels=labels).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```

    Example of multi-label classification:

    ```
    >>> import torch
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", problem_type="multi_label_classification")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

    >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    >>> num_labels = len(model.config.id2label)
    >>> model = {model_class}.from_pretrained("{checkpoint}", num_labels=num_labels)
    # 调用模型的预测方法，传入输入数据和检查点参数，返回预测结果
    >>> labels = torch.sum(
        # 使用 torch.nn.functional.one_hot 函数将预测的类别转换为独热编码
        torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels),
        # 在维度1上求和，得到每个样本的类别标签向量
        dim=1
    ).to(torch.float)
    # 调用模型的训练方法，传入输入数据和预测的标签，返回损失值
    >>> loss = model(**inputs, labels=labels).loss
"""

PT_MASKED_LM_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> # retrieve index of {mask}
    >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    >>> tokenizer.decode(predicted_token_id)
    {expected_output}

    >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
    >>> # mask labels of non-{mask} tokens
    >>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

    >>> outputs = model(**inputs, labels=labels)
    >>> round(outputs.loss.item(), 2)
    {expected_loss}
    ```
"""

PT_BASE_MODEL_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

PT_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."
    >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
    >>> outputs = model(**{{k: v.unsqueeze(0) for k, v in encoding.items()}}, labels=labels)  # batch size is 1

    >>> # the linear classifier still needs to be trained
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""

PT_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```
    >>> import torch
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs, labels=inputs["input_ids"])
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""

PT_SPEECH_BASE_MODEL_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoProcessor, {model_class}
    >>> import torch
    >>> from datasets import load_dataset
    
    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    # 加载指定的数据集，这里是从 Hugging Face 内部测试数据集中加载 LibriSpeech ASR Demo 的干净验证集数据
    
    >>> dataset = dataset.sort("id")
    # 按照数据集中的 "id" 列对数据集进行排序
    
    >>> sampling_rate = dataset.features["audio"].sampling_rate
    # 获取数据集中音频特征的采样率
    
    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    # 使用预训练的 AutoProcessor 加载指定的检查点
    
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    # 使用预训练的模型类加载指定的检查点
    
    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    # 在使用处理器时，音频文件会动态解码，这里将音频数据作为输入，返回 PyTorch 张量
    
    >>> with torch.no_grad():
    ...     outputs = model(**inputs)
    # 在不计算梯度的上下文中，使用模型进行推断，输入为处理后的数据
    
    >>> last_hidden_states = outputs.last_hidden_state
    # 从模型的输出中获取最后一层隐藏状态
    
    >>> list(last_hidden_states.shape)
    {expected_output}
    # 输出最后隐藏状态的形状
PT_SPEECH_FRAME_CLASS_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoFeatureExtractor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly


注释：
# 导入必要的库和模块，例如从transformers库中导入AutoFeatureExtractor和模型类（model_class），从datasets库中导入load_dataset函数，导入torch库
# 加载LibriSpeech ASR演示数据集的验证集（"clean"），并按ID排序
# 获取数据集中音频特征的采样率
# 从预训练的模型检查点中实例化AutoFeatureExtractor对象和模型对象
# 在使用时，音频文件会实时解码
    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
    # 使用特征提取器从数据集中的第一个音频样本提取特征，并返回PyTorch张量，使用给定的采样率
    
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits
    # 使用torch.no_grad()上下文管理器，禁止梯度计算，计算模型对输入数据的logits（未经softmax的预测值）
    
    >>> probabilities = torch.sigmoid(logits[0])
    # 计算logits的sigmoid函数值，得到预测概率
    
    >>> # labels is a one-hot array of shape (num_frames, num_speakers)
    >>> labels = (probabilities > 0.5).long()
    # 基于概率值大于0.5的条件，将其转换为long整数类型的标签数组，表示每帧对应的发言者
    
    >>> labels[0].tolist()
    # 将第一帧的标签数组转换为Python列表形式，并输出
    {expected_output}
# 示例代码段，展示如何使用预训练模型进行音频向量提取和相似性比较

"""
PT_SPEECH_XVECTOR_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoFeatureExtractor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(
    ...     [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
    ... )
    >>> with torch.no_grad():
    ...     embeddings = model(**inputs).embeddings

    >>> embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

    >>> # the resulting embeddings can be used for cosine similarity-based retrieval
    >>> cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    >>> similarity = cosine_sim(embeddings[0], embeddings[1])
    >>> threshold = 0.7  # the optimal threshold is dataset-dependent
    >>> if similarity < threshold:
    ...     print("Speakers are not the same!")
    >>> round(similarity.item(), 2)
    {expected_output}
    ```
"""


PT_VISION_BASE_MODEL_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> import torch
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="pt")

    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

PT_VISION_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> import torch
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_label = logits.argmax(-1).item()
    >>> print(model.config.id2label[predicted_label])
    {expected_output}
    ```
"""


PT_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": PT_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": PT_QUESTION_ANSWERING_SAMPLE,
}
    "TokenClassification": PT_TOKEN_CLASSIFICATION_SAMPLE,
    # 定义键为 "TokenClassification"，值为 PT_TOKEN_CLASSIFICATION_SAMPLE 的样本

    "MultipleChoice": PT_MULTIPLE_CHOICE_SAMPLE,
    # 定义键为 "MultipleChoice"，值为 PT_MULTIPLE_CHOICE_SAMPLE 的样本

    "MaskedLM": PT_MASKED_LM_SAMPLE,
    # 定义键为 "MaskedLM"，值为 PT_MASKED_LM_SAMPLE 的样本

    "LMHead": PT_CAUSAL_LM_SAMPLE,
    # 定义键为 "LMHead"，值为 PT_CAUSAL_LM_SAMPLE 的样本

    "BaseModel": PT_BASE_MODEL_SAMPLE,
    # 定义键为 "BaseModel"，值为 PT_BASE_MODEL_SAMPLE 的样本

    "SpeechBaseModel": PT_SPEECH_BASE_MODEL_SAMPLE,
    # 定义键为 "SpeechBaseModel"，值为 PT_SPEECH_BASE_MODEL_SAMPLE 的样本

    "CTC": PT_SPEECH_CTC_SAMPLE,
    # 定义键为 "CTC"，值为 PT_SPEECH_CTC_SAMPLE 的样本

    "AudioClassification": PT_SPEECH_SEQ_CLASS_SAMPLE,
    # 定义键为 "AudioClassification"，值为 PT_SPEECH_SEQ_CLASS_SAMPLE 的样本

    "AudioFrameClassification": PT_SPEECH_FRAME_CLASS_SAMPLE,
    # 定义键为 "AudioFrameClassification"，值为 PT_SPEECH_FRAME_CLASS_SAMPLE 的样本

    "AudioXVector": PT_SPEECH_XVECTOR_SAMPLE,
    # 定义键为 "AudioXVector"，值为 PT_SPEECH_XVECTOR_SAMPLE 的样本

    "VisionBaseModel": PT_VISION_BASE_MODEL_SAMPLE,
    # 定义键为 "VisionBaseModel"，值为 PT_VISION_BASE_MODEL_SAMPLE 的样本

    "ImageClassification": PT_VISION_SEQ_CLASS_SAMPLE,
    # 定义键为 "ImageClassification"，值为 PT_VISION_SEQ_CLASS_SAMPLE 的样本
}

TF_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer(
    ...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="tf"
    ... )

    >>> logits = model(**inputs).logits
    >>> predicted_token_class_ids = tf.math.argmax(logits, axis=-1)

    >>> # Note that tokens are classified rather then input words which means that
    >>> # there might be more predicted token classes than words.
    >>> # Multiple token classes might account for the same word
    >>> predicted_tokens_classes = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
    >>> predicted_tokens_classes
    {expected_output}
    ```

    ```
    >>> labels = predicted_token_class_ids
    >>> loss = tf.math.reduce_mean(model(**inputs, labels=labels).loss)
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    >>> inputs = tokenizer(question, text, return_tensors="tf")
    >>> outputs = model(**inputs)

    >>> answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    >>> answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

    >>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    >>> tokenizer.decode(predict_answer_tokens)
    {expected_output}
    ```

    ```
    >>> # target is "nice puppet"
    >>> target_start_index = tf.constant([{qa_target_start_index}])
    >>> target_end_index = tf.constant([{qa_target_end_index}])

    >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
    >>> loss = tf.math.reduce_mean(outputs.loss)
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

    >>> logits = model(**inputs).logits

    >>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
    >>> model.config.id2label[predicted_class_id]
    {expected_output}
    ```

    ```
    # Placeholder for additional code related to sequence classification if needed
    # 获取模型配置中标签的数量，以确定需要训练的类别数
    num_labels = len(model.config.id2label)
    
    # 使用预训练的模型类 `{model_class}` 从指定的检查点 `"{checkpoint}"` 加载模型，
    # 并设置分类任务中的类别数为 `num_labels`
    model = {model_class}.from_pretrained("{checkpoint}", num_labels=num_labels)
    
    # 创建一个 TensorFlow 常量张量 `labels`，其中值为 1，表示输入数据的真实标签
    labels = tf.constant(1)
    
    # 使用模型进行前向传播计算损失，通过传入 `inputs` 作为输入数据，同时指定真实标签 `labels`，
    # 获取计算得到的损失值 `loss`
    loss = model(**inputs, labels=labels).loss
    
    # 将损失值转换为浮点数，并保留两位小数，返回结果
    round(float(loss), 2)
    {expected_loss}
# 定义了一个字符串常量 TF_MASKED_LM_SAMPLE，包含了一个示例的Transformer模型使用案例，展示了如何使用Masked Language Modeling功能。
TF_MASKED_LM_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="tf")
    >>> logits = model(**inputs).logits

    >>> # retrieve index of {mask}
    >>> mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[0])
    >>> selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)

    >>> predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
    >>> tokenizer.decode(predicted_token_id)
    {expected_output}
    ```

    ```
    >>> labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
    >>> # mask labels of non-{mask} tokens
    >>> labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

    >>> outputs = model(**inputs, labels=labels)
    >>> round(float(outputs.loss), 2)
    {expected_loss}
    ```
"""

# 定义了一个字符串常量 TF_BASE_MODEL_SAMPLE，包含了一个示例的Transformer模型使用案例，展示了基础模型的输入输出操作。
TF_BASE_MODEL_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> outputs = model(inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

# 定义了一个字符串常量 TF_MULTIPLE_CHOICE_SAMPLE，包含了一个示例的Transformer模型使用案例，展示了多项选择任务的处理方式。
TF_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."

    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="tf", padding=True)
    >>> inputs = {{k: tf.expand_dims(v, 0) for k, v in encoding.items()}}
    >>> outputs = model(inputs)  # batch size is 1

    >>> # the linear classifier still needs to be trained
    >>> logits = outputs.logits
    ```
"""

# 定义了一个字符串常量 TF_CAUSAL_LM_SAMPLE，包含了一个示例的Transformer模型使用案例，展示了因果语言建模任务的操作流程。
TF_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> outputs = model(inputs)
    >>> logits = outputs.logits
    ```
"""

# 定义了一个字符串常量 TF_SPEECH_BASE_MODEL_SAMPLE，包含了一个示例的Transformer模型使用案例，展示了语音处理基础模型的初始化。
TF_SPEECH_BASE_MODEL_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoProcessor, {model_class}
    >>> from datasets import load_dataset  # 导入加载数据集的函数
    
    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")  # 按照"id"字段对数据集进行排序
    >>> sampling_rate = dataset.features["audio"].sampling_rate  # 获取音频数据的采样率
    
    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")  # 从预训练模型加载音频处理器
    >>> model = {model_class}.from_pretrained("{checkpoint}")  # 从预训练模型加载模型
    
    >>> # 在线解码音频文件
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="tf")
    >>> outputs = model(**inputs)  # 使用模型进行推断
    
    >>> last_hidden_states = outputs.last_hidden_state  # 获取输出中的最后隐藏状态
    >>> list(last_hidden_states.shape)  # 输出最后隐藏状态的形状，这里是期望的输出
"""

TF_SPEECH_CTC_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoProcessor, {model_class}  # 导入自动处理器和模型类
    >>> from datasets import load_dataset  # 导入数据集加载函数
    >>> import tensorflow as tf  # 导入 TensorFlow 库

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")  # 加载数据集
    >>> dataset = dataset.sort("id")  # 按照 ID 排序数据集
    >>> sampling_rate = dataset.features["audio"].sampling_rate  # 获取音频采样率

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")  # 根据预训练模型加载自动处理器
    >>> model = {model_class}.from_pretrained("{checkpoint}")  # 根据预训练模型加载模型

    >>> # audio file is decoded on the fly  # 在线解码音频文件
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="tf")  # 处理输入数据并转换为 TensorFlow 张量
    >>> logits = model(**inputs).logits  # 获取模型预测的 logits
    >>> predicted_ids = tf.math.argmax(logits, axis=-1)  # 获取预测的标签索引

    >>> # transcribe speech  # 转录语音
    >>> transcription = processor.batch_decode(predicted_ids)  # 批量解码预测结果
    >>> transcription[0]  # 输出第一个样本的转录文本
    {expected_output}
    ```

    ```
    >>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="tf").input_ids  # 处理文本输入并转换为 TensorFlow 张量

    >>> # compute loss  # 计算损失
    >>> loss = model(**inputs).loss  # 计算模型的损失
    >>> round(float(loss), 2)  # 将损失值四舍五入到两位小数并输出
    {expected_loss}
    ```
"""

TF_VISION_BASE_MODEL_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoImageProcessor, {model_class}  # 导入自动图像处理器和模型类
    >>> from datasets import load_dataset  # 导入数据集加载函数

    >>> dataset = load_dataset("huggingface/cats-image")  # 加载图像数据集
    >>> image = dataset["test"]["image"][0]  # 获取测试集中的第一张图像

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")  # 根据预训练模型加载自动图像处理器
    >>> model = {model_class}.from_pretrained("{checkpoint}")  # 根据预训练模型加载模型

    >>> inputs = image_processor(image, return_tensors="tf")  # 处理图像输入并转换为 TensorFlow 张量
    >>> outputs = model(**inputs)  # 获取模型的输出结果

    >>> last_hidden_states = outputs.last_hidden_state  # 获取模型的最后隐藏状态
    >>> list(last_hidden_states.shape)  # 输出最后隐藏状态的形状
    {expected_output}
    ```
"""

TF_VISION_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoImageProcessor, {model_class}  # 导入自动图像处理器和模型类
    >>> import tensorflow as tf  # 导入 TensorFlow 库
    >>> from datasets import load_dataset  # 导入数据集加载函数

    >>> dataset = load_dataset("huggingface/cats-image")  # 加载图像数据集
    >>> image = dataset["test"]["image"][0]  # 获取测试集中的第一张图像

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")  # 根据预训练模型加载自动图像处理器
    >>> model = {model_class}.from_pretrained("{checkpoint}")  # 根据预训练模型加载模型

    >>> inputs = image_processor(image, return_tensors="tf")  # 处理图像输入并转换为 TensorFlow 张量
    >>> logits = model(**inputs).logits  # 获取模型预测的 logits

    >>> # model predicts one of the 1000 ImageNet classes  # 模型预测其中一个 1000 个 ImageNet 类别
    >>> predicted_label = int(tf.math.argmax(logits, axis=-1))  # 获取预测标签的索引
    >>> print(model.config.id2label[predicted_label])  # 输出预测标签对应的类别名称
    {expected_output}
    ```
"""

TF_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": TF_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": TF_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": TF_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": TF_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": TF_MASKED_LM_SAMPLE,
    "LMHead": TF_CAUSAL_LM_SAMPLE,
    "BaseModel": TF_BASE_MODEL_SAMPLE,
    "SpeechBaseModel": TF_SPEECH_BASE_MODEL_SAMPLE,
}
    # 将字符串"TF_SPEECH_CTC_SAMPLE"赋值给键"CTC"
    "CTC": TF_SPEECH_CTC_SAMPLE,
    # 将字符串"TF_VISION_BASE_MODEL_SAMPLE"赋值给键"VisionBaseModel"
    "VisionBaseModel": TF_VISION_BASE_MODEL_SAMPLE,
    # 将字符串"TF_VISION_SEQ_CLASS_SAMPLE"赋值给键"ImageClassification"
    "ImageClassification": TF_VISION_SEQ_CLASS_SAMPLE,
FLAX_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    # 使用给定的检查点加载预训练的分词器
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    # 使用给定的检查点加载预训练的模型

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
    # 使用分词器对输入文本进行编码，并返回JAX张量
    >>> outputs = model(**inputs)
    # 对编码后的输入文本进行模型推理

    >>> last_hidden_states = outputs.last_hidden_state
    # 提取模型输出中的最终隐藏状态
    ```
"""
    # 从transformers库中导入AutoTokenizer类和指定的模型类（这里需要替换为实际的模型类名）
    >>> from transformers import AutoTokenizer, {model_class}

    # 使用指定的checkpoint加载预训练的分词器（tokenizer）
    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")

    # 使用指定的checkpoint加载预训练的模型（需要替换为实际的模型类名）
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    # 使用分词器处理输入文本，并将结果转换为NumPy格式的张量（Tensor）
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")

    # 使用加载的模型进行推理，传入处理过的输入，并获取模型的输出
    >>> outputs = model(**inputs)

    # 从模型的输出中提取出下一个预测的token的logits（logits是对应于每个词汇表中单词的分数）
    >>> next_token_logits = outputs.logits[:, -1]
"""

FLAX_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": FLAX_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": FLAX_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": FLAX_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": FLAX_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": FLAX_MASKED_LM_SAMPLE,
    "BaseModel": FLAX_BASE_MODEL_SAMPLE,
    "LMHead": FLAX_CAUSAL_LM_SAMPLE,
}

# 定义一个函数，用于过滤示例中输出为 `None` 的行
def filter_outputs_from_example(docstring, **kwargs):
    """
    Removes the lines testing an output with the doctest syntax in a code sample when it's set to `None`.
    """
    # 遍历关键字参数
    for key, value in kwargs.items():
        # 如果值不为 None，则跳过
        if value is not None:
            continue

        # 构建匹配示例输出的正则表达式
        doc_key = "{" + key + "}"
        # 使用正则表达式替换文档字符串中匹配的内容为空行
        docstring = re.sub(rf"\n([^\n]+)\n\s+{doc_key}\n", "\n", docstring)

    return docstring


# 定义一个装饰器函数，用于添加代码示例的文档字符串
def add_code_sample_docstrings(
    *docstr,
    processor_class=None,
    checkpoint=None,
    output_type=None,
    config_class=None,
    mask="[MASK]",
    qa_target_start_index=14,
    qa_target_end_index=15,
    model_cls=None,
    modality=None,
    expected_output=None,
    expected_loss=None,
    real_checkpoint=None,
    revision=None,
):
    return docstring_decorator


# 定义一个函数，用于替换返回值相关的文档字符串
def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        # 获取函数的文档字符串
        func_doc = fn.__doc__
        lines = func_doc.split("\n")
        i = 0
        # 查找第一个以 'Returns:' 或 'Return:' 开头的行
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            # 获取缩进
            indent = len(_get_indent(lines[i]))
            # 替换找到的 'Returns:' 行为预设的输出文档字符串
            lines[i] = _prepare_output_docstrings(output_type, config_class, min_indent=indent)
            func_doc = "\n".join(lines)
        else:
            # 如果找不到 'Returns:' 或 'Return:' 行，则抛出异常
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, "
                f"current docstring is:\n{func_doc}"
            )
        # 更新函数的文档字符串
        fn.__doc__ = func_doc
        return fn

    return docstring_decorator


# 定义一个函数，用于复制函数 f 并返回其副本
def copy_func(f):
    """Returns a copy of a function f."""
    # 基于 http://stackoverflow.com/a/6528148/190597 (Glenn Maynard) 的实现
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
```