# `.\transformers\utils\doc.py`

```
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意，分发的软件是基于"按原样"的基础分发的
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息

"""
Doc utilities: Utilities related to documentation
"""

# 导入必要的库
import functools
import re
import types

# 添加起始文档字符串的装饰器
def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        # 将传入的文档字符串与函数原有的文档字符串合并
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator

# 添加起始文档字符串到模型前向方法的装饰器
def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        # 合并传入的文档字符串与函数原有的文档字符串
        docstring = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        class_name = f"[`{fn.__qualname__.split('.')[0]}`]"
        intro = f"   The {class_name} forward method, overrides the `__call__` special method."
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

# 添加结束文档字符串的装饰器
def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        # 将传入的文档字符串与函数原有的文档字符串合并
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator

# PyTorch 返回说明文本
PT_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(torch.FloatTensor)`: A [`{full_output_type}`] or a tuple of
        `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
        elements depending on the configuration ([`{config_class}`]) and inputs.

"""

# TensorFlow 返回说明文本
TF_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(tf.Tensor)`: A [`{full_output_type}`] or a tuple of `tf.Tensor` (if
        `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the
        configuration ([`{config_class}`]) and inputs.

"""

# 获取缩进
def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search(r"^(\s*)\S", t)
    return "" if search is None else search.groups()[0]

# 转换输出参数文档以正确显示
def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    # 分割参数/描述块
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ""
    # 遍历输出参数文档的每一行
    for line in output_args_doc.split("\n"):
        # 如果缩进与开头相同，说明该行是新参数的名称
        if _get_indent(line) == indent:
            # 如果当前块不为空，将其添加到块列表中
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f"{line}\n"
        else:
            # 否则该行是当前参数描述的一部分
            # 需要将缩进减少2个空格
            current_block += f"{line[2:]}\n"
    blocks.append(current_block[:-1])

    # 格式化每个块以便正确渲染
    for i in range(len(blocks)):
        # 使用正则表达式将参数名称加粗
        blocks[i] = re.sub(r"^(\s+)(\S+)(\s+)", r"\1- **\2**\3", blocks[i])
        # 使用正则表达式添加参数描述的分隔符
        blocks[i] = re.sub(r":\s*\n\s*(\S)", r" -- \1", blocks[i])

    # 返回格式化后的块组成的字符串
    return "\n".join(blocks)
def _prepare_output_docstrings(output_type, config_class, min_indent=None):
    """
    准备文档字符串的返回部分，使用 `output_type`。
    """
    # 获取输出类型的文档字符串
    output_docstring = output_type.__doc__

    # 移除文档字符串的头部，保留参数列表
    lines = output_docstring.split("\n")
    i = 0
    while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
        i += 1
    if i < len(lines):
        params_docstring = "\n".join(lines[(i + 1) :])
        params_docstring = _convert_output_args_doc(params_docstring)
    else:
        raise ValueError(
            f"No `Args` or `Parameters` section is found in the docstring of `{output_type.__name__}`. Make sure it has "
            "docstring and contain either `Args` or `Parameters`."
        )

    # 添加返回值介绍
    full_output_type = f"{output_type.__module__}.{output_type.__name__}"
    intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith("TF") else PT_RETURN_INTRODUCTION
    intro = intro.format(full_output_type=full_output_type, config_class=config_class)
    result = intro + params_docstring

    # 如果需要，应用最小缩进
    if min_indent is not None:
        lines = result.split("\n")
        # 找到第一行非空行的缩进
        i = 0
        while len(lines[i]) == 0:
            i += 1
        indent = len(_get_indent(lines[i]))
        # 如果太小，对所有非空行添加缩进
        if indent < min_indent:
            to_add = " " * (min_indent - indent)
            lines = [(f"{to_add}{line}" if len(line) > 0 else line) for line in lines]
            result = "\n".join(lines)

    return result


FAKE_MODEL_DISCLAIMER = """
    <Tip warning={true}>

    This example uses a random model as the real ones are all very big. To get proper results, you should use
    {real_checkpoint} instead of {fake_checkpoint}. If you get out-of-memory when loading that checkpoint, you can try
    adding `device_map="auto"` in the `from_pretrained` call.

    </Tip>
"""

PT_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer(
    ...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
    ... )

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_token_class_ids = logits.argmax(-1)

    >>> # Note that tokens are classified rather then input words which means that
    >>> # there might be more predicted token classes than words.
    >>> # Multiple token classes might account for the same word
    >>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
    >>> predicted_tokens_classes
    # 将预测的标记类别ID赋值给labels变量
    labels = predicted_token_class_ids
    # 使用模型对输入数据进行预测，并计算损失值
    loss = model(**inputs, labels=labels).loss
    # 对损失值进行四舍五入保留两位小数
    round(loss.item(), 2)
    # 返回损失值
    {expected_loss}
"""

PT_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
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

    ```python
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

    ```python
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



注释：
    # 使用字符串格式化将变量checkpoint插入到字符串中
    "{checkpoint}", num_labels=num_labels, problem_type="multi_label_classification"
    # 使用torch.sum计算预测类别的one-hot编码
    labels = torch.sum(
        torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
    ).to(torch.float)
    # 使用模型进行训练，传入输入和标签，获取损失值
    loss = model(**inputs, labels=labels).loss
# 定义一个字符串常量，包含示例代码片段，用于展示如何使用预训练模型进行掩码语言建模
PT_MASKED_LM_SAMPLE = r"""
    Example:

    ```python
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

# 定义一个字符串常量，包含示例代码片段，用于展示如何使用预训练模型进行基础模型操作
PT_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

# 定义一个字符串常量，包含示例代码片段，用于展示如何使用预训练模型进行多项选择任务
PT_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```python
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

# 定义一个字符串常量，包含示例代码片段，用于展示如何使用预训练模型进行因果语言建模
PT_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```python
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

# 定义一个字符串常量，包含示例代码片段，用于展示如何使用预训练模型进行语音基础模型操作
PT_SPEECH_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    # 导入所需的库
    from transformers import AutoProcessor, {model_class}
    import torch
    from datasets import load_dataset
    
    # 加载数据集
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    # 根据"id"字段对数据集进行排序
    dataset = dataset.sort("id")
    # 获取采样率
    sampling_rate = dataset.features["audio"].sampling_rate
    
    # 加载预训练的处理器和模型
    processor = AutoProcessor.from_pretrained("{checkpoint}")
    model = {model_class}.from_pretrained("{checkpoint}")
    
    # 对音频文件进行即时解码
    inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    # 禁用梯度计算
    with torch.no_grad():
        # 将输入传递给模型并获取输出
        outputs = model(**inputs)
    
    # 获取最后一层隐藏状态
    last_hidden_states = outputs.last_hidden_state
    # 打印最后隐藏状态的形状
    list(last_hidden_states.shape)
"""

PT_SPEECH_CTC_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits
    >>> predicted_ids = torch.argmax(logits, dim=-1)

    >>> # transcribe speech
    >>> transcription = processor.batch_decode(predicted_ids)
    >>> transcription[0]
    {expected_output}

    >>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

    >>> # compute loss
    >>> loss = model(**inputs).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""

PT_SPEECH_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoFeatureExtractor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_class_ids = torch.argmax(logits, dim=-1).item()
    >>> predicted_label = model.config.id2label[predicted_class_ids]
    >>> predicted_label
    {expected_output}

    >>> # compute loss - target_label is e.g. "down"
    >>> target_label = model.config.id2label[0]
    >>> inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
    >>> loss = model(**inputs).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""


PT_SPEECH_FRAME_CLASS_SAMPLE = r"""
    Example:

    ```python
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
    # 使用特征提取器从音频数组中提取特征，并返回PyTorch张量
    inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用模型进行推理，得到logits
        logits = model(**inputs).logits

    # 对logits进行sigmoid操作，得到概率值
    probabilities = torch.sigmoid(logits[0])
    # labels是一个形状为(num_frames, num_speakers)的one-hot数组
    # 将概率值大于0.5的设为1，小于等于0.5的设为0，并转换为整数类型
    labels = (probabilities > 0.5).long()
    # 将第一个帧的标签转换为列表形式并输出
    labels[0].tolist()
    {expected_output}
# PyTorch示例文档字符串，包含有关语音x-vector模型的示例
PT_SPEECH_XVECTOR_SAMPLE = r"""
    Example:

    ```python
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



# PyTorch示例文档字符串，包含有关视觉基础模型的示例
PT_VISION_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
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



# PyTorch示例文档字符串，包含有关视觉序列分类模型的示例
PT_VISION_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```python
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
    # 定义不同任务类型对应的示例模型
    "TokenClassification": PT_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": PT_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": PT_MASKED_LM_SAMPLE,
    "LMHead": PT_CAUSAL_LM_SAMPLE,
    "BaseModel": PT_BASE_MODEL_SAMPLE,
    "SpeechBaseModel": PT_SPEECH_BASE_MODEL_SAMPLE,
    "CTC": PT_SPEECH_CTC_SAMPLE,
    "AudioClassification": PT_SPEECH_SEQ_CLASS_SAMPLE,
    "AudioFrameClassification": PT_SPEECH_FRAME_CLASS_SAMPLE,
    "AudioXVector": PT_SPEECH_XVECTOR_SAMPLE,
    "VisionBaseModel": PT_VISION_BASE_MODEL_SAMPLE,
    "ImageClassification": PT_VISION_SEQ_CLASS_SAMPLE,
# 结束代码块
}

# 定义一个包含示例的字符串常量，用于展示如何使用模型进行标记分类任务
TF_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
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

    ```python
    >>> labels = predicted_token_class_ids
    >>> loss = tf.math.reduce_mean(model(**inputs, labels=labels).loss)
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

# 定义一个包含示例的字符串常量，用于展示如何使用模型进行问答任务
TF_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
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

    ```python
    >>> # target is "nice puppet"
    >>> target_start_index = tf.constant([{qa_target_start_index}])
    >>> target_end_index = tf.constant([{qa_target_end_index}])

    >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
    >>> loss = tf.math.reduce_mean(outputs.loss)
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

# 定义一个包��示例的字符串常量，用于展示如何使用模型进行序列分类任务
TF_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
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
    # 获取模型配置中标签的数量
    num_labels = len(model.config.id2label)
    # 从预训练模型中加载模型，并指定类别数量为num_labels
    model = {model_class}.from_pretrained("{checkpoint}", num_labels=num_labels)

    # 创建一个常量张量，代表标签为1
    labels = tf.constant(1)
    # 使用模型计算损失，传入输入和标签信息，获取损失值
    loss = model(**inputs, labels=labels).loss
    # 将损失值转换为浮点数并保留两位小数
    round(float(loss), 2)
    # 期望的损失值
    {expected_loss}
# Transformer 模型的掩码语言建模示例
TF_MASKED_LM_SAMPLE = r"""
    Example:

    ```python
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

    ```python
    >>> labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
    >>> # mask labels of non-{mask} tokens
    >>> labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

    >>> outputs = model(**inputs, labels=labels)
    >>> round(float(outputs.loss), 2)
    {expected_loss}
    ```
"""

# Transformer 模型的基础模型示例
TF_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> outputs = model(inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

# Transformer 模型的多项选择示例
TF_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```python
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

# Transformer 模型��因果语言建模示例
TF_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> outputs = model(inputs)
    >>> logits = outputs.logits
    ```
"""

# Transformer 模型的语音基础模型示例
TF_SPEECH_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    # 导入 load_dataset 函数从 datasets 模块中
    from datasets import load_dataset
    
    # 加载名为 "hf-internal-testing/librispeech_asr_demo" 的数据集的验证集部分
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    # 根据 "id" 列对数据集进行排序
    dataset = dataset.sort("id")
    # 获取音频特征中的采样率
    sampling_rate = dataset.features["audio"].sampling_rate
    
    # 从预训练模型中加载处理器
    processor = AutoProcessor.from_pretrained("{checkpoint}")
    # 从预训练模型中加载模型
    model = {model_class}.from_pretrained("{checkpoint}")
    
    # 在处理器中对音频文件进行即时解码
    inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="tf")
    # 使用模型处理输入数据
    outputs = model(**inputs)
    
    # 获取模型输出中的最后一个隐藏状态
    last_hidden_states = outputs.last_hidden_state
    # 打印最后一个隐藏状态的形状
    list(last_hidden_states.shape)
    {expected_output}
"""

TF_SPEECH_CTC_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> from datasets import load_dataset
    >>> import tensorflow as tf

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="tf")
    >>> logits = model(**inputs).logits
    >>> predicted_ids = tf.math.argmax(logits, axis=-1)

    >>> # transcribe speech
    >>> transcription = processor.batch_decode(predicted_ids)
    >>> transcription[0]
    {expected_output}
    ```

    ```python
    >>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="tf").input_ids

    >>> # compute loss
    >>> loss = model(**inputs).loss
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_VISION_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="tf")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

TF_VISION_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> import tensorflow as tf
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="tf")
    >>> logits = model(**inputs).logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_label = int(tf.math.argmax(logits, axis=-1))
    >>> print(model.config.id2label[predicted_label])
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


注释：这部分代码定义了一些 TensorFlow 模型的示例代码，包括语音识别、图像处理和序列分类等。示例代码展示了如何使用 Transformers 库加载模型、处理数据和进行推理。
    # 将字符串"CTC"映射到TF_SPEECH_CTC_SAMPLE常量
    "CTC": TF_SPEECH_CTC_SAMPLE,
    # 将字符串"VisionBaseModel"映射到TF_VISION_BASE_MODEL_SAMPLE常量
    "VisionBaseModel": TF_VISION_BASE_MODEL_SAMPLE,
    # 将字符串"ImageClassification"映射到TF_VISION_SEQ_CLASS_SAMPLE常量
    "ImageClassification": TF_VISION_SEQ_CLASS_SAMPLE,
# 定义 FLAX_TOKEN_CLASSIFICATION_SAMPLE 字符串，包含示例代码和说明
FLAX_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""

# 定义 FLAX_QUESTION_ANSWERING_SAMPLE 字符串，包含示例代码和说明
FLAX_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> inputs = tokenizer(question, text, return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> start_scores = outputs.start_logits
    >>> end_scores = outputs.end_logits
    ```
"""

# 定义 FLAX_SEQUENCE_CLASSIFICATION_SAMPLE 字符串，包含示例代码和说明
FLAX_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""

# 定义 FLAX_MASKED_LM_SAMPLE 字符串，包含示例代码和说明
FLAX_MASKED_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""

# 定义 FLAX_BASE_MODEL_SAMPLE 字符串，包含示例代码和说明
FLAX_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

# 定义 FLAX_MULTIPLE_CHOICE_SAMPLE 字符串，包含示例代码和说明
FLAX_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."

    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="jax", padding=True)
    >>> outputs = model(**{{k: v[None, :] for k, v in encoding.items()}})

    >>> logits = outputs.logits
    ```
"""

# 定义 FLAX_CAUSAL_LM_SAMPLE 字符串，包含示例代码和说明
FLAX_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```python
    # 从transformers库中导入AutoTokenizer类和{model_class}类
    from transformers import AutoTokenizer, {model_class}
    
    # 使用预训练的模型检查点名称实例化一个AutoTokenizer对象
    tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    # 使用预训练的模型检查点名称实例化一个{model_class}对象
    model = {model_class}.from_pretrained("{checkpoint}")
    
    # 使用tokenizer对输入文本进行编码，返回numpy数组格式的张量
    inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
    # 使用model对编码后的输入进行推理，得到输出结果
    outputs = model(**inputs)
    
    # 提取下一个标记的logits（对数概率）
    next_token_logits = outputs.logits[:, -1]
# 定义一个包含不同模型类型和对应示例代码的字典
FLAX_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": FLAX_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": FLAX_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": FLAX_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": FLAX_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": FLAX_MASKED_LM_SAMPLE,
    "BaseModel": FLAX_BASE_MODEL_SAMPLE,
    "LMHead": FLAX_CAUSAL_LM_SAMPLE,
}

# 过滤示例代码中使用 doctest 语法测试输出的行，当输出为 `None` 时
def filter_outputs_from_example(docstring, **kwargs):
    """
    Removes the lines testing an output with the doctest syntax in a code sample when it's set to `None`.
    """
    for key, value in kwargs.items():
        if value is not None:
            continue

        doc_key = "{" + key + "}"
        docstring = re.sub(rf"\n([^\n]+)\n\s+{doc_key}\n", "\n", docstring)

    return docstring

# 添加代码示例的文档字符串
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

# 替换返回值的文档字符串
def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        func_doc = fn.__doc__
        lines = func_doc.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = len(_get_indent(lines[i]))
            lines[i] = _prepare_output_docstrings(output_type, config_class, min_indent=indent)
            func_doc = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, "
                f"current docstring is:\n{func_doc}"
            )
        fn.__doc__ = func_doc
        return fn

    return docstring_decorator

# 复制函数 f 并返回其副本
def copy_func(f):
    """Returns a copy of a function f."""
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
```