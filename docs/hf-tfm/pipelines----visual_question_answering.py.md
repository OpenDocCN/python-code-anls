# `.\pipelines\visual_question_answering.py`

```py
from typing import Union

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging
# 导入必要的模块和函数

from .base import Pipeline, build_pipeline_init_args
# 从当前目录下的base模块导入Pipeline类和build_pipeline_init_args函数

if is_vision_available():
    from PIL import Image
    # 如果PIL库可用，则从PIL模块导入Image类

    from ..image_utils import load_image
    # 导入load_image函数，从上一级目录中的image_utils模块

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES
    # 如果torch可用，则从models.auto.modeling_auto模块导入MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象

@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True, has_image_processor=True))
# 使用装饰器为VisualQuestionAnsweringPipeline类添加文档字符串，调用build_pipeline_init_args函数生成参数

class VisualQuestionAnsweringPipeline(Pipeline):
    """
    Visual Question Answering pipeline using a `AutoModelForVisualQuestionAnswering`. This pipeline is currently only
    available in PyTorch.

    Example:

    ```
    >>> from transformers import pipeline

    >>> oracle = pipeline(model="dandelin/vilt-b32-finetuned-vqa")
    >>> image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png"
    >>> oracle(question="What is she wearing ?", image=image_url)
    [{'score': 0.948, 'answer': 'hat'}, {'score': 0.009, 'answer': 'fedora'}, {'score': 0.003, 'answer': 'clothes'}, {'score': 0.003, 'answer': 'sun hat'}, {'score': 0.002, 'answer': 'nothing'}]

    >>> oracle(question="What is she wearing ?", image=image_url, top_k=1)
    [{'score': 0.948, 'answer': 'hat'}]

    >>> oracle(question="Is this a person ?", image=image_url, top_k=1)
    [{'score': 0.993, 'answer': 'yes'}]

    >>> oracle(question="Is this a man ?", image=image_url, top_k=1)
    [{'score': 0.996, 'answer': 'no'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This visual question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifiers: `"visual-question-answering", "vqa"`.

    The models that this pipeline can use are models that have been fine-tuned on a visual question answering task. See
    the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=visual-question-answering).
    """
    # Visual Question Answering Pipeline类的文档字符串，描述了使用AutoModelForVisualQuestionAnswering的视觉问答流水线

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 调用父类Pipeline的初始化方法，传递所有位置参数和关键字参数

        self.check_model_type(MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES)
        # 调用当前对象的check_model_type方法，检查模型类型是否匹配MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES

    def _sanitize_parameters(self, top_k=None, padding=None, truncation=None, timeout=None, **kwargs):
        preprocess_params, postprocess_params = {}, {}
        # 初始化预处理和后处理参数字典

        if padding is not None:
            preprocess_params["padding"] = padding
        # 如果padding参数不为None，则将其加入预处理参数字典中

        if truncation is not None:
            preprocess_params["truncation"] = truncation
        # 如果truncation参数不为None，则将其加入预处理参数字典中

        if timeout is not None:
            preprocess_params["timeout"] = timeout
        # 如果timeout参数不为None，则将其加入预处理参数字典中

        if top_k is not None:
            postprocess_params["top_k"] = top_k
        # 如果top_k参数不为None，则将其加入后处理参数字典中

        return preprocess_params, {}, postprocess_params
        # 返回预处理参数字典、空字典和后处理参数字典作为元组的形式
    def __call__(self, image: Union["Image.Image", str], question: str = None, **kwargs):
        r"""
        Answers open-ended questions about images. The pipeline accepts several types of inputs which are detailed
        below:

        - `pipeline(image=image, question=question)`
        - `pipeline({"image": image, "question": question})`
        - `pipeline([{"image": image, "question": question}])`
        - `pipeline([{"image": image, "question": question}, {"image": image, "question": question}])`

        Args:
            image (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. If given a single image, it can be
                broadcasted to multiple questions.
            question (`str`, `List[str]`):
                The question(s) asked. If given a single question, it can be broadcasted to multiple images.
            top_k (`int`, *optional*, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.
        Return:
            A dictionary or a list of dictionaries containing the result. The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        """
        if isinstance(image, (Image.Image, str)) and isinstance(question, str):
            # 如果 `image` 是 PIL.Image 或者字符串，且 `question` 是字符串，则组装成单个输入字典
            inputs = {"image": image, "question": question}
        else:
            """
            如果输入不符合上述条件，支持以下格式：
            - {"image": image, "question": question}
            - [{"image": image, "question": question}]
            - 生成器和数据集
            """
            # 否则，直接使用给定的输入作为 `inputs`
            inputs = image
        # 调用父类方法处理输入并返回结果
        results = super().__call__(inputs, **kwargs)
        return results

    def preprocess(self, inputs, padding=False, truncation=False, timeout=None):
        # 加载图像，并根据超时设置加载图像
        image = load_image(inputs["image"], timeout=timeout)
        # 使用分词器处理问题文本，并返回模型输入
        model_inputs = self.tokenizer(
            inputs["question"], return_tensors=self.framework, padding=padding, truncation=truncation
        )
        # 使用图像处理器处理图像特征，并更新模型输入
        image_features = self.image_processor(images=image, return_tensors=self.framework)
        model_inputs.update(image_features)
        return model_inputs
    # 定义一个私有方法 `_forward`，用于模型推理过程中的前向传播
    def _forward(self, model_inputs, **generate_kwargs):
        # 如果模型支持生成任务
        if self.model.can_generate():
            # 调用模型的生成方法，生成模型输出
            model_outputs = self.model.generate(**model_inputs, **generate_kwargs)
        else:
            # 否则，调用模型的正常推理方法
            model_outputs = self.model(**model_inputs)
        # 返回模型的输出结果
        return model_outputs

    # 定义后处理方法 `postprocess`，用于处理模型输出并返回结果
    def postprocess(self, model_outputs, top_k=5):
        # 如果模型支持生成任务
        if self.model.can_generate():
            # 对每个模型输出的标识符进行解码，生成答案字符串，并去除特殊标记
            return [
                {"answer": self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()}
                for output_ids in model_outputs
            ]
        else:
            # 如果 `top_k` 大于模型配置的标签数量，则将其设置为标签数量
            if top_k > self.model.config.num_labels:
                top_k = self.model.config.num_labels

            # 根据不同框架进行后处理
            if self.framework == "pt":
                # 计算模型输出的概率，并进行逻辑斯蒂处理，取第一个元素
                probs = model_outputs.logits.sigmoid()[0]
                # 获取概率最高的 `top_k` 个分数及其对应的标识符
                scores, ids = probs.topk(top_k)
            else:
                # 如果框架不支持，则抛出错误
                raise ValueError(f"Unsupported framework: {self.framework}")

            # 将分数和标识符转换为列表形式，并构建结果列表
            scores = scores.tolist()
            ids = ids.tolist()
            return [{"score": score, "answer": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
```