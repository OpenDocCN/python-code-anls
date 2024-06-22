# `.\transformers\pipelines\visual_question_answering.py`

```py
# 从 typing 模块中导入 Union 类型
from typing import Union

# 从 ..utils 模块中导入 add_end_docstrings, is_torch_available, is_vision_available, logging
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging
# 从 .base 模块中导入 PIPELINE_INIT_ARGS, Pipeline 类
from .base import PIPELINE_INIT_ARGS, Pipeline

# 如果 is_vision_available 返回 True，导入 Image 类和 load_image 函数
if is_vision_available():
    from PIL import Image
    from ..image_utils import load_image

# 如果 is_torch_available 返回 True，从 ..models.auto.modeling_auto 中导入 MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES

# 从 logging 模块中获取 logger 对象
logger = logging.get_logger(__name__)

# 使用 @add_end_docstrings(PIPELINE_INIT_ARGS) 装饰器添加文档字符串到 VisualQuestionAnsweringPipeline 类
@add_end_docstrings(PIPELINE_INIT_ARGS)
class VisualQuestionAnsweringPipeline(Pipeline):
    """
    Visual Question Answering pipeline using a `AutoModelForVisualQuestionAnswering`. This pipeline is currently only
    available in PyTorch.
    ...
    """

    # 定义初始化方法
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 检查模型类型是否为 MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES 中定义的类型
        self.check_model_type(MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES)

    # 定义私有方法 _sanitize_parameters
    def _sanitize_parameters(self, top_k=None, padding=None, truncation=None, timeout=None, **kwargs):
        # 初始化预处理参数和后处理参数
        preprocess_params, postprocess_params = {}, {}
        # 如果 padding 不为 None，则将其添加到预处理参数中
        if padding is not None:
            preprocess_params["padding"] = padding
        # 如果 truncation 不为 None，则将其添加到预处理参数中
        if truncation is not None:
            preprocess_params["truncation"] = truncation
        # 如果 timeout 不为 None，则将其添加到预处理参数中
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        # 如果 top_k 不为 None，则将其添加到后处理参数中
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        # 返回预处理参数，空字典，后处理参数
        return preprocess_params, {}, postprocess_params
    # 定义一个调用函数，接受图片和问题作为参数，可以传入不同格式的输入
    def __call__(self, image: Union["Image.Image", str], question: str = None, **kwargs):
        # 回答关于图片的开放性问题，接受几种不同类型的输入
        r"""
        Answers open-ended questions about images. The pipeline accepts several types of inputs which are detailed
        below:

        - `pipeline(image=image, question=question)`
        - `pipeline({"image": image, "question": question})`
        - `pipeline([{"image": image, "question": question}])`
        - `pipeline([{"image": image, "question": question}, {"image": image, "question": question}])`

        Args:
            # 图片参数，可以是字符串的链接，本地路径，也可以是PIL格式的图片
            image (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:
                
                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly
                
                The pipeline accepts either a single image or a batch of images. If given a single image, it can be
                broadcasted to multiple questions.
            # 问题参数，可以是字符串或字符串列表
            question (`str`, `List[str]`):
                The question(s) asked. If given a single question, it can be broadcasted to multiple images.
            # 返回结果的标签数量，默认为5
            top_k (`int`, *optional*, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.
            # 超时时间，单位为秒，默认为无限
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.
        Return:
            # 返回结果，包含标签和得分
            A dictionary or a list of dictionaries containing the result. The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        """
        # 判断输入的图片和问题是否符合要求
        if isinstance(image, (Image.Image, str)) and isinstance(question, str):
            inputs = {"image": image, "question": question}
        else:
            """
            Supports the following format
            - {"image": image, "question": question}
            - [{"image": image, "question": question}]
            - Generator and datasets
            """
            inputs = image
        # 调用父类的调用函数
        results = super().__call__(inputs, **kwargs)
        return results

    # 预处理函数，用于对输入进行处理
    def preprocess(self, inputs, padding=False, truncation=False, timeout=None):
        # 加载图片
        image = load_image(inputs["image"], timeout=timeout)
        # 使用分词器处理问题
        model_inputs = self.tokenizer(
            inputs["question"], return_tensors=self.framework, padding=padding, truncation=truncation
        )
        # 使用图片处理器处理图片
        image_features = self.image_processor(images=image, return_tensors=self.framework)
        # 将问题和图片特征合并
        model_inputs.update(image_features)
        return model_inputs
    # 定义一个方法用于模型的前向传播，接受模型输入作为参数
    def _forward(self, model_inputs):
        # 如果模型支持生成输出，则调用模型的生成方法，否则调用模型的普通前向传播方法
        if self.model.can_generate():
            model_outputs = self.model.generate(**model_inputs)
        else:
            model_outputs = self.model(**model_inputs)
        # 返回模型的输出
        return model_outputs

    # 定义一个方法用于对模型的输出进行后处理，可选参数top_k用于指定返回前几个结果
    def postprocess(self, model_outputs, top_k=5):
        # 如果模型支持生成输出，则将每个输出id对应的文本进行解码并返回
        if self.model.can_generate():
            return [
                {"answer": self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()}
                for output_ids in model_outputs
            ]
        # 如果模型不支持生成输出
        else:
            # 如果指定的top_k大于模型配置中的标签数量，则将top_k限制为模型配置中的标签数量
            if top_k > self.model.config.num_labels:
                top_k = self.model.config.num_labels

            # 如果使用的框架是PyTorch，则计算输出logits的sigmoid，并获取top_k的分数和id
            if self.framework == "pt":
                probs = model_outputs.logits.sigmoid()[0]
                scores, ids = probs.topk(top_k)
            # 如果使用的框架不是PyTorch，则抛出不支持的框架错误
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")

            # 将分数和id转换为列表形式，并返回每个结果的分数和对应的标签
            scores = scores.tolist()
            ids = ids.tolist()
            return [{"score": score, "answer": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
```