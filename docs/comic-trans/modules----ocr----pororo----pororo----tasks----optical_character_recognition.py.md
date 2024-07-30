# `.\comic-translate\modules\ocr\pororo\pororo\tasks\optical_character_recognition.py`

```py
"""
OCR related modeling class
"""

from typing import Optional

from ...pororo.tasks import download_or_load
from ...pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase


class PororoOcrFactory(PororoFactoryBase):
    """
    Recognize optical characters in image file
    Currently support Korean language

    English + Korean (`brainocr`)

        - dataset: Internal data + AI hub Font Image dataset
        - metric: TBU
        - ref: https://www.aihub.or.kr/aidata/133

    Examples:
        >>> ocr = Pororo(task="ocr", lang="ko")
        >>> ocr(IMAGE_PATH)
        ["사이렌'(' 신마'", "내가 말했잖아 속지열라고 이 손을 잡는 너는 위협해질 거라고"]

        >>> ocr = Pororo(task="ocr", lang="ko")
        >>> ocr(IMAGE_PATH, detail=True)
        {
            'description': ["사이렌'(' 신마'", "내가 말했잖아 속지열라고 이 손을 잡는 너는 위협해질 거라고"],
            'bounding_poly': [
                {
                    'description': "사이렌'(' 신마'",
                    'vertices': [
                        {'x': 93, 'y': 7},
                        {'x': 164, 'y': 7},
                        {'x': 164, 'y': 21},
                        {'x': 93, 'y': 21}
                    ]
                },
                {
                    'description': "내가 말했잖아 속지열라고 이 손을 잡는 너는 위협해질 거라고",
                    'vertices': [
                        {'x': 0, 'y': 30},
                        {'x': 259, 'y': 30},
                        {'x': 259, 'y': 194},
                        {'x': 0, 'y': 194}
                    ]
                }
            ]
        }
    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)
        # 设置 OCR 检测模型为 "craft"
        self.detect_model = "craft"
        # 设置 OCR 选项为 "ocr-opt"
        self.ocr_opt = "ocr-opt"

    @staticmethod
    def get_available_langs():
        # 返回支持的语言列表 ["en", "ko"]
        return ["en", "ko"]

    @staticmethod
    def get_available_models():
        # 返回支持的模型字典 {"en": ["brainocr"], "ko": ["brainocr"]}
        return {
            "en": ["brainocr"],
            "ko": ["brainocr"],
        }
    # 加载用户选择的特定任务模型

    # 如果配置中的模型名称为 "brainocr"
    if self.config.n_model == "brainocr":
        # 从指定路径导入 brainOCR 模型
        from ...pororo.models.brainOCR import brainocr

        # 如果配置中的语言不在支持的语言列表中
        if self.config.lang not in self.get_available_langs():
            # 抛出值错误并显示不支持的语言和支持的语言列表
            raise ValueError(
                f"Unsupported Language : {self.config.lang}",
                'Support Languages : ["en", "ko"]',
            )

        # 下载或加载检测模型的路径
        det_model_path = download_or_load(
            f"{self.detect_model}.pt",
            self.config.lang,
        )
        # 下载或加载识别模型的路径
        rec_model_path = download_or_load(
            f"{self.config.n_model}.pt",
            self.config.lang,
        )
        # 下载或加载OCR选项文件的路径
        opt_fp = download_or_load(
            f"{self.ocr_opt}.txt",
            self.config.lang,
        )

        # 创建 brainOCR.Reader 模型对象
        model = brainocr.Reader(
            self.config.lang,
            det_model_ckpt_fp=det_model_path,
            rec_model_ckpt_fp=rec_model_path,
            opt_fp=opt_fp,
            device=device,
        )

        # 将检测器模型移动到指定的设备上
        model.detector.to(device)
        # 将识别器模型移动到指定的设备上
        model.recognizer.to(device)

        # 返回 PororoOCR 类的实例，使用加载的模型和配置信息
        return PororoOCR(model, self.config)
    # 定义名为 PororoOCR 的类，继承自 PororoSimpleBase 类
    class PororoOCR(PororoSimpleBase):

        # 初始化方法，接受模型和配置参数，并调用父类的初始化方法
        def __init__(self, model, config):
            super().__init__(config)
            self._model = model  # 将传入的模型保存到实例变量 _model 中

        # 对 OCR 结果进行后处理的方法
        def _postprocess(self, ocr_results, detail: bool = False):
            """
            Post-process for OCR result

            Args:
                ocr_results (list): list contains result of OCR
                detail (bool): if True, returned to include details. (bounding poly, vertices, etc)

            """
            # 按照顶点坐标从小到大的顺序对 OCR 结果进行排序
            sorted_ocr_results = sorted(
                ocr_results,
                key=lambda x: (
                    x[0][0][1],  # 根据顶点的 y 坐标排序
                    x[0][0][0],  # 在 y 坐标相同时，根据顶点的 x 坐标排序
                ),
            )

            # 如果 detail 参数为 False，则返回排序后的 OCR 结果中的文本部分
            if not detail:
                return [
                    sorted_ocr_results[i][-1]  # 取出每个结果的文本部分
                    for i in range(len(sorted_ocr_results))
                ]

            # 如果 detail 参数为 True，则构建详细结果字典
            result_dict = {
                "description": list(),  # 初始化一个存储文本描述的列表
                "bounding_poly": list(),  # 初始化一个存储边界多边形信息的列表
            }

            # 遍历排序后的 OCR 结果
            for ocr_result in sorted_ocr_results:
                vertices = list()  # 初始化一个存储顶点信息的列表

                # 遍历每个顶点并以字典形式保存 x 和 y 坐标
                for vertice in ocr_result[0]:
                    vertices.append({
                        "x": vertice[0],  # 顶点的 x 坐标
                        "y": vertice[1],  # 顶点的 y 坐标
                    })

                # 将文本描述和顶点信息添加到结果字典中
                result_dict["description"].append(ocr_result[1])  # 添加文本描述
                result_dict["bounding_poly"].append({
                    "description": ocr_result[1],  # 边界多边形的描述信息
                    "vertices": vertices  # 边界多边形的顶点信息
                })

            # 返回构建好的详细结果字典
            return result_dict

        # 执行光学字符识别（OCR）的方法
        def predict(self, image_path: str, **kwargs):
            """
            Conduct Optical Character Recognition (OCR)

            Args:
                image_path (str): the image file path
                detail (bool): if True, returned to include details. (bounding poly, vertices, etc)

            """
            # 获取 detail 参数，默认为 False
            detail = kwargs.get("detail", False)

            # 调用模型进行预测，并调用 _postprocess 方法处理结果
            return self._postprocess(
                self._model(
                    image_path,
                    skip_details=False,  # 设置为不跳过详细信息
                    batch_size=1,  # 设置批处理大小为 1
                    paragraph=True,  # 启用段落识别
                ),
                detail,  # 传入 detail 参数
            )
```