# `stable-diffusion-webui\modules\deepbooru.py`

```py
# 导入所需的库
import os
import re

import torch
import numpy as np

# 导入自定义模块
from modules import modelloader, paths, deepbooru_model, devices, images, shared

# 编译正则表达式，用于匹配特殊字符
re_special = re.compile(r'([\\()])')

# 定义 DeepDanbooru 类
class DeepDanbooru:
    def __init__(self):
        self.model = None

    # 加载模型
    def load(self):
        # 如果模型已加载，则直接返回
        if self.model is not None:
            return

        # 加载模型文件
        files = modelloader.load_models(
            model_path=os.path.join(paths.models_path, "torch_deepdanbooru"),
            model_url='https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt',
            ext_filter=[".pt"],
            download_name='model-resnet_custom_v3.pt',
        )

        # 创建 DeepDanbooruModel 实例并加载模型参数
        self.model = deepbooru_model.DeepDanbooruModel()
        self.model.load_state_dict(torch.load(files[0], map_location="cpu"))

        # 设置模型为评估模式，并将模型移动到指定设备
        self.model.eval()
        self.model.to(devices.cpu, devices.dtype)

    # 启动模型
    def start(self):
        self.load()
        self.model.to(devices.device)

    # 停止模型
    def stop():
        # 如果不需要保留模型在内存中，则将模型移动到 CPU，并进行内存回收
        if not shared.opts.interrogate_keep_models_in_memory:
            self.model.to(devices.cpu)
            devices.torch_gc()

    # 对图像进行标记
    def tag(self, pil_image):
        # 启动模型
        self.start()
        # 对图像进行多标签标记
        res = self.tag_multi(pil_image)
        # 停止模型
        self.stop()

        return res
    # 对输入的 PIL 图像进行多标签分类，返回标签字符串
    def tag_multi(self, pil_image, force_disable_ranks=False):
        # 获取阈值、使用空格、使用转义、是否按字母排序、是否包含排名的设置
        threshold = shared.opts.interrogate_deepbooru_score_threshold
        use_spaces = shared.opts.deepbooru_use_spaces
        use_escape = shared.opts.deepbooru_escape
        alpha_sort = shared.opts.deepbooru_sort_alpha
        include_ranks = shared.opts.interrogate_return_ranks and not force_disable_ranks

        # 将 PIL 图像转换为 RGB 模式并调整大小为 512x512
        pic = images.resize_image(2, pil_image.convert("RGB"), 512, 512)
        # 将图像转换为 numpy 数组，并进行归一化处理
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        # 使用 torch.no_grad() 和 devices.autocast() 上下文管理器
        with torch.no_grad(), devices.autocast():
            # 将 numpy 数组转换为 torch 张量，并移动到指定设备上
            x = torch.from_numpy(a).to(devices.device)
            # 使用模型进行推理，获取预测结果
            y = self.model(x)[0].detach().cpu().numpy()

        # 初始化概率字典
        probability_dict = {}

        # 遍历模型的标签和对应的概率
        for tag, probability in zip(self.model.tags, y):
            # 如果概率低于阈值，则跳过
            if probability < threshold:
                continue
            # 如果标签以"rating:"开头，则跳过
            if tag.startswith("rating:"):
                continue
            # 将标签和概率添加到概率字典中
            probability_dict[tag] = probability

        # 根据设置对概率字典中的标签进行排序
        if alpha_sort:
            tags = sorted(probability_dict)
        else:
            tags = [tag for tag, _ in sorted(probability_dict.items(), key=lambda x: -x[1])]

        # 初始化结果列表
        res = []

        # 获取需要过滤的标签集合
        filtertags = {x.strip().replace(' ', '_') for x in shared.opts.deepbooru_filter_tags.split(",")}

        # 遍历排序后的标签，根据设置进行处理并添加到结果列表中
        for tag in [x for x in tags if x not in filtertags]:
            probability = probability_dict[tag]
            tag_outformat = tag
            # 如果使用空格，则将下划线替换为空格
            if use_spaces:
                tag_outformat = tag_outformat.replace('_', ' ')
            # 如果使用转义，则对特殊字符进行转义处理
            if use_escape:
                tag_outformat = re.sub(re_special, r'\\\1', tag_outformat)
            # 如果包含排名，则格式化输出标签和概率
            if include_ranks:
                tag_outformat = f"({tag_outformat}:{probability:.3f})"
            # 将处理后的标签添加到结果列表中
            res.append(tag_outformat)

        # 将结果列表中的标签字符串用逗号连接起来返回
        return ", ".join(res)
# 创建一个 DeepDanbooru 模型的实例
model = DeepDanbooru()
```