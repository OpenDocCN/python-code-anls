# `stable-diffusion-webui\modules\interrogate.py`

```
# 导入必要的库
import os
import sys
from collections import namedtuple
from pathlib import Path
import re

# 导入 PyTorch 库
import torch
import torch.hub

# 导入 torchvision 库中的相关模块和函数
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# 导入自定义模块
from modules import devices, paths, shared, lowvram, modelloader, errors

# 设置图片评估大小和 CLIP 模型名称
blip_image_eval_size = 384
clip_model_name = 'ViT-L/14'

# 创建一个命名元组 Category，包含名称、topn 和 items 字段
Category = namedtuple("Category", ["name", "topn", "items"])

# 编译正则表达式，用于匹配文件名中的 .topn. 部分
re_topn = re.compile(r"\.top(\d+)\.")

# 返回可用的类别类型列表
def category_types():
    return [f.stem for f in Path(shared.interrogator.content_dir).glob('*.txt')]

# 下载默认的 CLIP 类别文件到指定目录
def download_default_clip_interrogate_categories(content_dir):
    print("Downloading CLIP categories...")

    # 创建临时目录
    tmpdir = f"{content_dir}_tmp"
    category_types = ["artists", "flavors", "mediums", "movements"]

    try:
        # 创建临时目录并下载类别文件
        os.makedirs(tmpdir, exist_ok=True)
        for category_type in category_types:
            torch.hub.download_url_to_file(f"https://raw.githubusercontent.com/pharmapsychotic/clip-interrogator/main/clip_interrogator/data/{category_type}.txt", os.path.join(tmpdir, f"{category_type}.txt"))
        # 将临时目录重命名为指定目录
        os.rename(tmpdir, content_dir)

    except Exception as e:
        # 处理下载过程中的异常
        errors.display(e, "downloading default CLIP interrogate categories")
    finally:
        # 最终清理临时目录
        if os.path.exists(tmpdir):
            os.removedirs(tmpdir)

# 定义 InterrogateModels 类
class InterrogateModels:
    blip_model = None
    clip_model = None
    clip_preprocess = None
    dtype = None
    running_on_cpu = None

    # 初始化方法，设置相关属性
    def __init__(self, content_dir):
        self.loaded_categories = None
        self.skip_categories = []
        self.content_dir = content_dir
        self.running_on_cpu = devices.device_interrogate == torch.device("cpu")
    # 返回可用的类别列表，如果内容目录不存在，则下载默认的 clip_interrogate_categories
    def categories(self):
        if not os.path.exists(self.content_dir):
            download_default_clip_interrogate_categories(self.content_dir)

        # 如果已加载类别不为空且跳过类别与共享选项中的相同，则返回已加载的类别
        if self.loaded_categories is not None and self.skip_categories == shared.opts.interrogate_clip_skip_categories:
           return self.loaded_categories

        # 初始化已加载的类别列表
        self.loaded_categories = []

        # 如果内容目录存在
        if os.path.exists(self.content_dir):
            # 设置跳过类别为共享选项中的跳过类别
            self.skip_categories = shared.opts.interrogate_clip_skip_categories
            # 初始化类别类型列表
            category_types = []
            # 遍历内容目录下的所有 .txt 文件
            for filename in Path(self.content_dir).glob('*.txt'):
                # 将文件名添加到类别类型列表中
                category_types.append(filename.stem)
                # 如果文件名在跳过类别列表中，则继续下一个循环
                if filename.stem in self.skip_categories:
                    continue
                # 从文件名中提取 topn 数量
                m = re_topn.search(filename.stem)
                topn = 1 if m is None else int(m.group(1))
                # 打开文件，读取每行内容并去除空格，存储到 lines 列表中
                with open(filename, "r", encoding="utf8") as file:
                    lines = [x.strip() for x in file.readlines()]

                # 将类别信息添加到已加载的类别列表中
                self.loaded_categories.append(Category(name=filename.stem, topn=topn, items=lines))

        # 返回已加载的类别列表
        return self.loaded_categories

    # 创建虚拟的 Fairscale 类
    def create_fake_fairscale(self):
        class FakeFairscale:
            def checkpoint_wrapper(self):
                pass

        # 将虚拟的 Fairscale 类添加到 sys.modules 中
        sys.modules["fairscale.nn.checkpoint.checkpoint_activations"] = FakeFairscale

    # 加载 BLIP 模型
    def load_blip_model(self):
        # 创建虚拟的 Fairscale 类
        self.create_fake_fairscale()
        # 导入 models.blip 模块
        import models.blip

        # 加载 BLIP 模型文件
        files = modelloader.load_models(
            model_path=os.path.join(paths.models_path, "BLIP"),
            model_url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth',
            ext_filter=[".pth"],
            download_name='model_base_caption_capfilt_large.pth',
        )

        # 初始化 BLIP 模型
        blip_model = models.blip.blip_decoder(pretrained=files[0], image_size=blip_image_eval_size, vit='base', med_config=os.path.join(paths.paths["BLIP"], "configs", "med_config.json"))
        # 设置 BLIP 模型为评估模式
        blip_model.eval()

        # 返回 BLIP 模型
        return blip_model
    # 加载 CLIP 模型
    def load_clip_model(self):
        # 导入 clip 模块
        import clip

        # 根据是否在 CPU 上运行加载不同的模型
        if self.running_on_cpu:
            model, preprocess = clip.load(clip_model_name, device="cpu", download_root=shared.cmd_opts.clip_models_path)
        else:
            model, preprocess = clip.load(clip_model_name, download_root=shared.cmd_opts.clip_models_path)

        # 将模型设置为评估模式
        model.eval()
        # 将模型移动到指定设备上
        model = model.to(devices.device_interrogate)

        # 返回加载的模型和预处理函数
        return model, preprocess

    # 加载模型
    def load(self):
        # 如果 BLIP 模型为空，则加载 BLIP 模型
        if self.blip_model is None:
            self.blip_model = self.load_blip_model()
            # 如果不禁用半精度并且不在 CPU 上运行，则将 BLIP 模型转换为半精度
            if not shared.cmd_opts.no_half and not self.running_on_cpu:
                self.blip_model = self.blip_model.half()

        # 将 BLIP 模型移动到指定设备上
        self.blip_model = self.blip_model.to(devices.device_interrogate)

        # 如果 CLIP 模型为空，则加载 CLIP 模型和预处理函数
        if self.clip_model is None:
            self.clip_model, self.clip_preprocess = self.load_clip_model()
            # 如果不禁用半精度并且不在 CPU 上运行，则将 CLIP 模型转换为半精度
            if not shared.cmd_opts.no_half and not self.running_on_cpu:
                self.clip_model = self.clip_model.half()

        # 将 CLIP 模型移动到指定设备上
        self.clip_model = self.clip_model.to(devices.device_interrogate)

        # 获取模型参数的数据类型
        self.dtype = next(self.clip_model.parameters()).dtype

    # 将 CLIP 模型发送到 RAM
    def send_clip_to_ram(self):
        # 如果不保持模型在内存中，则将 CLIP 模型发送到 CPU
        if not shared.opts.interrogate_keep_models_in_memory:
            if self.clip_model is not None:
                self.clip_model = self.clip_model.to(devices.cpu)

    # 将 BLIP 模型发送到 RAM
    def send_blip_to_ram(self):
        # 如果不保持模型在内存中，则将 BLIP 模型发送到 CPU
        if not shared.opts.interrogate_keep_models_in_memory:
            if self.blip_model is not None:
                self.blip_model = self.blip_model.to(devices.cpu)

    # 卸载模型
    def unload(self):
        # 将 CLIP 模型发送到 RAM
        self.send_clip_to_ram()
        # 将 BLIP 模型发送到 RAM
        self.send_blip_to_ram()

        # 执行 PyTorch 的垃圾回收
        devices.torch_gc()
    # 根据图像特征和文本数组对文本进行排名，返回排名结果
    def rank(self, image_features, text_array, top_count=1):
        # 导入 clip 模块
        import clip

        # 释放 torch 的缓存
        devices.torch_gc()

        # 如果设置了 clip 字典限制，则截取文本数组
        if shared.opts.interrogate_clip_dict_limit != 0:
            text_array = text_array[0:int(shared.opts.interrogate_clip_dict_limit)]

        # 确定排名数量不超过文本数组长度
        top_count = min(top_count, len(text_array))
        # 对文本数组进行分词并转换为设备上的张量
        text_tokens = clip.tokenize(list(text_array), truncate=True).to(devices.device_interrogate)
        # 使用 clip 模型对文本进行编码
        text_features = self.clip_model.encode_text(text_tokens).type(self.dtype)
        # 对文本特征进行归一化处理
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 初始化相似度张量
        similarity = torch.zeros((1, len(text_array))).to(devices.device_interrogate)
        # 计算图像特征与文本特征的相似度
        for i in range(image_features.shape[0]):
            similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
        similarity /= image_features.shape[0]

        # 获取相似度最高的文本和对应概率
        top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)
        return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy()*100)) for i in range(top_count)]

    # 生成图像描述
    def generate_caption(self, pil_image):
        # 对 PIL 图像进行预处理，转换为 GPU 张量
        gpu_image = transforms.Compose([
            transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).type(self.dtype).to(devices.device_interrogate)

        # 使用 blip 模型生成图像描述
        with torch.no_grad():
            caption = self.blip_model.generate(gpu_image, sample=False, num_beams=shared.opts.interrogate_clip_num_beams, min_length=shared.opts.interrogate_clip_min_length, max_length=shared.opts.interrogate_clip_max_length)

        return caption[0]
    # 定义一个方法，用于对给定的 PIL 图像进行分析
    def interrogate(self, pil_image):
        # 初始化结果字符串
        res = ""
        # 开始一个新的任务，标记为“interrogate”
        shared.state.begin(job="interrogate")
        try:
            # 将所有内容从低 VRAM 发送到 CPU
            lowvram.send_everything_to_cpu()
            # 执行 Torch 的垃圾回收
            devices.torch_gc()

            # 载入模型
            self.load()

            # 生成图像描述
            caption = self.generate_caption(pil_image)
            # 将生成的描述发送到 RAM
            self.send_blip_to_ram()
            # 执行 Torch 的垃圾回收
            devices.torch_gc()

            # 将生成的描述添加到结果字符串中
            res = caption

            # 对 PIL 图像进行预处理，并转换为指定类型和设备
            clip_image = self.clip_preprocess(pil_image).unsqueeze(0).type(self.dtype).to(devices.device_interrogate)

            # 关闭 Torch 的梯度计算，并使用自动混合精度
            with torch.no_grad(), devices.autocast():
                # 使用 CLIP 模型对图像进行编码
                image_features = self.clip_model.encode_image(clip_image).type(self.dtype)

                # 对图像特征进行归一化处理
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # 遍历所有类别
                for cat in self.categories():
                    # 对图像特征进行排名，并返回前几个匹配项
                    matches = self.rank(image_features, cat.items, top_count=cat.topn)
                    # 遍历匹配项和对应的分数
                    for match, score in matches:
                        # 如果设置了返回排名信息，则将匹配项和分数添加到结果字符串中
                        if shared.opts.interrogate_return_ranks:
                            res += f", ({match}:{score/100:.3f})"
                        # 否则，只将匹配项添加到结果字符串中
                        else:
                            res += f", {match}"

        # 捕获异常并报告错误
        except Exception:
            errors.report("Error interrogating", exc_info=True)
            # 将错误信息添加到结果字符串中
            res += "<error>"

        # 卸载模型
        self.unload()
        # 结束当前任务
        shared.state.end()

        # 返回最终结果字符串
        return res
```