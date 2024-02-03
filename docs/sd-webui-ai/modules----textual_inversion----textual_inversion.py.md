# `stable-diffusion-webui\modules\textual_inversion\textual_inversion.py`

```py
# 导入所需的模块
import os
from collections import namedtuple
from contextlib import closing

import torch
import tqdm
import html
import datetime
import csv
import safetensors.torch

import numpy as np
from PIL import Image, PngImagePlugin
from torch.utils.tensorboard import SummaryWriter

# 导入自定义模块
from modules import shared, devices, sd_hijack, sd_models, images, sd_samplers, sd_hijack_checkpoint, errors, hashes
import modules.textual_inversion.dataset
from modules.textual_inversion.learn_schedule import LearnRateScheduler

# 导入自定义函数和类
from modules.textual_inversion.image_embedding import embedding_to_b64, embedding_from_b64, insert_image_data_embed, extract_image_data_embed, caption_image_overlay
from modules.textual_inversion.logging import save_settings_to_file

# 定义命名元组 TextualInversionTemplate
TextualInversionTemplate = namedtuple("TextualInversionTemplate", ["name", "path"])
# 存储文本反演模板的字典
textual_inversion_templates = {}

# 列出文本反演模板
def list_textual_inversion_templates():
    # 清空文本反演模板字典
    textual_inversion_templates.clear()

    # 遍历文本反演模板目录下的文件
    for root, _, fns in os.walk(shared.cmd_opts.textual_inversion_templates_dir):
        for fn in fns:
            path = os.path.join(root, fn)

            # 将文件名和路径存储到文本反演模板字典中
            textual_inversion_templates[fn] = TextualInversionTemplate(fn, path)

    return textual_inversion_templates

# 定义 Embedding 类
class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.shape = None
        self.vectors = 0
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None
        self.filename = None
        self.hash = None
        self.shorthash = None
    # 将嵌入数据保存到文件中
    def save(self, filename):
        # 构建嵌入数据字典
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        # 使用 torch 保存嵌入数据到文件
        torch.save(embedding_data, filename)

        # 如果需要保存优化器状态并且优化器状态字典不为空
        if shared.opts.save_optimizer_state and self.optimizer_state_dict is not None:
            # 构建优化器状态字典
            optimizer_saved_dict = {
                'hash': self.checksum(),
                'optimizer_state_dict': self.optimizer_state_dict,
            }
            # 使用 torch 保存优化器状态字典到文件
            torch.save(optimizer_saved_dict, f"{filename}.optim")

    # 计算数据的校验和
    def checksum(self):
        # 如果已经计算过校验和，则直接返回缓存的校验和
        if self.cached_checksum is not None:
            return self.cached_checksum

        # 定义一个常量哈希函数
        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        # 计算数据的校验和并缓存起来
        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum

    # 设置哈希值
    def set_hash(self, v):
        # 设置哈希值和截断哈希值
        self.hash = v
        self.shorthash = self.hash[0:12]
# 定义一个类，用于处理具有文本反转嵌入的目录
class DirWithTextualInversionEmbeddings:
    # 初始化方法，接收一个路径参数
    def __init__(self, path):
        # 将传入的路径赋值给实例变量 path
        self.path = path
        # 初始化实例变量 mtime 为 None
        self.mtime = None

    # 检查目录是否发生变化的方法
    def has_changed(self):
        # 如果路径不是一个目录，则返回 False
        if not os.path.isdir(self.path):
            return False

        # 获取路径的最后修改时间
        mt = os.path.getmtime(self.path)
        # 如果实例变量 mtime 为 None 或者路径的最后修改时间大于 mtime，则返回 True
        if self.mtime is None or mt > self.mtime:
            return True

    # 更新方法，更新实例变量 mtime
    def update(self):
        # 如果路径不是一个目录，则返回
        if not os.path.isdir(self.path):
            return

        # 获取路径的最后修改时间，赋值给实例变量 mtime
        self.mtime = os.path.getmtime(self.path)


# 定义一个类，用于处理嵌入数据库
class EmbeddingDatabase:
    # 初始化方法，初始化各个实例变量
    def __init__(self):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.skipped_embeddings = {}
        self.expected_shape = -1
        self.embedding_dirs = {}
        self.previously_displayed_embeddings = ()

    # 添加嵌入目录的方法
    def add_embedding_dir(self, path):
        # 将路径和 DirWithTextualInversionEmbeddings 类的实例关联起来
        self.embedding_dirs[path] = DirWithTextualInversionEmbeddings(path)

    # 清空嵌入目录的方法
    def clear_embedding_dirs(self):
        # 清空嵌入目录字典
        self.embedding_dirs.clear()

    # 注册嵌入的方法
    def register_embedding(self, embedding, model):
        # 调用 register_embedding_by_name 方法，传入嵌入、模型和嵌入名称
        return self.register_embedding_by_name(embedding, model, embedding.name)
    # 根据给定的嵌入、模型和名称注册嵌入
    def register_embedding_by_name(self, embedding, model, name):
        # 使用模型对名称进行分词处理，获取对应的 ID 列表
        ids = model.cond_stage_model.tokenize([name])[0]
        # 获取第一个 ID
        first_id = ids[0]
        # 如果第一个 ID 不在 ids_lookup 中，则将其添加到 ids_lookup 中
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []
        # 如果名称已经在 word_embeddings 中存在
        if name in self.word_embeddings:
            # 从 ids_lookup 中移除旧的名称
            lookup = [x for x in self.ids_lookup[first_id] if x[1].name!=name]
        else:
            lookup = self.ids_lookup[first_id]
        # 如果嵌入不为空，则将其添加到 lookup 中
        if embedding is not None:
            lookup += [(ids, embedding)]
        # 根据嵌入的长度对 lookup 进行排序，长度越长越靠前
        self.ids_lookup[first_id] = sorted(lookup, key=lambda x: len(x[0]), reverse=True)
        # 如果嵌入为空
        if embedding is None:
            # 取消注册指定名称的嵌入
            if name in self.word_embeddings:
                del self.word_embeddings[name]
            # 如果 ids_lookup[first_id] 为空，则将其从 ids_lookup 中删除
            if len(self.ids_lookup[first_id])==0:
                del self.ids_lookup[first_id]
            return None
        # 将嵌入添加到 word_embeddings 中
        self.word_embeddings[name] = embedding
        return embedding

    # 获取预期的形状
    def get_expected_shape(self):
        # 使用模型对指定文本进行编码，获取嵌入的初始化向量
        vec = shared.sd_model.cond_stage_model.encode_embedding_init_text(",", 1)
        # 返回向量的形状
        return vec.shape[1]
    # 从指定路径和文件名中获取文件名和扩展名
    def load_from_file(self, path, filename):
        name, ext = os.path.splitext(filename)
        # 将扩展名转换为大写形式
        ext = ext.upper()

        # 如果扩展名是图片格式之一
        if ext in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
            # 获取文件名中的第二个扩展名
            _, second_ext = os.path.splitext(name)
            # 如果第二个扩展名是'.PREVIEW'，则返回
            if second_ext.upper() == '.PREVIEW':
                return

            # 打开嵌入的图片文件
            embed_image = Image.open(path)
            # 如果图片包含'sd-ti-embedding'属性
            if hasattr(embed_image, 'text') and 'sd-ti-embedding' in embed_image.text:
                # 从base64编码的文本中提取嵌入数据
                data = embedding_from_b64(embed_image.text['sd-ti-embedding'])
                name = data.get('name', name)
            else:
                # 提取嵌入图片数据
                data = extract_image_data_embed(embed_image)
                if data:
                    name = data.get('name', name)
                else:
                    # 如果data为None，则表示这不是一个嵌入图片，而是一个预览图片
                    return
        # 如果扩展名是'.BIN'或'.PT'
        elif ext in ['.BIN', '.PT']:
            # 从文件中加载torch数据
            data = torch.load(path, map_location="cpu")
        # 如果扩展名是'.SAFETENSORS'
        elif ext in ['.SAFETENSORS']:
            # 从文件中加载safetensors数据
            data = safetensors.torch.load_file(path, device="cpu")
        else:
            return

        # 从数据中创建嵌入
        embedding = create_embedding_from_data(data, name, filename=filename, filepath=path)

        # 如果期望的形状为-1或与嵌入的形状相同，则注册嵌入
        if self.expected_shape == -1 or self.expected_shape == embedding.shape:
            self.register_embedding(embedding, shared.sd_model)
        else:
            # 否则将嵌入添加到跳过的嵌入字典中
            self.skipped_embeddings[name] = embedding

    # 从指定目录加载嵌入
    def load_from_dir(self, embdir):
        # 如果目录不存在，则返回
        if not os.path.isdir(embdir.path):
            return

        # 遍历目录中的文件
        for root, _, fns in os.walk(embdir.path, followlinks=True):
            for fn in fns:
                try:
                    fullfn = os.path.join(root, fn)

                    # 如果文件大小为0，则跳过
                    if os.stat(fullfn).st_size == 0:
                        continue

                    # 加载文件中的嵌入
                    self.load_from_file(fullfn, fn)
                except Exception:
                    # 报告加载嵌入时的错误
                    errors.report(f"Error loading embedding {fn}", exc_info=True)
                    continue
    # 加载文本反转嵌入，可选择是否强制重新加载
    def load_textual_inversion_embeddings(self, force_reload=False):
        # 如果不强制重新加载
        if not force_reload:
            # 初始化是否需要重新加载的标志为 False
            need_reload = False
            # 遍历所有嵌入目录
            for embdir in self.embedding_dirs.values():
                # 如果嵌入目录发生了变化
                if embdir.has_changed():
                    # 设置需要重新加载的标志为 True
                    need_reload = True
                    break

            # 如果不需要重新加载，则直接返回
            if not need_reload:
                return

        # 清空 ids_lookup、word_embeddings 和 skipped_embeddings
        self.ids_lookup.clear()
        self.word_embeddings.clear()
        self.skipped_embeddings.clear()
        # 获取预期的嵌入形状
        self.expected_shape = self.get_expected_shape()

        # 遍历所有嵌入目录
        for embdir in self.embedding_dirs.values():
            # 从目录加载嵌入
            self.load_from_dir(embdir)
            # 更新目录状态
            embdir.update()

        # 重新排序 word_embeddings，因为 load_from_dir 可能不按字母顺序加载
        # 使用临时副本以防止重新初始化 self.word_embeddings，以防其他对象引用它
        sorted_word_embeddings = {e.name: e for e in sorted(self.word_embeddings.values(), key=lambda e: e.name.lower())}
        self.word_embeddings.clear()
        self.word_embeddings.update(sorted_word_embeddings)

        # 显示加载的嵌入，如果设置了打印加载时的文本反转
        displayed_embeddings = (tuple(self.word_embeddings.keys()), tuple(self.skipped_embeddings.keys()))
        if shared.opts.textual_inversion_print_at_load and self.previously_displayed_embeddings != displayed_embeddings:
            self.previously_displayed_embeddings = displayed_embeddings
            print(f"Textual inversion embeddings loaded({len(self.word_embeddings)}): {', '.join(self.word_embeddings.keys())}")
            if self.skipped_embeddings:
                print(f"Textual inversion embeddings skipped({len(self.skipped_embeddings)}): {', '.join(self.skipped_embeddings.keys())}")
    # 在给定位置查找嵌入
    def find_embedding_at_position(self, tokens, offset):
        # 获取当前位置的标记
        token = tokens[offset]
        # 获取可能匹配的嵌入
        possible_matches = self.ids_lookup.get(token, None)

        # 如果没有可能匹配的嵌入，则返回 None
        if possible_matches is None:
            return None, None

        # 遍历可能匹配的嵌入
        for ids, embedding in possible_matches:
            # 如果标记序列与当前位置开始的标记序列匹配，则返回嵌入和匹配的标记长度
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        # 如果没有匹配的嵌入，则返回 None
        return None, None
# 创建一个嵌入向量，用于表示标记的向量化表示
def create_embedding(name, num_vectors_per_token, overwrite_old, init_text='*'):
    # 获取共享的条件模型
    cond_model = shared.sd_model.cond_stage_model

    # 使用自动混合精度上下文
    with devices.autocast():
        # 将条件模型发送到 GPU（如果低内存/中内存处于活动状态）
        cond_model([""])

    # 如果没有提供初始文本，则使用 '*' 作为备用
    embedded = cond_model.encode_embedding_init_text(init_text or '*', num_vectors_per_token)
    # 创建一个形状为 (num_vectors_per_token, embedded.shape[1]) 的零张量
    vec = torch.zeros((num_vectors_per_token, embedded.shape[1]), device=devices.device)

    # 只有在提供了初始文本时才复制，否则保持向量为零
    if init_text:
        for i in range(num_vectors_per_token):
            vec[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]

    # 从名称中删除非法字符
    name = "".join( x for x in name if (x.isalnum() or x in "._- "))
    # 构建文件路径
    fn = os.path.join(shared.cmd_opts.embeddings_dir, f"{name}.pt")
    # 如果不覆盖旧文件，则确保文件不存在
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    # 创建嵌入对象
    embedding = Embedding(vec, name)
    embedding.step = 0
    # 保存嵌入对象到文件
    embedding.save(fn)

    # 返回文件路径
    return fn


# 从数据中创建嵌入向量
def create_embedding_from_data(data, name, filename='unknown embedding file', filepath=None):
    # 如果数据中包含 'string_to_param'，则为文本反演嵌入
    if 'string_to_param' in data:
        param_dict = data['string_to_param']
        param_dict = getattr(param_dict, '_parameters', param_dict)  # 修复 torch 1.12.1 从 torch 1.11 加载保存文件的问题
        assert len(param_dict) == 1, 'embedding file has multiple terms in it'
        emb = next(iter(param_dict.items()))[1]
        vec = emb.detach().to(devices.device, dtype=torch.float32)
        shape = vec.shape[-1]
        vectors = vec.shape[0]
    # 如果数据类型为字典且包含 'clip_g' 和 'clip_l'，则为 SDXL 嵌入
    elif type(data) == dict and 'clip_g' in data and 'clip_l' in data:
        vec = {k: v.detach().to(devices.device, dtype=torch.float32) for k, v in data.items()}
        shape = data['clip_g'].shape[-1] + data['clip_l'].shape[-1]
        vectors = data['clip_g'].shape[0]
    # 如果数据类型为字典且值的类型为 torch.Tensor，则表示为 diffuser 概念
    elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:  
        # 断言字典中键的数量为1，即嵌入文件中只有一个术语
        assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

        # 获取嵌入向量
        emb = next(iter(data.values()))
        # 如果嵌入向量的维度为1，则在第0维度上增加一个维度
        if len(emb.shape) == 1:
            emb = emb.unsqueeze(0)
        # 将嵌入向量转移到指定设备上，并转换为 torch.float32 类型
        vec = emb.detach().to(devices.device, dtype=torch.float32)
        # 获取嵌入向量的形状
        shape = vec.shape[-1]
        # 获取嵌入向量的数量
        vectors = vec.shape[0]
    else:
        # 如果既不是文本反转嵌入也不是 diffuser 概念，则抛出异常
        raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")

    # 创建 Embedding 对象
    embedding = Embedding(vec, name)
    # 设置嵌入对象的步骤
    embedding.step = data.get('step', None)
    # 设置嵌入对象的 sd_checkpoint
    embedding.sd_checkpoint = data.get('sd_checkpoint', None)
    # 设置嵌入对象的 sd_checkpoint_name
    embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
    # 设置嵌入对象的向量数量
    embedding.vectors = vectors
    # 设置嵌入对象的形状
    embedding.shape = shape

    # 如果存在文件路径，则设置嵌入对象的文件名和哈希值
    if filepath:
        embedding.filename = filepath
        embedding.set_hash(hashes.sha256(filepath, "textual_inversion/" + name) or '')

    # 返回嵌入对象
    return embedding
# 写入损失值到 CSV 文件
def write_loss(log_directory, filename, step, epoch_len, values):
    # 如果设置为不写入 CSV 文件，则直接返回
    if shared.opts.training_write_csv_every == 0:
        return

    # 如果步数不是指定的写入频率的倍数，则直接返回
    if step % shared.opts.training_write_csv_every != 0:
        return
    # 判断是否需要写入 CSV 文件头部
    write_csv_header = False if os.path.exists(os.path.join(log_directory, filename)) else True

    # 打开 CSV 文件，追加写入数据
    with open(os.path.join(log_directory, filename), "a+", newline='') as fout:
        # 创建 CSV 写入对象
        csv_writer = csv.DictWriter(fout, fieldnames=["step", "epoch", "epoch_step", *(values.keys())])

        # 如果需要写入 CSV 文件头部，则写入
        if write_csv_header:
            csv_writer.writeheader()

        # 计算当前步数所在的周期和周期内步数
        epoch = (step - 1) // epoch_len
        epoch_step = (step - 1) % epoch_len

        # 写入数据到 CSV 文件
        csv_writer.writerow({
            "step": step,
            "epoch": epoch,
            "epoch_step": epoch_step,
            **values,
        })

# 设置 TensorBoard 日志目录
def tensorboard_setup(log_directory):
    # 创建 TensorBoard 日志目录
    os.makedirs(os.path.join(log_directory, "tensorboard"), exist_ok=True)
    # 返回 SummaryWriter 对象
    return SummaryWriter(
            log_dir=os.path.join(log_directory, "tensorboard"),
            flush_secs=shared.opts.training_tensorboard_flush_every)

# 向 TensorBoard 中添加数据
def tensorboard_add(tensorboard_writer, loss, global_step, step, learn_rate, epoch_num):
    # 向 TensorBoard 中添加损失值数据
    tensorboard_add_scaler(tensorboard_writer, "Loss/train", loss, global_step)
    tensorboard_add_scaler(tensorboard_writer, f"Loss/train/epoch-{epoch_num}", loss, step)
    # 向 TensorBoard 中添加学习率数据
    tensorboard_add_scaler(tensorboard_writer, "Learn rate/train", learn_rate, global_step)
    tensorboard_add_scaler(tensorboard_writer, f"Learn rate/train/epoch-{epoch_num}", learn_rate, step)

# 向 TensorBoard 中添加标量数据
def tensorboard_add_scaler(tensorboard_writer, tag, value, step):
    # 向 TensorBoard 中添加标量数据
    tensorboard_writer.add_scalar(tag=tag,
        scalar_value=value, global_step=step)

# 向 TensorBoard 中添加图像数据
def tensorboard_add_image(tensorboard_writer, tag, pil_image, step):
    # 将 PIL 图像转换为 Torch 张量
    img_tensor = torch.as_tensor(np.array(pil_image, copy=True))
    img_tensor = img_tensor.view(pil_image.size[1], pil_image.size[0],
        len(pil_image.getbands()))
    # 调整图像张量的维度顺序，将通道维度放在第一位
    img_tensor = img_tensor.permute((2, 0, 1))

    # 将图像张量添加到 TensorBoard 中，指定标签和全局步数
    tensorboard_writer.add_image(tag, img_tensor, global_step=step)
# 验证训练输入参数的有效性
def validate_train_inputs(model_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_model_every, create_image_every, log_directory, name="embedding"):
    # 确保选择了模型名称
    assert model_name, f"{name} not selected"
    # 确保学习率不为空且大于0
    assert learn_rate, "Learning rate is empty or 0"
    # 确保批量大小为整数
    assert isinstance(batch_size, int), "Batch size must be integer"
    # 确保批量大小为正数
    assert batch_size > 0, "Batch size must be positive"
    # 确保梯度累积步数为整数
    assert isinstance(gradient_step, int), "Gradient accumulation step must be integer"
    # 确保梯度累积步数为正数
    assert gradient_step > 0, "Gradient accumulation step must be positive"
    # 确保数据集目录不为空
    assert data_root, "Dataset directory is empty"
    # 确保数据集目录存在
    assert os.path.isdir(data_root), "Dataset directory doesn't exist"
    # 确保数据集目录不为空
    assert os.listdir(data_root), "Dataset directory is empty"
    # 确保选择了提示模板文件名
    assert template_filename, "Prompt template file not selected"
    # 确保找到了提示模板文件
    assert template_file, f"Prompt template file {template_filename} not found"
    # 确保提示模板文件存在
    assert os.path.isfile(template_file.path), f"Prompt template file {template_filename} doesn't exist"
    # 确保最大步数不为空且大于0
    assert steps, "Max steps is empty or 0"
    # 确保最大步数为整数
    assert isinstance(steps, int), "Max steps must be integer"
    # 确保最大步数为正数
    assert steps > 0, "Max steps must be positive"
    # 确保保存模型频率为整数
    assert isinstance(save_model_every, int), "Save {name} must be integer"
    # 确保保存模型频率大于等于0
    assert save_model_every >= 0, "Save {name} must be positive or 0"
    # 确保生成图像频率为整数
    assert isinstance(create_image_every, int), "Create image must be integer"
    # 确保生成图像频率大于等于0
    assert create_image_every >= 0, "Create image must be positive or 0"
    # 如果需要保存模型或生成图像，则确保日志目录不为空
    if save_model_every or create_image_every:
        assert log_directory, "Log directory is empty"
# 定义训练嵌入的函数，接受多个参数
def train_embedding(id_task, embedding_name, learn_rate, batch_size, gradient_step, data_root, log_directory, training_width, training_height, varsize, steps, clip_grad_mode, clip_grad_value, shuffle_tags, tag_drop_out, latent_sampling_method, use_weight, create_image_every, save_embedding_every, template_filename, save_image_with_stored_embedding, preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_name, preview_cfg_scale, preview_seed, preview_width, preview_height):
    # 导入 processing 模块
    from modules import processing

    # 如果 save_embedding_every 为假值，则设为 0
    save_embedding_every = save_embedding_every or 0
    # 如果 create_image_every 为假值，则设为 0
    create_image_every = create_image_every or 0
    # 获取指定模板文件
    template_file = textual_inversion_templates.get(template_filename, None)
    # 验证训练输入参数的有效性
    validate_train_inputs(embedding_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_embedding_every, create_image_every, log_directory, name="embedding")
    # 获取模板文件的路径
    template_file = template_file.path

    # 设置共享状态的作业为 "train-embedding"
    shared.state.job = "train-embedding"
    # 设置共享状态的文本信息为 "Initializing textual inversion training..."
    shared.state.textinfo = "Initializing textual inversion training..."
    # 设置共享状态的作业计数为 steps
    shared.state.job_count = steps

    # 生成嵌入文件的路径
    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')

    # 设置日志目录为当前日期下的 embedding_name 目录
    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)
    # 根据 shared.opts.unload_models_when_training 的值，决定是否卸载模型
    unload = shared.opts.unload_models_when_training

    # 如果 save_embedding_every 大于 0，则创建 embeddings 目录
    if save_embedding_every > 0:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None

    # 如果 create_image_every 大于 0，则创建 images 目录
    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    # 如果 create_image_every 大于 0 且 save_image_with_stored_embedding 为真，则创建 image_embeddings 目录
    if create_image_every > 0 and save_image_with_stored_embedding:
        images_embeds_dir = os.path.join(log_directory, "image_embeddings")
        os.makedirs(images_embeds_dir, exist_ok=True)
    else:
        images_embeds_dir = None
    # 从 sd_hijack 模块中导入 model_hijack 对象
    hijack = sd_hijack.model_hijack

    # 从 embedding_db 中获取指定名称的词嵌入
    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    
    # 选择模型的检查点
    checkpoint = sd_models.select_checkpoint()

    # 获取词嵌入的初始步数，如果已经超过指定的步数，则返回已经训练完成的信息
    initial_step = embedding.step or 0
    if initial_step >= steps:
        shared.state.textinfo = "Model has already been trained beyond specified max steps"
        return embedding, filename

    # 创建学习率调度器
    scheduler = LearnRateScheduler(learn_rate, steps, initial_step)
    
    # 根据 clip_grad_mode 的不同值选择不同的梯度裁剪方法
    clip_grad = torch.nn.utils.clip_grad_value_ if clip_grad_mode == "value" else \
        torch.nn.utils.clip_grad_norm_ if clip_grad_mode == "norm" else \
        None
    if clip_grad:
        # 如果需要梯度裁剪，则创建梯度裁剪的学习率调度器
        clip_grad_sched = LearnRateScheduler(clip_grad_value, steps, initial_step, verbose=False)
    
    # 在加载数据集之前，进行输入验证和提前返回
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    old_parallel_processing_allowed = shared.parallel_processing_allowed

    # 如果启用了 TensorBoard，则设置 TensorBoard 写入器
    if shared.opts.training_enable_tensorboard:
        tensorboard_writer = tensorboard_setup(log_directory)

    # 设置是否将数据加载到固定内存中
    pin_memory = shared.opts.pin_memory

    # 创建数据集对象
    ds = modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=shared.opts.training_image_repeats_per_epoch, placeholder_token=embedding_name, model=shared.sd_model, cond_model=shared.sd_model.cond_stage_model, device=devices.device, template_file=template_file, batch_size=batch_size, gradient_step=gradient_step, shuffle_tags=shuffle_tags, tag_drop_out=tag_drop_out, latent_sampling_method=latent_sampling_method, varsize=varsize, use_weight)

    # 如果需要将训练设置保存到文本文件中，则保存
    if shared.opts.save_training_settings_to_txt:
        save_settings_to_file(log_directory, {**dict(model_name=checkpoint.model_name, model_hash=checkpoint.shorthash, num_of_dataset_images=len(ds), num_vectors_per_token=len(embedding.vec)), **locals()})

    # 获取数据集的潜在采样方法
    latent_sampling_method = ds.latent_sampling_method
    # 创建个性化数据加载器，用于加载数据集并进行个性化处理
    dl = modules.textual_inversion.dataset.PersonalizedDataLoader(ds, latent_sampling_method=latent_sampling_method, batch_size=ds.batch_size, pin_memory=pin_memory)

    # 如果需要卸载模型
    if unload:
        # 禁止并行处理
        shared.parallel_processing_allowed = False
        # 将第一阶段模型移动到 CPU
        shared.sd_model.first_stage_model.to(devices.cpu)

    # 设置嵌入向量需要梯度计算
    embedding.vec.requires_grad = True
    # 使用 AdamW 优化器优化嵌入向量
    optimizer = torch.optim.AdamW([embedding.vec], lr=scheduler.learn_rate, weight_decay=0.0)
    # 如果需要保存优化器状态
    if shared.opts.save_optimizer_state:
        optimizer_state_dict = None
        # 如果存在优化器状态文件
        if os.path.exists(f"{filename}.optim"):
            # 加载优化器状态字典
            optimizer_saved_dict = torch.load(f"{filename}.optim", map_location='cpu')
            # 如果嵌入向量校验和与优化器状态字典中的哈希值匹配
            if embedding.checksum() == optimizer_saved_dict.get('hash', None):
                optimizer_state_dict = optimizer_saved_dict.get('optimizer_state_dict', None)

        # 如果存在优化器状态字典
        if optimizer_state_dict is not None:
            # 加载优化器状态
            optimizer.load_state_dict(optimizer_state_dict)
            print("Loaded existing optimizer from checkpoint")
        else:
            print("No saved optimizer exists in checkpoint")

    # 创建梯度缩放器
    scaler = torch.cuda.amp.GradScaler()

    # 获取批处理大小和梯度步数
    batch_size = ds.batch_size
    gradient_step = ds.gradient_step
    # 计算每个 epoch 的步数
    steps_per_epoch = len(ds) // batch_size // gradient_step
    max_steps_per_epoch = len(ds) // batch_size - (len(ds) // batch_size) % gradient_step
    loss_step = 0
    _loss_step = 0 #internal

    # 初始化最后保存的文件和图像名称
    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"
    embedding_yet_to_be_embedded = False

    # 判断是否正在训练修复模型
    is_training_inpainting_model = shared.sd_model.model.conditioning_key in {'hybrid', 'concat'}
    img_c = None

    # 创建进度条
    pbar = tqdm.tqdm(total=steps - initial_step)
<p>
Loss: {loss_step:.7f}<br/>
Step: {steps_done}<br/>
Last prompt: {html.escape(batch.cond_text[0])}<br/>
Last saved embedding: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""
        filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')
        # 保存嵌入到指定文件中
        save_embedding(embedding, optimizer, checkpoint, embedding_name, filename, remove_cached_checksum=True)
    except Exception:
        # 报告训练嵌入时出现的错误
        errors.report("Error training embedding", exc_info=True)
    finally:
        # 设置进度条的状态为 False
        pbar.leave = False
        # 关闭进度条
        pbar.close()
        # 将模型切换到指定设备
        shared.sd_model.first_stage_model.to(devices.device)
        # 恢复并行处理的允许状态
        shared.parallel_processing_allowed = old_parallel_processing_allowed
        # 移除 hijack 的检查点
        sd_hijack_checkpoint.remove()

    return embedding, filename


def save_embedding(embedding, optimizer, checkpoint, embedding_name, filename, remove_cached_checksum=True):
    # 保存旧的嵌入名称、检查点、检查点名称和缓存校验和
    old_embedding_name = embedding.name
    old_sd_checkpoint = embedding.sd_checkpoint if hasattr(embedding, "sd_checkpoint") else None
    old_sd_checkpoint_name = embedding.sd_checkpoint_name if hasattr(embedding, "sd_checkpoint_name") else None
    old_cached_checksum = embedding.cached_checksum if hasattr(embedding, "cached_checksum") else None
    try:
        # 设置嵌入的检查点和检查点名称，如果需要移除缓存校验和则设置为 None
        embedding.sd_checkpoint = checkpoint.shorthash
        embedding.sd_checkpoint_name = checkpoint.model_name
        if remove_cached_checksum:
            embedding.cached_checksum = None
        embedding.name = embedding_name
        embedding.optimizer_state_dict = optimizer.state_dict()
        # 保存嵌入到指定文件中
        embedding.save(filename)
    except:
        # 如果保存出错，则恢复旧的嵌入名称、检查点、检查点名称和缓存校验和，并抛出异常
        embedding.sd_checkpoint = old_sd_checkpoint
        embedding.sd_checkpoint_name = old_sd_checkpoint_name
        embedding.name = old_embedding_name
        embedding.cached_checksum = old_cached_checksum
        raise
```