# `.\pytorch\benchmarks\dynamo\timm_models.py`

```py
#!/usr/bin/env python3
# 导入所需的模块
import importlib
import logging
import os
import re
import subprocess
import sys
import warnings

try:
    # 尝试从当前目录下的common模块导入BenchmarkRunner, download_retry_decorator, main函数
    from .common import BenchmarkRunner, download_retry_decorator, main
except ImportError:
    # 如果导入失败，则从全局模块common中导入相应函数
    from common import BenchmarkRunner, download_retry_decorator, main

import torch

# 导入torch._dynamo.testing和torch._dynamo.utils模块下的特定函数和类
from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs

# 启用FX图缓存
if "TORCHINDUCTOR_FX_GRAPH_CACHE" not in os.environ:
    torch._inductor.config.fx_graph_cache = True

# 定义一个函数，用于通过pip安装指定的Python包
def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    # 尝试导入名为"timm"的Python模块
    importlib.import_module("timm")
except ModuleNotFoundError:
    # 如果模块未找到，输出安装信息，并安装特定版本的"pytorch-image-models"
    print("Installing PyTorch Image Models...")
    pip_install("git+https://github.com/rwightman/pytorch-image-models")
finally:
    # 无论导入是否成功，都从timm模块中导入其版本号和其他必要函数
    from timm import __version__ as timmversion
    from timm.data import resolve_data_config
    from timm.models import create_model

# 定义一个空的字典TIMM_MODELS，用于存储模型名称及其批处理大小
TIMM_MODELS = dict()
# 读取包含模型名称和批处理大小的文件"timm_models_list.txt"
filename = os.path.join(os.path.dirname(__file__), "timm_models_list.txt")

with open(filename) as fh:
    # 逐行读取文件内容，并去除每行的末尾换行符
    lines = fh.readlines()
    lines = [line.rstrip() for line in lines]
    # 遍历每一行，将模型名称和批处理大小分别存入TIMM_MODELS字典中
    for line in lines:
        model_name, batch_size = line.split(" ")
        TIMM_MODELS[model_name] = int(batch_size)

# TODO - Figure out the reason of cold start memory spike
# 根据模型名称定义不同的批处理大小除数
BATCH_SIZE_DIVISORS = {
    "beit_base_patch16_224": 2,
    "convit_base": 2,
    "convmixer_768_32": 2,
    "convnext_base": 2,
    "cspdarknet53": 2,
    "deit_base_distilled_patch16_224": 2,
    "gluon_xception65": 2,
    "mobilevit_s": 2,
    "pnasnet5large": 2,
    "poolformer_m36": 2,
    "resnest101e": 2,
    "swin_base_patch4_window7_224": 2,
    "swsl_resnext101_32x16d": 2,
    "vit_base_patch16_224": 2,
    "volo_d1_224": 2,
    "jx_nest_base": 4,
}

# 需要更高容忍度的模型集合
REQUIRE_HIGHER_TOLERANCE = {
    "fbnetv3_b",
    "gmixer_24_224",
    "hrnet_w18",
    "inception_v3",
    "mixer_b16_224",
    "mobilenetv3_large_100",
    "sebotnet33ts_256",
    "selecsls42b",
    "cspdarknet53",
}

# 冻结时需要更高容忍度的模型集合
REQUIRE_HIGHER_TOLERANCE_FOR_FREEZING = {
    "adv_inception_v3",
    "botnet26t_256",
    "gluon_inception_v3",
    "selecsls42b",
    "swsl_resnext101_32x16d",
}

# 需要缩放计算损失的模型集合
SCALED_COMPUTE_LOSS = {
    "ese_vovnet19b_dw",
    "fbnetc_100",
    "mnasnet_100",
    "mobilevit_s",
    "sebotnet33ts_256",
}

# 强制FP16或BF16模型使用AMP加速的集合
FORCE_AMP_FOR_FP16_BF16_MODELS = {
    "convit_base",
    "xcit_large_24_p8_224",
}

# 作为即时非确定性模型跳过准确性检查的集合
SKIP_ACCURACY_CHECK_AS_EAGER_NON_DETERMINISTIC_MODELS = {
    "xcit_large_24_p8_224",
}

# 定义一个函数，用于刷新模型名称列表
def refresh_model_names():
    import glob

    # 从timm模块中导入list_models函数
    from timm.models import list_models
    def read_models_from_docs():
        # 创建一个空集合用于存储模型名称
        models = set()
        # 遍历指定路径下所有以 .md 结尾的文件
        for fn in glob.glob("../pytorch-image-models/docs/models/*.md"):
            # 打开文件
            with open(fn) as f:
                # 循环读取文件的每一行
                while True:
                    line = f.readline()
                    # 如果到达文件末尾，跳出循环
                    if not line:
                        break
                    # 如果当前行不是以 "model = timm.create_model(" 开头，则继续下一行
                    if not line.startswith("model = timm.create_model("):
                        continue

                    # 提取模型名称（假设模型名称被单引号包围）
                    model = line.split("'")[1]
                    # 将模型名称添加到集合中
                    models.add(model)
        # 返回模型名称集合
        return models

    def get_family_name(name):
        # 已知的模型家族列表
        known_families = [
            "darknet",
            "densenet",
            "dla",
            "dpn",
            "ecaresnet",
            "halo",
            "regnet",
            "efficientnet",
            "deit",
            "mobilevit",
            "mnasnet",
            "convnext",
            "resnet",
            "resnest",
            "resnext",
            "selecsls",
            "vgg",
            "xception",
        ]

        # 遍历已知的模型家族列表
        for known_family in known_families:
            # 如果模型名称中包含已知的模型家族名称，则返回该家族名称
            if known_family in name:
                return known_family

        # 如果模型名称以 "gluon_" 开头，则返回 "gluon_" 后接下划线分割后的第二部分
        if name.startswith("gluon_"):
            return "gluon_" + name.split("_")[1]
        # 否则返回模型名称按下划线分割后的第一部分
        return name.split("_")[0]

    def populate_family(models):
        # 创建一个空字典用于存储模型家族信息
        family = dict()
        # 遍历模型名称列表
        for model_name in models:
            # 获取模型名称对应的家族名称
            family_name = get_family_name(model_name)
            # 如果家族名称不在字典中，则初始化空列表
            if family_name not in family:
                family[family_name] = []
            # 将模型名称添加到相应家族的列表中
            family[family_name].append(model_name)
        # 返回模型家族字典
        return family

    # 从文档中读取模型名称集合
    docs_models = read_models_from_docs()
    # 获取所有模型列表（假设这里有一个 list_models 函数）
    all_models = list_models(pretrained=True, exclude_filters=["*in21k"])

    # 构建所有模型的家族信息字典
    all_models_family = populate_family(all_models)
    # 构建文档中模型的家族信息字典
    docs_models_family = populate_family(docs_models)

    # 从所有模型家族中删除文档中已有的模型家族信息
    for key in docs_models_family:
        del all_models_family[key]

    # 选取首个模型作为已选择模型的集合
    chosen_models = set()
    chosen_models.update(value[0] for value in docs_models_family.values())

    # 将所有模型家族的首个模型添加到已选择模型的集合中
    chosen_models.update(value[0] for key, value in all_models_family.items())

    # 指定输出文件名
    filename = "timm_models_list.txt"
    # 如果存在 benchmarks 文件夹，则将文件名加入到 benchmarks 文件夹下
    if os.path.exists("benchmarks"):
        filename = "benchmarks/" + filename
    # 打开文件，将已选择模型按字母顺序写入文件中
    with open(filename, "w") as fw:
        for model_name in sorted(chosen_models):
            fw.write(model_name + "\n")
class TimmRunner(BenchmarkRunner):
    # TimmRunner 类，继承自 BenchmarkRunner 类，用于运行基准测试
    def __init__(self):
        super().__init__()  # 调用父类 BenchmarkRunner 的初始化方法
        self.suite_name = "timm_models"  # 设置属性 suite_name 为 "timm_models"

    @property
    def force_amp_for_fp16_bf16_models(self):
        # 返回是否强制对 FP16 和 BF16 模型使用 AMP 的布尔值
        return FORCE_AMP_FOR_FP16_BF16_MODELS

    @property
    def force_fp16_for_bf16_models(self):
        # 返回需要强制使用 FP16 的 BF16 模型集合
        return set()

    @property
    def get_output_amp_train_process_func(self):
        # 返回输出 AMP 训练过程函数的字典
        return {}

    @property
    def skip_accuracy_check_as_eager_non_deterministic(self):
        # 如果设置了参数 accuracy 和 training，则返回需要跳过精度检查的模型集合
        if self.args.accuracy and self.args.training:
            return SKIP_ACCURACY_CHECK_AS_EAGER_NON_DETERMINISTIC_MODELS
        return set()

    @property
    def guard_on_nn_module_models(self):
        # 返回需要保护的 NN 模块模型集合
        return {
            "convit_base",
        }

    @property
    def inline_inbuilt_nn_modules_models(self):
        # 返回内置 NN 模块模型集合
        return {
            "lcnet_050",
        }

    @download_retry_decorator
    def _download_model(self, model_name):
        # 下载指定模型的方法，使用 download_retry_decorator 进行装饰
        model = create_model(
            model_name,
            in_chans=3,
            scriptable=False,
            num_classes=None,
            drop_rate=0.0,
            drop_path_rate=None,
            drop_block_rate=None,
            pretrained=True,
        )
        return model  # 返回创建的模型对象

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        extra_args=None,
        ):
            # 如果启用激活检查点，则抛出未实现的错误，因为 Timm 模型不支持激活检查点
            if self.args.enable_activation_checkpointing:
                raise NotImplementedError(
                    "Activation checkpointing not implemented for Timm models"
                )

        # 根据参数确定模型是否处于训练状态
        is_training = self.args.training
        # 根据参数确定是否使用评估模式
        use_eval_mode = self.args.use_eval_mode

        # 获取参数中的通道顺序配置
        channels_last = self._args.channels_last
        # 下载指定模型
        model = self._download_model(model_name)

        # 如果模型下载失败，则抛出运行时错误
        if model is None:
            raise RuntimeError(f"Failed to load model '{model_name}'")
        # 将模型转移到指定设备上，并根据通道顺序配置设置内存格式
        model.to(
            device=device,
            memory_format=torch.channels_last if channels_last else None,
        )

        # 获取模型的类别数
        self.num_classes = model.num_classes

        # 解析数据配置，根据参数和模型配置获取输入尺寸等信息
        data_config = resolve_data_config(
            vars(self._args) if timmversion >= "0.8.0" else self._args,
            model=model,
            use_test_size=not is_training,
        )
        # 获取输入尺寸
        input_size = data_config["input_size"]
        # 获取记录的批量大小
        recorded_batch_size = TIMM_MODELS[model_name]

        # 如果模型在批量大小除数字典中，则根据其值更新记录的批量大小
        if model_name in BATCH_SIZE_DIVISORS:
            recorded_batch_size = max(
                int(recorded_batch_size / BATCH_SIZE_DIVISORS[model_name]), 1
            )
        # 如果没有指定批量大小，则使用记录的批量大小
        batch_size = batch_size or recorded_batch_size

        # 设置随机种子
        torch.manual_seed(1337)
        # 生成随机整数张量作为示例输入，并根据设备类型和数据类型进行设置
        input_tensor = torch.randint(
            256, size=(batch_size,) + input_size, device=device
        ).to(dtype=torch.float32)
        # 计算输入张量的均值和标准差
        mean = torch.mean(input_tensor)
        std_dev = torch.std(input_tensor)
        # 标准化输入张量
        example_inputs = (input_tensor - mean) / std_dev

        # 如果使用通道顺序模式，则确保输入张量的连续性设置为通道顺序
        if channels_last:
            example_inputs = example_inputs.contiguous(
                memory_format=torch.channels_last
            )
        # 将示例输入放入列表中
        example_inputs = [
            example_inputs,
        ]

        # 生成目标数据，用于模型验证
        self.target = self._gen_target(batch_size, device)

        # 设置损失函数为交叉熵损失函数，并移至指定设备
        self.loss = torch.nn.CrossEntropyLoss().to(device)

        # 如果模型在缩放计算损失模型列表中，则使用缩放版本的损失计算函数
        if model_name in SCALED_COMPUTE_LOSS:
            self.compute_loss = self.scaled_compute_loss

        # 如果处于训练状态且未使用评估模式，则将模型设为训练模式；否则设为评估模式
        if is_training and not use_eval_mode:
            model.train()
        else:
            model.eval()

        # 验证模型并生成示例输入
        self.validate_model(model, example_inputs)

        # 返回设备类型、模型名称、模型对象、示例输入、批量大小
        return device, model_name, model, example_inputs, batch_size

    def iter_model_names(self, args):
        # 获取所有预训练模型的名称列表，并按字母顺序排列
        model_names = sorted(TIMM_MODELS.keys())
        # 根据参数获取起始和结束索引，用于限制模型名称的迭代范围
        start, end = self.get_benchmark_indices(len(model_names))
        # 遍历模型名称列表中指定范围内的模型名称
        for index, model_name in enumerate(model_names):
            # 如果当前索引不在指定的起始和结束范围内，则跳过
            if index < start or index >= end:
                continue
            # 如果模型名称不匹配过滤器列表中的任何一个条件，则跳过当前模型
            if (
                not re.search("|".join(args.filter), model_name, re.I)
                or re.search("|".join(args.exclude), model_name, re.I)
                or model_name in args.exclude_exact
                or model_name in self.skip_models
            ):
                continue

            # 返回符合条件的模型名称
            yield model_name
    def pick_grad(self, name, is_training):
        # 根据是否处于训练模式选择梯度计算策略
        if is_training:
            # 如果处于训练模式，启用梯度计算
            return torch.enable_grad()
        else:
            # 如果非训练模式，关闭梯度计算
            return torch.no_grad()

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        # 默认余弦参数
        cosine = self.args.cosine
        # 容忍度初始值
        tolerance = 1e-3

        # 如果开启了冻结并且名称在需要更高容忍度的列表中
        if self.args.freezing and name in REQUIRE_HIGHER_TOLERANCE_FOR_FREEZING:
            # 在冻结状态下，卷积-批量归一化融合可能导致较大的数值差异，需要更高的容忍度
            # 参考 https://github.com/pytorch/pytorch/issues/120545
            tolerance = 8 * 1e-2

        # 如果处于训练模式
        if is_training:
            # 如果名称为 "levit_128"
            if name in ["levit_128"]:
                tolerance = 8 * 1e-2
            # 如果名称在需要更高容忍度的列表中
            elif name in REQUIRE_HIGHER_TOLERANCE:
                tolerance = 4 * 1e-2
            else:
                # 否则使用默认容忍度
                tolerance = 1e-2
        # 返回容忍度和余弦标志
        return tolerance, cosine

    def _gen_target(self, batch_size, device):
        # 生成目标张量
        return torch.empty((batch_size,) + (), device=device, dtype=torch.long).random_(
            self.num_classes
        )

    def compute_loss(self, pred):
        # 计算损失值，高损失值会使梯度检查更困难，因为积累顺序的微小变化会影响精度检查
        return reduce_to_scalar_loss(pred)

    def scaled_compute_loss(self, pred):
        # 缩放后计算损失值，损失值需要进一步放大
        return reduce_to_scalar_loss(pred) / 1000.0

    def forward_pass(self, mod, inputs, collect_outputs=True):
        # 前向传播过程，自动使用混合精度加速
        with self.autocast(**self.autocast_arg):
            return mod(*inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        # 前向和反向传播过程
        cloned_inputs = clone_inputs(inputs)
        # 优化器梯度清零
        self.optimizer_zero_grad(mod)
        # 前向传播，使用自动混合精度
        with self.autocast(**self.autocast_arg):
            pred = mod(*cloned_inputs)
            # 如果预测结果是元组，取第一个元素作为预测值
            if isinstance(pred, tuple):
                pred = pred[0]
            # 计算损失值
            loss = self.compute_loss(pred)
        # 使用梯度放大器进行梯度缩放
        self.grad_scaler.scale(loss).backward()
        # 优化器执行一步优化
        self.optimizer_step()
        # 如果需要收集输出结果，则返回模型、预测值、损失值和克隆的输入
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        # 否则返回空值
        return None
# 定义一个函数 `timm_main()`，用于执行特定的任务，通常是运行一个主程序或工作流程
def timm_main():
    # 配置日志系统，设置日志级别为警告（Warning）
    logging.basicConfig(level=logging.WARNING)
    # 忽略所有警告
    warnings.filterwarnings("ignore")
    # 调用 `main()` 函数，并传入 `TimmRunner()` 的实例作为参数
    main(TimmRunner())


# 如果脚本作为主程序执行，则执行 `timm_main()` 函数
if __name__ == "__main__":
    timm_main()
```