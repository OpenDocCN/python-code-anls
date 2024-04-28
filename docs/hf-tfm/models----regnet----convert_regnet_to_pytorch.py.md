# `.\transformers\models\regnet\convert_regnet_to_pytorch.py`

```
# 设置文件编码和版权信息
# 导入所需模块和类
# 定义数据类，用于存储信息
# 初始化跟踪器对象，用于记录模块和钩子
# 定义前向钩子函数，用于记录模块
# 定义跟踪器的调用函数，用于注册前向钩子并执行模块的前向传播
# 定义跟踪器的parametrized属性，用于筛选具有可学习参数的模块
# 定义模块迁移类，用于将源模块的参数迁移到目标模块中
    # 将权重从 self.src 模块传递到 self.dest 模块，通过对 x 进行前向传递实现。在底层，我们追踪了两个模块中的所有操作。
    def __call__(self, x: Tensor):
        # 创建用于追踪的 self.dest 模块的副本，并对其进行参数化
        dest_traced = Tracker(self.dest)(x).parametrized
        # 创建用于追踪的 self.src 模块的副本，并对其进行参数化
        src_traced = Tracker(self.src)(x).parametrized

        # 从 src_traced 中过滤掉类型在 self.src_skip 中的操作
        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        # 从 dest_traced 中过滤掉类型在 self.dest_skip 中的操作
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))

        # 如果 dest_traced 和 src_traced 的长度不相等，并且 self.raise_if_mismatch 为 True，则抛出异常
        if len(dest_traced) != len(src_traced) and self.raise_if_mismatch:
            raise Exception(
                f"Numbers of operations are different. Source module has {len(src_traced)} operations while"
                f" destination module has {len(dest_traced)}."
            )

        # 将 dest_traced 和 src_traced 中的模块逐个进行权重加载
        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            # 如果 self.verbose 为 1，则打印从 src_m 到 dest_m 的迁移信息
            if self.verbose == 1:
                print(f"Transfered from={src_m} to={dest_m}")
class FakeRegNetVisslWrapper(nn.Module):
    """
    伪装的 RegNet 包装器，模仿 vissl 的行为而无需传递配置文件。
    """

    def __init__(self, model: nn.Module):
        super().__init__()

        feature_blocks: List[Tuple[str, nn.Module]] = []
        # - 获取网络结构的主干部分
        feature_blocks.append(("conv1", model.stem))
        # - 获取所有的特征块
        for k, v in model.trunk_output.named_children():
            assert k.startswith("block"), f"Unexpected layer name {k}"
            block_index = len(feature_blocks) + 1
            feature_blocks.append((f"res{block_index}", v))

        self._feature_blocks = nn.ModuleDict(feature_blocks)

    def forward(self, x: Tensor):
        return get_trunk_forward_outputs(
            x,
            out_feat_keys=None,
            feature_blocks=self._feature_blocks,
        )



class NameToFromModelFuncMap(dict):
    """
    一个带有额外逻辑的字典，用于返回一个创建正确的原始模型的函数。
    """

    def convert_name_to_timm(self, x: str) -> str:
        x_split = x.split("-")
        return x_split[0] + x_split[1] + "_" + "".join(x_split[2:])

    def __getitem__(self, x: str) -> Callable[[], Tuple[nn.Module, Dict]]:
        # 默认使用 timm！
        if x not in self:
            x = self.convert_name_to_timm(x)
            val = partial(lambda: (timm.create_model(x, pretrained=True).eval(), None))

        else:
            val = super().__getitem__(x)

        return val
    # 如果存在已有的状态字典，则创建空列表以备使用
    if from_state_dict is not None:
        keys = []
        # 对于“seer - in1k finetuned”模型，需要手动复制头部权重和偏置
        if "seer" in name and "in1k" in name:
            keys = [("0.clf.0.weight", "classifier.1.weight"), ("0.clf.0.bias", "classifier.1.bias")]
        # 手动复制 vissl 头部并返回到状态字典中
        to_state_dict = manually_copy_vissl_head(from_state_dict, our_model.state_dict(), keys)
        # 加载新的状态字典
        our_model.load_state_dict(to_state_dict)

    # 对输入进行模型预测，输出隐藏状态
    our_outputs = our_model(x, output_hidden_states=True)
    # 如果模型是 RegNetForImageClassification 类型，则输出 logits；否则输出最后隐藏状态
    our_output = (
        our_outputs.logits if isinstance(our_model, RegNetForImageClassification) else our_outputs.last_hidden_state
    )

    # 对输入进行原始模型预测
    from_output = from_model(x)
    # 如果输出是列表类型，则取最后一个元素
    from_output = from_output[-1] if isinstance(from_output, list) else from_output

    # 如果模型名中包含"seer"和"in1k"，则使用隐藏状态的最后一个来作为输出
    if "seer" in name and "in1k" in name:
        our_output = our_outputs.hidden_states[-1]

    # 断言两个输出是否很接近，如果不是则触发异常
    assert torch.allclose(from_output, our_output), "The model logits don't match the original one."

    # 如果要推送到托管服务
    if push_to_hub:
        # 将模型推送到指定目录或名称的仓库中
        our_model.push_to_hub(
            repo_path_or_name=save_directory / name,
            commit_message="Add model",
            use_temp_dir=True,
        )
        
        # 根据模型名称和大小创建图像处理器实例，并推送到仓库
        size = 224 if "seer" not in name else 384
        image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k-1k", size=size)
        image_processor.push_to_hub(
            repo_path_or_name=save_directory / name,
            commit_message="Add image processor",
            use_temp_dir=True,
        )
        
        # 打印推送成功的信息
        print(f"Pushed {name}")
# 定义函数，用于转换权重并推送到 Hub
def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    # 声明文件名和预期的形状
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000
    expected_shape = (1, num_labels)

    # 设置仓库标识、标签数量，加载并处理标签数据
    repo_id = "huggingface/label-files"
    num_labels = num_labels
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}

    # 配置 ImageNet 预训练模型
    ImageNetPreTrainedConfig = partial(RegNetConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)

    # 初始化模型映射
    names_to_ours_model_map = NameToOurModelFuncMap()
    names_to_from_model_map = NameToFromModelFuncMap()

    # 定义使用 classy vision 加载模型的函数
    def load_using_classy_vision(checkpoint_url: str, model_func: Callable[[], nn.Module]) -> Tuple[nn.Module, Dict]:
        # 从给定 URL 加载模型状态字典
        files = torch.hub.load_state_dict_from_url(checkpoint_url, model_dir=str(save_directory), map_location="cpu")
        # 调用模型函数，初始化模型
        model = model_func()
        # 检查是否有头部，如果有，就添加
        model_state_dict = files["classy_state_dict"]["base_model"]["model"]
        state_dict = model_state_dict["trunk"]
        model.load_state_dict(state_dict)
        # 返回模型和头部状态字典
        return model.eval(), model_state_dict["heads"]

    # 预训练的模型加载信息
    names_to_from_model_map["regnet-y-320-seer"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet32d/seer_regnet32gf_model_iteration244000.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY32gf()),
    )

    names_to_from_model_map["regnet-y-640-seer"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet64/seer_regnet64gf_model_final_checkpoint_phase0.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY64gf()),
    )

    names_to_from_model_map["regnet-y-1280-seer"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_regnet128Gf_cnstant_bs32_node16_sinkhorn10_proto16k_syncBN64_warmup8k/model_final_checkpoint_phase0.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY128gf()),
    )

    names_to_from_model_map["regnet-y-10b-seer"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet10B/model_iteration124500_conso.torch",
        lambda: FakeRegNetVisslWrapper(
            RegNet(RegNetParams(depth=27, group_width=1010, w_0=1744, w_a=620.83, w_m=2.52))
        ),
    )

    # IN1K finetuned 模型加载信息
    names_to_from_model_map["regnet-y-320-seer-in1k"] = partial(
        load_using_classy_vision,
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet32_finetuned_in1k_model_final_checkpoint_phase78.torch",
        lambda: FakeRegNetVisslWrapper(RegNetY32gf()),
    )
    # 将模型名称映射到模型加载函数的部分
    names_to_from_model_map["regnet-y-640-seer-in1k"] = partial(
        # 使用 Classy Vision 的加载函数加载模型
        load_using_classy_vision,
        # 模型的 URL 地址
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet64_finetuned_in1k_model_final_checkpoint_phase78.torch",
        # 创建一个假的 RegNetVisslWrapper 对象，使用 RegNetY64gf 模型
        lambda: FakeRegNetVisslWrapper(RegNetY64gf()),
    )
    
    # 将模型名称映射到模型加载函数的部分
    names_to_from_model_map["regnet-y-1280-seer-in1k"] = partial(
        # 使用 Classy Vision 的加载函数加载模型
        load_using_classy_vision,
        # 模型的 URL 地址
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet128_finetuned_in1k_model_final_checkpoint_phase78.torch",
        # 创建一个假的 RegNetVisslWrapper 对象，使用 RegNetY128gf 模型
        lambda: FakeRegNetVisslWrapper(RegNetY128gf()),
    )
    
    # 将模型名称映射到模型加载函数的部分
    names_to_from_model_map["regnet-y-10b-seer-in1k"] = partial(
        # 使用 Classy Vision 的加载函数加载模型
        load_using_classy_vision,
        # 模型的 URL 地址
        "https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_10b_finetuned_in1k_model_phase28_conso.torch",
        # 创建一个假的 RegNetVisslWrapper 对象，使用自定义的 RegNet 参数
        lambda: FakeRegNetVisslWrapper(
            RegNet(RegNetParams(depth=27, group_width=1010, w_0=1744, w_a=620.83, w_m=2.52))
        ),
    )
    
    # 如果存在模型名称，则转换权重并推送到 Hub
    if model_name:
        convert_weight_and_push(
            # 模型名称
            model_name,
            # 使用模型名称从映射中获取加载函数
            names_to_from_model_map[model_name],
            # 使用模型名称从另一个映射中获取我们的模型映射
            names_to_ours_model_map[model_name],
            # 使用模型名称从配置映射中获取配置
            names_to_config[model_name],
            # 保存目录
            save_directory,
            # 是否推送到 Hub
            push_to_hub,
        )
    else:
        # 否则，对每个模型名称及其配置执行转换权重并推送到 Hub
        for model_name, config in names_to_config.items():
            convert_weight_and_push(
                # 模型名称
                model_name,
                # 使用模型名称从映射中获取加载函数
                names_to_from_model_map[model_name],
                # 使用模型名称从另一个映射中获取我们的模型映射
                names_to_ours_model_map[model_name],
                # 配置
                config,
                # 保存目录
                save_directory,
                # 是否推送到 Hub
                push_to_hub,
            )
    # 返回配置和预期形状
    return config, expected_shape
if __name__ == "__main__":
    # 创建命令行解析器对象
    parser = argparse.ArgumentParser()
    # 添加必填参数
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help=(
            "The name of the model you wish to convert, it must be one of the supported regnet* architecture,"
            " currently: regnetx-*, regnety-*. If `None`, all of them will the converted."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=Path,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        default=True,
        type=bool,
        required=False,
        help="If True, push model and image processor to the hub.",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 设置 PyTorch 模型存储路径
    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    # 如果路径不存在，则创建目录
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    # 转换权重并推送到 Hub
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
```