# `.\models\glpn\convert_glpn_to_pytorch.py`

```py
# 设置代码文件的编码格式为UTF-8
# 版权声明，指明了代码的版权归属和使用许可
# 引入必要的库和模块
# 从指定路径导入Path类
# 从requests模块导入请求函数
# 导入torch模块
# 从PIL库导入Image类
# 从transformers模块导入GLPNConfig, GLPNForDepthEstimation, GLPNImageProcessor类
# 从transformers.utils模块导入logging模块
# 设置日志的输出级别为INFO
# 获取指定名称的日志器logger
# 定义一个函数，用于重命名模型的参数名
    # 遍历状态字典中的键值对
    for key, value in state_dict.items():
        # 如果键以 "module.encoder" 开头，则替换为 "glpn.encoder"
        if key.startswith("module.encoder"):
            key = key.replace("module.encoder", "glpn.encoder")
        # 如果键以 "module.decoder" 开头，则替换为 "decoder.stages"
        if key.startswith("module.decoder"):
            key = key.replace("module.decoder", "decoder.stages")
        # 如果键中包含 "patch_embed"，则进行替换
        if "patch_embed" in key:
            # 例如将 "patch_embed1" 替换为 "patch_embeddings.0"
            idx = key[key.find("patch_embed") + len("patch_embed")]
            key = key.replace(f"patch_embed{idx}", f"patch_embeddings.{int(idx)-1}")
        # 如果键中包含 "norm"，则替换为 "layer_norm"
        if "norm" in key:
            key = key.replace("norm", "layer_norm")
        # 如果键中包含 "glpn.encoder.layer_norm"，则进行替换
        if "glpn.encoder.layer_norm" in key:
            idx = key[key.find("glpn.encoder.layer_norm") + len("glpn.encoder.layer_norm")]
            # 例如将 "layer_norm1" 替换为 "layer_norm.0"
            key = key.replace(f"layer_norm{idx}", f"layer_norm.{int(idx)-1}")
        # 如果键为 "layer_norm1"，则替换为 "layer_norm_1"
        if "layer_norm1" in key:
            key = key.replace("layer_norm1", "layer_norm_1")
        # 如果键为 "layer_norm2"，则替换为 "layer_norm_2"
        if "layer_norm2" in key:
            key = key.replace("layer_norm2", "layer_norm_2")
        # 如果键中包含 "block"，则进行替换
        if "block" in key:
            idx = key[key.find("block") + len("block")]
            # 例如将 "block1" 替换为 "block.0"
            key = key.replace(f"block{idx}", f"block.{int(idx)-1}")
        # 如果键中包含 "attn.q"，则替换为 "attention.self.query"
        if "attn.q" in key:
            key = key.replace("attn.q", "attention.self.query")
        # 如果键中包含 "attn.proj"，则替换为 "attention.output.dense"
        if "attn.proj" in key:
            key = key.replace("attn.proj", "attention.output.dense")
        # 如果键中包含 "attn"，则替换为 "attention.self"
        if "attn" in key:
            key = key.replace("attn", "attention.self")
        # 如果键中包含 "fc1"，则替换为 "dense1"
        if "fc1" in key:
            key = key.replace("fc1", "dense1")
        # 如果键中包含 "fc2"，则替换为 "dense2"
        if "fc2" in key:
            key = key.replace("fc2", "dense2")
        # 如果键中包含 "linear_pred"，则替换为 "classifier"
        if "linear_pred" in key:
            key = key.replace("linear_pred", "classifier")
        # 如果键中包含 "linear_fuse"，则进行替换
        if "linear_fuse" in key:
            key = key.replace("linear_fuse.conv", "linear_fuse")
            key = key.replace("linear_fuse.bn", "batch_norm")
        # 如果键中包含 "linear_c"，则进行替换
        if "linear_c" in key:
            idx = key[key.find("linear_c") + len("linear_c")]
            # 例如将 "linear_c4" 替换为 "linear_c.3"
            key = key.replace(f"linear_c{idx}", f"linear_c.{int(idx)-1}")
        # 如果键中包含 "bot_conv"，则替换为 "0.convolution"
        if "bot_conv" in key:
            key = key.replace("bot_conv", "0.convolution")
        # 如果键中包含 "skip_conv1"，则替换为 "1.convolution"
        if "skip_conv1" in key:
            key = key.replace("skip_conv1", "1.convolution")
        # 如果键中包含 "skip_conv2"，则替换为 "2.convolution"
        if "skip_conv2" in key:
            key = key.replace("skip_conv2", "2.convolution")
        # 如果键中包含 "fusion1"，则替换为 "1.fusion"
        if "fusion1" in key:
            key = key.replace("fusion1", "1.fusion")
        # 如果键中包含 "fusion2"，则替换为 "2.fusion"
        if "fusion2" in key:
            key = key.replace("fusion2", "2.fusion")
        # 如果键中包含 "fusion3"，则替换为 "3.fusion"
        if "fusion3" in key:
            key = key.replace("fusion3", "3.fusion")
        # 如果键中包含 "fusion" 且 "conv"，则进行替换
        if "fusion" in key and "conv" in key:
            key = key.replace("conv", "convolutional_layer")
        # 如果键以 "module.last_layer_depth" 开头，则替换为 "head.head"
        if key.startswith("module.last_layer_depth"):
            key = key.replace("module.last_layer_depth", "head.head")
        # 将经过修改后的键值对加入新的状态字典
        new_state_dict[key] = value

    # 返回修改后的状态字典
    return new_state_dict
# 从状态字典中读取键和值的权重和偏置，然后添加到状态字典中
def read_in_k_v(state_dict, config):
    # 对于每个编码器块：
    for i in range(config.num_encoder_blocks):
        for j in range(config.depths[i]):
            # 读入键和值的权重和偏置（在原始实现中是单独的矩阵）
            kv_weight = state_dict.pop(f"glpn.encoder.block.{i}.{j}.attention.self.kv.weight")
            kv_bias = state_dict.pop(f"glpn.encoder.block.{i}.{j}.attention.self.kv.bias")
            # 接下来，按顺序将键和值添加到状态字典中
            state_dict[f"glpn.encoder.block.{i}.{j}.attention.self.key.weight"] = kv_weight[
                : config.hidden_sizes[i], :
            ]
            state_dict[f"glpn.encoder.block.{i}.{j}.attention.self.key.bias"] = kv_bias[: config.hidden_sizes[i]]
            state_dict[f"glpn.encoder.block.{i}.{j}.attention.self.value.weight"] = kv_weight[
                config.hidden_sizes[i] :, :
            ]
            state_dict[f"glpn.encoder.block.{i}.{j}.attention.self.value.bias"] = kv_bias[config.hidden_sizes[i] :]


# 我们将在 COCO 图像上验证我们的结果
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    return image


@torch.no_grad()
def convert_glpn_checkpoint(checkpoint_path, pytorch_dump_folder_path, push_to_hub=False, model_name=None):
    """
    将模型的权重复制/粘贴/调整到我们的 GLPN 结构中。
    """

    # 加载 GLPN 配置（Segformer-B4 大小）
    config = GLPNConfig(hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=64, depths=[3, 8, 27, 3])

    # 加载图像处理器（仅调整大小 + 重新缩放）
    image_processor = GLPNImageProcessor()

    # 准备图像
    image = prepare_img()
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    logger.info("转换模型...")

    # 加载原始状态字典
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # 重命名键
    state_dict = rename_keys(state_dict)

    # 键和值矩阵需要特殊处理
    read_in_k_v(state_dict, config)

    # 创建 HuggingFace 模型并加载状态字典
    model = GLPNForDepthEstimation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 前向传播
    outputs = model(pixel_values)
    predicted_depth = outputs.predicted_depth

    # 验证输出
    # 如果模型名称不为空
    if model_name is not None:
        # 如果模型名称中包含“nyu”
        if "nyu" in model_name:
            # 设定预期的切片值
            expected_slice = torch.tensor(
                [[4.4147, 4.0873, 4.0673], [3.7890, 3.2881, 3.1525], [3.7674, 3.5423, 3.4913]]
            )
        # 如果模型名称中包含“kitti”
        elif "kitti" in model_name:
            # 设定预期的切片值
            expected_slice = torch.tensor(
                [[3.4291, 2.7865, 2.5151], [3.2841, 2.7021, 2.3502], [3.1147, 2.4625, 2.2481]]
            )
        else:
            # 抛出数值错误，指明未知的模型名称
            raise ValueError(f"Unknown model name: {model_name}")

        # 设定预期的形状
        expected_shape = torch.Size([1, 480, 640])

        # 断言预测的深度数据的形状是否与预期形状相同
        assert predicted_depth.shape == expected_shape
        # 断言预测的深度数据的前3x3部分是否与预期切片相近，允许误差为1e-4
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=1e-4)
        # 打印“Looks ok!”，表示一切正常

    # 最后，如果需要推送到hub
    if push_to_hub:
        # 记录信息，表示正在将模型和图像处理器推送到hub上
        logger.info("Pushing model and image processor to the hub...")
        # 将模型推送到hub
        model.push_to_hub(
            repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
            organization="nielsr",
            commit_message="Add model",
            use_temp_dir=True,
        )
        # 将图像处理器推送到hub
        image_processor.push_to_hub(
            repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
            organization="nielsr",
            commit_message="Add image processor",
            use_temp_dir=True,
        )
# 如果当前文件是作为主程序执行
if __name__ == "__main__":
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser()

    # 添加命令行参数，指定原始PyTorch检查点（.pth文件）的路径
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="Path to the original PyTorch checkpoint (.pth file).",
    )
    # 添加命令行参数，指定输出PyTorch模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    # 添加命令行参数，指定是否将模型上传到HuggingFace hub
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether to upload the model to the HuggingFace hub."
    )
    # 添加命令行参数，指定模型名称（用于上传到hub时）
    parser.add_argument(
        "--model_name",
        default="glpn-kitti",
        type=str,
        help="Name of the model in case you're pushing to the hub.",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用convert_glpn_checkpoint函数，传入命令行参数作为参数
    convert_glpn_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub, args.model_name)
```