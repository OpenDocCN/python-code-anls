# `.\models\bert\convert_bert_pytorch_checkpoint_to_original_tf.py`

```
# 设置 Python 文件的编码格式为 UTF-8
# Copyright 2018 The HuggingFace Inc. team.
# 声明脚本使用 Apache 2.0 版本许可协议，详见链接
# 只有符合许可协议的情况下才能使用本文件
# 请访问上述链接获取详细信息
# 除非法律另有规定或书面同意，否则不得使用此文件
# 此文件按原样发布，不附带任何形式的保证或条件
# 详见许可协议以了解更多信息

"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint."""
# 导入需要的库和模块
import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统相关功能模块

import numpy as np  # 导入数值计算库 NumPy
import tensorflow as tf  # 导入 TensorFlow 深度学习框架
import torch  # 导入 PyTorch 深度学习框架

from transformers import BertModel  # 从 transformers 库中导入 BertModel 类


def convert_pytorch_checkpoint_to_tf(model: BertModel, ckpt_dir: str, model_name: str):
    """
    Args:
        model: BertModel Pytorch model instance to be converted
        ckpt_dir: Tensorflow model directory
        model_name: model name

    Currently supported HF models:

        - Y BertModel
        - N BertForMaskedLM
        - N BertForPreTraining
        - N BertForMultipleChoice
        - N BertForNextSentencePrediction
        - N BertForSequenceClassification
        - N BertForQuestionAnswering
    """
    # 定义需要转置的张量名称列表
    tensors_to_transpose = ("dense.weight", "attention.self.query", "attention.self.key", "attention.self.value")

    # 定义变量名称映射规则列表
    var_map = (
        ("layer.", "layer_"),
        ("word_embeddings.weight", "word_embeddings"),
        ("position_embeddings.weight", "position_embeddings"),
        ("token_type_embeddings.weight", "token_type_embeddings"),
        (".", "/"),
        ("LayerNorm/weight", "LayerNorm/gamma"),
        ("LayerNorm/bias", "LayerNorm/beta"),
        ("weight", "kernel"),
    )

    # 如果指定的 TensorFlow 模型目录不存在，则创建该目录
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # 获取 PyTorch 模型的状态字典
    state_dict = model.state_dict()

    def to_tf_var_name(name: str):
        # 根据变量映射规则将 PyTorch 变量名转换为 TensorFlow 变量名
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return f"bert/{name}"

    def create_tf_var(tensor: np.ndarray, name: str, session: tf.Session):
        # 根据张量的数据类型和形状，在 TensorFlow 中创建新的变量
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    # 重置 TensorFlow 默认计算图
    tf.reset_default_graph()
    # 使用 TensorFlow 创建一个会话（Session），并将其命名为 session
    with tf.Session() as session:
        # 遍历 state_dict 中的每一个变量名
        for var_name in state_dict:
            # 将变量名 var_name 转换为 TensorFlow 的变量名 tf_name
            tf_name = to_tf_var_name(var_name)
            # 将 PyTorch 张量转换为 NumPy 数组，存储在 torch_tensor 中
            torch_tensor = state_dict[var_name].numpy()
            # 如果 var_name 中包含在 tensors_to_transpose 中的任何字符串，则对 torch_tensor 进行转置操作
            if any(x in var_name for x in tensors_to_transpose):
                torch_tensor = torch_tensor.T
            # 使用 create_tf_var 函数在 TensorFlow 中创建变量 tf_var，使用 session 进行管理
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            # 将 torch_tensor 转换为 tf_var 的数据类型，并赋值给 tf_var
            tf_var.assign(tf.cast(torch_tensor, tf_var.dtype))
            # 在 TensorFlow 中运行 tf_var，将结果存储在 tf_weight 中
            tf_weight = session.run(tf_var)
            # 打印成功创建的 TensorFlow 变量 tf_name 和其与 torch_tensor 是否全部接近的比较结果
            print(f"Successfully created {tf_name}: {np.allclose(tf_weight, torch_tensor)}")

        # 使用 tf.train.Saver() 创建一个 Saver 对象，保存所有可训练变量的状态
        saver = tf.train.Saver(tf.trainable_variables())
        # 将这些变量的状态保存到指定的文件路径下，文件名为 model_name 替换 '-' 为 '_' 后加上 '.ckpt' 后缀
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))
# 主函数，程序的入口点
def main(raw_args=None):
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加解析器的命令行参数：模型名称，必须提供，用于指定模型的名称，如 google-bert/bert-base-uncased
    parser.add_argument("--model_name", type=str, required=True, help="model name e.g. google-bert/bert-base-uncased")
    # 添加解析器的命令行参数：缓存目录，可选，默认为 None，用于指定包含 PyTorch 模型的目录
    parser.add_argument(
        "--cache_dir", type=str, default=None, required=False, help="Directory containing pytorch model"
    )
    # 添加解析器的命令行参数：PyTorch 模型路径，必须提供，用于指定 PyTorch 模型的路径，如 /path/to/<pytorch-model-name>.bin
    parser.add_argument("--pytorch_model_path", type=str, required=True, help="/path/to/<pytorch-model-name>.bin")
    # 添加解析器的命令行参数：TensorFlow 缓存目录，必须提供，用于指定保存 TensorFlow 模型的目录
    parser.add_argument("--tf_cache_dir", type=str, required=True, help="Directory in which to save tensorflow model")
    # 解析命令行参数，将结果存储在 args 中
    args = parser.parse_args(raw_args)

    # 从预训练模型中加载 BertModel 对象
    model = BertModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name,  # 使用命令行参数指定的模型名称或路径
        state_dict=torch.load(args.pytorch_model_path),  # 加载指定路径下的 PyTorch 模型参数
        cache_dir=args.cache_dir,  # 使用命令行参数指定的缓存目录
    )

    # 将 PyTorch 模型转换为 TensorFlow 格式的检查点文件
    convert_pytorch_checkpoint_to_tf(model=model, ckpt_dir=args.tf_cache_dir, model_name=args.model_name)


if __name__ == "__main__":
    main()
```