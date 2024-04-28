# `.\transformers\models\bert\convert_bert_pytorch_checkpoint_to_original_tf.py`

```
# 设置文件编码为utf-8
# 版权声明
# 根据Apache许可证2.0版授权使用此文件
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 基于"按原样"分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

"""将Huggingface Pytorch检查点转换为Tensorflow检查点。"""

import argparse
import os

import numpy as np
import tensorflow as tf
import torch

from transformers import BertModel

# 将Pytorch模型转换为Tensorflow模型
def convert_pytorch_checkpoint_to_tf(model: BertModel, ckpt_dir: str, model_name: str):
    """
    Args:
        model: 要转换的BertModel Pytorch模型实例
        ckpt_dir: Tensorflow模型目录
        model_name: 模型名称

    当前支持的HF模型:

        - Y BertModel
        - N BertForMaskedLM
        - N BertForPreTraining
        - N BertForMultipleChoice
        - N BertForNextSentencePrediction
        - N BertForSequenceClassification
        - N BertForQuestionAnswering
    """

    # 需要转置的张量
    tensors_to_transpose = ("dense.weight", "attention.self.query", "attention.self.key", "attention.self.value")

    # 变量映射
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

    # 如果目录不存在，则创建目录
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # 获取模型的状态字典
    state_dict = model.state_dict()

    # 将Pytorch变量名转换为Tensorflow变量名
    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return f"bert/{name}"

    # 创建Tensorflow变量
    def create_tf_var(tensor: np.ndarray, name: str, session: tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    # 重置默认图
    tf.reset_default_graph()
    # 创建 TensorFlow 会话
    with tf.Session() as session:
        # 遍历状态字典中的变量名
        for var_name in state_dict:
            # 将变量名转换为 TensorFlow 变量名
            tf_name = to_tf_var_name(var_name)
            # 将 PyTorch 张量转换为 NumPy 数组
            torch_tensor = state_dict[var_name].numpy()
            # 如果变量名中包含需要转置的关键词，则对张量进行转置操作
            if any(x in var_name for x in tensors_to_transpose):
                torch_tensor = torch_tensor.T
            # 创建 TensorFlow 变量
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            # 设置 TensorFlow 变量的值为 PyTorch 张量的值
            tf.keras.backend.set_value(tf_var, torch_tensor)
            # 运行 TensorFlow 变量，获取权重值
            tf_weight = session.run(tf_var)
            # 打印成功创建的 TensorFlow 变量及其与 PyTorch 张量是否相似
            print(f"Successfully created {tf_name}: {np.allclose(tf_weight, torch_tensor)}")

        # 创建 TensorFlow 可训练变量的保存器
        saver = tf.train.Saver(tf.trainable_variables())
        # 保存 TensorFlow 会话的检查点文件
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))
# 主函数，用于执行程序的主要逻辑
def main(raw_args=None):
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数：模型名称，必需的字符串类型参数
    parser.add_argument("--model_name", type=str, required=True, help="model name e.g. bert-base-uncased")
    # 添加命令行参数：缓存目录，可选的字符串类型参数，默认为None
    parser.add_argument(
        "--cache_dir", type=str, default=None, required=False, help="Directory containing pytorch model"
    )
    # 添加命令行参数：PyTorch 模型路径，必需的字符串类型参数
    parser.add_argument("--pytorch_model_path", type=str, required=True, help="/path/to/<pytorch-model-name>.bin")
    # 添加命令行参数：TensorFlow 缓存目录，必需的字符串类型参数
    parser.add_argument("--tf_cache_dir", type=str, required=True, help="Directory in which to save tensorflow model")
    # 解析命令行参数
    args = parser.parse_args(raw_args)

    # 从预训练模型中加载 BertModel
    model = BertModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name,  # 预训练模型名称或路径
        state_dict=torch.load(args.pytorch_model_path),  # PyTorch 模型的状态字典
        cache_dir=args.cache_dir,  # 缓存目录
    )

    # 将 PyTorch 检查点文件转换为 TensorFlow 模型
    convert_pytorch_checkpoint_to_tf(model=model, ckpt_dir=args.tf_cache_dir, model_name=args.model_name)


if __name__ == "__main__":
    # 调用主函数
    main()
```