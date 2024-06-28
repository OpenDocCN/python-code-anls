# `.\models\tapas\convert_tapas_original_tf_checkpoint_to_pytorch.py`

```
# 引入命令行参数解析模块
import argparse

# 从transformers库中引入各种Tapas相关类和函数
from transformers import (
    TapasConfig,
    TapasForMaskedLM,
    TapasForQuestionAnswering,
    TapasForSequenceClassification,
    TapasModel,
    TapasTokenizer,
    load_tf_weights_in_tapas,
)
# 从transformers的工具模块中引入日志记录功能
from transformers.utils import logging

# 设置日志记录的详细程度为info级别
logging.set_verbosity_info()

# 定义函数convert_tf_checkpoint_to_pytorch，用于将TensorFlow模型转换为PyTorch模型
def convert_tf_checkpoint_to_pytorch(
    task, reset_position_index_per_cell, tf_checkpoint_path, tapas_config_file, pytorch_dump_path
):
    # 初始化PyTorch模型配置
    # 如果要转换使用绝对位置嵌入的检查点，请确保将TapasConfig的reset_position_index_per_cell设置为False。
    
    # 从json文件加载TapasConfig配置
    config = TapasConfig.from_json_file(tapas_config_file)
    # 设置绝对/相对位置嵌入参数
    config.reset_position_index_per_cell = reset_position_index_per_cell

    # 根据任务设置TapasConfig的其余参数以及模型类型
    if task == "SQA":
        model = TapasForQuestionAnswering(config=config)
    elif task == "WTQ":
        # WTQ任务的特定配置
        config.num_aggregation_labels = 4
        config.use_answer_as_supervision = True
        config.answer_loss_cutoff = 0.664694
        config.cell_selection_preference = 0.207951
        config.huber_loss_delta = 0.121194
        config.init_cell_selection_weights_to_zero = True
        config.select_one_column = True
        config.allow_empty_column_selection = False
        config.temperature = 0.0352513

        model = TapasForQuestionAnswering(config=config)
    elif task == "WIKISQL_SUPERVISED":
        # WIKISQL_SUPERVISED任务的特定配置
        config.num_aggregation_labels = 4
        config.use_answer_as_supervision = False
        config.answer_loss_cutoff = 36.4519
        config.cell_selection_preference = 0.903421
        config.huber_loss_delta = 222.088
        config.init_cell_selection_weights_to_zero = True
        config.select_one_column = True
        config.allow_empty_column_selection = True
        config.temperature = 0.763141

        model = TapasForQuestionAnswering(config=config)
    elif task == "TABFACT":
        # TABFACT任务使用序列分类模型
        model = TapasForSequenceClassification(config=config)
    elif task == "MLM":
        # MLM任务使用遮蔽语言建模模型
        model = TapasForMaskedLM(config=config)
    # 如果任务是INTERMEDIATE_PRETRAINING，则创建一个TapasModel模型对象，使用给定的配置
    elif task == "INTERMEDIATE_PRETRAINING":
        model = TapasModel(config=config)
    # 如果任务不是INTERMEDIATE_PRETRAINING，抛出异常，显示不支持的任务类型
    else:
        raise ValueError(f"Task {task} not supported.")

    # 打印消息，显示正在根据配置构建PyTorch模型
    print(f"Building PyTorch model from configuration: {config}")

    # 从TensorFlow的检查点中加载权重到TapasModel模型中
    load_tf_weights_in_tapas(model, config, tf_checkpoint_path)

    # 保存PyTorch模型（包括权重和配置）到指定路径
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

    # 保存tokenizer文件到指定路径
    print(f"Save tokenizer files to {pytorch_dump_path}")
    # 创建一个TapasTokenizer对象，使用给定的词汇文件和最大长度，保存到pytorch_dump_path
    tokenizer = TapasTokenizer(vocab_file=tf_checkpoint_path[:-10] + "vocab.txt", model_max_length=512)
    tokenizer.save_pretrained(pytorch_dump_path)

    # 打印消息，显示是否使用了相对位置嵌入
    print("Used relative position embeddings:", model.config.reset_position_index_per_cell)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # Required parameters
    parser.add_argument(
        "--task", default="SQA", type=str, help="Model task for which to convert a checkpoint. Defaults to SQA."
    )
    # 添加一个必选参数 --task，用于指定模型任务类型，默认为 "SQA"

    parser.add_argument(
        "--reset_position_index_per_cell",
        default=False,
        action="store_true",
        help="Whether to use relative position embeddings or not. Defaults to True.",
    )
    # 添加一个可选参数 --reset_position_index_per_cell，用于控制是否使用相对位置嵌入，默认为 False

    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    # 添加一个必选参数 --tf_checkpoint_path，指定 TensorFlow checkpoint 的路径

    parser.add_argument(
        "--tapas_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained TAPAS model. \n"
            "This specifies the model architecture."
        ),
    )
    # 添加一个必选参数 --tapas_config_file，指定预训练 TAPAS 模型的配置文件路径，用于指定模型架构

    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加一个必选参数 --pytorch_dump_path，指定输出 PyTorch 模型的路径

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 变量中

    convert_tf_checkpoint_to_pytorch(
        args.task,
        args.reset_position_index_per_cell,
        args.tf_checkpoint_path,
        args.tapas_config_file,
        args.pytorch_dump_path,
    )
    # 调用函数 convert_tf_checkpoint_to_pytorch，传入解析得到的参数，执行 TensorFlow 到 PyTorch 模型的转换
```