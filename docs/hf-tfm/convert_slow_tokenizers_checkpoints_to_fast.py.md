# `.\convert_slow_tokenizers_checkpoints_to_fast.py`

```
# 设置脚本的编码格式为 UTF-8
# 版权声明，此代码归 HuggingFace Inc. 团队所有，使用 Apache 许可证 2.0 版本
#
# 根据许可证，除非符合许可证的要求，否则不能使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 如果适用法律要求或书面同意，本软件根据“现状”分发，不提供任何明示或暗示的保证或条件
# 请查阅许可证了解具体的使用条款和限制
""" 转换慢速分词器检查点为快速分词器的序列化格式（tokenizers 库的序列化格式）"""

# 导入必要的库和模块
import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统功能模块

import transformers  # 导入 transformers 库

from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS  # 从当前目录导入慢速分词器转换器
from .utils import logging  # 从当前目录导入日志记录工具

# 设置日志输出为信息级别
logging.set_verbosity_info()

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 创建一个字典，将慢速分词器名称映射到其对应的快速分词器类
TOKENIZER_CLASSES = {name: getattr(transformers, name + "Fast") for name in SLOW_TO_FAST_CONVERTERS}


def convert_slow_checkpoint_to_fast(tokenizer_name, checkpoint_name, dump_path, force_download):
    # 如果指定的分词器名称不在 TOKENIZER_CLASSES 中，则引发 ValueError
    if tokenizer_name is not None and tokenizer_name not in TOKENIZER_CLASSES:
        raise ValueError(f"Unrecognized tokenizer name, should be one of {list(TOKENIZER_CLASSES.keys())}.")

    # 如果未指定分词器名称，则使用 TOKENIZER_CLASSES 中的所有分词器
    if tokenizer_name is None:
        tokenizer_names = TOKENIZER_CLASSES
    else:
        tokenizer_names = {tokenizer_name: getattr(transformers, tokenizer_name + "Fast")}

    # 记录日志，显示正在加载的分词器类信息
    logger.info(f"Loading tokenizer classes: {tokenizer_names}")
    # 遍历每个给定的分词器名称
    for tokenizer_name in tokenizer_names:
        # 获取与分词器名称对应的分词器类
        tokenizer_class = TOKENIZER_CLASSES[tokenizer_name]

        # 根据是否提供了检查点名称决定加载哪些检查点
        add_prefix = True
        if checkpoint_name is None:
            # 如果未提供检查点名称，则加载所有可用的检查点名称列表
            checkpoint_names = list(tokenizer_class.max_model_input_sizes.keys())
        else:
            # 否则，只加载指定的检查点名称
            checkpoint_names = [checkpoint_name]

        # 记录日志，显示正在加载哪个分词器类的哪些检查点
        logger.info(f"For tokenizer {tokenizer_class.__class__.__name__} loading checkpoints: {checkpoint_names}")

        # 遍历每个指定的检查点名称
        for checkpoint in checkpoint_names:
            # 记录日志，显示正在加载哪个分词器类的哪个具体检查点
            logger.info(f"Loading {tokenizer_class.__class__.__name__} {checkpoint}")

            # 加载分词器对象
            tokenizer = tokenizer_class.from_pretrained(checkpoint, force_download=force_download)

            # 记录日志，显示正在将快速分词器保存到指定路径，并指定前缀和是否添加前缀
            logger.info(f"Save fast tokenizer to {dump_path} with prefix {checkpoint} add_prefix {add_prefix}")

            # 根据检查点名称是否包含斜杠来决定文件保存路径
            if "/" in checkpoint:
                checkpoint_directory, checkpoint_prefix_name = checkpoint.split("/")
                dump_path_full = os.path.join(dump_path, checkpoint_directory)
            elif add_prefix:
                checkpoint_prefix_name = checkpoint
                dump_path_full = dump_path
            else:
                checkpoint_prefix_name = None
                dump_path_full = dump_path

            # 记录日志，显示保存路径和前缀信息
            logger.info(f"=> {dump_path_full} with prefix {checkpoint_prefix_name}, add_prefix {add_prefix}")

            # 检查是否需要添加额外路径，以适应特定的文件保存结构
            if checkpoint in list(tokenizer.pretrained_vocab_files_map.values())[0]:
                file_path = list(tokenizer.pretrained_vocab_files_map.values())[0][checkpoint]
                next_char = file_path.split(checkpoint)[-1][0]
                if next_char == "/":
                    dump_path_full = os.path.join(dump_path_full, checkpoint_prefix_name)
                    checkpoint_prefix_name = None

                # 记录日志，显示最终的保存路径和前缀信息
                logger.info(f"=> {dump_path_full} with prefix {checkpoint_prefix_name}, add_prefix {add_prefix}")

            # 保存预训练模型文件，并返回保存的文件名列表
            file_names = tokenizer.save_pretrained(
                dump_path_full, legacy_format=False, filename_prefix=checkpoint_prefix_name
            )
            # 记录日志，显示保存的文件名列表
            logger.info(f"=> File names {file_names}")

            # 遍历保存的文件列表，删除非tokenizer.json结尾的文件
            for file_name in file_names:
                if not file_name.endswith("tokenizer.json"):
                    os.remove(file_name)
                    logger.info(f"=> removing {file_name}")
if __name__ == "__main__":
    # 如果脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必选参数
    parser.add_argument(
        "--dump_path", default=None, type=str, required=True, help="Path to output generated fast tokenizer files."
    )
    # 添加名为 --dump_path 的参数，类型为字符串，必选，用于指定生成的快速标记器文件的输出路径

    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help=(
            f"Optional tokenizer type selected in the list of {list(TOKENIZER_CLASSES.keys())}. If not given, will "
            "download and convert all the checkpoints from AWS."
        ),
    )
    # 添加名为 --tokenizer_name 的参数，类型为字符串，可选，用于选择标记器类型。如果未提供，则将从 AWS 下载并转换所有检查点。

    parser.add_argument(
        "--checkpoint_name",
        default=None,
        type=str,
        help="Optional checkpoint name. If not given, will download and convert the canonical checkpoints from AWS.",
    )
    # 添加名为 --checkpoint_name 的参数，类型为字符串，可选，用于指定检查点的名称。如果未提供，则将从 AWS 下载并转换规范的检查点。

    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download checkpoints.",
    )
    # 添加名为 --force_download 的参数，动作为存储真值，用于强制重新下载检查点。

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 对象中

    convert_slow_checkpoint_to_fast(args.tokenizer_name, args.checkpoint_name, args.dump_path, args.force_download)
    # 调用函数 convert_slow_checkpoint_to_fast，传递参数：标记器名称、检查点名称、输出路径和是否强制重新下载的标志
```