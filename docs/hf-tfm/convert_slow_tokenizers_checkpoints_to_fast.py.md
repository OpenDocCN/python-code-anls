# `.\transformers\convert_slow_tokenizers_checkpoints_to_fast.py`

```py
# 设置文件编码为 UTF-8
# 这个文件的版权声明，使用了 Apache 许可证 2.0 版
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
# 除非适用法律要求或书面同意，否则按"原样"提供，不提供任何明示或暗示的保证或条件
# 有关许可下的语言，请参阅许可证

# 导入必要的库
import argparse  # 导入用于解析命令行参数的模块
import os  # 导入用于处理文件路径的模块

# 导入 transformers 库
import transformers  

# 从本地模块中导入 convert_slow_tokenizer 和 logging 工具
from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS  
from .utils import logging  

# 设置日志级别为 info
logging.set_verbosity_info()

# 获取或创建名为 __name__ 的 logger 对象
logger = logging.get_logger(__name__)

# 将 tokenizer 类名映射到对应的 Fast tokenizer 类，存储在 TOKENIZER_CLASSES 字典中
TOKENIZER_CLASSES = {name: getattr(transformers, name + "Fast") for name in SLOW_TO_FAST_CONVERTERS}

# 定义一个函数，将慢速 tokenizer 的检查点转换为快速格式
def convert_slow_checkpoint_to_fast(tokenizer_name, checkpoint_name, dump_path, force_download):
    # 如果指定了 tokenizer_name 且不在 TOKENIZER_CLASSES 中，则引发 ValueError
    if tokenizer_name is not None and tokenizer_name not in TOKENIZER_CLASSES:
        raise ValueError(f"Unrecognized tokenizer name, should be one of {list(TOKENIZER_CLASSES.keys())}.")

    # 如果未指定 tokenizer_name，则使用 TOKENIZER_CLASSES 中的所有 tokenizer 名称
    if tokenizer_name is None:
        tokenizer_names = TOKENIZER_CLASSES
    else:
        tokenizer_names = {tokenizer_name: getattr(transformers, tokenizer_name + "Fast")}

    # 记录信息，显示正在加载的 tokenizer 类
    logger.info(f"Loading tokenizer classes: {tokenizer_names}")
    # 遍历所有的分词器名称
    for tokenizer_name in tokenizer_names:
        # 根据分词器名称获取对应的分词器类
        tokenizer_class = TOKENIZER_CLASSES[tokenizer_name]
    
        # 默认添加前缀
        add_prefix = True
        
        # 如果未指定检查点名称
        if checkpoint_name is None:
            # 获取分词器类支持的所有检查点名称
            checkpoint_names = list(tokenizer_class.max_model_input_sizes.keys())
        else:
            # 将指定的检查点名称作为列表
            checkpoint_names = [checkpoint_name]
    
        # 记录日志，提示加载检查点
        logger.info(f"For tokenizer {tokenizer_class.__class__.__name__} loading checkpoints: {checkpoint_names}")
    
        # 遍历每个检查点名称
        for checkpoint in checkpoint_names:
            # 记录日志，提示加载分词器和对应检查点
            logger.info(f"Loading {tokenizer_class.__class__.__name__} {checkpoint}")
    
            # 加载分词器
            tokenizer = tokenizer_class.from_pretrained(checkpoint, force_download=force_download)
    
            # 保存快速分词器，记录日志
            logger.info(f"Save fast tokenizer to {dump_path} with prefix {checkpoint} add_prefix {add_prefix}")
    
            # 对于包含斜杠的检查点名称，创建子目录
            if "/" in checkpoint:
                # 分割出目录和前缀名称
                checkpoint_directory, checkpoint_prefix_name = checkpoint.split("/")
                # 拼接完整的保存路径
                dump_path_full = os.path.join(dump_path, checkpoint_directory)
            # 如果需要添加前缀
            elif add_prefix:
                # 将检查点名称作为前缀
                checkpoint_prefix_name = checkpoint
                dump_path_full = dump_path
            else:
                checkpoint_prefix_name = None
                dump_path_full = dump_path
    
            # 记录日志，提示保存路径和前缀
            logger.info(f"=> {dump_path_full} with prefix {checkpoint_prefix_name}, add_prefix {add_prefix}")
    
            # 如果检查点存在于预训练词汇文件映射的值中
            if checkpoint in list(tokenizer.pretrained_vocab_files_map.values())[0]:
                # 获取对应的文件路径
                file_path = list(tokenizer.pretrained_vocab_files_map.values())[0][checkpoint]
                # 获取检查点后的下一个字符
                next_char = file_path.split(checkpoint)[-1][0]
                # 如果下一个字符是斜杠
                if next_char == "/":
                    # 更新保存路径为子目录
                    dump_path_full = os.path.join(dump_path_full, checkpoint_prefix_name)
                    checkpoint_prefix_name = None
    
                # 记录日志，提示保存路径和前缀
                logger.info(f"=> {dump_path_full} with prefix {checkpoint_prefix_name}, add_prefix {add_prefix}")
    
            # 保存预训练模型
            file_names = tokenizer.save_pretrained(
                dump_path_full, legacy_format=False, filename_prefix=checkpoint_prefix_name
            )
            # 记录日志，提示文件名
            logger.info(f"=> File names {file_names}")
    
            # 遍历保存的文件名列表
            for file_name in file_names:
                # 如果文件名不以 "tokenizer.json" 结尾
                if not file_name.endswith("tokenizer.json"):
                    # 移除该文件，并记录日志
                    os.remove(file_name)
                    logger.info(f"=> removing {file_name}")
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--dump_path", default=None, type=str, required=True, help="Path to output generated fast tokenizer files."
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help=(
            f"Optional tokenizer type selected in the list of {list(TOKENIZER_CLASSES.keys())}. If not given, will "
            "download and convert all the checkpoints from AWS."
        ),
    )
    parser.add_argument(
        "--checkpoint_name",
        default=None,
        type=str,
        help="Optional checkpoint name. If not given, will download and convert the canonical checkpoints from AWS.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download checkpoints.",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数将慢速tokenizer转换为快速tokenizer
    convert_slow_checkpoint_to_fast(args.tokenizer_name, args.checkpoint_name, args.dump_path, args.force_download)
```