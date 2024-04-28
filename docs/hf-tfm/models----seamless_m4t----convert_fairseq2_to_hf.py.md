# `.\transformers\models\seamless_m4t\convert_fairseq2_to_hf.py`

```
# coding=utf-8
# 版权声明，版权归 HuggingFace Inc. 团队所有
#
# 基于 Apache 许可证 2.0 版本使用本文件；
# 除非遵守许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 不附带任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。
""" 将 Meta SeamlessM4T 检查点从 seamless_communication 转换为 HF。"""


import argparse  # 导入命令行参数解析库
import os  # 导入操作系统功能模块
from pathlib import Path  # 导入 Path 对象

import torch  # 导入 PyTorch 库
from accelerate.utils.modeling import find_tied_parameters  # 导入加速库中的参数查找函数
from seamless_communication.models.inference.translator import Translator  # 从 seamless_communication 模块中导入 Translator 类

from transformers import (  # 导入 transformers 库中的以下模块和类
    SeamlessM4TConfig,  # 导入 SeamlessM4TConfig 类
    SeamlessM4TFeatureExtractor,  # 导入 SeamlessM4TFeatureExtractor 类
    SeamlessM4TModel,  # 导入 SeamlessM4TModel 类
    SeamlessM4TProcessor,  # 导入 SeamlessM4TProcessor 类
    SeamlessM4TTokenizer,  # 导入 SeamlessM4TTokenizer 类
)
from transformers.utils import logging  # 导入日志记录模块

UNIT_SUPPORTED_LANGUAGES = [  # 定义 UNIT 支持的语言列表
    "__arb__", "__ben__", "__cat__", "__ces__", "__cmn__", "__cym__", "__dan__", "__deu__", "__eng__",
    "__est__", "__fin__", "__fra__", "__hin__", "__ind__", "__ita__", "__jpn__", "__kan__", "__kor__",
    "__mlt__", "__nld__", "__pes__", "__pol__", "__por__", "__ron__", "__rus__", "__slk__", "__spa__",
    "__swe__", "__swh__", "__tam__", "__tel__", "__tgl__", "__tha__", "__tur__", "__ukr__", "__urd__",
    "__uzn__", "__vie__",
]  # 格式: 跳过
VOCODER_SUPPORTED_LANGUAGES = [  # 定义 VOCODER 支持的语言列表
    "__arb__", "__ben__", "__cat__", "__ces__", "__cmn__", "__cym__", "__dan__", "__deu__", "__eng__",
    "__est__", "__fin__", "__fra__", "__hin__", "__ind__", "__ita__", "__jpn__", "__kor__", "__mlt__",
    "__nld__", "__pes__", "__pol__", "__por__", "__ron__", "__rus__", "__slk__", "__spa__", "__swe__",
    "__swh__", "__tel__", "__tgl__", "__tha__", "__tur__", "__ukr__", "__urd__", "__uzn__", "__vie__",
]  # 格式: 跳过
# 支持的中等大小模型的语言列表
MEDIUM_SUPPORTED_LANGUAGES = ["ace","ace_Latn","acm","acq","aeb","afr","ajp","aka","amh","apc","arb","ars","ary","arz","asm","ast","awa","ayr","azb","azj","bak","bam","ban","bel","bem","ben","bho","bjn","bjn_Latn","bod","bos","bug","bul","cat","ceb","ces","cjk","ckb","crh","cym","dan","deu","dik","dyu","dzo","ell","eng","epo","est","eus","ewe","fao","pes","fij","fin","fon","fra","fur","fuv","gla","gle","glg","grn","guj","hat","hau","heb","hin","hne","hrv","hun","hye","ibo","ilo","ind","isl","ita","jav","jpn","kab","kac","kam","kan","kas","kas_Deva","kat","knc","knc_Latn","kaz","kbp","kea","khm","kik","kin","kir","kmb","kon","kor","kmr","lao","lvs","lij","lim","lin","lit","lmo","ltg","ltz","lua","lug","luo","lus","mag","mai","mal","mar","min","mkd","plt","mlt","mni","khk","mos","mri","zsm","mya","nld","nno","nob","npi","nso","nus","nya","oci","gaz","ory","pag","pan","pap","pol","por","prs","pbt","quy","ron","run","rus","sag","san","sat","scn","shn","sin","slk","slv","smo","sna","snd","som","sot","spa","als","srd","srp","ssw","sun","swe","swh","szl","tam","tat","tel","tgk","tgl","tha","tir","taq","taq_Tfng","tpi","tsn","tso","tuk","tum","tur","twi","tzm","uig","ukr","umb","urd","uzn","vec","vie","war","wol","xho","ydd","yor","yue","cmn","cmn_Hant","zul",]  # fmt: skip
# 支持的大型模型的语言列表
LARGE_SUPPORTED_LANGUAGES = ["afr","amh","arb","ary","arz","asm","azj","bel","ben","bos","bul","cat","ceb","ces","ckb","cmn","cmn_Hant","cym","dan","deu","ell","eng","est","eus","fin","fra","fuv","gaz","gle","glg","guj","heb","hin","hrv","hun","hye","ibo","ind","isl","ita","jav","jpn","kan","kat","kaz","khk","khm","kir","kor","lao","lit","lug","luo","lvs","mai","mal","mar","mkd","mlt","mni","mya","nld","nno","nob","npi","nya","ory","pan","pbt","pes","pol","por","ron","rus","sat","slk","slv","sna","snd","som","spa","srp","swe","swh","tam","tel","tgk","tgl","tha","tur","ukr","urd","uzn","vie","yor","yue","zlm","zul",]  # fmt: skip


# 检查模型参数数量是否相同
def assert_param_count(model_1, model_2):
    # 计算模型1参数数量，排除final_proj参数
    count_1 = sum(p[1].numel() for p in model_1.named_parameters() if "final_proj" not in p[0])
    # 计算模型2参数数量，排除final_proj参数
    count_2 = sum(p[1].numel() for p in model_2.named_parameters() if "final_proj" not in p[0])
    # 断言模型1和模型2的参数数量相同，否则抛出异常
    assert count_1 == count_2, f"{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}"


# 计算模型参数数量，排除final_proj参数
def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters() if "final_proj" not in p[0])


# 获取最佳的设备（GPU或CPU）
def _grab_best_device(use_gpu=True):
    # 如果存在GPU且使用GPU标志为True，则选择cuda设备，否则选择cpu设备
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    # 返回torch设备对象
    return torch.device(device)


# 设置日志记录的详细程度为INFO
logging.set_verbosity_info()
# 获取与当前模块相关联的logger对象
logger = logging.get_logger(__name__)

# 声码器转换列表
vocoder_convert_list = [
    ("ups", "hifi_gan.upsampler"),
    ("conv_pre", "hifi_gan.conv_pre"),
    ("resblocks", "hifi_gan.resblocks"),
    ("conv_post", "hifi_gan.conv_post"),
    ("lang", "language_embedding"),
    ("spkr", "speaker_embedding"),
    ("dict.", "unit_embedding."),
    ("dur_predictor.conv1.0", "dur_predictor.conv1"),
]
    # 创建一个包含两个字符串的元组，字符串分别为 "dur_predictor.conv2.0" 和 "dur_predictor.conv2"
    ("dur_predictor.conv2.0", "dur_predictor.conv2"),
# 将根据不同的模型类型设置 HuggingFace 模型的配置信息
def _load_hf_config(model_type="medium"):
    # 定义 wav2vec_convert_list、t2u_convert_list 和 text_convert_list 三个列表，用于转换模型权重的键名
    # 这些列表中记录了原模型中的键名与目标模型中的对应键名之间的映射关系
    
    # wav2vec_convert_list 列表定义了 wav2vec 模型权重转换时的键名映射关系
    wav2vec_convert_list = [
        # 这些映射关系定义了如何将原模型中的参数转换到目标模型中
        ("speech_encoder_frontend.model_dim_proj", "feature_projection.projection"),
        ("speech_encoder_frontend.post_extract_layer_norm", "feature_projection.layer_norm"),
        # 其他映射关系...
    ]
    
    # t2u_convert_list 列表定义了 t2u 模型权重转换时的键名映射关系
    t2u_convert_list = [
        # 这些映射关系定义了如何将原模型中的参数转换到目标模型中
        ("t2u_model.final_proj", "lm_head"),
        ("t2u_model.", "model."),
        # 其他映射关系...
    ]
    
    # text_convert_list 列表定义了文本模型权重转换时的键名映射关系
    text_convert_list = [
        # 这些映射关系定义了如何将原模型中的参数转换到目标模型中
        ("text_encoder.", ""),
        ("text_decoder.", ""),
        # 其他映射关系...
    ]
    
    # 定义 CACHE_DIR 变量，用于存储缓存的模型文件
    CUR_PATH = os.path.dirname(os.path.abspath(__file__))
    default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
    CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "huggingface", "hub")
    
    # 这个函数主要用于加载 HuggingFace 模型的配置信息
    # 根据传入的 model_type 参数，设置不同的配置信息
    # 如果模型类型是"medium"
    if model_type == "medium":
        # 设置参数字典
        kwargs = {
            "vocab_size": 256206,
            "t2u_vocab_size": 10082,
            "hidden_size": 1024,
            "max_position_embeddings": 4096,
            "encoder_layers": 12,
            "decoder_layers": 12,
            "encoder_ffn_dim": 4096,
            "decoder_ffn_dim": 4096,
            "t2u_encoder_layers": 4,
            "t2u_decoder_layers": 4,
            "speech_encoder_layers": 12,
        }
        # 使用参数字典创建并返回SeamlessM4TConfig对象
        return SeamlessM4TConfig(**kwargs)
    # 如果模型类型不是"medium"
    else:
        # 返回默认的SeamlessM4TConfig对象
        return SeamlessM4TConfig()
def _convert_model(
    original_model,
    hf_model,
    convert_list,
    device,
    unwanted_prefix="model.",
    filter_state_dict="speech",
    exclude_state_dict=None,
):
    state_dict = original_model.state_dict()

    # 定义过滤函数
    if isinstance(filter_state_dict, str):
        # 如果过滤条件是字符串，则使用简单的过滤函数
        def filter_func(x):
            return filter_state_dict in x[0]
    else:
        # 如果过滤条件是列表，则使用更复杂的过滤函数
        def filter_func(item):
            # 排除指定的状态字典
            if exclude_state_dict is not None and exclude_state_dict in item[0]:
                return False
            for filter_el in filter_state_dict:
                # 判断状态字典中是否包含过滤条件中的元素
                if filter_el in item[0]:
                    return True
            return False

    # 对状态字典进行过滤
    state_dict = dict(filter(filter_func, state_dict.items()))

    # 对状态字典中的键进行转换
    for k, v in list(state_dict.items()):
        new_k = k[len(unwanted_prefix):]
        for old_layer_name, new_layer_name in convert_list:
            if old_layer_name in new_k:
                new_k = new_k.replace(old_layer_name, new_layer_name)

        # 针对特定命名规则的处理
        if ".layer_norm" in new_k and new_k.split(".layer_norm")[0][-1].isnumeric():
            new_k = new_k.replace("layer_norm", "final_layer_norm")

        # 更新状态字典的键
        state_dict[new_k] = state_dict.pop(k)

    # 检查额外的键和缺失的键
    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    # 检查是否有额外的键
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    # 检查是否有缺失的键
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")

    # 载入模型状态字典到预训练模型
    hf_model.load_state_dict(state_dict, strict=False)

    # 统计参数数量
    n_params = param_count(hf_model)

    # 记录模型加载完成并显示参数数量
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params")

    # 将模型设置为评估模式并移到指定设备
    hf_model.eval()
    hf_model.to(device)

    # 清除临时变量
    del state_dict

    # 返回转换后的模型
    return hf_model


def load_model(save_dir, model_type, repo_id):
    """
    Meta SeamlessM4T is made of 8 main components:
    - speech_encoder (#1) and speech_encoder_frontend (#2)
    - t2u_model (#3)
    - text_encoder (#4) and text_encoder_frontend (#5)
    - text_decoder (#6) [and text_decoder_frontend (#5) = equals to text_encoder_frontend]
    - final_proj (#7)
    - vocoder (#8)
    """
    # 获取最佳设备
    device = _grab_best_device()

    # 根据模型类型选择模型
    if model_type == "medium":
        name = "seamlessM4T_medium"
    else:
        name = "seamlessM4T_large"

    # 创建原始模型实例
    original_model = Translator(name, "vocoder_36langs", device, torch.float32)

    # 设置支持的语言列表
    langs = MEDIUM_SUPPORTED_LANGUAGES if model_type == "medium" else LARGE_SUPPORTED_LANGUAGES
    langs = [f"__{lang}__" for lang in langs]

    # 设置词汇文件路径
    vocab_file = os.path.join(os.path.expanduser("~"), "tokenizer", model_type, "tokenizer.model")

    # 创建保存目录
    save_dir = os.path.join(save_dir, name)
    Path(save_dir).mkdir(exist_ok=True)

    # 创建分词器实例
    tokenizer = SeamlessM4TTokenizer(vocab_file, additional_special_tokens=langs)
    # 获取 "__fra__" 对应的语言 ID，用于后续检查 tokenizer 保存/加载的一致性
    sanity_check_lang_id = tokenizer.convert_tokens_to_ids("__fra__")

    # 将 tokenizer 保存到指定目录
    tokenizer.save_pretrained(save_dir)
    # 从指定目录加载 SeamlessM4TTokenizer
    tokenizer = SeamlessM4TTokenizer.from_pretrained(save_dir)

    # 检查加载后的 "__fra__" 语言 ID 是否与之前一致，若不一致则引发异常
    if sanity_check_lang_id != tokenizer.convert_tokens_to_ids("__fra__"):
        raise ValueError(
            f"Error in tokenizer saving/loading - __fra__ lang id is not coherent: {sanity_check_lang_id} vs {tokenizer.convert_tokens_to_ids('__fra__')}"
        )

    ####### 获取语言到 ID 的字典
    # 使用 tokenizer 将语言转换为对应的 ID，并构建语言到 ID 的字典
    text_decoder_lang_code_to_id = {lang.replace("__", ""): tokenizer.convert_tokens_to_ids(lang) for lang in langs}
    # 计算 t2u 语言到 ID 的字典
    # offset: vocoder unit vocab size + 5 (for EOS/PAD/BOS/UNK/MSK) + len(supported_languages)
    t2u_lang_code_to_id = {
        code.replace("__", ""): i + 10005 + len(UNIT_SUPPORTED_LANGUAGES)
        for i, code in enumerate(UNIT_SUPPORTED_LANGUAGES)
    }
    # 构建 vocoder 语言到 ID 的字典
    vocoder_lang_code_to_id = {code.replace("__", ""): i for i, code in enumerate(VOCODER_SUPPORTED_LANGUAGES)}

    ######### FE

    # 初始化特征提取器，使用指定语言列表
    fe = SeamlessM4TFeatureExtractor(language_code=langs)

    # 将特征提取器保存到指定目录
    fe.save_pretrained(save_dir)
    # 从指定目录加载特征提取器
    fe = SeamlessM4TFeatureExtractor.from_pretrained(save_dir)

    # 初始化处理器，使用特征提取器和 tokenizer
    processor = SeamlessM4TProcessor(feature_extractor=fe, tokenizer=tokenizer)
    # 将处理器保存到指定目录
    processor.save_pretrained(save_dir)
    # 将处理器推送到 Hub
    processor.push_to_hub(repo_id=repo_id, create_pr=True)

    # 从指定目录加载处理器
    processor = SeamlessM4TProcessor.from_pretrained(save_dir)

    ######## Model

    # 初始化模型，加载指定类型的配置
    hf_config = _load_hf_config(model_type)
    hf_model = SeamlessM4TModel(hf_config)

    # 设置生成配置的语言到代码 ID 的映射
    hf_model.generation_config.__setattr__("text_decoder_lang_to_code_id", text_decoder_lang_code_to_id)
    hf_model.generation_config.__setattr__("t2u_lang_code_to_id", t2u_lang_code_to_id)
    hf_model.generation_config.__setattr__("vocoder_lang_code_to_id", vocoder_lang_code_to_id)

    # -1. 处理 vocoder
    # 类似于语音 T5，需要应用和移除权重规范化
    hf_model.vocoder.apply_weight_norm()
    # 将原始模型的 vocoder 部分转换成当前模型的 vocoder
    hf_model.vocoder = _convert_model(
        original_model,
        hf_model.vocoder,
        vocoder_convert_list,
        device,
        unwanted_prefix="vocoder.code_generator.",
        filter_state_dict="vocoder",
    )
    # 移除权重规范化
    hf_model.vocoder.remove_weight_norm()

    # 1. 处理语音编码器
    wav2vec = hf_model.speech_encoder
    hf_model.speech_encoder = _convert_model(
        original_model, wav2vec, wav2vec_convert_list, device, unwanted_prefix="model.", filter_state_dict="speech"
    )

    # 2. 处理 t2u

    hf_model.t2u_model = _convert_model(
        original_model,
        hf_model.t2u_model,
        t2u_convert_list,
        device,
        unwanted_prefix="model.",
        filter_state_dict="t2u_model",
    )

    # 3. 处理文本编码器
    hf_model.text_encoder = _convert_model(
        original_model,
        hf_model.text_encoder,
        text_convert_list,
        device,
        unwanted_prefix="model.",
        filter_state_dict=["model.text_encoder"],
        exclude_state_dict="t2u_model",
    )
    )

    # 4. 处理文本解码器
    hf_model.text_decoder = _convert_model(
        original_model,
        hf_model.text_decoder,
        text_convert_list,
        device,
        unwanted_prefix="model.",
        filter_state_dict=["model.text_decoder"],
        exclude_state_dict="t2u_model",
    )

    # 5. 处理最终的投影层
    hf_model.lm_head = _convert_model(
        original_model,
        hf_model.lm_head,
        [("final_proj.", "")],
        device,
        unwanted_prefix="model.",
        filter_state_dict=["model.final_proj"],
        exclude_state_dict="t2u_model",
    )

    # 检查参数绑定是否正确
    print(find_tied_parameters(hf_model))

    # 统计模型参数数量
    count_1 = param_count(hf_model)
    count_2 = param_count(original_model)

    # 打印模型参数数量对比
    print(f"HF MODEL:{count_1}, ORIGINAL_MODEL: {count_2}, diff:{count_1 - count_2}")
    # 打印不包括嵌入层的模型参数数量
    print(f"HF MODEL excluding embeddings:{hf_model.num_parameters(exclude_embeddings=True)}")

    # 删除原始模型
    del original_model

    # 设置生成配置的属性为 False
    hf_model.generation_config._from_model_config = False
    # 保存预训练模型到指定目录
    hf_model.save_pretrained(save_dir)
    # 将模型推送到 Hub
    hf_model.push_to_hub(repo_id=repo_id, create_pr=True)
    # 从指定目录加载无缝模型
    hf_model = SeamlessM4TModel.from_pretrained(save_dir)
# 如果当前脚本被直接执行而非被导入，则执行以下代码块
if __name__ == "__main__":
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加必需的参数

    # 添加名为 "--model_type" 的参数，类型为字符串，默认值为 "medium"，帮助信息为 "Model type."
    parser.add_argument(
        "--model_type",
        default="medium",
        type=str,
        help="Model type.",
    )

    # 添加名为 "--save_dir" 的参数，类型为字符串，默认值为 "/home/ubuntu/weights"，帮助信息为 "Path to the output PyTorch model."
    parser.add_argument(
        "--save_dir",
        default="/home/ubuntu/weights",
        type=str,
        help="Path to the output PyTorch model.",
    )

    # 添加名为 "--repo_id" 的参数，类型为字符串，默认值为 "facebook/hf-seamless-m4t-medium"，帮助信息为 "Repo ID."
    parser.add_argument(
        "--repo_id",
        default="facebook/hf-seamless-m4t-medium",
        type=str,
        help="Repo ID.",
    )

    # 从命令行中解析参数，并将它们存储在args对象中
    args = parser.parse_args()

    # 调用 load_model 函数，传入解析后的参数
    load_model(args.save_dir, args.model_type, args.repo_id)
```