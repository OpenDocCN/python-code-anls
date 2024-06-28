# `.\models\seamless_m4t\convert_fairseq2_to_hf.py`

```py
# 指定 Python 文件的编码格式为 UTF-8
# 版权声明，声明代码版权归 The HuggingFace Inc. 团队所有，保留所有权利
#
# 根据 Apache 许可证 2.0 版本授权使用此文件；除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 如果没有适用法律要求或书面同意，软件将基于“原样”分发，没有任何明示或暗示的担保或条件
# 请查阅许可证了解具体语言规定及限制
""" 将 Meta SeamlessM4T 检查点从 seamless_communication 转换为 HF."""

import argparse  # 导入解析命令行参数的模块
import os  # 导入操作系统相关功能的模块
from pathlib import Path  # 导入处理文件路径的模块

import torch  # 导入 PyTorch 深度学习框架
from accelerate.utils.modeling import find_tied_parameters  # 导入加速库中的模型参数查找工具
from seamless_communication.models.inference.translator import Translator  # 导入翻译模型

from transformers import (  # 导入 Transformers 库中的相关模块
    SeamlessM4TConfig,  # 导入 Meta SeamlessM4T 模型的配置类
    SeamlessM4TFeatureExtractor,  # 导入 Meta SeamlessM4T 模型的特征提取器类
    SeamlessM4TModel,  # 导入 Meta SeamlessM4T 模型类
    SeamlessM4TProcessor,  # 导入 Meta SeamlessM4T 模型的处理器类
    SeamlessM4TTokenizer,  # 导入 Meta SeamlessM4T 模型的分词器类
)
from transformers.utils import logging  # 导入 Transformers 的日志模块


UNIT_SUPPORTED_LANGUAGES = [  # 支持单元翻译的语言列表，使用双下划线包裹语言代码
    "__arb__", "__ben__", "__cat__", "__ces__", "__cmn__", "__cym__", "__dan__", "__deu__",
    "__eng__", "__est__", "__fin__", "__fra__", "__hin__", "__ind__", "__ita__", "__jpn__",
    "__kan__", "__kor__", "__mlt__", "__nld__", "__pes__", "__pol__", "__por__", "__ron__",
    "__rus__", "__slk__", "__spa__", "__swe__", "__swh__", "__tam__", "__tel__", "__tgl__",
    "__tha__", "__tur__", "__ukr__", "__urd__", "__uzn__", "__vie__",
]  # 支持的语言列表，忽略格式

VOCODER_SUPPORTED_LANGUAGES = [  # 支持语音合成的语言列表，使用双下划线包裹语言代码
    "__arb__", "__ben__", "__cat__", "__ces__", "__cmn__", "__cym__", "__dan__", "__deu__",
    "__eng__", "__est__", "__fin__", "__fra__", "__hin__", "__ind__", "__ita__", "__jpn__",
    "__kor__", "__mlt__", "__nld__", "__pes__", "__pol__", "__por__", "__ron__", "__rus__",
    "__slk__", "__spa__", "__swe__", "__swh__", "__tel__", "__tgl__", "__tha__", "__tur__",
    "__ukr__", "__urd__", "__uzn__", "__vie__",
]  # 支持的语言列表，忽略格式
# 支持的中等大小模型的语言列表，使用列表存储字符串形式的语言代码
MEDIUM_SUPPORTED_LANGUAGES = ["ace","ace_Latn","acm","acq","aeb","afr","ajp","aka","amh","apc","arb","ars","ary","arz","asm","ast","awa","ayr","azb","azj","bak","bam","ban","bel","bem","ben","bho","bjn","bjn_Latn","bod","bos","bug","bul","cat","ceb","ces","cjk","ckb","crh","cym","dan","deu","dik","dyu","dzo","ell","eng","epo","est","eus","ewe","fao","pes","fij","fin","fon","fra","fur","fuv","gla","gle","glg","grn","guj","hat","hau","heb","hin","hne","hrv","hun","hye","ibo","ilo","ind","isl","ita","jav","jpn","kab","kac","kam","kan","kas","kas_Deva","kat","knc","knc_Latn","kaz","kbp","kea","khm","kik","kin","kir","kmb","kon","kor","kmr","lao","lvs","lij","lim","lin","lit","lmo","ltg","ltz","lua","lug","luo","lus","mag","mai","mal","mar","min","mkd","plt","mlt","mni","khk","mos","mri","zsm","mya","nld","nno","nob","npi","nso","nus","nya","oci","gaz","ory","pag","pan","pap","pol","por","prs","pbt","quy","ron","run","rus","sag","san","sat","scn","shn","sin","slk","slv","smo","sna","snd","som","sot","spa","als","srd","srp","ssw","sun","swe","swh","szl","tam","tat","tel","tgk","tgl","tha","tir","taq","taq_Tfng","tpi","tsn","tso","tuk","tum","tur","twi","tzm","uig","ukr","umb","urd","uzn","vec","vie","war","wol","xho","ydd","yor","yue","cmn","cmn_Hant","zul",]

# 支持的大型模型的语言列表，使用列表存储字符串形式的语言代码
LARGE_SUPPORTED_LANGUAGES = ["afr","amh","arb","ary","arz","asm","azj","bel","ben","bos","bul","cat","ceb","ces","ckb","cmn","cmn_Hant","cym","dan","deu","ell","eng","est","eus","fin","fra","fuv","gaz","gle","glg","guj","heb","hin","hrv","hun","hye","ibo","ind","isl","ita","jav","jpn","kan","kat","kaz","khk","khm","kir","kor","lao","lit","lug","luo","lvs","mai","mal","mar","mkd","mlt","mni","mya","nld","nno","nob","npi","nya","ory","pan","pbt","pes","pol","por","ron","rus","sat","slk","slv","sna","snd","som","spa","srp","swe","swh","tam","tel","tgk","tgl","tha","tur","ukr","urd","uzn","vie","yor","yue","zlm","zul",]

# 对比两个模型的参数数量是否相等，若不相等则抛出异常
def assert_param_count(model_1, model_2):
    # 计算模型1中除去包含"final_proj"的参数数量总和
    count_1 = sum(p[1].numel() for p in model_1.named_parameters() if "final_proj" not in p[0])
    # 计算模型2中除去包含"final_proj"的参数数量总和
    count_2 = sum(p[1].numel() for p in model_2.named_parameters() if "final_proj" not in p[0])
    # 断言两个模型的参数数量相等，否则输出错误信息
    assert count_1 == count_2, f"{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}"

# 计算模型参数中除去"final_proj"的部分的总数量
def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters() if "final_proj" not in p[0])

# 获取最佳的计算设备，如果支持GPU且use_gpu为True，则选择cuda，否则选择cpu
def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)

# 设置日志记录的详细程度为info级别
logging.set_verbosity_info()

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 定义用于声码器转换的列表，包含元组，每个元组包含旧名称和新名称
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
    # 定义一个元组，包含两个字符串元素："dur_predictor.conv2.0" 和 "dur_predictor.conv2"
    ("dur_predictor.conv2.0", "dur_predictor.conv2"),
# 重命名用于转换的列表，将源名称映射到目标名称
# 顺序很重要，源名称在前，目标名称在后
wav2vec_convert_list = [
    ("speech_encoder_frontend.model_dim_proj", "feature_projection.projection"),
    ("speech_encoder_frontend.post_extract_layer_norm", "feature_projection.layer_norm"),
    ("speech_encoder_frontend.pos_encoder.conv", "encoder.pos_conv_embed.conv"),
    ("speech_encoder.inner.layers", "encoder.layers"),
    ("speech_encoder.inner_layer_norm", "encoder.layer_norm"),
    ("speech_encoder.adaptor_layers", "adapter.layers"),
    ("inner_proj", "intermediate_dense"),
    ("self_attn.output_proj", "self_attn.linear_out"),
    ("output_proj", "output_dense"),
    ("self_attn.k_proj", "self_attn.linear_k"),
    ("self_attn.v_proj", "self_attn.linear_v"),
    ("self_attn.q_proj", "self_attn.linear_q"),
    ("self_attn.sdpa.u_bias", "self_attn.pos_bias_u"),
    ("self_attn.sdpa.v_bias", "self_attn.pos_bias_v"),
    ("self_attn.sdpa.r_proj", "self_attn.linear_pos"),
    ("conv.pointwise_conv1", "conv_module.pointwise_conv1"),
    ("conv.pointwise_conv2", "conv_module.pointwise_conv2"),
    ("conv.depthwise_conv", "conv_module.depthwise_conv"),
    ("conv.batch_norm", "conv_module.batch_norm"),
    ("conv_layer_norm", "conv_module.layer_norm"),
    ("speech_encoder.proj1", "intermediate_ffn.intermediate_dense"),
    ("speech_encoder.proj2", "intermediate_ffn.output_dense"),
    ("speech_encoder.layer_norm", "inner_layer_norm"),
]

# 用于文本模型的转换列表，将源名称映射到目标名称
t2u_convert_list = [
    ("t2u_model.final_proj", "lm_head"),
    ("t2u_model.", "model."),
    ("encoder_decoder_attn_layer_norm", "cross_attention_layer_norm"),
    ("encoder_decoder_attn", "cross_attention"),
    ("linear_k", "k_proj"),
    ("linear_v", "v_proj"),
    ("linear_q", "q_proj"),
    ("ffn.inner_proj", "ffn.fc1"),
    ("ffn.output_proj", "ffn.fc2"),
    ("output_proj", "out_proj"),
    ("decoder_frontend.embed", "decoder.embed_tokens"),
]

# 用于文本模型的转换列表，将源名称映射到目标名称
text_convert_list = [
    ("text_encoder.", ""),
    ("text_decoder.", ""),
    ("text_encoder_frontend.embed", "embed_tokens"),
    ("text_decoder_frontend.embed", "embed_tokens"),
    ("encoder_decoder_attn_layer_norm", "cross_attention_layer_norm"),
    ("encoder_decoder_attn", "cross_attention"),
    ("linear_k", "k_proj"),
    ("linear_v", "v_proj"),
    ("linear_q", "q_proj"),
    ("ffn.inner_proj", "ffn.fc1"),
    ("ffn.output_proj", "ffn.fc2"),
    ("output_proj", "out_proj"),
    ("final_proj", "lm_head"),
]

# 当前文件所在路径
CUR_PATH = os.path.dirname(os.path.abspath(__file__))

# 默认的缓存目录，如果没有设置 XDG_CACHE_HOME 则放在用户的主目录下
default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")

# 缓存文件夹的完整路径，根据环境变量 XDG_CACHE_HOME 设置，存放在 huggingface/hub 目录下
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "huggingface", "hub")


def _load_hf_config(model_type="medium"):
    # 这个函数加载 Hugging Face 模型的配置
    # 如果模型类型是 "medium"，则设置参数 kwargs 为以下数值
    kwargs = {
        "vocab_size": 256206,                   # 词汇表大小设为 256206
        "t2u_vocab_size": 10082,                # t2u 词汇表大小设为 10082
        "hidden_size": 1024,                    # 隐藏层大小设为 1024
        "max_position_embeddings": 4096,        # 最大位置嵌入数设为 4096
        "encoder_layers": 12,                   # 编码器层数设为 12
        "decoder_layers": 12,                   # 解码器层数设为 12
        "encoder_ffn_dim": 4096,                # 编码器中的前馈网络维度设为 4096
        "decoder_ffn_dim": 4096,                # 解码器中的前馈网络维度设为 4096
        "t2u_encoder_layers": 4,                # t2u 编码器层数设为 4
        "t2u_decoder_layers": 4,                # t2u 解码器层数设为 4
        "speech_encoder_layers": 12,            # 语音编码器层数设为 12
    }
    # 根据 kwargs 创建 SeamlessM4TConfig 对象并返回
    return SeamlessM4TConfig(**kwargs)
def _convert_model(
    original_model,
    hf_model,
    convert_list,
    device,
    unwanted_prefix="model.",
    filter_state_dict="speech",
    exclude_state_dict=None,
):
    # 获取原始模型的状态字典
    state_dict = original_model.state_dict()

    # 定义过滤函数
    if isinstance(filter_state_dict, str):
        # 如果过滤条件是字符串，创建对应的过滤函数
        def filter_func(x):
            return filter_state_dict in x[0]

    else:
        # 如果过滤条件是列表，则创建对应的过滤函数
        def filter_func(item):
            # 检查是否需要排除特定的状态字典项
            if exclude_state_dict is not None and exclude_state_dict in item[0]:
                return False
            # 遍历过滤条件列表，检查状态字典项是否符合任一条件
            for filter_el in filter_state_dict:
                if filter_el in item[0]:
                    return True
            return False

    # 使用过滤函数过滤状态字典，得到新的状态字典
    state_dict = dict(filter(filter_func, state_dict.items()))

    # 遍历状态字典中的每个键值对
    for k, v in list(state_dict.items()):
        # 移除不需要的前缀
        new_k = k[len(unwanted_prefix):]
        # 根据转换列表，将旧层名称替换为新层名称
        for old_layer_name, new_layer_name in convert_list:
            if old_layer_name in new_k:
                new_k = new_k.replace(old_layer_name, new_layer_name)

        # 手动处理特定情况，例如处理 ".layer_norm" 的替换
        if ".layer_norm" in new_k and new_k.split(".layer_norm")[0][-1].isnumeric():
            new_k = new_k.replace("layer_norm", "final_layer_norm")

        # 更新状态字典，移除旧键，添加新键
        state_dict[new_k] = state_dict.pop(k)

    # 检查额外的键和缺失的键，并确保模型状态字典加载成功
    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    extra_keys = set(extra_keys)
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set({k for k in missing_keys if "final_logits_bias" not in k})
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")

    # 加载新的状态字典到预训练模型中（允许非严格匹配）
    hf_model.load_state_dict(state_dict, strict=False)

    # 计算加载后的模型参数数量
    n_params = param_count(hf_model)

    # 记录模型加载信息，包括参数数量
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params")

    # 设置模型为评估模式，并移动到指定设备
    hf_model.eval()
    hf_model.to(device)

    # 释放状态字典占用的内存
    del state_dict

    # 返回加载并转换后的模型
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

    # 根据模型类型选择模型名称
    if model_type == "medium":
        name = "seamlessM4T_medium"
    else:
        name = "seamlessM4T_large"

    # 创建原始模型实例
    original_model = Translator(name, "vocoder_36langs", device, torch.float32)

    ######### TOKENIZER

    # 根据模型类型选择支持的语言列表
    langs = MEDIUM_SUPPORTED_LANGUAGES if model_type == "medium" else LARGE_SUPPORTED_LANGUAGES
    langs = [f"__{lang}__" for lang in langs]

    # 构建词汇文件路径
    vocab_file = os.path.join(os.path.expanduser("~"), "tokenizer", model_type, "tokenizer.model")

    # 确保保存目录存在
    save_dir = os.path.join(save_dir, name)
    Path(save_dir).mkdir(exist_ok=True)

    # 创建并配置Tokenizer实例
    tokenizer = SeamlessM4TTokenizer(vocab_file, additional_special_tokens=langs)
    # 获取 "__fra__" 标记的语言ID，用于验证分词器的一致性
    sanity_check_lang_id = tokenizer.convert_tokens_to_ids("__fra__")

    # 将分词器保存到指定目录
    tokenizer.save_pretrained(save_dir)
    # 从指定目录加载分词器为 SeamlessM4TTokenizer 类型的对象
    tokenizer = SeamlessM4TTokenizer.from_pretrained(save_dir)

    # 检查加载后的 "__fra__" 语言ID是否与之前保存的一致，若不一致则引发 ValueError 异常
    if sanity_check_lang_id != tokenizer.convert_tokens_to_ids("__fra__"):
        raise ValueError(
            f"Error in tokenizer saving/loading - __fra__ lang id is not coherent: {sanity_check_lang_id} vs {tokenizer.convert_tokens_to_ids('__fra__')}"
        )

    ####### 获取语言到ID的映射字典

    # 根据 langs 列表生成文本解码器的语言代码到ID的映射字典
    text_decoder_lang_code_to_id = {lang.replace("__", ""): tokenizer.convert_tokens_to_ids(lang) for lang in langs}

    # 根据 UNIT_SUPPORTED_LANGUAGES 列表生成 t2u 模型的语言代码到ID的映射字典
    # 计算公式为：vocoder 单元词汇表大小 + 5（用于 EOS/PAD/BOS/UNK/MSK） + 支持的语言数量
    t2u_lang_code_to_id = {
        code.replace("__", ""): i + 10005 + len(UNIT_SUPPORTED_LANGUAGES)
        for i, code in enumerate(UNIT_SUPPORTED_LANGUAGES)
    }

    # 根据 VOCODER_SUPPORTED_LANGUAGES 列表生成 vocoder 模型的语言代码到ID的映射字典
    vocoder_lang_code_to_id = {code.replace("__", ""): i for i, code in enumerate(VOCODER_SUPPORTED_LANGUAGES)}

    ######### FE

    # 初始化特征提取器，使用 langs 参数指定语言代码
    fe = SeamlessM4TFeatureExtractor(language_code=langs)

    # 将特征提取器保存到指定目录
    fe.save_pretrained(save_dir)
    # 从指定目录加载特征提取器为 SeamlessM4TFeatureExtractor 类型的对象
    fe = SeamlessM4TFeatureExtractor.from_pretrained(save_dir)

    # 使用特征提取器和分词器初始化处理器
    processor = SeamlessM4TProcessor(feature_extractor=fe, tokenizer=tokenizer)
    # 将处理器保存到指定目录
    processor.save_pretrained(save_dir)
    # 将处理器推送到 Hub 上的指定仓库，并创建 pull request（PR）
    processor.push_to_hub(repo_id=repo_id, create_pr=True)

    # 从指定目录加载处理器为 SeamlessM4TProcessor 类型的对象
    processor = SeamlessM4TProcessor.from_pretrained(save_dir)

    ######## Model

    # 初始化 hf_model，加载指定类型的配置文件
    hf_config = _load_hf_config(model_type)
    # 使用 hf_config 初始化 SeamlessM4TModel 类型的 hf_model
    hf_model = SeamlessM4TModel(hf_config)

    # 设置生成配置的特定属性：text_decoder_lang_to_code_id、t2u_lang_code_to_id、vocoder_lang_code_to_id
    hf_model.generation_config.__setattr__("text_decoder_lang_to_code_id", text_decoder_lang_code_to_id)
    hf_model.generation_config.__setattr__("t2u_lang_code_to_id", t2u_lang_code_to_id)
    hf_model.generation_config.__setattr__("vocoder_lang_code_to_id", vocoder_lang_code_to_id)

    # -1. 处理 vocoder
    # 类似于 speech T5，必须应用并移除权重归一化
    hf_model.vocoder.apply_weight_norm()
    # 将原始模型转换为 hf_model.vocoder，采用指定的转换列表和设备，并过滤掉不需要的前缀和状态字典
    hf_model.vocoder = _convert_model(
        original_model,
        hf_model.vocoder,
        vocoder_convert_list,
        device,
        unwanted_prefix="vocoder.code_generator.",
        filter_state_dict="vocoder",
    )
    # 移除 vocoder 模型的权重归一化
    hf_model.vocoder.remove_weight_norm()

    # 1. 处理语音编码器
    wav2vec = hf_model.speech_encoder
    # 将原始模型转换为 hf_model.speech_encoder，采用指定的转换列表和设备，并过滤掉不需要的前缀和状态字典
    hf_model.speech_encoder = _convert_model(
        original_model, wav2vec, wav2vec_convert_list, device, unwanted_prefix="model.", filter_state_dict="speech"
    )

    # 2. 处理 t2u 模型
    # 将原始模型转换为 hf_model.t2u_model，采用指定的转换列表和设备，并过滤掉不需要的前缀和状态字典
    hf_model.t2u_model = _convert_model(
        original_model,
        hf_model.t2u_model,
        t2u_convert_list,
        device,
        unwanted_prefix="model.",
        filter_state_dict="t2u_model",
    )

    # 3. 处理文本编码器
    # 将原始模型转换为 hf_model.text_encoder，采用指定的转换列表和设备，并过滤掉不需要的前缀和状态字典，
    # 同时排除 t2u_model 的状态字典
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

    # 4. take care of text decoder
    hf_model.text_decoder = _convert_model(
        original_model,
        hf_model.text_decoder,
        text_convert_list,
        device,
        unwanted_prefix="model.",
        filter_state_dict=["model.text_decoder"],
        exclude_state_dict="t2u_model",
    )

    # 5. take care of final proj
    hf_model.lm_head = _convert_model(
        original_model,
        hf_model.lm_head,
        [("final_proj.", "")],
        device,
        unwanted_prefix="model.",
        filter_state_dict=["model.final_proj"],
        exclude_state_dict="t2u_model",
    )

    # sanity check: print tied parameters in hf_model
    print(find_tied_parameters(hf_model))

    # count parameters in both hf_model and original_model
    count_1 = param_count(hf_model)
    count_2 = param_count(original_model)

    # print parameter counts and their difference
    print(f"HF MODEL:{count_1}, ORIGINAL_MODEL: {count_2}, diff:{count_1 - count_2}")

    # print parameter count of hf_model excluding embeddings
    print(f"HF MODEL excluding embeddings:{hf_model.num_parameters(exclude_embeddings=True)}")

    # delete original_model to free up memory
    del original_model

    # set _from_model_config attribute to False
    hf_model.generation_config._from_model_config = False

    # save hf_model to specified directory
    hf_model.save_pretrained(save_dir)

    # push hf_model to a hub repository with specified repo_id and create a pull request
    hf_model.push_to_hub(repo_id=repo_id, create_pr=True)

    # load SeamlessM4TModel from the saved directory
    hf_model = SeamlessM4TModel.from_pretrained(save_dir)
if __name__ == "__main__":
    # 如果当前脚本被直接执行（而非被导入为模块），则执行以下代码

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_type",
        default="medium",
        type=str,
        help="Model type.",
    )
    # 添加一个命令行参数，指定模型类型，默认为'medium'，类型为字符串，帮助文本为“Model type.”

    parser.add_argument(
        "--save_dir",
        default="/home/ubuntu/weights",
        type=str,
        help="Path to the output PyTorch model.",
    )
    # 添加一个命令行参数，指定保存模型的目录路径，默认为'/home/ubuntu/weights'，类型为字符串，帮助文本为“Path to the output PyTorch model.”

    parser.add_argument(
        "--repo_id",
        default="facebook/hf-seamless-m4t-medium",
        type=str,
        help="Repo ID.",
    )
    # 添加一个命令行参数，指定仓库 ID，默认为'facebook/hf-seamless-m4t-medium'，类型为字符串，帮助文本为“Repo ID.”

    # 解析命令行参数并将其保存到args对象
    args = parser.parse_args()

    # 调用load_model函数，传入解析后的参数
    load_model(args.save_dir, args.model_type, args.repo_id)
```