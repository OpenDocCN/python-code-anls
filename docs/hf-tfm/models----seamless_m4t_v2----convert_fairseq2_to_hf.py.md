# `.\transformers\models\seamless_m4t_v2\convert_fairseq2_to_hf.py`

```py
# 导入必要的模块和类
import argparse
import os
from pathlib import Path

import torch
from accelerate.utils.modeling import find_tied_parameters
from seamless_communication.inference import Translator

from transformers import (
    SeamlessM4TFeatureExtractor,
    SeamlessM4TProcessor,
    SeamlessM4TTokenizer,
    SeamlessM4Tv2Config,
    SeamlessM4Tv2Model,
)
from transformers.utils import logging

# 定义支持的语言列表
# fmt: off
UNIT_SUPPORTED_LANGUAGES = ["__arb__", "__ben__", "__cat__", "__ces__", "__cmn__", "__cym__", "__dan__", "__deu__", "__eng__", "__est__", "__fin__", "__fra__", "__hin__", "__ind__", "__ita__", "__jpn__", "__kan__", "__kor__", "__mlt__", "__nld__", "__pes__", "__pol__", "__por__", "__ron__", "__rus__", "__slk__", "__spa__", "__swe__", "__swh__", "__tam__", "__tel__", "__tgl__", "__tha__", "__tur__", "__ukr__", "__urd__", "__uzn__", "__vie__", ]
# fmt: on

# fmt: off
VOCODER_SUPPORTED_LANGUAGES = ["__arb__", "__ben__", "__cat__", "__ces__", "__cmn__", "__cym__", "__dan__", "__deu__", "__eng__", "__est__", "__fin__", "__fra__", "__hin__", "__ind__", "__ita__", "__jpn__", "__kor__", "__mlt__", "__nld__", "__pes__", "__pol__", "__por__", "__ron__", "__rus__", "__slk__", "__spa__", "__swe__", "__swh__", "__tel__", "__tgl__", "__tha__", "__tur__", "__ukr__", "__urd__", "__uzn__", "__vie__",]
# fmt: on

# fmt: off
LARGE_SUPPORTED_LANGUAGES = ["afr","amh","arb","ary","arz","asm","azj","bel","ben","bos","bul","cat","ceb","ces","ckb","cmn","cmn_Hant","cym","dan","deu","ell","eng","est","eus","fin","fra","fuv","gaz","gle","glg","guj","heb","hin","hrv","hun","hye","ibo","ind","isl","ita","jav","jpn","kan","kat","kaz","khk","khm","kir","kor","lao","lit","lug","luo","lvs","mai","mal","mar","mkd","mlt","mni","mya","nld","nno","nob","npi","nya","ory","pan","pbt","pes","pol","por","ron","rus","sat","slk","slv","sna","snd","som","spa","srp","swe","swh","tam","tel","tgk","tgl","tha","tur","ukr","urd","uzn","vie","yor","yue","zlm","zul",]
# fmt: on

# 定义一个函数, 用来比较两个模型的参数数量
def assert_param_count(model_1, model_2):
    count_1 = sum(p[1].numel() for p in model_1.named_parameters() if "final_proj" not in p[0])
    count_2 = sum(p[1].numel() for p in model_2.named_parameters() if "final_proj" not in p[0])
    assert count_1 == count_2, f"{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}"


这段代码主要做了以下几件事:

1. 导入了必要的模块和类, 包括 `argparse`、`os`、`Path`、`torch`、`find_tied_parameters`、`Translator`、以及一些 `transformers` 相关的类。

2. 定义了三个列表, 分别是 `UNIT_SUPPORTED_LANGUAGES`、`VOCODER_SUPPORTED_LANGUAGES` 和 `LARGE_SUPPORTED_LANGUAGES`, 这些列表包含了支持的语言。

3. 定义了一个名为 `assert_param_count` 的函数, 用于比较两个模型的参数数量是否相等, 不包括 `final_proj` 层的参数。

整体来看, 这段代码主要是为了处理 Meta 公司的 SeamlessM4Tv2 模型, 并将其转换为 Hugging Face 的格式。
# 计算模型参数数量，不包括名字中包含"final_proj"的参数
def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters() if "final_proj" not in p[0])

# 获取最佳设备，如果可用 GPU 且 use_gpu 参数为 True，则返回 cuda 设备，否则返回 cpu 设备
def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)

# 设置日志记录级别为 INFO
logging.set_verbosity_info()
# 获取名称为 __name__ 的记录器
logger = logging.get_logger(__name__)

# 声码器的转换列表，将原模型的层名映射到新模型的层名
vocoder_convert_list = [
    ("ups", "hifi_gan.upsampler"),
    ("conv_pre", "hifi_gan.conv_pre"),
    ("resblocks", "hifi_gan.resblocks"),
    ("conv_post", "hifi_gan.conv_post"),
    ("lang", "language_embedding"),
    ("spkr", "speaker_embedding"),
    ("dict.", "unit_embedding."),
    ("dur_predictor.conv1.0", "dur_predictor.conv1"),
    ("dur_predictor.conv2.0", "dur_predictor.conv2"),
]

# 声码器转换列表，将原模型的层名映射到新模型的层名
# 转换的顺序很重要
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
    ("self_attn.sdpa.rel_k_embed", "self_attn.distance_embedding"),
    ("self_attn.sdpa.r_proj", "self_attn.linear_pos"),
    ("conv.pointwise_conv1", "conv_module.pointwise_conv1"),
    ("conv.pointwise_conv2", "conv_module.pointwise_conv2"),
    ("conv.depthwise_conv", "conv_module.depthwise_conv"),
    ("conv.batch_norm", "conv_module.batch_norm"),
    ("conv.layer_norm", "conv_module.depthwise_layer_norm"),
    ("conv_layer_norm", "conv_module.layer_norm"),
    ("speech_encoder.proj1", "intermediate_ffn.intermediate_dense"),
    ("speech_encoder.proj2", "intermediate_ffn.output_dense"),
    ("speech_encoder.layer_norm", "inner_layer_norm"),
]

# 文本到单位转换器的转换列表，将原模型的层名映射到新模型的层名
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
    ("decoder_frontend.embed_char", "decoder.embed_char"),
    ("decoder_frontend.pos_emb_alpha_char", "decoder.pos_emb_alpha_char"),
    ("decoder_frontend.embed", "decoder.embed_tokens"),
    ("decoder_frontend.pos_emb_alpha", "decoder.pos_emb_alpha"),
    ("conv1d.conv", "conv"),
]
    # 将元组中的两个字符串作为键值对添加到字典中，第一个字符串作为键，第二个字符串作为值
    ("conv1d_layer_norm", "conv_layer_norm"),
    # 将元组中的两个字符串作为键值对添加到字典中，第一个字符串作为键，第二个字符串作为值
    ("decoder_frontend.variance_adaptor", "decoder"),
    # 将元组中的两个字符串作为键值对添加到字典中，第一个字符串作为键，第二个字符串作为值
    ("duration_predictor.conv1.0", "duration_predictor.conv1"),
    # 将元组中的两个字符串作为键值对添加到字典中，第一个字符串作为键，第二个字符串作为值
    ("duration_predictor.conv2.0", "duration_predictor.conv2"),
# 定义一个将模型参数名称从原始格式转换为 HuggingFace 格式的列表
text_convert_list = [
    # 将 "text_encoder." 前缀替换为空字符串
    ("text_encoder.", ""),
    # 将 "text_decoder." 前缀替换为空字符串
    ("text_decoder.", ""),
    # 将 "text_encoder_frontend.embed" 替换为 "embed_tokens"
    ("text_encoder_frontend.embed", "embed_tokens"),
    # 将 "text_decoder_frontend.embed" 替换为 "embed_tokens"
    ("text_decoder_frontend.embed", "embed_tokens"),
    # 将 "encoder_decoder_attn_layer_norm" 替换为 "cross_attention_layer_norm"
    ("encoder_decoder_attn_layer_norm", "cross_attention_layer_norm"),
    # 将 "encoder_decoder_attn" 替换为 "cross_attention"
    ("encoder_decoder_attn", "cross_attention"),
    # 将 "linear_k" 替换为 "k_proj"
    ("linear_k", "k_proj"),
    # 将 "linear_v" 替换为 "v_proj"
    ("linear_v", "v_proj"),
    # 将 "linear_q" 替换为 "q_proj"
    ("linear_q", "q_proj"),
    # 将 "ffn.inner_proj" 替换为 "ffn.fc1"
    ("ffn.inner_proj", "ffn.fc1"),
    # 将 "ffn.output_proj" 替换为 "ffn.fc2"
    ("ffn.output_proj", "ffn.fc2"),
    # 将 "output_proj" 替换为 "out_proj"
    ("output_proj", "out_proj"),
    # 将 "final_proj" 替换为 "lm_head"
    ("final_proj", "lm_head"),
]

# 获取当前脚本所在的目录路径
CUR_PATH = os.path.dirname(os.path.abspath(__file__))

# 定义默认的缓存目录路径
default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")

# 获取 HuggingFace 缓存目录路径
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "huggingface", "hub")


# 加载 HuggingFace 配置
def _load_hf_config():
    return SeamlessM4Tv2Config()


# 转换模型参数权重
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

    # 定义过滤状态字典的函数
    if isinstance(filter_state_dict, str):
        def filter_func(x):
            return filter_state_dict in x[0]
    else:
        def filter_func(item):
            if exclude_state_dict is not None and exclude_state_dict in item[0]:
                return False
            for filter_el in filter_state_dict:
                if filter_el in item[0]:
                    return True
            return False

    # 根据过滤函数过滤状态字典
    state_dict = dict(filter(filter_func, state_dict.items()))

    # 将状态字典的键名转换为 HuggingFace 格式
    for k, v in list(state_dict.items()):
        new_k = k[len(unwanted_prefix) :]
        for old_layer_name, new_layer_name in convert_list:
            if old_layer_name in new_k:
                new_k = new_k.replace(old_layer_name, new_layer_name)

        # 手动处理 ".layer_norm" 中带有数字的情况
        if ".layer_norm" in new_k and new_k.split(".layer_norm")[0][-1].isnumeric():
            new_k = new_k.replace("layer_norm", "final_layer_norm")

        state_dict[new_k] = state_dict.pop(k)

    # 检查状态字典中是否有多余或缺失的键
    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    extra_keys = set(extra_keys)
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set({k for k in missing_keys if "final_logits_bias" not in k})
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")

    # 将状态字典加载到 HuggingFace 模型中
    hf_model.load_state_dict(state_dict, strict=False)
    n_params = param_count(hf_model)

    # 输出模型参数数量
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params")

    # 将模型设置为评估模式并移动到指定设备
    hf_model.eval()
    hf_model.to(device)
    del state_dict

    return hf_model


# 加载模型
def load_model(save_dir, model_type, repo_id):
    """
    Meta SeamlessM4Tv2 is made of 8 main components:
    - speech_encoder (#1) and speech_encoder_frontend (#2)
    - t2u_model (#3)
    - text_encoder (#4) and text_encoder_frontend (#5)
    """
    # 定义字符串变量
    text_decoder (#6) [and text_decoder_frontend (#5) = equals to text_encoder_frontend]
    final_proj (#7)
    vocoder (#8)
    """
    # 获取最佳设备
    device = _grab_best_device()
    # 定义模型名称
    name = "seamlessM4T_v2_large"
    # 使用Translator类创建原始模型
    original_model = Translator(name, "vocoder_v2", device, dtype=torch.float32)
    
    ######### TOKENIZER
    
    # 定义支持的语言列表
    langs = LARGE_SUPPORTED_LANGUAGES
    langs = [f"__{lang}__" for lang in langs]
    # 指定词汇文件路径
    vocab_file = os.path.join(os.path.expanduser("~"), "tokenizer", model_type, "tokenizer.model")
    
    # 保存目录
    save_dir = os.path.join(save_dir, name)
    Path(save_dir).mkdir(exist_ok=True)
    
    # 创建tokenizer对象
    tokenizer = SeamlessM4TTokenizer(vocab_file, additional_special_tokens=langs)
    
    # 对特定语言进行ID检查
    sanity_check_lang_id = tokenizer.convert_tokens_to_ids("__fra__")
    
    # 保存tokenizer
    tokenizer.save_pretrained(save_dir)
    tokenizer = SeamlessM4TTokenizer.from_pretrained(save_dir)
    
    # 检查ID是否一致
    if sanity_check_lang_id != tokenizer.convert_tokens_to_ids("__fra__"):
        raise ValueError(
            f"Error in tokenizer saving/loading - __fra__ lang id is not coherent: {sanity_check_lang_id} vs {tokenizer.convert_tokens_to_ids('__fra__')}"
        )
    
    ####### get language to ids dict
    # 获取语言到ID的映射字典
    text_decoder_lang_code_to_id = {lang.replace("__", ""): tokenizer.convert_tokens_to_ids(lang) for lang in langs}
    # 获取语言到ID的映射字典
    t2u_lang_code_to_id = {
        code.replace("__", ""): i + 10005 + len(UNIT_SUPPORTED_LANGUAGES)
        for i, code in enumerate(UNIT_SUPPORTED_LANGUAGES)
    }
    # 获取语言到ID的映射字典
    vocoder_lang_code_to_id = {code.replace("__", ""): i for i, code in enumerate(VOCODER_SUPPORTED_LANGUAGES)}
    
    ######### FE
    
    # 创建特征提取器对象
    fe = SeamlessM4TFeatureExtractor(language_code=langs)
    
    # 保存特征提取器
    fe.save_pretrained(save_dir)
    fe = SeamlessM4TFeatureExtractor.from_pretrained(save_dir)
    
    # 创建处理器
    processor = SeamlessM4TProcessor(feature_extractor=fe, tokenizer=tokenizer)
    # 保存处理器
    processor.save_pretrained(save_dir)
    # 推送到Hub
    processor.push_to_hub(repo_id=repo_id, create_pr=True)
    
    # 从保存的路径中还原处理器
    processor = SeamlessM4TProcessor.from_pretrained(save_dir)
    
    ######## Model
    
    # 初始化配置
    hf_config = _load_hf_config()
    
    # 从原始模型的分词器中获取id_to_text和char_to_id
    id_to_text = {i: original_model.text_tokenizer.model.index_to_token(i) for i in range(hf_config.vocab_size)}
    char_to_id = {
        original_model.model.t2u_model.decoder_frontend.char_tokenizer.model.index_to_token(i): i for i in range(10904)
    }
    
    # 初始化模型
    hf_model = SeamlessM4Tv2Model(hf_config)
    
    # 设置��成配置
    hf_model.generation_config.__setattr__("text_decoder_lang_to_code_id", text_decoder_lang_code_to_id)
    hf_model.generation_config.__setattr__("t2u_lang_code_to_id", t2u_lang_code_to_id)
    hf_model.generation_config.__setattr__("vocoder_lang_code_to_id", vocoder_lang_code_to_id)
    hf_model.generation_config.__setattr__("id_to_text", id_to_text)
    hf_model.generation_config.__setattr__("char_to_id", char_to_id)
    
    # -1. take care of vocoder
    # 类似于语音 T5，必须应用和移除权重归一化
    hf_model.vocoder.apply_weight_norm()
    # 将原始模型中与语音生成器相关的部分转换为新模型的语音生成器部分
    hf_model.vocoder = _convert_model(
        original_model,
        hf_model.vocoder,
        vocoder_convert_list,
        device,
        unwanted_prefix="vocoder.code_generator.",
        filter_state_dict="vocoder",
    )
    # 移除语音生成器的权重归一化
    hf_model.vocoder.remove_weight_norm()

    # 1. 处理语音编码器
    wav2vec = hf_model.speech_encoder
    hf_model.speech_encoder = _convert_model(
        original_model, wav2vec, wav2vec_convert_list, device, unwanted_prefix="model.", filter_state_dict="speech"
    )

    # 2. 处理 T2U 模型
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

    # 打印检查绑定的参数
    print(find_tied_parameters(hf_model))

    # 计算模型参数数量
    count_1 = param_count(hf_model)
    count_2 = param_count(original_model)

    # 打印模型参数数量以及差异
    print(f"HF MODEL:{count_1}, ORIGINAL_MODEL: {count_2}, diff:{count_1 - count_2}")
    # 打印除去嵌入层的模型参数数量
    print(f"HF MODEL excluding embeddings:{hf_model.num_parameters(exclude_embeddings=True)}")

    # 删除原始模型
    del original_model

    # 设置生成配置中的属性
    hf_model.generation_config._from_model_config = False
    # 保存修改后的模型
    hf_model.save_pretrained(save_dir)
    # 将模型推送到 Hub
    hf_model.push_to_hub(repo_id=repo_id, create_pr=True)
    # 从保存的目录中加载 SeamlessM4Tv2Model 模型
    hf_model = SeamlessM4Tv2Model.from_pretrained(save_dir)
# 如果当前脚本被直接执行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数

    # 添加模型类型参数，设置默认值为"large"，类型为字符串，帮助文本指定了模型类型
    parser.add_argument(
        "--model_type",
        default="large",
        type=str,
        help="Model type.",
    )

    # 添加保存目录参数，设置默认值为"/home/ubuntu/weights_v2"，类型为字符串，帮助文本指定了输出PyTorch模型的路径
    parser.add_argument(
        "--save_dir",
        default="/home/ubuntu/weights_v2",
        type=str,
        help="Path to the output PyTorch model.",
    )

    # 添加仓库ID参数，设置默认值为"facebook/seamless-m4t-v2-large"，类型为字符串，帮助文本指定了仓库ID
    parser.add_argument(
        "--repo_id",
        default="facebook/seamless-m4t-v2-large",
        type=str,
        help="Repo ID.",
    )

    # 解析命令行参数并将其存储在args对象中
    args = parser.parse_args()

    # 调用load_model函数，传入保存目录、模型类型和仓库ID作为参数
    load_model(args.save_dir, args.model_type, args.repo_id)
```