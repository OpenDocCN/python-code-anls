# `.\models\seamless_m4t_v2\convert_fairseq2_to_hf.py`

```
# coding=utf-8
# 版权声明和许可证信息

""" 将 Meta SeamlessM4Tv2 检查点从 seamless_communication 转换为 HF."""

import argparse  # 导入处理命令行参数的模块
import os  # 导入操作系统功能的模块
from pathlib import Path  # 导入处理路径的模块

import torch  # 导入 PyTorch 深度学习框架
from accelerate.utils.modeling import find_tied_parameters  # 导入加速训练相关的模块
from seamless_communication.inference import Translator  # 导入无缝通信的翻译器

from transformers import (  # 导入 Hugging Face Transformers 相关模块
    SeamlessM4TFeatureExtractor,
    SeamlessM4TProcessor,
    SeamlessM4TTokenizer,
    SeamlessM4Tv2Config,
    SeamlessM4Tv2Model,
)
from transformers.utils import logging  # 导入日志记录相关的模块

# fmt: off
UNIT_SUPPORTED_LANGUAGES = [  # 支持单元模型的语言列表
    "__arb__", "__ben__", "__cat__", "__ces__", "__cmn__", "__cym__", "__dan__", "__deu__", "__eng__",
    "__est__", "__fin__", "__fra__", "__hin__", "__ind__", "__ita__", "__jpn__", "__kan__", "__kor__",
    "__mlt__", "__nld__", "__pes__", "__pol__", "__por__", "__ron__", "__rus__", "__slk__", "__spa__",
    "__swe__", "__swh__", "__tam__", "__tel__", "__tgl__", "__tha__", "__tur__", "__ukr__", "__urd__",
    "__uzn__", "__vie__",
]
# fmt: on

# fmt: off
VOCODER_SUPPORTED_LANGUAGES = [  # 支持语音合成器模型的语言列表，与 UNIT_SUPPORTED_LANGUAGES 相同
    "__arb__", "__ben__", "__cat__", "__ces__", "__cmn__", "__cym__", "__dan__", "__deu__", "__eng__",
    "__est__", "__fin__", "__fra__", "__hin__", "__ind__", "__ita__", "__jpn__", "__kor__", "__mlt__",
    "__nld__", "__pes__", "__pol__", "__por__", "__ron__", "__rus__", "__slk__", "__spa__", "__swe__",
    "__swh__", "__tel__", "__tgl__", "__tha__", "__tur__", "__ukr__", "__urd__", "__uzn__", "__vie__",
]
# fmt: on

# fmt: off
LARGE_SUPPORTED_LANGUAGES = [  # 支持大型模型的语言列表
    "afr", "amh", "arb", "ary", "arz", "asm", "azj", "bel", "ben", "bos", "bul", "cat", "ceb", "ces", "ckb",
    "cmn", "cmn_Hant", "cym", "dan", "deu", "ell", "eng", "est", "eus", "fin", "fra", "fuv", "gaz", "gle",
    "glg", "guj", "heb", "hin", "hrv", "hun", "hye", "ibo", "ind", "isl", "ita", "jav", "jpn", "kan", "kat",
    "kaz", "khk", "khm", "kir", "kor", "lao", "lit", "lug", "luo", "lvs", "mai", "mal", "mar", "mkd", "mlt",
    "mni", "mya", "nld", "nno", "nob", "npi", "nya", "ory", "pan", "pbt", "pes", "pol", "por", "ron", "rus",
    "sat", "slk", "slv", "sna", "snd", "som", "spa", "srp", "swe", "swh", "tam", "tel", "tgk", "tgl", "tha",
    "tur", "ukr", "urd", "uzn", "vie", "yor", "yue", "zlm", "zul",
]
# fmt: on

def assert_param_count(model_1, model_2):
    # 检查两个模型的参数数量是否相等（不包括 "final_proj" 在内的参数）
    count_1 = sum(p[1].numel() for p in model_1.named_parameters() if "final_proj" not in p[0])
    count_2 = sum(p[1].numel() for p in model_2.named_parameters() if "final_proj" not in p[0])
    assert count_1 == count_2, f"{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}"
# 计算模型中除了包含"final_proj"的参数外的总数量
def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters() if "final_proj" not in p[0])


# 根据是否使用GPU返回最佳设备
def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)


# 设置日志级别为信息
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 音码器（vocoder）模型参数转换列表
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

# 顺序很重要
# wav2vec模型参数转换列表
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

# t2u模型参数转换列表
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
    # 定义一个包含多个元组的元组，每个元组包含两个字符串作为元素
    ("conv1d_layer_norm", "conv_layer_norm"),
    # 另一个元组，包含两个字符串作为元素
    ("decoder_frontend.variance_adaptor", "decoder"),
    # 另一个元组，包含两个字符串作为元素，用于指定 duration_predictor 模块中的第一个卷积层
    ("duration_predictor.conv1.0", "duration_predictor.conv1"),
    # 另一个元组，包含两个字符串作为元素，用于指定 duration_predictor 模块中的第二个卷积层
    ("duration_predictor.conv2.0", "duration_predictor.conv2"),
# 转换文本列表，将元组中的第一个字符串替换为第二个字符串
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

# 默认缓存目录为用户主目录下的.cache/huggingface/hub
default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "huggingface", "hub")


def _load_hf_config():
    # 返回一个SeamlessM4Tv2Config对象，用于加载配置
    return SeamlessM4Tv2Config()


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

    # 筛选函数
    if isinstance(filter_state_dict, str):
        # 如果filter_state_dict是字符串，则定义一个筛选函数，根据字符串在键名中的存在与否来筛选
        def filter_func(x):
            return filter_state_dict in x[0]

    else:
        # 如果filter_state_dict不是字符串，则定义另一个筛选函数，根据列表中的元素在键名中的存在与否来筛选
        def filter_func(item):
            if exclude_state_dict is not None and exclude_state_dict in item[0]:
                return False
            for filter_el in filter_state_dict:
                if filter_el in item[0]:
                    return True
            return False

    # 使用筛选函数过滤状态字典，生成新的状态字典
    state_dict = dict(filter(filter_func, state_dict.items()))

    # 遍历状态字典中的每一个键值对
    for k, v in list(state_dict.items()):
        # 新键名去除unwanted_prefix前缀
        new_k = k[len(unwanted_prefix) :]
        # 根据转换列表，将符合条件的旧层名替换为新层名
        for old_layer_name, new_layer_name in convert_list:
            if old_layer_name in new_k:
                new_k = new_k.replace(old_layer_name, new_layer_name)

        # 手动处理包含".layer_norm"且前缀为数字的情况
        if ".layer_norm" in new_k and new_k.split(".layer_norm")[0][-1].isnumeric():
            new_k = new_k.replace("layer_norm", "final_layer_norm")

        # 更新状态字典，将旧键名对应的值移至新键名下
        state_dict[new_k] = state_dict.pop(k)

    # 计算多余的和缺失的键，并确保模型状态字典的一致性
    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")

    # 加载处理后的状态字典到hf_model中
    hf_model.load_state_dict(state_dict, strict=False)

    # 计算模型参数数量并记录日志
    n_params = param_count(hf_model)
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params")

    # 将模型设置为评估模式，并移动到指定设备
    hf_model.eval()
    hf_model.to(device)

    # 删除状态字典以释放内存
    del state_dict

    # 返回加载并转换后的hf_model
    return hf_model


def load_model(save_dir, model_type, repo_id):
    """
    加载模型函数，通过指定保存目录、模型类型和库ID来加载模型
    Meta SeamlessM4Tv2由8个主要组件组成:
    - speech_encoder (#1) 和 speech_encoder_frontend (#2)
    - t2u_model (#3)
    - text_encoder (#4) 和 text_encoder_frontend (#5)
    """
    pass  # 这里的函数体并未提供，在实际应用中应根据具体需要进行实现
    # 从内部函数 _grab_best_device() 获取最佳设备
    device = _grab_best_device()
    # 设置模型名称
    name = "seamlessM4T_v2_large"

    # 使用 Translator 类创建原始模型对象，用于翻译器和声码器
    original_model = Translator(name, "vocoder_v2", device, dtype=torch.float32)

    ######### TOKENIZER

    # 获取支持的大语言列表
    langs = LARGE_SUPPORTED_LANGUAGES
    # 为每种语言生成特殊的标记
    langs = [f"__{lang}__" for lang in langs]
    # 指定词汇文件的路径
    vocab_file = os.path.join(os.path.expanduser("~"), "tokenizer", model_type, "tokenizer.model")

    # 创建保存目录路径
    save_dir = os.path.join(save_dir, name)
    # 如果保存目录不存在，则创建它
    Path(save_dir).mkdir(exist_ok=True)

    # 使用 SeamlessM4TTokenizer 类创建 tokenizer 对象，加入额外的特殊标记
    tokenizer = SeamlessM4TTokenizer(vocab_file, additional_special_tokens=langs)

    # 检查 "__fra__" 标记是否正确转换为对应的语言 id
    sanity_check_lang_id = tokenizer.convert_tokens_to_ids("__fra__")

    # 将 tokenizer 的配置保存到指定的目录
    tokenizer.save_pretrained(save_dir)
    # 从预训练的保存目录中加载 tokenizer 对象
    tokenizer = SeamlessM4TTokenizer.from_pretrained(save_dir)

    # 如果 "__fra__" 标记的 id 转换不正确，则抛出 ValueError 异常
    if sanity_check_lang_id != tokenizer.convert_tokens_to_ids("__fra__"):
        raise ValueError(
            f"Error in tokenizer saving/loading - __fra__ lang id is not coherent: {sanity_check_lang_id} vs {tokenizer.convert_tokens_to_ids('__fra__')}"
        )

    ####### get language to ids dict

    # 创建文本解码器的语言代码到 id 的映射字典
    text_decoder_lang_code_to_id = {lang.replace("__", ""): tokenizer.convert_tokens_to_ids(lang) for lang in langs}
    # 计算 t2u 语言代码到 id 的偏移量，考虑词汇大小和支持的语言数目
    t2u_lang_code_to_id = {
        code.replace("__", ""): i + 10005 + len(UNIT_SUPPORTED_LANGUAGES)
        for i, code in enumerate(UNIT_SUPPORTED_LANGUAGES)
    }
    # 创建声码器的语言代码到 id 的映射字典
    vocoder_lang_code_to_id = {code.replace("__", ""): i for i, code in enumerate(VOCODER_SUPPORTED_LANGUAGES)}

    ######### FE

    # 使用 SeamlessM4TFeatureExtractor 类创建特征提取器对象，指定语言代码
    fe = SeamlessM4TFeatureExtractor(language_code=langs)

    # 将特征提取器的配置保存到指定的目录
    fe.save_pretrained(save_dir)
    # 从预训练的保存目录中加载特征提取器对象
    fe = SeamlessM4TFeatureExtractor.from_pretrained(save_dir)

    # 使用 SeamlessM4TProcessor 类创建处理器对象，传入特征提取器和 tokenizer 对象
    processor = SeamlessM4TProcessor(feature_extractor=fe, tokenizer=tokenizer)
    # 将处理器的配置保存到指定的目录，并推送到 Hub 仓库中
    processor.save_pretrained(save_dir)
    processor.push_to_hub(repo_id=repo_id, create_pr=True)

    # 从预训练的保存目录中加载处理器对象
    processor = SeamlessM4TProcessor.from_pretrained(save_dir)

    ######## Model

    # 初始化 hf_config 配置对象
    hf_config = _load_hf_config()

    ######## get id_to_text and char_to_id from original model tokenizers

    # 创建 id 到文本和字符到 id 的映射字典，从原始模型的 tokenizer 中获取
    id_to_text = {i: original_model.text_tokenizer.model.index_to_token(i) for i in range(hf_config.vocab_size)}
    char_to_id = {
        original_model.model.t2u_model.decoder_frontend.char_tokenizer.model.index_to_token(i): i for i in range(10904)
    }

    # 初始化 SeamlessM4Tv2Model 模型对象
    hf_model = SeamlessM4Tv2Model(hf_config)

    # 设置生成配置对象的文本解码器、t2u、声码器的语言到 id 的映射字典以及 id 到文本、字符到 id 的映射字典
    hf_model.generation_config.__setattr__("text_decoder_lang_to_code_id", text_decoder_lang_code_to_id)
    hf_model.generation_config.__setattr__("t2u_lang_code_to_id", t2u_lang_code_to_id)
    hf_model.generation_config.__setattr__("vocoder_lang_code_to_id", vocoder_lang_code_to_id)
    hf_model.generation_config.__setattr__("id_to_text", id_to_text)
    hf_model.generation_config.__setattr__("char_to_id", char_to_id)

    # -1. take care of vocoder
    # 类似于语音模型 T5，必须应用和移除权重规范化
    hf_model.vocoder.apply_weight_norm()
    
    # 转换语音编码器模型，将原始模型的状态转移到 hf_model.speech_encoder 上
    hf_model.vocoder = _convert_model(
        original_model,
        hf_model.vocoder,
        vocoder_convert_list,
        device,
        unwanted_prefix="vocoder.code_generator.",
        filter_state_dict="vocoder",
    )
    
    # 移除语音模型的权重规范化
    hf_model.vocoder.remove_weight_norm()

    # 1. 处理语音编码器
    wav2vec = hf_model.speech_encoder
    hf_model.speech_encoder = _convert_model(
        original_model, wav2vec, wav2vec_convert_list, device, unwanted_prefix="model.", filter_state_dict="speech"
    )

    # 2. 处理 t2u 模型

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

    # 5. 处理最终投影层
    hf_model.lm_head = _convert_model(
        original_model,
        hf_model.lm_head,
        [("final_proj.", "")],
        device,
        unwanted_prefix="model.",
        filter_state_dict=["model.final_proj"],
        exclude_state_dict="t2u_model",
    )

    # 检查模型参数绑定情况
    print(find_tied_parameters(hf_model))

    # 统计参数数量
    count_1 = param_count(hf_model)
    count_2 = param_count(original_model)

    # 打印参数数量差异
    print(f"HF MODEL:{count_1}, ORIGINAL_MODEL: {count_2}, diff:{count_1 - count_2}")
    
    # 打印不包括嵌入层的 HF 模型参数数量
    print(f"HF MODEL excluding embeddings:{hf_model.num_parameters(exclude_embeddings=True)}")

    # 删除原始模型
    del original_model

    # 禁止从模型配置加载
    hf_model.generation_config._from_model_config = False
    
    # 保存转换后的模型到指定目录
    hf_model.save_pretrained(save_dir)
    
    # 推送模型到指定的 Hub 仓库，并创建 PR（Pull Request）
    hf_model.push_to_hub(repo_id=repo_id, create_pr=True)
    
    # 从保存的目录加载 SeamlessM4Tv2Model 模型
    hf_model = SeamlessM4Tv2Model.from_pretrained(save_dir)
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_type",
        default="large",
        type=str,
        help="Model type.",
    )
    # 添加一个名为 --model_type 的参数选项，类型为字符串，默认为 "large"
    # 用于指定模型类型，例如大型模型、中型模型等

    parser.add_argument(
        "--save_dir",
        default="/home/ubuntu/weights_v2",
        type=str,
        help="Path to the output PyTorch model.",
    )
    # 添加一个名为 --save_dir 的参数选项，类型为字符串，默认为 "/home/ubuntu/weights_v2"
    # 用于指定 PyTorch 模型的输出路径

    parser.add_argument(
        "--repo_id",
        default="facebook/seamless-m4t-v2-large",
        type=str,
        help="Repo ID.",
    )
    # 添加一个名为 --repo_id 的参数选项，类型为字符串，默认为 "facebook/seamless-m4t-v2-large"
    # 用于指定代码仓库的 ID 或名称

    args = parser.parse_args()
    # 解析命令行参数，并将其存储在 args 对象中

    load_model(args.save_dir, args.model_type, args.repo_id)
    # 调用 load_model 函数，传递参数 args.save_dir、args.model_type 和 args.repo_id
    # 用于加载指定路径下的模型，指定的模型类型和相关代码仓库
```