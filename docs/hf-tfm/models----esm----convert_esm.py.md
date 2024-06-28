# `.\models\esm\convert_esm.py`

```
# coding=utf-8
# 版权 2022 年 HuggingFace Inc. 团队所有。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）授权；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据"原样"的基础分发，
# 不提供任何形式的明示或暗示保证或条件。
# 有关详细信息，请参阅许可证。

"""Convert ESM checkpoint."""

# 导入必要的库和模块
import argparse  # 导入命令行参数解析模块
import pathlib   # 导入路径操作模块
from pathlib import Path  # 导入路径操作模块中的Path类
from tempfile import TemporaryDirectory  # 导入临时目录模块

# 导入ESM相关的模块和类
import esm as esm_module  # 导入ESM模块
import torch  # 导入PyTorch库
from esm.esmfold.v1.misc import batch_encode_sequences as esmfold_encode_sequences  # 导入序列编码函数
from esm.esmfold.v1.pretrained import esmfold_v1  # 导入ESM-Fold v1预训练模型

# 导入Transformers相关的类和函数
from transformers.models.esm.configuration_esm import EsmConfig, EsmFoldConfig  # 导入ESM和ESM-Fold的配置类
from transformers.models.esm.modeling_esm import (  # 导入ESM模型相关类
    EsmForMaskedLM,
    EsmForSequenceClassification,
    EsmIntermediate,
    EsmLayer,
    EsmOutput,
    EsmSelfAttention,
    EsmSelfOutput,
)
from transformers.models.esm.modeling_esmfold import EsmForProteinFolding  # 导入蛋白质折叠相关的ESM模型类
from transformers.models.esm.tokenization_esm import EsmTokenizer  # 导入ESM的分词器类
from transformers.utils import logging  # 导入日志记录模块

# 设置日志的详细级别为信息级别
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义样本数据，包含蛋白质序列和标识
SAMPLE_DATA = [
    (
        "protein1",
        "MNGTEGPNFYVPFSNATGVVRSPFEYPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVLGGFTSTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLAGWSRYIPEGLQCSCGIDYYTLKPEVNNESFVIYMFVVHFTIPMIIIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIAFLICWVPYASVAFYIFTHQGSNFGPIFMTIPAFFAKSAAIYNPVIYIMMNKQFRNCMLTTICCGKNPLGDDEASATVSKTETSQVAPA",
    ),
    ("protein2", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA"),
    ("protein3", "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLAGG"),
    ("protein4", "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLA"),
]

# 定义ESM模型的名称与模型对象的映射关系
MODEL_MAPPING = {
    "esm1b_t33_650M_UR50S": esm_module.pretrained.esm1b_t33_650M_UR50S,
    "esm1v_t33_650M_UR90S_1": esm_module.pretrained.esm1v_t33_650M_UR90S_1,
    "esm1v_t33_650M_UR90S_2": esm_module.pretrained.esm1v_t33_650M_UR90S_2,
    "esm1v_t33_650M_UR90S_3": esm_module.pretrained.esm1v_t33_650M_UR90S_3,
    "esm1v_t33_650M_UR90S_4": esm_module.pretrained.esm1v_t33_650M_UR90S_4,
    "esm1v_t33_650M_UR90S_5": esm_module.pretrained.esm1v_t33_650M_UR90S_5,
    "esm2_t48_15B_UR50D": esm_module.pretrained.esm2_t48_15B_UR50D,
    "esm2_t36_3B_UR50D": esm_module.pretrained.esm2_t36_3B_UR50D,
    "esm2_t33_650M_UR50D": esm_module.pretrained.esm2_t33_650M_UR50D,
    "esm2_t30_150M_UR50D": esm_module.pretrained.esm2_t30_150M_UR50D,
    "esm2_t12_35M_UR50D": esm_module.pretrained.esm2_t12_35M_UR50D,
}
    # 将模型名称映射到预训练模型对象的引用："esm2_t6_8M_UR50D"映射到esm_module.pretrained.esm2_t6_8M_UR50D
    "esm2_t6_8M_UR50D": esm_module.pretrained.esm2_t6_8M_UR50D,
    # 将模型名称映射到预训练模型对象的引用："esmfold_v1"映射到esmfold_v1
    "esmfold_v1": esmfold_v1,
}

# 定义氨基酸类型列表
restypes = list("ARNDCQEGHILKMFPSTWYV")

# 在氨基酸类型列表中加入额外的字符 'X'
restypes_with_x = restypes + ["X"]

# 在带有 'X' 的氨基酸类型列表中再加入特殊的 token
restypes_with_extras = restypes_with_x + ["<pad>", "<mask>", "<cls>", "<sep>", "<eos>"]

# 返回一个 ESM 模型的 tokenizer 对象
def get_esmfold_tokenizer():
    # 使用临时目录创建词汇表文件并写入字符列表
    with TemporaryDirectory() as tempdir:
        vocab = "\n".join(restypes_with_extras)
        vocab_file = Path(tempdir) / "vocab.txt"
        vocab_file.write_text(vocab)
        # 使用词汇表文件创建 ESM tokenizer 对象
        hf_tokenizer = EsmTokenizer(vocab_file=str(vocab_file))
    # 设置 padding token 的 ID
    hf_tokenizer.pad_token_id = 0  # 与 'A' 重叠，但这似乎是他们想要的
    return hf_tokenizer

# 将原始模型的权重转移并检查到我们的模型中
def transfer_and_check_weights(original_module, our_module):
    status = our_module.load_state_dict(original_module.state_dict())
    # 如果有缺失的键，则引发 ValueError 异常
    if status.missing_keys:
        raise ValueError(f"Missing keys: {status.missing_keys}")
    # 如果有意外的键，则引发 ValueError 异常
    if status.unexpected_keys:
        raise ValueError(f"Unexpected keys: {status.unexpected_keys}")

# 将 ESM 模型检查点转换为 PyTorch 的格式
def convert_esm_checkpoint_to_pytorch(
    model: str, pytorch_dump_folder_path: str, classification_head: bool, push_to_repo: str, auth_token: str
):
    """
    复制/粘贴/调整 esm 的权重到我们的 BERT 结构中。
    """
    # 如果模型以 "esmfold" 开头，则创建相应的 ESM 模型实例
    if model.startswith("esmfold"):
        esm = MODEL_MAPPING[model]()
    else:
        esm, alphabet = MODEL_MAPPING[model]()
    
    # 将模型设为评估模式，禁用 dropout
    esm.eval()

    # 根据模型类型设置各种参数和配置
    if model.startswith("esmfold"):
        embed_dim = esm.esm.embed_dim
        num_layers = esm.esm.num_layers
        num_attention_heads = esm.esm.attention_heads
        intermediate_size = 4 * embed_dim
        token_dropout = esm.esm.token_dropout
        emb_layer_norm_before = False  # 这条代码路径在 ESM-2 中不存在
        position_embedding_type = "rotary"
        is_folding_model = True
        esmfold_config = EsmFoldConfig()
        # 更新 ESMFoldConfig 对象的配置项
        for key, val in esm.cfg.items():
            if hasattr(esmfold_config, key) and key != "trunk":
                setattr(esmfold_config, key, val)
        for key, val in esm.cfg.trunk.items():
            if hasattr(esmfold_config.trunk, key) and key != "structure_module":
                setattr(esmfold_config.trunk, key, val)
        for key, val in esm.cfg.trunk.structure_module.items():
            if hasattr(esmfold_config.trunk.structure_module, key):
                setattr(esmfold_config.trunk.structure_module, key, val)
    elif hasattr(esm, "args"):
        # 表明是 ESM-1b 或 ESM-1v 模型
        embed_dim = esm.args.embed_dim
        num_layers = esm.args.layers
        num_attention_heads = esm.args.attention_heads
        intermediate_size = esm.args.ffn_embed_dim
        token_dropout = esm.args.token_dropout
        emb_layer_norm_before = True if esm.emb_layer_norm_before else False
        position_embedding_type = "absolute"
        is_folding_model = False
        esmfold_config = None
    else:
        # 表示这是一个 ESM-2 模型
        embed_dim = esm.embed_dim
        num_layers = esm.num_layers
        num_attention_heads = esm.attention_heads
        intermediate_size = 4 * embed_dim  # 这个值在 ESM-2 中是硬编码的
        token_dropout = esm.token_dropout
        emb_layer_norm_before = False  # 这个代码路径在 ESM-2 中不存在
        position_embedding_type = "rotary"
        is_folding_model = False
        esmfold_config = None

    if is_folding_model:
        alphabet = esm.esm.alphabet
    vocab_list = tuple(alphabet.all_toks)
    mask_token_id = alphabet.mask_idx
    pad_token_id = alphabet.padding_idx

    if is_folding_model:
        original_esm_model = esm.esm
    else:
        original_esm_model = esm

    config = EsmConfig(
        vocab_size=original_esm_model.embed_tokens.num_embeddings,
        mask_token_id=mask_token_id,
        hidden_size=embed_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=1026,
        layer_norm_eps=1e-5,  # 在 fairseq 中使用的 PyTorch 默认值
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        pad_token_id=pad_token_id,
        emb_layer_norm_before=emb_layer_norm_before,
        token_dropout=token_dropout,
        position_embedding_type=position_embedding_type,
        is_folding_model=is_folding_model,
        esmfold_config=esmfold_config,
        vocab_list=vocab_list,
    )
    if classification_head:
        config.num_labels = esm.classification_heads["mnli"].out_proj.weight.shape[0]
    print("Our ESM config:", config)

    if model.startswith("esmfold"):
        model_class = EsmForProteinFolding
    elif classification_head:
        model_class = EsmForSequenceClassification
    else:
        model_class = EsmForMaskedLM
    model = model_class(config)
    model.eval()

    # 现在我们来复制所有的权重。
    # Embeddings
    model.esm.embeddings.word_embeddings.weight = original_esm_model.embed_tokens.weight
    if position_embedding_type == "absolute":
        model.esm.embeddings.position_embeddings.weight = original_esm_model.embed_positions.weight

    if config.emb_layer_norm_before:
        model.esm.embeddings.layer_norm.weight = original_esm_model.emb_layer_norm_before.weight
        model.esm.embeddings.layer_norm.bias = original_esm_model.emb_layer_norm_before.bias

    model.esm.encoder.emb_layer_norm_after.weight = original_esm_model.emb_layer_norm_after.weight
    model.esm.encoder.emb_layer_norm_after.bias = original_esm_model.emb_layer_norm_after.bias
    # 如果是折叠模型（folding model），则执行以下操作
    if is_folding_model:
        # 将 ESM 模型的 esm_s_combine 数据传输到 model 的 esm_s_combine 中
        model.esm_s_combine.data = esm.esm_s_combine.data
        # 将 ESM 模型的 af2_to_esm 数据传输到 model 的 af2_to_esm 中
        model.af2_to_esm.data = esm.af2_to_esm.data
        # 将 ESM 模型的 embedding 数据传输到 model 的 embedding 中，并检查权重
        transfer_and_check_weights(esm.embedding, model.embedding)
        # 将 ESM 模型的 esm_s_mlp 数据传输到 model 的 esm_s_mlp 中，并检查权重
        transfer_and_check_weights(esm.esm_s_mlp, model.esm_s_mlp)
        # 将 ESM 模型的 trunk 数据传输到 model 的 trunk 中，并检查权重
        transfer_and_check_weights(esm.trunk, model.trunk)
        # 将 ESM 模型的 distogram_head 数据传输到 model 的 distogram_head 中，并检查权重
        transfer_and_check_weights(esm.distogram_head, model.distogram_head)
        # 将 ESM 模型的 ptm_head 数据传输到 model 的 ptm_head 中，并检查权重
        transfer_and_check_weights(esm.ptm_head, model.ptm_head)
        # 将 ESM 模型的 lm_head 数据传输到 model 的 lm_head 中，并检查权重
        transfer_and_check_weights(esm.lm_head, model.lm_head)
        # 将 ESM 模型的 lddt_head 数据传输到 model 的 lddt_head 中，并检查权重
        transfer_and_check_weights(esm.lddt_head, model.lddt_head)

    # 否则，如果是分类头（classification head），执行以下操作
    elif classification_head:
        # 将 ESM 模型的 "mnli" 分类头的权重传输到 model 的 classifier.dense.weight 中
        model.classifier.dense.weight = esm.esm.classification_heads["mnli"].dense.weight
        # 将 ESM 模型的 "mnli" 分类头的偏置传输到 model 的 classifier.dense.bias 中
        model.classifier.dense.bias = esm.classification_heads["mnli"].dense.bias
        # 将 ESM 模型的 "mnli" 分类头的输出投影权重传输到 model 的 classifier.out_proj.weight 中
        model.classifier.out_proj.weight = esm.classification_heads["mnli"].out_proj.weight
        # 将 ESM 模型的 "mnli" 分类头的输出投影偏置传输到 model 的 classifier.out_proj.bias 中
        model.classifier.out_proj.bias = esm.classification_heads["mnli"].out_proj.bias

    # 否则，执行以下操作（通常是语言模型头）
    else:
        # 将 ESM 模型的 lm_head 的 dense.weight 数据传输到 model 的 lm_head.dense.weight 中
        model.lm_head.dense.weight = esm.lm_head.dense.weight
        # 将 ESM 模型的 lm_head 的 dense.bias 数据传输到 model 的 lm_head.dense.bias 中
        model.lm_head.dense.bias = esm.lm_head.dense.bias
        # 将 ESM 模型的 lm_head 的 layer_norm.weight 数据传输到 model 的 lm_head.layer_norm.weight 中
        model.lm_head.layer_norm.weight = esm.lm_head.layer_norm.weight
        # 将 ESM 模型的 lm_head 的 layer_norm.bias 数据传输到 model 的 lm_head.layer_norm.bias 中
        model.lm_head.layer_norm.bias = esm.lm_head.layer_norm.bias
        # 将 ESM 模型的 lm_head 的 weight 数据传输到 model 的 lm_head.decoder.weight 中
        model.lm_head.decoder.weight = esm.lm_head.weight
        # 将 ESM 模型的 lm_head 的 bias 数据传输到 model 的 lm_head.bias 中
        model.lm_head.bias = esm.lm_head.bias

    # 将 ESM 模型的 contact_head 数据传输到 model 的 esm.contact_head 中，并检查权重
    transfer_and_check_weights(esm.contact_head, model.esm.contact_head)

    # 准备数据（来自 ESMStructuralSplitDataset 超家族的前两个序列 / 4）
    if is_folding_model:
        # 对于折叠模型，采样前两个数据样本，因为折叠模型不会使用掩码输入且不喜欢掩码令牌
        sample_data = SAMPLE_DATA[:2]
    else:
        # 对于其他模型，采样全部数据样本
        sample_data = SAMPLE_DATA

    if is_folding_model:
        # 获取 ESMFold 的 tokenizer
        hf_tokenizer = get_esmfold_tokenizer()
        # 使用 ESMFold tokenizer 处理样本数据，返回 PyTorch 张量格式，进行填充，并不添加特殊令牌
        hf_tokens = hf_tokenizer(
            [row[1] for row in sample_data], return_tensors="pt", padding=True, add_special_tokens=False
        )
        # 使用 ESMFold 编码函数处理样本数据，获取氨基酸序列、掩码等信息
        esmfold_aas, esmfold_mask, _, _, _ = esmfold_encode_sequences([row[1] for row in sample_data])
        # 检查是否成功匹配 ESMFold tokenizer 输出的 input_ids 和 attention_mask 与 hf_tokens 中的对应值
        success = torch.all(hf_tokens["input_ids"] == esmfold_aas) and torch.all(
            hf_tokens["attention_mask"] == esmfold_mask
        )
    else:
        # 否则，检查两个模型的 tokenizer 是否输出相同的 tokens
        batch_converter = alphabet.get_batch_converter()
        # 使用 batch_converter 处理样本数据，返回批次标签、字符串和 tokens
        batch_labels, batch_strs, batch_tokens = batch_converter(sample_data)
        # 准备 tokenizer，并确保其与 batch_tokens 匹配
        with TemporaryDirectory() as tempdir:
            # 创建临时目录，写入 alphabet 的全部 tokens 作为 vocab
            vocab = "\n".join(alphabet.all_toks)
            vocab_file = Path(tempdir) / "vocab.txt"
            vocab_file.write_text(vocab)
            # 使用 EsmTokenizer 初始化 hf_tokenizer
            hf_tokenizer = EsmTokenizer(vocab_file=str(vocab_file))

        # 使用 hf_tokenizer 处理样本数据，返回 PyTorch 张量格式，进行填充
        hf_tokens = hf_tokenizer([row[1] for row in sample_data], return_tensors="pt", padding=True)
        # 检查是否成功匹配 hf_tokens 的 input_ids 与 batch_tokens 中的对应值
        success = torch.all(hf_tokens["input_ids"] == batch_tokens)

    # 打印是否两个模型的 tokenizer 输出相同的 tokens，如果相同则输出 "🔥"，否则输出 "💩"
    print("Do both models tokenizers output the same tokens?", "🔥" if success else "💩")
    # 如果成功标志为假，则引发异常并显示消息
    if not success:
        raise Exception("Tokenization does not match!")

    # 禁用梯度计算，因为这是推断阶段
    with torch.no_grad():
        # 如果是折叠模型
        if is_folding_model:
            # 分阶段测试模型
            # ESMFold 总是将 ESM stem 转换为 float16，需要在 GPU 上执行 float16 操作
            # 这在 CPU 上不支持。因此，我们需要在 GPU 上运行它。然而，
            # ESMFold 是社区中所谓的“大型模型”，因此我们强烈避免同时将原始模型和转换后的模型放在 GPU 上。
            their_output = esm.cuda().infer([row[1] for row in sample_data])
            # 使用模型在 GPU 上运行推理
            our_output = model.cuda()(
                input_ids=hf_tokens["input_ids"].cuda(), attention_mask=hf_tokens["attention_mask"].cuda()
            )
        else:
            # 在模型上运行输入以生成输出隐藏状态
            our_output = model(**hf_tokens, output_hidden_states=True)
            # 从输出中提取逻辑回归层结果
            our_output = our_output["logits"]
            if classification_head:
                # 如果是分类头，使用 ESM 模型的多功能自然语言推理分类
                their_output = esm.model.classification_heads["mnli"](esm.extract_features(batch_tokens))
            else:
                # 使用 ESM 模型对输入进行推理并返回逻辑回归结果
                their_output = esm(hf_tokens["input_ids"], repr_layers=list(range(999)))
                their_output = their_output["logits"]

        # 如果是折叠模型，则计算位置差的最大绝对值，并检查输出是否全部接近
        if is_folding_model:
            max_absolute_diff = torch.max(torch.abs(our_output["positions"] - their_output["positions"])).item()
            success = torch.allclose(our_output["positions"], their_output["positions"], atol=1e-5)
        else:
            # 否则计算输出差的最大绝对值，并检查输出是否全部接近
            max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
            success = torch.allclose(our_output, their_output, atol=1e-5)

        # 打印最大绝对差异的值
        print(f"max_absolute_diff = {max_absolute_diff}")  # 大约为 1e-5
        # 打印模型是否输出相同的张量
        print("Do both models output the same tensors?", "🔥" if success else "💩")

        # 如果没有成功匹配输出，则引发异常
        if not success:
            raise Exception("Something went wRoNg")

        # 如果不是折叠模型，进行接触预测测试
        if not is_folding_model:
            # 使用模型预测接触点
            our_output = model.predict_contacts(hf_tokens["input_ids"], hf_tokens["attention_mask"])
            # 使用 ESM 模型预测接触点
            their_output = esm.predict_contacts(hf_tokens["input_ids"])
            # 计算接触预测的最大绝对值差异，并检查输出是否全部接近
            max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
            success = torch.allclose(our_output, their_output, atol=1e-5)

            # 打印接触预测测试结果
            print("Contact prediction testing:")
            print(f"max_absolute_diff = {max_absolute_diff}")  # 大约为 1e-5
            print("Do both models output the same tensors?", "🔥" if success else "💩")

            # 如果没有成功匹配输出，则引发异常
            if not success:
                raise Exception("Something went wRoNg")

        # 创建目录以保存 PyTorch 模型
        pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
        # 打印正在保存模型的消息
        print(f"Saving model to {pytorch_dump_folder_path}")
        # 保存模型的预训练参数到指定路径
        model.save_pretrained(pytorch_dump_folder_path)

        # 在继续之前释放部分内存
        del esm

    # 打印正在保存分词器的消息
    print(f"Saving tokenizer to {pytorch_dump_folder_path}")
    # 保存分词器的预训练参数到指定路径
    hf_tokenizer.save_pretrained(pytorch_dump_folder_path)
    # 如果 push_to_repo 为真，则执行下面的操作
    if push_to_repo:
        # 调用 model 对象的 push_to_hub 方法，将模型推送到指定的仓库
        model.push_to_hub(repo_id=push_to_repo, token_token=auth_token)
        # 调用 hf_tokenizer 对象的 push_to_hub 方法，将 tokenizer 推送到指定的仓库
        hf_tokenizer.push_to_hub(repo_id=push_to_repo, token_token=auth_token)
if __name__ == "__main__":
    # 如果脚本作为主程序执行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必填参数
    parser.add_argument(
        "--pytorch_dump_folder_path", type=str, required=True, help="Path to the output PyTorch model."
    )
    # 参数：pytorch_dump_folder_path，类型为字符串，必填，用于指定输出 PyTorch 模型的路径

    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    # 参数：classification_head，如果存在则设置为 True，用于指定是否转换最终分类头部

    parser.add_argument("--model", default=None, type=str, required=True, help="Name of model to convert.")
    # 参数：model，类型为字符串，默认值为 None，必填，用于指定要转换的模型的名称

    parser.add_argument("--push_to_repo", type=str, help="Repo to upload to (including username!).")
    # 参数：push_to_repo，类型为字符串，用于指定要上传的仓库（包括用户名）

    parser.add_argument("--auth_token", type=str, help="HuggingFace auth token.")
    # 参数：auth_token，类型为字符串，用于指定 HuggingFace 的认证令牌

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 对象中

    convert_esm_checkpoint_to_pytorch(
        args.model, args.pytorch_dump_folder_path, args.classification_head, args.push_to_repo, args.auth_token
    )
    # 调用函数 convert_esm_checkpoint_to_pytorch，传入解析后的参数来执行模型转换操作
```