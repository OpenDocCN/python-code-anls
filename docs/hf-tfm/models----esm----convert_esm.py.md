# `.\models\esm\convert_esm.py`

```
# coding=utf-8
# 版权声明
# 使用 Apache License, Version 2.0 进行许可
# 如果要使用这个文件，必须符合许可协议
# 可以通过 http://www.apache.org/licenses/LICENSE-2.0 获取许可协议的副本
#
# 除非适用的法律要求或书面同意，本软件是基于“按原样”提供的，
# 不提供任何明示或暗示的保证或条件
# 参见许可协议获取更多详细信息
"""转换 ESM 检查点。"""


import argparse
import pathlib
from pathlib import Path
from tempfile import TemporaryDirectory

import esm as esm_module
import torch
from esm.esmfold.v1.misc import batch_encode_sequences as esmfold_encode_sequences
from esm.esmfold.v1.pretrained import esmfold_v1

from transformers.models.esm.configuration_esm import EsmConfig, EsmFoldConfig
from transformers.models.esm.modeling_esm import (
    EsmForMaskedLM,
    EsmForSequenceClassification,
    EsmIntermediate,
    EsmLayer,
    EsmOutput,
    EsmSelfAttention,
    EsmSelfOutput,
)
from transformers.models.esm.modeling_esmfold import EsmForProteinFolding
from transformers.models.esm.tokenization_esm import EsmTokenizer
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_DATA = [
    (
        "protein1",
        "MNGTEGPNFYVPFSNATGVVRSPFEYPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVLGGFTSTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLAGWSRYIPEGLQCSCGIDYYTLKPEVNNESFVIYMFVVHFTIPMIIIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIAFLICWVPYASVAFYIFTHQGSNFGPIFMTIPAFFAKSAAIYNPVIYIMMNKQFRNCMLTTICCGKNPLGDDEASATVSKTETSQVAPA",
    ),
    ("protein2", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA"),
    ("protein3", "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLAGG"),
    ("protein4", "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLA"),
]

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
    # 将预训练模型“esm2_t6_8M_UR50D”添加到字典中，键为"esm2_t6_8M_UR50D"，值为该预训练模型
    "esm2_t6_8M_UR50D": esm_module.pretrained.esm2_t6_8M_UR50D,
    # 将预训练模型“esmfold_v1”添加到字典中，键为"esmfold_v1"，值为该预训练模型
    "esmfold_v1": esmfold_v1,
```  
}

# 定义氨基酸残基类型的列表
restypes = list("ARNDCQEGHILKMFPSTWYV")

# 扩展包含'X'的氨基酸残基类型列表
restypes_with_x = restypes + ["X"]

# 添加特殊标记后的氨基酸残基类型列表
restypes_with_extras = restypes_with_x + ["
    else:
        # 表示为 ESM-2 模型
        embed_dim = esm.embed_dim
        num_layers = esm.num_layers
        num_attention_heads = esm.attention_heads
        intermediate_size = 4 * embed_dim  # 这在 ESM-2 中是硬编码的
        token_dropout = esm.token_dropout
        emb_layer_norm_before = False  # 在 ESM-2 中不存在此代码路径
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
        layer_norm_eps=1e-5,  # 在 fairseq 中使用 PyTorch 默认值
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

    # 现在让我们复制所有权重。
    # Embeddings
    model.esm.embeddings.word_embeddings.weight = original_esm_model.embed_tokens.weight
    if position_embedding_type == "absolute":
        model.esm.embeddings.position_embeddings.weight = original_esm_model.embed_positions.weight

    if config.emb_layer_norm_before:
        model.esm.embeddings.layer_norm.weight = original_esm_model.emb_layer_norm_before.weight
        model.esm.embeddings.layer_norm.bias = original_esm_model.emb_layer_norm_before.bias

    model.esm.encoder.emb_layer_norm_after.weight = original_esm_model.emb_layer_norm_after.weight
    model.esm.encoder.emb_layer_norm_after.bias = original_esm_model.emb_layer_norm_after.bias
    # 如果是折叠模型
    if is_folding_model:
        # 将ESM模型的数据赋值给模型的属性
        model.esm_s_combine.data = esm.esm_s_combine.data
        model.af2_to_esm.data = esm.af2_to_esm.data
        # 迁移并检查权重
        transfer_and_check_weights(esm.embedding, model.embedding)
        transfer_and_check_weights(esm.esm_s_mlp, model.esm_s_mlp)
        transfer_and_check_weights(esm.trunk, model.trunk)
        transfer_and_check_weights(esm.distogram_head, model.distogram_head)
        transfer_and_check_weights(esm.ptm_head, model.ptm_head)
        transfer_and_check_weights(esm.lm_head, model.lm_head)
        transfer_and_check_weights(esm.lddt_head, model.lddt_head)

    # 如果是分类头
    elif classification_head:
        # 将ESM模型的分类头数据赋值给模型的属性
        model.classifier.dense.weight = esm.esm.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = esm.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = esm.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = esm.classification_heads["mnli"].out_proj.bias
    else:
        # LM头
        model.lm_head.dense.weight = esm.lm_head.dense.weight
        model.lm_head.dense.bias = esm.lm_head.dense.bias
        model.lm_head.layer_norm.weight = esm.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = esm.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = esm.lm_head.weight
        model.lm_head.bias = esm.lm_head.bias

    # 联络预测头
    transfer_and_check_weights(esm.contact_head, model.esm.contact_head)

    # 准备数据（从ESMStructuralSplitDataset超级家族中获取前2个序列/ 4）
    if is_folding_model:
        # 折叠模型不是在屏蔽输入上进行训练，也不喜欢掩码标记。
        sample_data = SAMPLE_DATA[:2]
    else:
        sample_data = SAMPLE_DATA

    if is_folding_model:
        # 获取ESM折叠的标记器和标记，并进行比较
        hf_tokenizer = get_esmfold_tokenizer()
        hf_tokens = hf_tokenizer(
            [row[1] for row in sample_data], return_tensors="pt", padding=True, add_special_tokens=False
        )
        esmfold_aas, esmfold_mask, _, _, _ = esmfold_encode_sequences([row[1] for row in sample_data])
        success = torch.all(hf_tokens["input_ids"] == esmfold_aas) and torch.all(
            hf_tokens["attention_mask"] == esmfold_mask
        )
    else:
        # 检查是否获得相同的结果
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(sample_data)
        # 准备标记器并确保其匹配
        with TemporaryDirectory() as tempdir:
            vocab = "\n".join(alphabet.all_toks)
            vocab_file = Path(tempdir) / "vocab.txt"
            vocab_file.write_text(vocab)
            hf_tokenizer = EsmTokenizer(vocab_file=str(vocab_file))

        hf_tokens = hf_tokenizer([row[1] for row in sample_data], return_tensors="pt", padding=True)
        success = torch.all(hf_tokens["input_ids"] == batch_tokens)

    # 输出两个模型标记器输出相同的标记
    print("Do both models tokenizers output the same tokens?", "🔥" if success else "💩")
    if not success:
        # 如果条件不满足，则抛出异常，指示令牌化不匹配
        raise Exception("Tokenization does not match!")

    with torch.no_grad():
        if is_folding_model:
            # 测试模型的各个部分
            # ESMFold 总是会将 ESM stem 转换为 float16，这需要 float16 运算，而在 CPU 上不支持。
            # 因此，为了测试它，我们需要在 GPU 上运行它。然而，大型模型（"big boy"）会导致原始模型和转换后的模型在同一时间放在 GPU 上，因此要尽量避免这种情况。
            their_output = esm.cuda().infer([row[1] for row in sample_data])
            our_output = model.cuda()(
                input_ids=hf_tokens["input_ids"].cuda(), attention_mask=hf_tokens["attention_mask"].cuda()
            )
        else:
            our_output = model(**hf_tokens, output_hidden_states=True)
            our_output = our_output["logits"]
            if classification_head:
                their_output = esm.model.classification_heads["mnli"](esm.extract_features(batch_tokens))
            else:
                their_output = esm(hf_tokens["input_ids"], repr_layers=list(range(999)))
                their_output = their_output["logits"]

        if is_folding_model:
            max_absolute_diff = torch.max(torch.abs(our_output["positions"] - their_output["positions"])).item()
            success = torch.allclose(our_output["positions"], their_output["positions"], atol=1e-5)
        else:
            max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
            success = torch.allclose(our_output, their_output, atol=1e-5)

        print(f"max_absolute_diff = {max_absolute_diff}")  # 最大绝对差 ~1e-5
        print("Do both models output the same tensors?", "🔥" if success else "💩")

        if not success:
            raise Exception("Something went wRoNg")

        if not is_folding_model:
            # 让我们也检查联系预测
            our_output = model.predict_contacts(hf_tokens["input_ids"], hf_tokens["attention_mask"])
            their_output = esm.predict_contacts(hf_tokens["input_ids"])
            max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
            success = torch.allclose(our_output, their_output, atol=1e-5)

            print("Contact prediction testing:")
            print(f"max_absolute_diff = {max_absolute_diff}")  # 最大绝对差 ~1e-5
            print("Do both models output the same tensors?", "🔥" if success else "💩")

            if not success:
                raise Exception("Something went wRoNg")

        pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
        print(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

        del esm  # 在继续之前释放一些内存

    print(f"Saving tokenizer to {pytorch_dump_folder_path}")
    hf_tokenizer.save_pretrained(pytorch_dump_folder_path)
    # 如果 push_to_repo 为真，则将模型推送到仓库
    if push_to_repo:
        # 调用 model 对象的 push_to_hub 方法，将模型推送到指定的仓库，使用指定的认证令牌
        model.push_to_hub(repo_id=push_to_repo, token_token=auth_token)
        # 调用 hf_tokenizer 对象的 push_to_hub 方法，将分词器推送到指定的仓库，使用指定的认证令牌
        hf_tokenizer.push_to_hub(repo_id=push_to_repo, token_token=auth_token)
# 如果该脚本被直接运行
if __name__ == "__main__":
    # 创建一个参数解析对象
    parser = argparse.ArgumentParser()
    # 添加必要参数
    parser.add_argument(
        "--pytorch_dump_folder_path", type=str, required=True, help="Path to the output PyTorch model."
    )
    # 是否转换最终的分类头部
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    # 要转换的模型的名称
    parser.add_argument("--model", default=None, type=str, required=True, help="Name of model to convert.")
    # 要上传到的仓库（包括用户名）
    parser.add_argument("--push_to_repo", type=str, help="Repo to upload to (including username!).")
    # HuggingFace 认证令牌
    parser.add_argument("--auth_token", type=str, help="HuggingFace auth token.")
    # 解析参数
    args = parser.parse_args()
    # 转换 ESM 模型检查点为 PyTorch 模型
    convert_esm_checkpoint_to_pytorch(
        args.model, args.pytorch_dump_folder_path, args.classification_head, args.push_to_repo, args.auth_token
    )
```