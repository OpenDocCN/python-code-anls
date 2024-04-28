# `.\transformers\models\__init__.py`

```py
# 导入各种模型
from . import (
    albert,  # 导入 ALBERT 模型
    align,  # 导入 Align 模型
    altclip,  # 导入 AltClip 模型
    audio_spectrogram_transformer,  # 导入音频频谱变换器模型
    auto,  # 导入 Auto 模型
    autoformer,  # 导入 AutoFormer 模型
    bark,  # 导入 Bark 模型
    bart,  # 导入 BART 模型
    barthez,  # 导入 Barthez 模型
    bartpho,  # 导入 Bartpho 模型
    beit,  # 导入 BEiT 模型
    bert,  # 导入 BERT 模型
    bert_generation,  # 导入 BERT Generation 模型
    bert_japanese,  # 导入 BERT Japanese 模型
    bertweet,  # 导入 BERTweet 模型
    big_bird,  # 导入 BigBird 模型
    bigbird_pegasus,  # 导入 BigBird Pegasus 模型
    biogpt,  # 导入 BioGPT 模型
    bit,  # 导入 BiT 模型
    blenderbot,  # 导入 BlenderBot 模型
    blenderbot_small,  # 导入 BlenderBot Small 模型
    blip,  # 导入 BLIP 模型
    blip_2,  # 导入 BLIP 2 模型
    bloom,  # 导入 Bloom 模型
    bridgetower,  # 导入 BridgeTower 模型
    bros,  # 导入 Bros 模型
    byt5,  # 导入 ByT5 模型
    camembert,  # 导入 Camembert 模型
    canine,  # 导入 Canine 模型
    chinese_clip,  # 导入 Chinese CLIP 模型
    clap,  # 导入 CLAP 模型
    clip,  # 导入 CLIP 模型
    clipseg,  # 导入 ClipSeg 模型
    clvp,  # 导入 CLVP 模型
    code_llama,  # 导入 Code Llama 模型
    codegen,  # 导入 CodeGen 模型
    conditional_detr,  # 导入 Conditional DETR 模型
    convbert,  # 导入 ConvBERT 模型
    convnext,  # 导入 ConvNeXT 模型
    convnextv2,  # 导入 ConvNeXTv2 模型
    cpm,  # 导入 CPM 模型
    cpmant,  # 导入 CPMAnt 模型
    ctrl,  # 导入 CTRL 模型
    cvt,  # 导入 CVT 模型
    data2vec,  # 导入 Data2Vec 模型
    deberta,  # 导入 DeBERTa 模型
    deberta_v2,  # 导入 DeBERTa-v2 模型
    decision_transformer,  # 导入 Decision Transformer 模型
    deformable_detr,  # 导入 Deformable DETR 模型
    deit,  # 导入 DeiT 模型
    deprecated,  # 导入 Deprecated 模型
    deta,  # 导入 DeTA 模型
    detr,  # 导入 DETR 模型
    dialogpt,  # 导入 DialoGPT 模型
    dinat,  # 导入 DiNAT 模型
    dinov2,  # 导入 DINOv2 模型
    distilbert,  # 导入 DistilBERT 模型
    dit,  # 导入 DIT 模型
    donut,  # 导入 Donut 模型
    dpr,  # 导入 DPR 模型
    dpt,  # 导入 DPT 模型
    efficientformer,  # 导入 EfficientFormer 模型
    efficientnet,  # 导入 EfficientNet 模型
    electra,  # 导入 Electra 模型
    encodec,  # 导入 Encodec 模型
    encoder_decoder,  # 导入 Encoder-Decoder 模型
    ernie,  # 导入 ERNIE 模型
    ernie_m,  # 导入 ERNIE-M 模型
    esm,  # 导入 ESM 模型
    falcon,  # 导入 Falcon 模型
    fastspeech2_conformer,  # 导入 FastSpeech2 Conformer 模型
    flaubert,  # 导入 FlauBERT 模型
    flava,  # 导入 Flava 模型
    fnet,  # 导入 FNet 模型
    focalnet,  # 导入 FocalNet 模型
    fsmt,  # 导入 FSMT 模型
    funnel,  # 导入 Funnel 模型
    fuyu,  # 导入 Fuyu 模型
    git,  # 导入 Git 模型
    glpn,  # 导入 GLPN 模型
    gpt2,  # 导入 GPT-2 模型
    gpt_bigcode,  # 导入 GPT BigCode 模型
    gpt_neo,  # 导入 GPT-Neo 模型
    gpt_neox,  # 导入 GPT-NeoX 模型
    gpt_neox_japanese,  # 导入 GPT-NeoX Japanese 模型
    gpt_sw3,  # 导入 GPT-SW3 模型
    gptj,  # 导入 GPTJ 模型
    gptsan_japanese,  # 导入 GPTSan Japanese 模型
    graphormer,  # 导入 Graphormer 模型
    groupvit,  # 导入 GroupViT 模型
    herbert,  # 导入 Herbert 模型
    hubert,  # 导入 Hubert 模型
    ibert,  # 导入 IBert 模型
    idefics,  # 导入 iDEFICS 模型
    imagegpt, 
    seamless_m4t, 
    # 定义变量 seamless_m4t

    seamless_m4t_v2, 
    # 定义变量 seamless_m4t_v2

    segformer, 
    # 定义变量 segformer

    sew, 
    # 定义变量 sew

    sew_d, 
    # 定义变量 sew_d

    siglip, 
    # 定义变量 siglip

    speech_encoder_decoder, 
    # 定义变量 speech_encoder_decoder

    speech_to_text, 
    # 定义变量 speech_to_text

    speech_to_text_2, 
    # 定义变量 speech_to_text_2

    speecht5, 
    # 定义变量 speecht5

    splinter, 
    # 定义变量 splinter

    squeezebert, 
    # 定义变量 squeezebert

    swiftformer, 
    # 定义变量 swiftformer

    swin, 
    # 定义变量 swin

    swin2sr, 
    # 定义变量 swin2sr

    swinv2, 
    # 定义变量 swinv2

    switch_transformers, 
    # 定义变量 switch_transformers

    t5, 
    # 定义变量 t5

    table_transformer, 
    # 定义变量 table_transformer

    tapas, 
    # 定义变量 tapas

    time_series_transformer, 
    # 定义变量 time_series_transformer

    timesformer, 
    # 定义变量 timesformer

    timm_backbone, 
    # 定义变量 timm_backbone

    trocr, 
    # 定义变量 trocr

    tvlt, 
    # 定义变量 tvlt

    tvp, 
    # 定义变量 tvp

    umt5, 
    # 定义变量 umt5

    unispeech, 
    # 定义变量 unispeech

    unispeech_sat, 
    # 定义变量 unispeech_sat

    univnet, 
    # 定义变量 univnet

    upernet, 
    # 定义变量 upernet

    videomae, 
    # 定义变量 videomae

    vilt, 
    # 定义变量 vilt

    vipllava, 
    # 定义变量 vipllava

    vision_encoder_decoder, 
    # 定义变量 vision_encoder_decoder

    vision_text_dual_encoder, 
    # 定义变量 vision_text_dual_encoder

    visual_bert, 
    # 定义变量 visual_bert

    vit, 
    # 定义变量 vit

    vit_hybrid, 
    # 定义变量 vit_hybrid

    vit_mae, 
    # 定义变量 vit_mae

    vit_msn, 
    # 定义变量 vit_msn

    vitdet, 
    # 定义变量 vitdet

    vitmatte, 
    # 定义变量 vitmatte

    vits, 
    # 定义变量 vits

    vivit, 
    # 定义变量 vivit

    wav2vec2, 
    # 定义变量 wav2vec2

    wav2vec2_bert, 
    # 定义变量 wav2vec2_bert

    wav2vec2_conformer, 
    # 定义变量 wav2vec2_conformer

    wav2vec2_phoneme, 
    # 定义变量 wav2vec2_phoneme

    wav2vec2_with_lm, 
    # 定义变量 wav2vec2_with_lm

    wavlm, 
    # 定义变量 wavlm

    whisper, 
    # 定义变量 whisper

    x_clip, 
    # 定义变量 x_clip

    xglm, 
    # 定义变量 xglm

    xlm, 
    # 定义变量 xlm

    xlm_prophetnet, 
    # 定义变�� xlm_prophetnet

    xlm_roberta, 
    # 定义变量 xlm_roberta

    xlm_roberta_xl, 
    # 定义变量 xlm_roberta_xl

    xlnet, 
    # 定义变量 xlnet

    xmod, 
    # 定义变量 xmod

    yolos, 
    # 定义变量 yolos

    yoso,
    # 定义变量 yoso
# 该行代码为一个空行，没有实际作用，可以忽略
```