# `.\models\__init__.py`

```py
# 导入模块和库

from . import (
    albert,  # 导入albert模块
    align,  # 导入align模块
    altclip,  # 导入altclip模块
    audio_spectrogram_transformer,  # 导入audio_spectrogram_transformer模块
    auto,  # 导入auto模块
    autoformer,  # 导入autoformer模块
    bark,  # 导入bark模块
    bart,  # 导入bart模块
    barthez,  # 导入barthez模块
    bartpho,  # 导入bartpho模块
    beit,  # 导入beit模块
    bert,  # 导入bert模块
    bert_generation,  # 导入bert_generation模块
    bert_japanese,  # 导入bert_japanese模块
    bertweet,  # 导入bertweet模块
    big_bird,  # 导入big_bird模块
    bigbird_pegasus,  # 导入bigbird_pegasus模块
    biogpt,  # 导入biogpt模块
    bit,  # 导入bit模块
    blenderbot,  # 导入blenderbot模块
    blenderbot_small,  # 导入blenderbot_small模块
    blip,  # 导入blip模块
    blip_2,  # 导入blip_2模块
    bloom,  # 导入bloom模块
    bridgetower,  # 导入bridgetower模块
    bros,  # 导入bros模块
    byt5,  # 导入byt5模块
    camembert,  # 导入camembert模块
    canine,  # 导入canine模块
    chinese_clip,  # 导入chinese_clip模块
    clap,  # 导入clap模块
    clip,  # 导入clip模块
    clipseg,  # 导入clipseg模块
    clvp,  # 导入clvp模块
    code_llama,  # 导入code_llama模块
    codegen,  # 导入codegen模块
    cohere,  # 导入cohere模块
    conditional_detr,  # 导入conditional_detr模块
    convbert,  # 导入convbert模块
    convnext,  # 导入convnext模块
    convnextv2,  # 导入convnextv2模块
    cpm,  # 导入cpm模块
    cpmant,  # 导入cpmant模块
    ctrl,  # 导入ctrl模块
    cvt,  # 导入cvt模块
    data2vec,  # 导入data2vec模块
    deberta,  # 导入deberta模块
    deberta_v2,  # 导入deberta_v2模块
    decision_transformer,  # 导入decision_transformer模块
    deformable_detr,  # 导入deformable_detr模块
    deit,  # 导入deit模块
    deprecated,  # 导入deprecated模块
    depth_anything,  # 导入depth_anything模块
    deta,  # 导入deta模块
    detr,  # 导入detr模块
    dialogpt,  # 导入dialogpt模块
    dinat,  # 导入dinat模块
    dinov2,  # 导入dinov2模块
    distilbert,  # 导入distilbert模块
    dit,  # 导入dit模块
    donut,  # 导入donut模块
    dpr,  # 导入dpr模块
    dpt,  # 导入dpt模块
    efficientformer,  # 导入efficientformer模块
    efficientnet,  # 导入efficientnet模块
    electra,  # 导入electra模块
    encodec,  # 导入encodec模块
    encoder_decoder,  # 导入encoder_decoder模块
    ernie,  # 导入ernie模块
    ernie_m,  # 导入ernie_m模块
    esm,  # 导入esm模块
    falcon,  # 导入falcon模块
    fastspeech2_conformer,  # 导入fastspeech2_conformer模块
    flaubert,  # 导入flaubert模块
    flava,  # 导入flava模块
    fnet,  # 导入fnet模块
    focalnet,  # 导入focalnet模块
    fsmt,  # 导入fsmt模块
    funnel,  # 导入funnel模块
    fuyu,  # 导入fuyu模块
    gemma,  # 导入gemma模块
    git,  # 导入git模块
    glpn,  # 导入glpn模块
    gpt2,  # 导入gpt2模块
    gpt_bigcode,  # 导入gpt_bigcode模块
    gpt_neo,  # 导入gpt_neo模块
    gpt_neox,  # 导入gpt_neox模块
    gpt_neox_japanese,  # 导入gpt_neox_japanese模块
    gpt_sw3,  # 导入gpt_sw3模块
    gptj,  # 导入gptj模块
    gptsan_japanese,  # 导入gptsan_japanese模块
    graphormer,  # 导入graphormer模块
    groupvit,  # 导入groupvit模块
    herbert,  # 导入herbert模块
    hubert,  # 导入hubert模块
    ibert,  # 导入ibert模块
    idefics,  # 导入idefics模块
    imagegpt,  # 导入imagegpt模块
    informer,  # 导入informer模块
    instructblip,  # 导入instructblip模块
    jukebox,  # 导入jukebox模块
    kosmos2,  # 导入kosmos2模块
    layoutlm,  # 导入layoutlm模块
    layoutlmv2,  # 导入layoutlmv2模块
    layoutlmv3,  # 导入layoutlmv3模块
    layoutxlm,  # 导入layoutxlm模块
    led,  # 导入led模块
    levit,  # 导入levit模块
    lilt,  # 导入lilt模块
    llama,  # 导入llama模块
    llava,  # 导入llava模块
    llava_next,  # 导入llava_next模块
    longformer,  # 导入longformer模块
    longt5,  # 导入longt5模块
    luke,  # 导入luke模块
    lxmert,  # 导入lxmert模块
    m2m_100,  # 导入m2m_100模块
    mamba,  # 导入mamba模块
    marian,  # 导入marian模块
    markuplm,  # 导入markuplm模块
    mask2former,  # 导入mask2former模块
    maskformer,  # 导入maskformer模块
    mbart,  # 导入mbart模块
    mbart50,  # 导入mbart50模块
    mega,  # 导入mega模块
    megatron_bert,  # 导入megatron_bert模块
    megatron_gpt2,  # 导入
    # 导入所有模型
    import rembert
    import resnet
    import roberta
    import roberta_prelayernorm
    import roc_bert
    import roformer
    import rwkv
    import sam
    import seamless_m4t
    import seamless_m4t_v2
    import segformer
    import seggpt
    import sew
    import sew_d
    import siglip
    import speech_encoder_decoder
    import speech_to_text
    import speech_to_text_2
    import speecht5
    import splinter
    import squeezebert
    import stablelm
    import starcoder2
    import superpoint
    import swiftformer
    import swin
    import swin2sr
    import swinv2
    import switch_transformers
    import t5
    import table_transformer
    import tapas
    import time_series_transformer
    import timesformer
    import timm_backbone
    import trocr
    import tvlt
    import tvp
    import udop
    import umt5
    import unispeech
    import unispeech_sat
    import univnet
    import upernet
    import videomae
    import vilt
    import vipllava
    import vision_encoder_decoder
    import vision_text_dual_encoder
    import visual_bert
    import vit
    import vit_hybrid
    import vit_mae
    import vit_msn
    import vitdet
    import vitmatte
    import vits
    import vivit
    import wav2vec2
    import wav2vec2_bert
    import wav2vec2_conformer
    import wav2vec2_phoneme
    import wav2vec2_with_lm
    import wavlm
    import whisper
    import x_clip
    import xglm
    import xlm
    import xlm_prophetnet
    import xlm_roberta
    import xlm_roberta_xl
    import xlnet
    import xmod
    import yolos
    import yoso
)
```