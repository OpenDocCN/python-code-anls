# CogVideo & CogVideoX 微调代码源码解析（五）



# SAT CogVideoX-2B

[中文阅读](./README_zh.md)

[日本語で読む](./README_ja.md)

This folder contains the inference code using [SAT](https://github.com/THUDM/SwissArmyTransformer) weights and the
fine-tuning code for SAT weights.

This code is the framework used by the team to train the model. It has few comments and requires careful study.

## Inference Model

### 1. Ensure that you have correctly installed the dependencies required by this folder.

```py
pip install -r requirements.txt
```

### 2. Download the model weights

### 2. Download model weights

First, go to the SAT mirror to download the model weights. For the CogVideoX-2B model, please download as follows:

```py
mkdir CogVideoX-2b-sat
cd CogVideoX-2b-sat
wget https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
mv 'index.html?dl=1' vae.zip
unzip vae.zip
wget https://cloud.tsinghua.edu.cn/f/556a3e1329e74f1bac45/?dl=1
mv 'index.html?dl=1' transformer.zip
unzip transformer.zip
```

For the CogVideoX-5B model, please download the `transformers` file as follows link:
(VAE files are the same as 2B)

+ [CogVideoX-5B](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
+ [CogVideoX-5B-I2V](https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/?p=%2F1&mode=list)

Next, you need to format the model files as follows:

```py
.
├── transformer
│   ├── 1000 (or 1)
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── 3d-vae.pt
```

Due to large size of model weight file, using `git lfs` is recommended. Installation of `git lfs` can be
found [here](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing)

Next, clone the T5 model, which is not used for training and fine-tuning, but must be used.
> T5 model is available on [Modelscope](https://modelscope.cn/models/ZhipuAI/CogVideoX-2b) as well.

```py
git clone https://huggingface.co/THUDM/CogVideoX-2b.git
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

By following the above approach, you will obtain a safetensor format T5 file. Ensure that there are no errors when
loading it into Deepspeed in Finetune.

```py
├── added_tokens.json
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── spiece.model
└── tokenizer_config.json

0 directories, 8 files
```

### 3. Modify the file in `configs/cogvideox_2b.yaml`.

```py
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  log_keys:
    - txt

  denoiser_config:
    target: sgm.modules.diffusionmodules.denoiser.DiscreteDenoiser
    params:
      num_idx: 1000
      quantize_c_noise: False

      weighting_config:
        target: sgm.modules.diffusionmodules.denoiser_weighting.EpsWeighting
      scaling_config:
        target: sgm.modules.diffusionmodules.denoiser_scaling.VideoScaling
      discretization_config:
        target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
        params:
          shift_scale: 3.0

  network_config:
    target: dit_video_concat.DiffusionTransformer
    params:
      time_embed_dim: 512
      elementwise_affine: True
      num_frames: 49
      time_compressed_rate: 4
      latent_width: 90
      latent_height: 60
      num_layers: 30
      patch_size: 2
      in_channels: 16
      out_channels: 16
      hidden_size: 1920
      adm_in_channels: 256
      num_attention_heads: 30

      transformer_args:
        checkpoint_activations: True ## using gradient checkpointing
        vocab_size: 1
        max_sequence_length: 64
        layernorm_order: pre
        skip_init: false
        model_parallel_size: 1
        is_decoder: false

      modules:
        pos_embed_config:
          target: dit_video_concat.Basic3DPositionEmbeddingMixin
          params:
            text_length: 226
            height_interpolation: 1.875
            width_interpolation: 1.875

        patch_embed_config:
          target: dit_video_concat.ImagePatchEmbeddingMixin
          params:
            text_hidden_size: 4096

        adaln_layer_config:
          target: dit_video_concat.AdaLNMixin
          params:
            qk_ln: True

        final_layer_config:
          target: dit_video_concat.FinalLayerMixin

  conditioner_config:
    target: sgm.modules.GeneralConditioner
    params:
      emb_models:
        - is_trainable: false
          input_key: txt
          ucg_rate: 0.1
          target: sgm.modules.encoders.modules.FrozenT5Embedder
          params:
            model_dir: "t5-v1_1-xxl" # Absolute path to the CogVideoX-2b/t5-v1_1-xxl weights folder
            max_length: 226

  first_stage_config:
    target: vae_modules.autoencoder.VideoAutoencoderInferenceWrapper
    params:
      cp_size: 1
      ckpt_path: "CogVideoX-2b-sat/vae/3d-vae.pt" # Absolute path to the CogVideoX-2b-sat/vae/3d-vae.pt folder
      ignore_keys: [ 'loss' ]

      loss_config:
        target: torch.nn.Identity

      regularizer_config:
        target: vae_modules.regularizers.DiagonalGaussianRegularizer

      encoder_config:
        target: vae_modules.cp_enc_dec.ContextParallelEncoder3D
        params:
          double_z: true
          z_channels: 16
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [ 1, 2, 2, 4 ]
          attn_resolutions: [ ]
          num_res_blocks: 3
          dropout: 0.0
          gather_norm: True

      decoder_config:
        target: vae_modules.cp_enc_dec.ContextParallelDecoder3D
        params:
          double_z: True
          z_channels: 16
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [ 1, 2, 2, 4 ]
          attn_resolutions: [ ]
          num_res_blocks: 3
          dropout: 0.0
          gather_norm: False

  loss_fn_config:
    target: sgm.modules.diffusionmodules.loss.VideoDiffusionLoss
    params:
      offset_noise_level: 0
      sigma_sampler_config:
        target: sgm.modules.diffusionmodules.sigma_sampling.DiscreteSampling
        params:
          uniform_sampling: True
          num_idx: 1000
          discretization_config:
            target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
            params:
              shift_scale: 3.0

  sampler_config:
    target: sgm.modules.diffusionmodules.sampling.VPSDEDPMPP2MSampler
    params:
      num_steps: 50
      verbose: True

      discretization_config:
        target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
        params:
          shift_scale: 3.0

      guider_config:
        target: sgm.modules.diffusionmodules.guiders.DynamicCFG
        params:
          scale: 6
          exp: 5
          num_steps: 50
```

### 4. Modify the file in `configs/inference.yaml`.

```py
args:
  latent_channels: 16
  mode: inference
  load: "{absolute_path/to/your}/transformer" # Absolute path to the CogVideoX-2b-sat/transformer folder
  # load: "{your lora folder} such as zRzRzRzRzRzRzR/lora-disney-08-20-13-28" # This is for Full model without lora adapter

  batch_size: 1
  input_type: txt # You can choose txt for pure text input, or change to cli for command line input
  input_file: configs/test.txt # Pure text file, which can be edited
  sampling_num_frames: 13  # Must be 13, 11 or 9
  sampling_fps: 8
  fp16: True # For CogVideoX-2B
  #  bf16: True # For CogVideoX-5B
  output_dir: outputs/
  force_inference: True
```

+ Modify `configs/test.txt` if multiple prompts is required, in which each line makes a prompt.
+ For better prompt formatting, refer to [convert_demo.py](../inference/convert_demo.py), for which you should set the
  OPENAI_API_KEY as your environmental variable.
+ Modify `input_type` in `configs/inference.yaml` if use command line as prompt iuput.

```py
input_type: cli
```

This allows input from the command line as prompts.

Change `output_dir` if you wish to modify the address of the output video

```py
output_dir: outputs/
```

It is saved by default in the `.outputs/` folder.

### 5. Run the inference code to perform inference.

```py
bash inference.sh
```

## Fine-tuning the Model

### Preparing the Dataset

The dataset format should be as follows:

```py
.
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...
```

Each text file shares the same name as its corresponding video, serving as the label for that video. Videos and labels
should be matched one-to-one. Generally, a single video should not be associated with multiple labels.

For style fine-tuning, please prepare at least 50 videos and labels with similar styles to ensure proper fitting.

### Modifying Configuration Files

We support two fine-tuning methods: `Lora` and full-parameter fine-tuning. Please note that both methods only fine-tune
the `transformer` part and do not modify the `VAE` section. `T5` is used solely as an Encoder. Please modify
the `configs/sft.yaml` (for full-parameter fine-tuning) file as follows:

```py
  # checkpoint_activations: True ## Using gradient checkpointing (Both checkpoint_activations in the config file need to be set to True)
  model_parallel_size: 1 # Model parallel size
  experiment_name: lora-disney  # Experiment name (do not modify)
  mode: finetune # Mode (do not modify)
  load: "{your_CogVideoX-2b-sat_path}/transformer" ## Transformer model path
  no_load_rng: True # Whether to load random seed
  train_iters: 1000 # Training iterations
  eval_iters: 1 # Evaluation iterations
  eval_interval: 100    # Evaluation interval
  eval_batch_size: 1  # Evaluation batch size
  save: ckpts # Model save path
  save_interval: 100 # Model save interval
  log_interval: 20 # Log output interval
  train_data: [ "your train data path" ]
  valid_data: [ "your val data path" ] # Training and validation datasets can be the same
  split: 1,0,0 # Training, validation, and test set ratio
  num_workers: 8 # Number of worker threads for data loader
  force_train: True # Allow missing keys when loading checkpoint (T5 and VAE are loaded separately)
  only_log_video_latents: True # Avoid memory overhead caused by VAE decode
  deepspeed:
    bf16:
      enabled: False # For CogVideoX-2B set to False and for CogVideoX-5B set to True
    fp16:
      enabled: True  # For CogVideoX-2B set to True and for CogVideoX-5B set to False
```

If you wish to use Lora fine-tuning, you also need to modify the `cogvideox_<model_parameters>_lora` file:

Here, take `CogVideoX-2B` as a reference:

```py
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  not_trainable_prefixes: [ 'all' ] ## Uncomment
  log_keys:
    - txt'

  lora_config: ## Uncomment
    target: sat.model.finetune.lora2.LoraMixin
    params:
      r: 256
```

### Modifying Run Scripts

Edit `finetune_single_gpu.sh` or `finetune_multi_gpus.sh` to select the configuration file. Below are two examples:

1. If you want to use the `CogVideoX-2B` model and the `Lora` method, you need to modify `finetune_single_gpu.sh`
   or `finetune_multi_gpus.sh`:

```py
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b_lora.yaml configs/sft.yaml --seed $RANDOM"
```

2. If you want to use the `CogVideoX-2B` model and the `full-parameter fine-tuning` method, you need to
   modify `finetune_single_gpu.sh` or `finetune_multi_gpus.sh`:

```py
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b.yaml configs/sft.yaml --seed $RANDOM"
```

### Fine-Tuning and Evaluation

Run the inference code to start fine-tuning.

```py
bash finetune_single_gpu.sh # Single GPU
bash finetune_multi_gpus.sh # Multi GPUs
```

### Using the Fine-Tuned Model

The fine-tuned model cannot be merged; here is how to modify the inference configuration file `inference.sh`:

```py
run_cmd="$environs python sample_video.py --base configs/cogvideox_<model_parameters>_lora.yaml configs/inference.yaml --seed 42"
```

Then, execute the code:

```py
bash inference.sh 
```

### Converting to Huggingface Diffusers Supported Weights

The SAT weight format is different from Huggingface's weight format and needs to be converted. Please run:

```py
python ../tools/convert_weight_sat2hf.py
```

### Exporting Huggingface Diffusers lora LoRA Weights from SAT Checkpoints

After completing the training using the above steps, we get a SAT checkpoint with LoRA weights. You can find the file
at `{args.save}/1000/1000/mp_rank_00_model_states.pt`.

The script for exporting LoRA weights can be found in the CogVideoX repository at `tools/export_sat_lora_weight.py`.
After exporting, you can use `load_cogvideox_lora.py` for inference.

Export command:

```py
python tools/export_sat_lora_weight.py --sat_pt_path {args.save}/{experiment_name}-09-09-21-10/1000/mp_rank_00_model_states.pt --lora_save_directory {args.save}/export_hf_lora_weights_1/
```

This training mainly modified the following model structures. The table below lists the corresponding structure mappings
for converting to the HF (Hugging Face) format LoRA structure. As you can see, LoRA adds a low-rank weight to the
model's attention structure.

```py
'attention.query_key_value.matrix_A.0': 'attn1.to_q.lora_A.weight',
'attention.query_key_value.matrix_A.1': 'attn1.to_k.lora_A.weight',
'attention.query_key_value.matrix_A.2': 'attn1.to_v.lora_A.weight',
'attention.query_key_value.matrix_B.0': 'attn1.to_q.lora_B.weight',
'attention.query_key_value.matrix_B.1': 'attn1.to_k.lora_B.weight',
'attention.query_key_value.matrix_B.2': 'attn1.to_v.lora_B.weight',
'attention.dense.matrix_A.0': 'attn1.to_out.0.lora_A.weight',
'attention.dense.matrix_B.0': 'attn1.to_out.0.lora_B.weight'
```

Using export_sat_lora_weight.py, you can convert the SAT checkpoint into the HF LoRA format.
![alt text](../resources/hf_lora_weights.png)


# SAT CogVideoX-2B

[Read this in English.](./README_zh)

[中文阅读](./README_zh.md)

このフォルダには、[SAT](https://github.com/THUDM/SwissArmyTransformer) ウェイトを使用した推論コードと、SAT
ウェイトのファインチューニングコードが含まれています。

このコードは、チームがモデルをトレーニングするために使用したフレームワークです。コメントが少なく、注意深く研究する必要があります。

## 推論モデル

### 1. このフォルダに必要な依存関係が正しくインストールされていることを確認してください。

```py
pip install -r requirements.txt
```

### 2. モデルウェイトをダウンロードします

まず、SAT ミラーに移動してモデルの重みをダウンロードします。 CogVideoX-2B モデルの場合は、次のようにダウンロードしてください。

```py
mkdir CogVideoX-2b-sat
cd CogVideoX-2b-sat
wget https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
mv 'index.html?dl=1' vae.zip
unzip vae.zip
wget https://cloud.tsinghua.edu.cn/f/556a3e1329e74f1bac45/?dl=1
mv 'index.html?dl=1' transformer.zip
unzip transformer.zip
```

CogVideoX-5B モデルの `transformers` ファイルを以下のリンクからダウンロードしてください （VAE ファイルは 2B と同じです）：

+ [CogVideoX-5B](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
+ [CogVideoX-5B-I2V](https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/?p=%2F1&mode=list)

次に、モデルファイルを以下の形式にフォーマットする必要があります：

```py
.
├── transformer
│   ├── 1000 (or 1)
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── 3d-vae.pt
```

モデルの重みファイルが大きいため、`git lfs`を使用することをお勧めいたします。`git lfs`
のインストールについては、[こちら](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing)をご参照ください。

```py
git lfs install
```

次に、T5 モデルをクローンします。これはトレーニングやファインチューニングには使用されませんが、使用する必要があります。
> モデルを複製する際には、[Modelscope](https://modelscope.cn/models/ZhipuAI/CogVideoX-2b)のモデルファイルの場所もご使用いただけます。

```py
git clone https://huggingface.co/THUDM/CogVideoX-2b.git #ハギングフェイス(huggingface.org)からモデルをダウンロードいただきます
# git clone https://www.modelscope.cn/ZhipuAI/CogVideoX-2b.git #Modelscopeからモデルをダウンロードいただきます
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

上記の方法に従うことで、safetensor 形式の T5 ファイルを取得できます。これにより、Deepspeed でのファインチューニング中にエラーが発生しないようにします。

```py
├── added_tokens.json
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── spiece.model
└── tokenizer_config.json

0 directories, 8 files
```

### 3. `configs/cogvideox_2b.yaml` ファイルを変更します。

```py
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  log_keys:
    - txt

  denoiser_config:
    target: sgm.modules.diffusionmodules.denoiser.DiscreteDenoiser
    params:
      num_idx: 1000
      quantize_c_noise: False

      weighting_config:
        target: sgm.modules.diffusionmodules.denoiser_weighting.EpsWeighting
      scaling_config:
        target: sgm.modules.diffusionmodules.denoiser_scaling.VideoScaling
      discretization_config:
        target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
        params:
          shift_scale: 3.0

  network_config:
    target: dit_video_concat.DiffusionTransformer
    params:
      time_embed_dim: 512
      elementwise_affine: True
      num_frames: 49
      time_compressed_rate: 4
      latent_width: 90
      latent_height: 60
      num_layers: 30
      patch_size: 2
      in_channels: 16
      out_channels: 16
      hidden_size: 1920
      adm_in_channels: 256
      num_attention_heads: 30

      transformer_args:
        checkpoint_activations: True ## グラデーション チェックポイントを使用する
        vocab_size: 1
        max_sequence_length: 64
        layernorm_order: pre
        skip_init: false
        model_parallel_size: 1
        is_decoder: false

      modules:
        pos_embed_config:
          target: dit_video_concat.Basic3DPositionEmbeddingMixin
          params:
            text_length: 226
            height_interpolation: 1.875
            width_interpolation: 1.875

        patch_embed_config:
          target: dit_video_concat.ImagePatchEmbeddingMixin
          params:
            text_hidden_size: 4096

        adaln_layer_config:
          target: dit_video_concat.AdaLNMixin
          params:
            qk_ln: True

        final_layer_config:
          target: dit_video_concat.FinalLayerMixin

  conditioner_config:
    target: sgm.modules.GeneralConditioner
    params:
      emb_models:
        - is_trainable: false
          input_key: txt
          ucg_rate: 0.1
          target: sgm.modules.encoders.modules.FrozenT5Embedder
          params:
            model_dir: "t5-v1_1-xxl" # CogVideoX-2b/t5-v1_1-xxlフォルダの絶対パス
            max_length: 226

  first_stage_config:
    target: vae_modules.autoencoder.VideoAutoencoderInferenceWrapper
    params:
      cp_size: 1
      ckpt_path: "CogVideoX-2b-sat/vae/3d-vae.pt" # CogVideoX-2b-sat/vae/3d-vae.ptフォルダの絶対パス
      ignore_keys: [ 'loss' ]

      loss_config:
        target: torch.nn.Identity

      regularizer_config:
        target: vae_modules.regularizers.DiagonalGaussianRegularizer

      encoder_config:
        target: vae_modules.cp_enc_dec.ContextParallelEncoder3D
        params:
          double_z: true
          z_channels: 16
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [ 1, 2, 2, 4 ]
          attn_resolutions: [ ]
          num_res_blocks: 3
          dropout: 0.0
          gather_norm: True

      decoder_config:
        target: vae_modules.cp_enc_dec.ContextParallelDecoder3D
        params:
          double_z: True
          z_channels: 16
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [ 1, 2, 2, 4 ]
          attn_resolutions: [ ]
          num_res_blocks: 3
          dropout: 0.0
          gather_norm: False

  loss_fn_config:
    target: sgm.modules.diffusionmodules.loss.VideoDiffusionLoss
    params:
      offset_noise_level: 0
      sigma_sampler_config:
        target: sgm.modules.diffusionmodules.sigma_sampling.DiscreteSampling
        params:
          uniform_sampling: True
          num_idx: 1000
          discretization_config:
            target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
            params:
              shift_scale: 3.0

  sampler_config:
    target: sgm.modules.diffusionmodules.sampling.VPSDEDPMPP2MSampler
    params:
      num_steps: 50
      verbose: True

      discretization_config:
        target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
        params:
          shift_scale: 3.0

      guider_config:
        target: sgm.modules.diffusionmodules.guiders.DynamicCFG
        params:
          scale: 6
          exp: 5
          num_steps: 50
```

### 4. `configs/inference.yaml` ファイルを変更します。

```py
args:
  latent_channels: 16
  mode: inference
  load: "{absolute_path/to/your}/transformer" # CogVideoX-2b-sat/transformerフォルダの絶対パス
  # load: "{your lora folder} such as zRzRzRzRzRzRzR/lora-disney-08-20-13-28" # This is for Full model without lora adapter

  batch_size: 1
  input_type: txt #TXTのテキストファイルを入力として選択されたり、CLIコマンドラインを入力として変更されたりいただけます
  input_file: configs/test.txt #テキストファイルのパスで、これに対して編集がさせていただけます
  sampling_num_frames: 13  # Must be 13, 11 or 9
  sampling_fps: 8
  fp16: True # For CogVideoX-2B
  #  bf16: True # For CogVideoX-5B
  output_dir: outputs/
  force_inference: True
```

+ 複数のプロンプトを保存するために txt を使用する場合は、`configs/test.txt`
  を参照して変更してください。1行に1つのプロンプトを記述します。プロンプトの書き方がわからない場合は、最初に [このコード](../inference/convert_demo.py)
  を使用して LLM によるリファインメントを呼び出すことができます。
+ コマンドラインを入力として使用する場合は、次のように変更します。

```py
input_type: cli
```

これにより、コマンドラインからプロンプトを入力できます。

出力ビデオのディレクトリを変更したい場合は、次のように変更できます：

```py
output_dir: outputs/
```

デフォルトでは `.outputs/` フォルダに保存されます。

### 5. 推論コードを実行して推論を開始します。

```py
bash inference.sh
```

## モデルのファインチューニング

### データセットの準備

データセットの形式は次のようになります：

```py
.
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...
```

各 txt ファイルは対応するビデオファイルと同じ名前であり、そのビデオのラベルを含んでいます。各ビデオはラベルと一対一で対応する必要があります。通常、1つのビデオに複数のラベルを持たせることはありません。

スタイルファインチューニングの場合、少なくとも50本のスタイルが似たビデオとラベルを準備し、フィッティングを容易にします。

### 設定ファイルの変更

`Lora` とフルパラメータ微調整の2つの方法をサポートしています。両方の微調整方法は、`transformer` 部分のみを微調整し、`VAE`
部分には変更を加えないことに注意してください。`T5` はエンコーダーとしてのみ使用されます。以下のように `configs/sft.yaml` (
フルパラメータ微調整用) ファイルを変更してください。

```py
  # checkpoint_activations: True ## 勾配チェックポイントを使用する場合 (設定ファイル内の2つの checkpoint_activations を True に設定する必要があります)
  model_parallel_size: 1 # モデル並列サイズ
  experiment_name: lora-disney  # 実験名 (変更しないでください)
  mode: finetune # モード (変更しないでください)
  load: "{your_CogVideoX-2b-sat_path}/transformer" ## Transformer モデルのパス
  no_load_rng: True # 乱数シードを読み込むかどうか
  train_iters: 1000 # トレーニングイテレーション数
  eval_iters: 1 # 評価イテレーション数
  eval_interval: 100    # 評価間隔
  eval_batch_size: 1  # 評価バッチサイズ
  save: ckpts # モデル保存パス
  save_interval: 100 # モデル保存間隔
  log_interval: 20 # ログ出力間隔
  train_data: [ "your train data path" ]
  valid_data: [ "your val data path" ] # トレーニングデータと評価データは同じでも構いません
  split: 1,0,0 # トレーニングセット、評価セット、テストセットの割合
  num_workers: 8 # データローダーのワーカースレッド数
  force_train: True # チェックポイントをロードするときに欠落したキーを許可 (T5 と VAE は別々にロードされます)
  only_log_video_latents: True # VAE のデコードによるメモリオーバーヘッドを回避
  deepspeed:
    bf16:
      enabled: False # CogVideoX-2B の場合は False に設定し、CogVideoX-5B の場合は True に設定
    fp16:
      enabled: True  # CogVideoX-2B の場合は True に設定し、CogVideoX-5B の場合は False に設定
```

Lora 微調整を使用したい場合は、`cogvideox_<model_parameters>_lora` ファイルも変更する必要があります。

ここでは、`CogVideoX-2B` を参考にします。

```py
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  not_trainable_prefixes: [ 'all' ] ## コメントを解除
  log_keys:
    - txt'

  lora_config: ## コメントを解除
    target: sat.model.finetune.lora2.LoraMixin
    params:
      r: 256
```

### 実行スクリプトの変更

設定ファイルを選択するために `finetune_single_gpu.sh` または `finetune_multi_gpus.sh` を編集します。以下に2つの例を示します。

1. `CogVideoX-2B` モデルを使用し、`Lora` 手法を利用する場合は、`finetune_single_gpu.sh` または `finetune_multi_gpus.sh`
   を変更する必要があります。

```py
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b_lora.yaml configs/sft.yaml --seed $RANDOM"
```

2. `CogVideoX-2B` モデルを使用し、`フルパラメータ微調整` 手法を利用する場合は、`finetune_single_gpu.sh`
   または `finetune_multi_gpus.sh` を変更する必要があります。

```py
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b.yaml configs/sft.yaml --seed $RANDOM"
```

### 微調整と評価

推論コードを実行して微調整を開始します。

```py
bash finetune_single_gpu.sh # シングルGPU
bash finetune_multi_gpus.sh # マルチGPU
```

### 微調整後のモデルの使用

微調整されたモデルは統合できません。ここでは、推論設定ファイル `inference.sh` を変更する方法を示します。

```py
run_cmd="$environs python sample_video.py --base configs/cogvideox_<model_parameters>_lora.yaml configs/inference.yaml --seed 42"
```

その後、次のコードを実行します。

```py
bash inference.sh 
```

### Huggingface Diffusers サポートのウェイトに変換

SAT ウェイト形式は Huggingface のウェイト形式と異なり、変換が必要です。次のコマンドを実行してください：

```py
python ../tools/convert_weight_sat2hf.py
```

### SATチェックポイントからHuggingface Diffusers lora LoRAウェイトをエクスポート

上記のステップを完了すると、LoRAウェイト付きのSATチェックポイントが得られます。ファイルは `{args.save}/1000/1000/mp_rank_00_model_states.pt` にあります。

LoRAウェイトをエクスポートするためのスクリプトは、CogVideoXリポジトリの `tools/export_sat_lora_weight.py` にあります。エクスポート後、`load_cogvideox_lora.py` を使用して推論を行うことができます。

エクスポートコマンド:

```py
python tools/export_sat_lora_weight.py --sat_pt_path {args.save}/{experiment_name}-09-09-21-10/1000/mp_rank_00_model_states.pt --lora_save_directory {args.save}/export_hf_lora_weights_1/
```

このトレーニングでは主に以下のモデル構造が変更されました。以下の表は、HF (Hugging Face) 形式のLoRA構造に変換する際の対応関係を示しています。ご覧の通り、LoRAはモデルの注意メカニズムに低ランクの重みを追加しています。

```py
'attention.query_key_value.matrix_A.0': 'attn1.to_q.lora_A.weight',
'attention.query_key_value.matrix_A.1': 'attn1.to_k.lora_A.weight',
'attention.query_key_value.matrix_A.2': 'attn1.to_v.lora_A.weight',
'attention.query_key_value.matrix_B.0': 'attn1.to_q.lora_B.weight',
'attention.query_key_value.matrix_B.1': 'attn1.to_k.lora_B.weight',
'attention.query_key_value.matrix_B.2': 'attn1.to_v.lora_B.weight',
'attention.dense.matrix_A.0': 'attn1.to_out.0.lora_A.weight',
'attention.dense.matrix_B.0': 'attn1.to_out.0.lora_B.weight'
```
  
export_sat_lora_weight.py を使用して、SATチェックポイントをHF LoRA形式に変換できます。


![alt text](../resources/hf_lora_weights.png)


# SAT CogVideoX-2B

[Read this in English.](./README_zh)

[日本語で読む](./README_ja.md)

本文件夹包含了使用 [SAT](https://github.com/THUDM/SwissArmyTransformer) 权重的推理代码，以及 SAT 权重的微调代码。

该代码是团队训练模型时使用的框架。注释较少，需要认真研究。

## 推理模型

### 1. 确保你已经正确安装本文件夹中的要求的依赖

```py
pip install -r requirements.txt
```

### 2. 下载模型权重

首先，前往 SAT 镜像下载模型权重。

对于 CogVideoX-2B 模型，请按照如下方式下载:

```py
mkdir CogVideoX-2b-sat
cd CogVideoX-2b-sat
wget https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
mv 'index.html?dl=1' vae.zip
unzip vae.zip
wget https://cloud.tsinghua.edu.cn/f/556a3e1329e74f1bac45/?dl=1
mv 'index.html?dl=1' transformer.zip
unzip transformer.zip
```

请按如下链接方式下载 CogVideoX-5B 模型的 `transformers` 文件（VAE 文件与 2B 相同）：

+ [CogVideoX-5B](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
+ [CogVideoX-5B-I2V](https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/?p=%2F1&mode=list)

接着，你需要将模型文件排版成如下格式：

```py
.
├── transformer
│   ├── 1000 (or 1)
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── 3d-vae.pt
```

由于模型的权重档案较大，建议使用`git lfs`。`git lfs`
安装参见[这里](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing)

```py
git lfs install
```

接着，克隆 T5 模型，该模型不用做训练和微调，但是必须使用。
> 克隆模型的时候也可以使用[Modelscope](https://modelscope.cn/models/ZhipuAI/CogVideoX-2b)上的模型文件位置。

```py
git clone https://huggingface.co/THUDM/CogVideoX-2b.git #从huggingface下载模型
# git clone https://www.modelscope.cn/ZhipuAI/CogVideoX-2b.git #从modelscope下载模型
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

通过上述方案，你将会得到一个 safetensor 格式的T5文件，确保在 Deepspeed微调过程中读入的时候不会报错。

```py
├── added_tokens.json
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── spiece.model
└── tokenizer_config.json

0 directories, 8 files
```

### 3. 修改`configs/cogvideox_2b.yaml`中的文件。

```py
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  log_keys:
    - txt

  denoiser_config:
    target: sgm.modules.diffusionmodules.denoiser.DiscreteDenoiser
    params:
      num_idx: 1000
      quantize_c_noise: False

      weighting_config:
        target: sgm.modules.diffusionmodules.denoiser_weighting.EpsWeighting
      scaling_config:
        target: sgm.modules.diffusionmodules.denoiser_scaling.VideoScaling
      discretization_config:
        target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
        params:
          shift_scale: 3.0

  network_config:
    target: dit_video_concat.DiffusionTransformer
    params:
      time_embed_dim: 512
      elementwise_affine: True
      num_frames: 49
      time_compressed_rate: 4
      latent_width: 90
      latent_height: 60
      num_layers: 30
      patch_size: 2
      in_channels: 16
      out_channels: 16
      hidden_size: 1920
      adm_in_channels: 256
      num_attention_heads: 30

      transformer_args:
        checkpoint_activations: True ## using gradient checkpointing
        vocab_size: 1
        max_sequence_length: 64
        layernorm_order: pre
        skip_init: false
        model_parallel_size: 1
        is_decoder: false

      modules:
        pos_embed_config:
          target: dit_video_concat.Basic3DPositionEmbeddingMixin
          params:
            text_length: 226
            height_interpolation: 1.875
            width_interpolation: 1.875

        patch_embed_config:
          target: dit_video_concat.ImagePatchEmbeddingMixin
          params:
            text_hidden_size: 4096

        adaln_layer_config:
          target: dit_video_concat.AdaLNMixin
          params:
            qk_ln: True

        final_layer_config:
          target: dit_video_concat.FinalLayerMixin

  conditioner_config:
    target: sgm.modules.GeneralConditioner
    params:
      emb_models:
        - is_trainable: false
          input_key: txt
          ucg_rate: 0.1
          target: sgm.modules.encoders.modules.FrozenT5Embedder
          params:
            model_dir: "t5-v1_1-xxl" # CogVideoX-2b/t5-v1_1-xxl 权重文件夹的绝对路径
            max_length: 226

  first_stage_config:
    target: vae_modules.autoencoder.VideoAutoencoderInferenceWrapper
    params:
      cp_size: 1
      ckpt_path: "CogVideoX-2b-sat/vae/3d-vae.pt" # CogVideoX-2b-sat/vae/3d-vae.pt文件夹的绝对路径
      ignore_keys: [ 'loss' ]

      loss_config:
        target: torch.nn.Identity

      regularizer_config:
        target: vae_modules.regularizers.DiagonalGaussianRegularizer

      encoder_config:
        target: vae_modules.cp_enc_dec.ContextParallelEncoder3D
        params:
          double_z: true
          z_channels: 16
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [ 1, 2, 2, 4 ]
          attn_resolutions: [ ]
          num_res_blocks: 3
          dropout: 0.0
          gather_norm: True

      decoder_config:
        target: vae_modules.cp_enc_dec.ContextParallelDecoder3D
        params:
          double_z: True
          z_channels: 16
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [ 1, 2, 2, 4 ]
          attn_resolutions: [ ]
          num_res_blocks: 3
          dropout: 0.0
          gather_norm: False

  loss_fn_config:
    target: sgm.modules.diffusionmodules.loss.VideoDiffusionLoss
    params:
      offset_noise_level: 0
      sigma_sampler_config:
        target: sgm.modules.diffusionmodules.sigma_sampling.DiscreteSampling
        params:
          uniform_sampling: True
          num_idx: 1000
          discretization_config:
            target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
            params:
              shift_scale: 3.0

  sampler_config:
    target: sgm.modules.diffusionmodules.sampling.VPSDEDPMPP2MSampler
    params:
      num_steps: 50
      verbose: True

      discretization_config:
        target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
        params:
          shift_scale: 3.0

      guider_config:
        target: sgm.modules.diffusionmodules.guiders.DynamicCFG
        params:
          scale: 6
          exp: 5
          num_steps: 50
```

### 4. 修改`configs/inference.yaml`中的文件。

```py
args:
  latent_channels: 16
  mode: inference
  load: "{absolute_path/to/your}/transformer" # CogVideoX-2b-sat/transformer文件夹的绝对路径
  # load: "{your lora folder} such as zRzRzRzRzRzRzR/lora-disney-08-20-13-28" # This is for Full model without lora adapter

  batch_size: 1
  input_type: txt #可以选择txt纯文字档作为输入，或者改成cli命令行作为输入
  input_file: configs/test.txt #纯文字档，可以对此做编辑
  sampling_num_frames: 13  # Must be 13, 11 or 9
  sampling_fps: 8
  fp16: True # For CogVideoX-2B
  #  bf16: True # For CogVideoX-5B
  output_dir: outputs/
  force_inference: True
```

+ 如果使用 txt 保存多个提示词，请参考`configs/test.txt`
  进行修改。每一行一个提示词。如果您不知道如何书写提示词，可以先使用[此代码](../inference/convert_demo.py)调用 LLM进行润色。
+ 如果使用命令行作为输入，请修改

```py
input_type: cli
```

这样就可以从命令行输入提示词。

如果你希望修改输出视频的地址，你可以修改:

```py
output_dir: outputs/
```

默认保存在`.outputs/`文件夹下。

### 5. 运行推理代码, 即可推理

```py
bash inference.sh
```

## 微调模型

### 准备数据集

数据集格式应该如下：

```py
.
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...
```

每个 txt 与视频同名，为视频的标签。视频与标签应该一一对应。通常情况下，不使用一个视频对应多个标签。

如果为风格微调，清准备至少50条风格相似的视频和标签，以利于拟合。

### 修改配置文件

我们支持 `Lora` 和 全参数微调两种方式。请注意，两种微调方式都仅仅对 `transformer` 部分进行微调。不改动 `VAE` 部分。`T5`仅作为
Encoder 使用。
部分。 请按照以下方式修改`configs/sft.yaml`(全量微调) 中的文件。

```py
  # checkpoint_activations: True ## using gradient checkpointing (配置文件中的两个checkpoint_activations都需要设置为True)
  model_parallel_size: 1 # 模型并行大小
  experiment_name: lora-disney  # 实验名称(不要改动)
  mode: finetune # 模式(不要改动)
  load: "{your_CogVideoX-2b-sat_path}/transformer" ## Transformer 模型路径
  no_load_rng: True # 是否加载随机数种子
  train_iters: 1000 # 训练迭代次数
  eval_iters: 1 # 验证迭代次数
  eval_interval: 100    # 验证间隔
  eval_batch_size: 1  # 验证集 batch size
  save: ckpts # 模型保存路径 
  save_interval: 100 # 模型保存间隔
  log_interval: 20 # 日志输出间隔
  train_data: [ "your train data path" ]
  valid_data: [ "your val data path" ] # 训练集和验证集可以相同
  split: 1,0,0 # 训练集，验证集，测试集比例
  num_workers: 8 # 数据加载器的工作线程数
  force_train: True # 在加载checkpoint时允许missing keys (T5 和 VAE 单独加载)
  only_log_video_latents: True # 避免VAE decode带来的显存开销
  deepspeed:
    bf16:
      enabled: False # For CogVideoX-2B Turn to False and For CogVideoX-5B Turn to True
    fp16:
      enabled: True  # For CogVideoX-2B Turn to True and For CogVideoX-5B Turn to False
```

如果你希望使用 Lora 微调，你还需要修改`cogvideox_<模型参数>_lora` 文件：

这里以 `CogVideoX-2B` 为参考:

```py
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  not_trainable_prefixes: [ 'all' ] ## 解除注释
  log_keys:
    - txt'

  lora_config: ##  解除注释
    target: sat.model.finetune.lora2.LoraMixin
    params:
      r: 256
```

### 修改运行脚本

编辑`finetune_single_gpu.sh` 或者 `finetune_multi_gpus.sh`，选择配置文件。下面是两个例子:

1. 如果您想使用 `CogVideoX-2B` 模型并使用`Lora`方案，您需要修改`finetune_single_gpu.sh` 或者 `finetune_multi_gpus.sh`:

```py
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b_lora.yaml configs/sft.yaml --seed $RANDOM"
```

2. 如果您想使用 `CogVideoX-2B` 模型并使用`全量微调`方案，您需要修改`finetune_single_gpu.sh`
   或者 `finetune_multi_gpus.sh`:

```py
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b.yaml configs/sft.yaml --seed $RANDOM"
```

### 微调和验证

运行推理代码,即可开始微调。

```py
bash finetune_single_gpu.sh # Single GPU
bash finetune_multi_gpus.sh # Multi GPUs
```

### 使用微调后的模型

微调后的模型无法合并，这里展现了如何修改推理配置文件 `inference.sh`

```py
run_cmd="$environs python sample_video.py --base configs/cogvideox_<模型参数>_lora.yaml configs/inference.yaml --seed 42"
```

然后，执行代码:

```py
bash inference.sh 
```

### 转换到 Huggingface Diffusers 库支持的权重

SAT 权重格式与 Huggingface 的权重格式不同，需要转换。请运行

```py
python ../tools/convert_weight_sat2hf.py
```

### 从SAT权重文件 导出Huggingface Diffusers lora权重

支持了从SAT权重文件
在经过上面这些步骤训练之后，我们得到了一个sat带lora的权重，在{args.save}/1000/1000/mp_rank_00_model_states.pt你可以看到这个文件

导出的lora权重脚本在CogVideoX仓库 tools/export_sat_lora_weight.py ,导出后使用 load_cogvideox_lora.py 推理

导出命令:

```py
python tools/export_sat_lora_weight.py --sat_pt_path {args.save}/{experiment_name}-09-09-21-10/1000/mp_rank_00_model_states.pt --lora_save_directory   {args.save}/export_hf_lora_weights_1/
```

这次训练主要修改了下面几个模型结构,下面列出了 转换为HF格式的lora结构对应关系,可以看到lora将模型注意力结构上增加一个低秩权重,

```py
'attention.query_key_value.matrix_A.0': 'attn1.to_q.lora_A.weight',
'attention.query_key_value.matrix_A.1': 'attn1.to_k.lora_A.weight',
'attention.query_key_value.matrix_A.2': 'attn1.to_v.lora_A.weight',
'attention.query_key_value.matrix_B.0': 'attn1.to_q.lora_B.weight',
'attention.query_key_value.matrix_B.1': 'attn1.to_k.lora_B.weight',
'attention.query_key_value.matrix_B.2': 'attn1.to_v.lora_B.weight',
'attention.dense.matrix_A.0': 'attn1.to_out.0.lora_A.weight',
'attention.dense.matrix_B.0': 'attn1.to_out.0.lora_B.weight'
```

通过export_sat_lora_weight.py将它转换为HF格式的lora结构
![alt text](../resources/hf_lora_weights.png)


# `.\cogvideo-finetune\sat\sample_video.py`

```py
# 导入操作系统相关功能
import os
# 导入数学运算库
import math
# 导入命令行参数解析库
import argparse
# 导入类型注解工具
from typing import List, Union
# 导入进度条库
from tqdm import tqdm
# 导入列表配置类
from omegaconf import ListConfig
# 导入图像输入输出库
import imageio

# 导入PyTorch库
import torch
# 导入NumPy库
import numpy as np
# 导入重排列工具
from einops import rearrange
# 导入图像变换库
import torchvision.transforms as TT

# 从自定义模型库导入获取模型的函数
from sat.model.base_model import get_model
# 从自定义训练库导入加载检查点的函数
from sat.training.model_io import load_checkpoint
# 导入多处理工具
from sat import mpu

# 从扩散视频模块导入视频扩散引擎
from diffusion_video import SATVideoDiffusionEngine
# 从参数模块导入获取参数的函数
from arguments import get_args
# 导入中心裁剪和调整大小功能
from torchvision.transforms.functional import center_crop, resize
# 导入插值模式
from torchvision.transforms import InterpolationMode
# 导入PIL库中的图像模块
from PIL import Image

# 定义从命令行读取输入的生成器函数
def read_from_cli():
    # 初始化计数器
    cnt = 0
    try:
        # 循环直到接收到EOF
        while True:
            # 提示用户输入英文文本
            x = input("Please input English text (Ctrl-D quit): ")
            # 返回处理后的文本和计数器值
            yield x.strip(), cnt
            # 增加计数器
            cnt += 1
    # 捕获EOF错误
    except EOFError as e:
        pass

# 定义从文件读取输入的生成器函数
def read_from_file(p, rank=0, world_size=1):
    # 以只读模式打开文件
    with open(p, "r") as fin:
        # 初始化计数器
        cnt = -1
        # 遍历文件中的每一行
        for l in fin:
            # 增加计数器
            cnt += 1
            # 根据rank和world_size决定是否继续
            if cnt % world_size != rank:
                continue
            # 返回处理后的行和计数器值
            yield l.strip(), cnt

# 定义从条件器获取唯一嵌入器键的函数
def get_unique_embedder_keys_from_conditioner(conditioner):
    # 返回唯一的输入键列表
    return list(set([x.input_key for x in conditioner.embedders]))

# 定义获取批次数据的函数
def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    # 初始化批次字典和未条件化批次字典
    batch = {}
    batch_uc = {}

    # 遍历所有键
    for key in keys:
        # 处理文本键
        if key == "txt":
            # 生成包含提示的批次数据
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            # 生成包含负提示的未条件化批次数据
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            # 将其他键的值直接添加到批次中
            batch[key] = value_dict[key]

    # 如果T不为None，则添加视频帧数信息
    if T is not None:
        batch["num_video_frames"] = T

    # 遍历批次字典中的所有键
    for key in batch.keys():
        # 如果未条件化字典中没有该键且其为张量，则克隆张量
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    # 返回批次和未条件化批次
    return batch, batch_uc

# 定义将视频保存为网格和MP4格式的函数
def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    # 如果保存路径不存在，则创建它
    os.makedirs(save_path, exist_ok=True)

    # 遍历视频批次
    for i, vid in enumerate(video_batch):
        # 初始化GIF帧列表
        gif_frames = []
        # 遍历每一帧
        for frame in vid:
            # 调整帧的维度顺序
            frame = rearrange(frame, "c h w -> h w c")
            # 将帧的数据转换为0-255的整型
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            # 添加帧到GIF帧列表
            gif_frames.append(frame)
        # 生成当前保存路径
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        # 使用imageio保存视频
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            # 遍历GIF帧并写入视频文件
            for frame in gif_frames:
                writer.append_data(frame)

# 定义调整图像大小以适应矩形裁剪的函数
def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    # 检查输入数组的宽高比与目标宽高比的关系
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        # 按照宽度进行调整大小
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        # 按照高度进行调整大小
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )
    # 获取数组的高度和宽度
        h, w = arr.shape[2], arr.shape[3]
        # 去掉数组的第一个维度，保持其他维度不变
        arr = arr.squeeze(0)
    
        # 计算高度和宽度的差值
        delta_h = h - image_size[0]
        delta_w = w - image_size[1]
    
        # 根据重塑模式确定裁剪的起始位置
        if reshape_mode == "random" or reshape_mode == "none":
            # 随机生成裁剪的顶部和左边位置
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            # 计算中心裁剪的顶部和左边位置
            top, left = delta_h // 2, delta_w // 2
        else:
            # 如果模式不被支持，抛出异常
            raise NotImplementedError
        # 裁剪数组为指定的高度和宽度
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        # 返回裁剪后的数组
        return arr
# 主函数，负责采样过程
def sampling_main(args, model_cls):
    # 检查 model_cls 是否为类型，若是则调用 get_model 函数获取模型
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    # 否则直接使用 model_cls
    else:
        model = model_cls

    # 加载模型的检查点
    load_checkpoint(model, args)
    # 设置模型为评估模式
    model.eval()

    # 根据输入类型读取数据
    if args.input_type == "cli":
        # 从命令行读取数据
        data_iter = read_from_cli()
    elif args.input_type == "txt":
        # 获取当前进程的排名和总进程数
        rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        print("rank and world_size", rank, world_size)
        # 从文件读取数据，带入排名和进程数
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        # 如果输入类型不被支持，抛出错误
        raise NotImplementedError

    # 设置图像大小
    image_size = [480, 720]

    # 如果需要将图像转换为视频
    if args.image2video:
        chained_trainsforms = []
        # 添加将图像转换为张量的变换
        chained_trainsforms.append(TT.ToTensor())
        # 组合变换
        transform = TT.Compose(chained_trainsforms)

    # 获取模型的采样函数
    sample_func = model.sample
    # 定义采样的相关参数
    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
    # 设置样本数量
    num_samples = [1]
    # 定义强制使用的嵌入类型
    force_uc_zero_embeddings = ["txt"]
    # 获取模型所使用的设备
    device = model.device

# 当脚本作为主程序执行时
if __name__ == "__main__":
    # 检查环境变量以获取进程相关信息
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    # 创建命令行参数解析器
    py_parser = argparse.ArgumentParser(add_help=False)
    # 解析已知参数
    known, args_list = py_parser.parse_known_args()

    # 获取完整的命令行参数
    args = get_args(args_list)
    # 将已知参数与其他参数合并
    args = argparse.Namespace(**vars(args), **vars(known))
    # 删除不需要的深度学习配置参数
    del args.deepspeed_config
    # 设置模型配置的检查点大小
    args.model_config.first_stage_config.params.cp_size = 1
    # 设置网络配置的模型并行大小
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    # 关闭检查点激活
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    # 关闭均匀采样
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    # 调用采样主函数
    sampling_main(args, model_cls=SATVideoDiffusionEngine)
```

# `.\cogvideo-finetune\sat\sgm\lr_scheduler.py`

```py
# 导入 NumPy 库，通常用于数值计算
import numpy as np


class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """  # 类的文档字符串，说明使用基础学习率为 1.0

    def __init__(  # 构造函数，初始化调度器
        self,
        warm_up_steps,  # 预热步骤数
        lr_min,  # 最小学习率
        lr_max,  # 最大学习率
        lr_start,  # 起始学习率
        max_decay_steps,  # 最大衰减步骤数
        verbosity_interval=0,  # 输出间隔
    ):
        # 将输入参数赋值给实例变量
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.0  # 初始化最近的学习率为 0
        self.verbosity_interval = verbosity_interval  # 设定输出间隔

    def schedule(self, n, **kwargs):  # 学习率调度函数
        if self.verbosity_interval > 0:  # 检查是否需要输出信息
            if n % self.verbosity_interval == 0:  # 如果当前步骤满足输出条件
                print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")  # 输出当前步骤和学习率
        if n < self.lr_warm_up_steps:  # 如果当前步骤在预热阶段
            # 计算线性增加的学习率
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr  # 更新最近的学习率
            return lr  # 返回计算的学习率
        else:  # 如果已过预热阶段
            # 计算归一化的时间参数 t
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)  # 确保 t 不超过 1.0
            # 计算余弦衰减的学习率
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))
            self.last_lr = lr  # 更新最近的学习率
            return lr  # 返回计算的学习率

    def __call__(self, n, **kwargs):  # 使得类实例可调用
        return self.schedule(n, **kwargs)  # 调用调度函数


class LambdaWarmUpCosineScheduler2:
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """  # 类的文档字符串，说明支持重复迭代且通过列表配置，基础学习率为 1.0

    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):  # 构造函数
        # 检查所有输入列表长度是否一致
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        # 将输入参数赋值给实例变量
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        # 计算循环长度的累积和
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.0  # 初始化最近的函数值为 0
        self.verbosity_interval = verbosity_interval  # 设定输出间隔

    def find_in_interval(self, n):  # 查找当前步骤所在的周期
        interval = 0  # 初始化周期计数
        for cl in self.cum_cycles[1:]:  # 遍历所有累积周期
            if n <= cl:  # 如果当前步骤在当前周期内
                return interval  # 返回周期索引
            interval += 1  # 递增周期计数

    def schedule(self, n, **kwargs):  # 学习率调度函数
        cycle = self.find_in_interval(n)  # 查找当前步骤所在的周期
        n = n - self.cum_cycles[cycle]  # 计算当前步骤在周期内的相对步骤
        if self.verbosity_interval > 0:  # 检查是否需要输出信息
            if n % self.verbosity_interval == 0:  # 如果当前步骤满足输出条件
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, " f"current cycle {cycle}")  # 输出信息
        if n < self.lr_warm_up_steps[cycle]:  # 如果当前步骤在预热阶段
            # 计算线性增加的函数值
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f  # 更新最近的函数值
            return f  # 返回计算的函数值
        else:  # 如果已过预热阶段
            # 计算归一化的时间参数 t
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)  # 确保 t 不超过 1.0
            # 计算余弦衰减的函数值
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + np.cos(t * np.pi))
            self.last_f = f  # 更新最近的函数值
            return f  # 返回计算的函数值
    # 定义可调用对象的魔术方法，接收一个参数 n 和可变关键字参数 kwargs
        def __call__(self, n, **kwargs):
            # 调用该对象的 schedule 方法，传入参数 n 和可变关键字参数 kwargs，并返回其结果
            return self.schedule(n, **kwargs)
# 定义一个新的类 LambdaLinearScheduler，继承自 LambdaWarmUpCosineScheduler2
class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    # 定义调度方法，接受当前步数 n 和额外参数 kwargs
    def schedule(self, n, **kwargs):
        # 找到 n 在累积周期中的区间
        cycle = self.find_in_interval(n)
        # 计算当前步数 n，减去已完成的累计周期
        n = n - self.cum_cycles[cycle]
        # 如果设置了详细输出间隔，则打印当前状态
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, " f"current cycle {cycle}")

        # 如果当前步数小于热身步数，计算线性增加的函数值
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            # 更新最近的函数值
            self.last_f = f
            # 返回计算得到的函数值
            return f
        else:
            # 在热身步数之后，计算函数值为周期长度的线性递减
            f = (
                self.f_min[cycle]
                + (self.f_max[cycle] - self.f_min[cycle])
                * (self.cycle_lengths[cycle] - n)
                / (self.cycle_lengths[cycle])
            )
            # 更新最近的函数值
            self.last_f = f
            # 返回计算得到的函数值
            return f
```

# `.\cogvideo-finetune\sat\sgm\models\autoencoder.py`

```py
# 导入标准库的日志模块
import logging
# 导入数学库
import math
# 导入正则表达式库
import re
# 导入随机数生成库
import random
# 从 abc 模块导入抽象方法装饰器
from abc import abstractmethod
# 从上下文管理库导入上下文管理器装饰器
from contextlib import contextmanager
# 导入类型提示相关的类型
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch Lightning 库
import pytorch_lightning as pl
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的分布式模块
import torch.distributed
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 从 einops 库导入重排函数
from einops import rearrange
# 导入版本管理库
from packaging import version

# 从自定义模块中导入所需的类和函数
from ..modules.autoencoding.regularizers import AbstractRegularizer
from ..modules.ema import LitEma
from ..util import (
    default,  # 默认值函数
    get_nested_attribute,  # 获取嵌套属性函数
    get_obj_from_str,  # 从字符串获取对象函数
    instantiate_from_config,  # 从配置实例化对象函数
    initialize_context_parallel,  # 初始化上下文并行函数
    get_context_parallel_group,  # 获取上下文并行组函数
    get_context_parallel_group_rank,  # 获取上下文并行组的排名函数
    is_context_parallel_initialized,  # 检查上下文并行是否已初始化函数
)
from ..modules.cp_enc_dec import _conv_split, _conv_gather  # 导入卷积拆分和聚合函数

# 创建日志记录器
logpy = logging.getLogger(__name__)

# 定义抽象自编码器类，继承自 PyTorch Lightning 模块
class AbstractAutoencoder(pl.LightningModule):
    """
    这是所有自编码器的基类，包括图像自编码器、带鉴别器的图像自编码器、unCLIP 模型等。
    因此，它是相当通用的，具体特性（例如，鉴别器训练、编码、解码）必须在子类中实现。
    """

    # 初始化方法，设置自编码器的属性
    def __init__(
        self,
        ema_decay: Union[None, float] = None,  # 指定 EMA 衰减参数
        monitor: Union[None, str] = None,  # 指定监控指标
        input_key: str = "jpg",  # 输入数据的键，默认为 "jpg"
    ):
        super().__init__()  # 调用父类的初始化方法

        self.input_key = input_key  # 存储输入键
        self.use_ema = ema_decay is not None  # 检查是否使用 EMA
        if monitor is not None:  # 如果监控指标不为 None
            self.monitor = monitor  # 存储监控指标

        if self.use_ema:  # 如果使用 EMA
            self.model_ema = LitEma(self, decay=ema_decay)  # 创建 EMA 实例
            logpy.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")  # 记录 EMA 缓冲区的数量

        # 检查 PyTorch 版本是否大于或等于 2.0.0
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            self.automatic_optimization = False  # 禁用自动优化

    # 应用检查点的方法
    def apply_ckpt(self, ckpt: Union[None, str, dict]):
        if ckpt is None:  # 如果检查点为 None
            return  # 直接返回
        if isinstance(ckpt, str):  # 如果检查点是字符串
            ckpt = {
                "target": "sgm.modules.checkpoint.CheckpointEngine",  # 设置目标为检查点引擎
                "params": {"ckpt_path": ckpt},  # 设置检查点路径
            }
        engine = instantiate_from_config(ckpt)  # 根据配置实例化引擎
        engine(self)  # 将自编码器传入引擎

    @abstractmethod  # 声明此方法为抽象方法
    def get_input(self, batch) -> Any:  # 获取输入的方法
        raise NotImplementedError()  # 抛出未实现错误

    # 训练批次结束后的回调方法
    def on_train_batch_end(self, *args, **kwargs):
        # 用于 EMA 计算
        if self.use_ema:  # 如果使用 EMA
            self.model_ema(self)  # 更新 EMA

    # 定义 EMA 上下文管理器
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:  # 如果使用 EMA
            self.model_ema.store(self.parameters())  # 存储当前参数
            self.model_ema.copy_to(self)  # 复制 EMA 权重到模型
            if context is not None:  # 如果有上下文信息
                logpy.info(f"{context}: Switched to EMA weights")  # 记录切换到 EMA 权重的信息
        try:
            yield None  # 允许在上下文中执行
        finally:
            if self.use_ema:  # 如果使用 EMA
                self.model_ema.restore(self.parameters())  # 恢复模型参数
                if context is not None:  # 如果有上下文信息
                    logpy.info(f"{context}: Restored training weights")  # 记录恢复训练权重的信息

    @abstractmethod  # 声明此方法为抽象方法
    # 定义一个编码方法，接受可变参数，返回一个张量
        def encode(self, *args, **kwargs) -> torch.Tensor:
            # 抛出未实现错误，指示这是一个抽象基类的方法
            raise NotImplementedError("encode()-method of abstract base class called")
    
        # 定义一个解码方法，接受可变参数，返回一个张量
        @abstractmethod
        def decode(self, *args, **kwargs) -> torch.Tensor:
            # 抛出未实现错误，指示这是一个抽象基类的方法
            raise NotImplementedError("decode()-method of abstract base class called")
    
        # 根据配置实例化优化器，接受参数列表、学习率和配置字典
        def instantiate_optimizer_from_config(self, params, lr, cfg):
            # 记录加载优化器的目标信息
            logpy.info(f"loading >>> {cfg['target']} <<< optimizer from config")
            # 从配置中获取优化器对象，并返回实例化的优化器
            return get_obj_from_str(cfg["target"])(params, lr=lr, **cfg.get("params", dict()))
    
        # 配置优化器，返回任意类型
        def configure_optimizers(self) -> Any:
            # 抛出未实现错误，指示这是一个抽象基类的方法
            raise NotImplementedError()
# 定义图像自编码器的基类，例如 VQGAN 或 AutoencoderKL
class AutoencodingEngine(AbstractAutoencoder):
    """
    所有图像自编码器的基类，我们训练的如 VQGAN 或 AutoencoderKL
    （出于遗留原因，我们也显式恢复它们作为特例）。
    正则化如 KL 或 VQ 被移动到正则化器类中。
    """

    # 初始化自编码器
    def __init__(
        self,
        *args,
        encoder_config: Dict,  # 编码器配置字典
        decoder_config: Dict,  # 解码器配置字典
        loss_config: Dict,  # 损失函数配置字典
        regularizer_config: Dict,  # 正则化器配置字典
        optimizer_config: Union[Dict, None] = None,  # 优化器配置字典，可选
        lr_g_factor: float = 1.0,  # 学习率缩放因子
        trainable_ae_params: Optional[List[List[str]]] = None,  # 可训练的自编码器参数
        ae_optimizer_args: Optional[List[dict]] = None,  # 自编码器优化器参数
        trainable_disc_params: Optional[List[List[str]]] = None,  # 可训练的判别器参数
        disc_optimizer_args: Optional[List[dict]] = None,  # 判别器优化器参数
        disc_start_iter: int = 0,  # 判别器开始迭代的初始迭代次数
        diff_boost_factor: float = 3.0,  # 差异提升因子
        ckpt_engine: Union[None, str, dict] = None,  # 检查点引擎配置
        ckpt_path: Optional[str] = None,  # 检查点路径
        additional_decode_keys: Optional[List[str]] = None,  # 额外解码键
        **kwargs,  # 其他参数
    ):
        super().__init__(*args, **kwargs)  # 调用父类构造函数
        self.automatic_optimization = False  # 禁用自动优化，适用于 PyTorch Lightning

        # 根据配置实例化编码器
        self.encoder: torch.nn.Module = instantiate_from_config(encoder_config)
        # 根据配置实例化解码器
        self.decoder: torch.nn.Module = instantiate_from_config(decoder_config)
        # 根据配置实例化损失函数
        self.loss: torch.nn.Module = instantiate_from_config(loss_config)
        # 根据配置实例化正则化器
        self.regularization: AbstractRegularizer = instantiate_from_config(regularizer_config)
        # 设置优化器配置，默认为 Adam
        self.optimizer_config = default(optimizer_config, {"target": "torch.optim.Adam"})
        # 设置差异提升因子
        self.diff_boost_factor = diff_boost_factor
        # 设置判别器开始迭代的初始值
        self.disc_start_iter = disc_start_iter
        # 设置学习率缩放因子
        self.lr_g_factor = lr_g_factor
        # 存储可训练的自编码器参数
        self.trainable_ae_params = trainable_ae_params
        if self.trainable_ae_params is not None:
            # 设置自编码器优化器参数，默认为空字典
            self.ae_optimizer_args = default(
                ae_optimizer_args,
                [{} for _ in range(len(self.trainable_ae_params))],
            )
            # 确保优化器参数和可训练参数数量一致
            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            self.ae_optimizer_args = [{}]  # 使类型一致

        # 存储可训练的判别器参数
        self.trainable_disc_params = trainable_disc_params
        if self.trainable_disc_params is not None:
            # 设置判别器优化器参数，默认为空字典
            self.disc_optimizer_args = default(
                disc_optimizer_args,
                [{} for _ in range(len(self.trainable_disc_params))],
            )
            # 确保优化器参数和可训练参数数量一致
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            self.disc_optimizer_args = [{}]  # 使类型一致

        # 如果设置了检查点路径
        if ckpt_path is not None:
            # 确保不能同时设置检查点引擎和检查点路径
            assert ckpt_engine is None, "Can't set ckpt_engine and ckpt_path"
            logpy.warn("Checkpoint path is deprecated, use `checkpoint_egnine` instead")  # 记录警告
        # 应用检查点，默认使用给定的路径或引擎
        self.apply_ckpt(default(ckpt_path, ckpt_engine))
        # 设置额外解码键的集合
        self.additional_decode_keys = set(default(additional_decode_keys, []))
    # 获取输入数据，返回一个张量
    def get_input(self, batch: Dict) -> torch.Tensor:
        # 假设统一的数据格式，数据加载器返回一个字典
        # 图像张量应缩放到 -1 到 1，并采用通道优先格式（例如 bchw 而不是 bhwc）
        return batch[self.input_key]  # 从批次中提取指定的输入键的数据

    # 获取自动编码器的可训练参数
    def get_autoencoder_params(self) -> list:
        params = []  # 初始化参数列表
        # 检查损失对象是否具有获取可训练自动编码器参数的方法
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            # 将损失对象的可训练参数添加到参数列表
            params += list(self.loss.get_trainable_autoencoder_parameters())
        # 检查正则化对象是否具有获取可训练参数的方法
        if hasattr(self.regularization, "get_trainable_parameters"):
            # 将正则化对象的可训练参数添加到参数列表
            params += list(self.regularization.get_trainable_parameters())
        # 添加编码器的参数到参数列表
        params = params + list(self.encoder.parameters())
        # 添加解码器的参数到参数列表
        params = params + list(self.decoder.parameters())
        return params  # 返回所有可训练参数的列表

    # 获取判别器的可训练参数
    def get_discriminator_params(self) -> list:
        # 检查损失对象是否具有获取可训练参数的方法
        if hasattr(self.loss, "get_trainable_parameters"):
            # 获取并返回损失对象的可训练参数（例如，判别器）
            params = list(self.loss.get_trainable_parameters())
        else:
            params = []  # 如果没有，初始化为空列表
        return params  # 返回判别器的可训练参数列表

    # 获取解码器的最后一层
    def get_last_layer(self):
        return self.decoder.get_last_layer()  # 返回解码器的最后一层

    # 编码输入张量并可选择返回正则化日志
    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x, **kwargs)  # 使用编码器对输入进行编码
        if unregularized:
            return z, dict()  # 如果未正则化，返回编码结果和空字典
        z, reg_log = self.regularization(z)  # 对编码结果进行正则化，并获取日志
        if return_reg_log:
            return z, reg_log  # 如果要求返回正则化日志，返回编码结果和日志
        return z  # 返回编码结果

    # 解码输入张量
    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.decoder(z, **kwargs)  # 使用解码器对编码结果进行解码
        return x  # 返回解码后的结果

    # 前向传播方法，处理输入并返回编码、解码结果和正则化日志
    def forward(self, x: torch.Tensor, **additional_decode_kwargs) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True)  # 编码输入并要求返回正则化日志
        dec = self.decode(z, **additional_decode_kwargs)  # 解码编码结果
        return z, dec, reg_log  # 返回编码结果、解码结果和正则化日志
    # 定义内部训练步骤方法，接受批次数据、批次索引和优化器索引
        def inner_training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0) -> torch.Tensor:
            # 从批次中获取输入数据
            x = self.get_input(batch)
            # 创建额外解码参数字典，包含批次中额外的解码键
            additional_decode_kwargs = {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
            # 执行前向传播，得到潜在变量 z、重构的输入 xrec 和正则化日志
            z, xrec, regularization_log = self(x, **additional_decode_kwargs)
            # 检查损失对象是否有 forward_keys 属性
            if hasattr(self.loss, "forward_keys"):
                # 构建额外信息字典，包括潜在变量和训练相关信息
                extra_info = {
                    "z": z,
                    "optimizer_idx": optimizer_idx,
                    "global_step": self.global_step,
                    "last_layer": self.get_last_layer(),
                    "split": "train",
                    "regularization_log": regularization_log,
                    "autoencoder": self,
                }
                # 仅保留在损失对象中定义的额外信息键
                extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
            else:
                # 初始化额外信息为空字典
                extra_info = dict()
    
            # 检查优化器索引，如果是第一个优化器
            if optimizer_idx == 0:
                # 计算自编码器损失
                out_loss = self.loss(x, xrec, **extra_info)
                # 检查损失是否为元组，分解损失和日志字典
                if isinstance(out_loss, tuple):
                    aeloss, log_dict_ae = out_loss
                else:
                    # 简单损失函数，初始化损失和日志字典
                    aeloss = out_loss
                    log_dict_ae = {"train/loss/rec": aeloss.detach()}
    
                # 记录字典中的损失信息
                self.log_dict(
                    log_dict_ae,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=False,
                )
                # 在进度条上记录平均损失
                self.log(
                    "loss",
                    aeloss.mean().detach(),
                    prog_bar=True,
                    logger=False,
                    on_epoch=False,
                    on_step=True,
                )
                # 返回自编码器损失
                return aeloss
            # 如果是第二个优化器
            elif optimizer_idx == 1:
                # 计算判别器损失
                discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
                # 判别器总是需要返回一个元组
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                # 返回判别器损失
                return discloss
            else:
                # 抛出未实现的错误，表示未知的优化器索引
                raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")
    
        # 定义训练步骤方法，接受批次数据和批次索引
        def training_step(self, batch: dict, batch_idx: int):
            # 获取当前模型的优化器
            opts = self.optimizers()
            # 检查优化器是否为列表，如果不是则将其转为列表
            if not isinstance(opts, list):
                # 非对抗情况，将优化器放入列表
                opts = [opts]
            # 根据批次索引计算当前优化器的索引
            optimizer_idx = batch_idx % len(opts)
            # 如果全局步骤小于判别器开始迭代，则使用第一个优化器
            if self.global_step < self.disc_start_iter:
                optimizer_idx = 0
            # 选择当前优化器
            opt = opts[optimizer_idx]
            # 将优化器的梯度置为零
            opt.zero_grad()
            # 在优化器的模型切换上下文中执行
            with opt.toggle_model():
                # 调用内部训练步骤，计算损失
                loss = self.inner_training_step(batch, batch_idx, optimizer_idx=optimizer_idx)
                # 手动进行反向传播
                self.manual_backward(loss)
            # 更新优化器的参数
            opt.step()
    
        # 定义验证步骤方法，接受批次数据和批次索引
        def validation_step(self, batch: dict, batch_idx: int) -> Dict:
            # 执行基本的验证步骤，获取日志字典
            log_dict = self._validation_step(batch, batch_idx)
            # 在 EMA（指数移动平均）上下文中执行验证步骤
            with self.ema_scope():
                # 获取 EMA 验证步骤的日志字典
                log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
                # 更新日志字典，合并 EMA 结果
                log_dict.update(log_dict_ema)
            # 返回合并后的日志字典
            return log_dict
    # 定义验证步骤的方法，接受一个批次的数据、批次索引和可选的后缀
    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> Dict:
        # 从批次数据中获取输入
        x = self.get_input(batch)
    
        # 前向传播，得到潜在变量 z、重建的输入 xrec 和正则化日志
        z, xrec, regularization_log = self(x)
        # 检查损失对象是否有前向键
        if hasattr(self.loss, "forward_keys"):
            # 构建额外信息字典，包含多个状态信息
            extra_info = {
                "z": z,  # 潜在变量
                "optimizer_idx": 0,  # 优化器索引初始化为 0
                "global_step": self.global_step,  # 全局步数
                "last_layer": self.get_last_layer(),  # 获取最后一层的输出
                "split": "val" + postfix,  # 验证数据集的标识
                "regularization_log": regularization_log,  # 正则化日志
                "autoencoder": self,  # 自编码器对象
            }
            # 仅保留损失对象中定义的前向键
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            # 如果没有前向键，初始化为空字典
            extra_info = dict()
        # 计算损失值
        out_loss = self.loss(x, xrec, **extra_info)
        # 检查损失值是否为元组
        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss  # 解包自编码器损失和日志字典
        else:
            # 简单的损失函数处理
            aeloss = out_loss  # 直接将损失赋值给 aeloss
            log_dict_ae = {f"val{postfix}/loss/rec": aeloss.detach()}  # 创建日志字典
        full_log_dict = log_dict_ae  # 初始化完整日志字典
    
        # 如果额外信息中有优化器索引
        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1  # 更新优化器索引
            # 计算判别器损失
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            # 更新完整日志字典，包含判别器的日志
            full_log_dict.update(log_dict_disc)
        # 记录重建损失
        self.log(
            f"val{postfix}/loss/rec",  # 日志名称
            log_dict_ae[f"val{postfix}/loss/rec"],  # 日志值
            sync_dist=True,  # 进行分布式同步
        )
        # 记录完整日志字典
        self.log_dict(full_log_dict, sync_dist=True)
        # 返回完整日志字典
        return full_log_dict
    
    # 定义获取参数组的方法，接受参数名称列表和优化器参数列表
    def get_param_groups(
        self, parameter_names: List[List[str]], optimizer_args: List[dict]
    ) -> Tuple[List[Dict[str, Any]], int]:
        groups = []  # 初始化参数组列表
        num_params = 0  # 初始化参数数量计数器
        # 遍历参数名称和优化器参数
        for names, args in zip(parameter_names, optimizer_args):
            params = []  # 初始化当前参数列表
            # 遍历每个参数模式
            for pattern_ in names:
                pattern_params = []  # 初始化匹配到的参数列表
                pattern = re.compile(pattern_)  # 编译正则表达式模式
                # 遍历命名参数
                for p_name, param in self.named_parameters():
                    # 检查参数名称是否与模式匹配
                    if re.match(pattern, p_name):
                        pattern_params.append(param)  # 添加匹配的参数
                        num_params += param.numel()  # 更新参数数量计数
                # 如果没有找到匹配的参数，发出警告
                if len(pattern_params) == 0:
                    logpy.warn(f"Did not find parameters for pattern {pattern_}")
                params.extend(pattern_params)  # 扩展当前参数列表
            # 将当前参数及其参数设置添加到参数组中
            groups.append({"params": params, **args})
        # 返回参数组和参数数量
        return groups, num_params
    # 配置优化器，返回优化器列表
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        # 检查自编码器可训练参数是否为 None
        if self.trainable_ae_params is None:
            # 获取自编码器参数
            ae_params = self.get_autoencoder_params()
        else:
            # 获取指定的参数组及其数量
            ae_params, num_ae_params = self.get_param_groups(self.trainable_ae_params, self.ae_optimizer_args)
            # 记录可训练自编码器参数的数量
            logpy.info(f"Number of trainable autoencoder parameters: {num_ae_params:,}")
        # 检查鉴别器可训练参数是否为 None
        if self.trainable_disc_params is None:
            # 获取鉴别器参数
            disc_params = self.get_discriminator_params()
        else:
            # 获取指定的参数组及其数量
            disc_params, num_disc_params = self.get_param_groups(self.trainable_disc_params, self.disc_optimizer_args)
            # 记录可训练鉴别器参数的数量
            logpy.info(f"Number of trainable discriminator parameters: {num_disc_params:,}")
        # 根据自编码器参数和学习率配置实例化优化器
        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        # 初始化优化器列表
        opts = [opt_ae]
        # 如果鉴别器参数不为空，则实例化鉴别器优化器
        if len(disc_params) > 0:
            opt_disc = self.instantiate_optimizer_from_config(disc_params, self.learning_rate, self.optimizer_config)
            opts.append(opt_disc)
        # 返回优化器列表
        return opts
    
    # 不计算梯度，记录图像
    @torch.no_grad()
    def log_images(self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs) -> dict:
        # 初始化日志字典
        log = dict()
        # 初始化额外解码参数字典
        additional_decode_kwargs = {}
        # 从批次中获取输入
        x = self.get_input(batch)
        # 更新额外解码参数字典
        additional_decode_kwargs.update({key: batch[key] for key in self.additional_decode_keys.intersection(batch)})
        
        # 获取输入的重构结果
        _, xrec, _ = self(x, **additional_decode_kwargs)
        # 记录输入
        log["inputs"] = x
        # 记录重构结果
        log["reconstructions"] = xrec
        # 计算输入与重构之间的差异
        diff = 0.5 * torch.abs(torch.clamp(xrec, -1.0, 1.0) - x)
        # 将差异值限制在 [0, 1] 范围内
        diff.clamp_(0, 1.0)
        # 记录差异值
        log["diff"] = 2.0 * diff - 1.0
        # 通过增强小误差的亮度来显示误差位置
        log["diff_boost"] = 2.0 * torch.clamp(self.diff_boost_factor * diff, 0.0, 1.0) - 1
        # 如果损失对象有 log_images 方法，则更新日志
        if hasattr(self.loss, "log_images"):
            log.update(self.loss.log_images(x, xrec))
        # 进入 EMA 作用域
        with self.ema_scope():
            # 获取 EMA 重构结果
            _, xrec_ema, _ = self(x, **additional_decode_kwargs)
            # 记录 EMA 重构结果
            log["reconstructions_ema"] = xrec_ema
            # 计算 EMA 输入与重构之间的差异
            diff_ema = 0.5 * torch.abs(torch.clamp(xrec_ema, -1.0, 1.0) - x)
            # 将差异值限制在 [0, 1] 范围内
            diff_ema.clamp_(0, 1.0)
            # 记录 EMA 差异值
            log["diff_ema"] = 2.0 * diff_ema - 1.0
            # 记录 EMA 差异增强值
            log["diff_boost_ema"] = 2.0 * torch.clamp(self.diff_boost_factor * diff_ema, 0.0, 1.0) - 1
        # 如果有额外的日志参数，则进行处理
        if additional_log_kwargs:
            additional_decode_kwargs.update(additional_log_kwargs)
            # 获取额外重构结果
            _, xrec_add, _ = self(x, **additional_decode_kwargs)
            # 构造日志字符串
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            # 记录额外重构结果
            log[log_str] = xrec_add
        # 返回日志字典
        return log
# 定义一个继承自 AutoencodingEngine 的类 AutoencodingEngineLegacy
class AutoencodingEngineLegacy(AutoencodingEngine):
    # 初始化方法，接受嵌入维度和其他关键字参数
    def __init__(self, embed_dim: int, **kwargs):
        # 从 kwargs 中提取最大批次大小，默认值为 None
        self.max_batch_size = kwargs.pop("max_batch_size", None)
        # 从 kwargs 中提取 ddconfig 参数
        ddconfig = kwargs.pop("ddconfig")
        # 从 kwargs 中提取检查点路径，默认值为 None
        ckpt_path = kwargs.pop("ckpt_path", None)
        # 从 kwargs 中提取检查点引擎，默认值为 None
        ckpt_engine = kwargs.pop("ckpt_engine", None)
        # 调用父类构造函数，设置编码器和解码器配置
        super().__init__(
            encoder_config={
                "target": "sgm.modules.diffusionmodules.model.Encoder",
                "params": ddconfig,
            },
            decoder_config={
                "target": "sgm.modules.diffusionmodules.model.Decoder",
                "params": ddconfig,
            },
            **kwargs,
        )
        # 定义量化卷积层，输入通道数由 ddconfig 决定
        self.quant_conv = torch.nn.Conv2d(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],
            (1 + ddconfig["double_z"]) * embed_dim,
            1,
        )
        # 定义后量化卷积层，输出通道数为 z_channels
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # 保存嵌入维度
        self.embed_dim = embed_dim
        # 应用检查点，初始化模型状态
        self.apply_ckpt(default(ckpt_path, ckpt_engine))

    # 获取自动编码器参数
    def get_autoencoder_params(self) -> list:
        # 调用父类方法获取参数
        params = super().get_autoencoder_params()
        return params

    # 编码输入张量，返回编码结果和可选的正则化日志
    def encode(self, x: torch.Tensor, return_reg_log: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        # 如果没有设置最大批次大小，直接编码并量化
        if self.max_batch_size is None:
            z = self.encoder(x)
            z = self.quant_conv(z)
        else:
            # 获取输入张量的批次大小
            N = x.shape[0]
            bs = self.max_batch_size
            # 计算需要的批次数量
            n_batches = int(math.ceil(N / bs))
            z = list()
            # 遍历每个批次进行编码和量化
            for i_batch in range(n_batches):
                z_batch = self.encoder(x[i_batch * bs : (i_batch + 1) * bs])
                z_batch = self.quant_conv(z_batch)
                z.append(z_batch)
            # 将所有批次的结果连接成一个张量
            z = torch.cat(z, 0)

        # 应用正则化到编码结果
        z, reg_log = self.regularization(z)
        # 根据参数决定返回结果
        if return_reg_log:
            return z, reg_log
        return z

    # 解码输入张量
    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        # 如果没有设置最大批次大小，直接解码
        if self.max_batch_size is None:
            dec = self.post_quant_conv(z)
            dec = self.decoder(dec, **decoder_kwargs)
        else:
            # 获取输入张量的批次大小
            N = z.shape[0]
            bs = self.max_batch_size
            # 计算需要的批次数量
            n_batches = int(math.ceil(N / bs))
            dec = list()
            # 遍历每个批次进行解码
            for i_batch in range(n_batches):
                dec_batch = self.post_quant_conv(z[i_batch * bs : (i_batch + 1) * bs])
                dec_batch = self.decoder(dec_batch, **decoder_kwargs)
                dec.append(dec_batch)
            # 将所有批次的结果连接成一个张量
            dec = torch.cat(dec, 0)

        # 返回解码结果
        return dec


# 定义一个继承自 AbstractAutoencoder 的类 IdentityFirstStage
class IdentityFirstStage(AbstractAutoencoder):
    # 初始化方法，调用父类构造函数
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 获取输入，返回原始输入
    def get_input(self, x: Any) -> Any:
        return x

    # 编码方法，返回原始输入
    def encode(self, x: Any, *args, **kwargs) -> Any:
        return x

    # 解码方法，什么也不返回
    def decode(self, x: Any, *args, **kwargs) -> Any:
        return


# 定义一个继承自 AutoencodingEngine 的类 VideoAutoencodingEngine
class VideoAutoencodingEngine(AutoencodingEngine):
    # 初始化方法，用于设置模型参数
        def __init__(
            self,
            ckpt_path: Union[None, str] = None,  # 可选的检查点路径，用于加载模型
            ignore_keys: Union[Tuple, list] = (),  # 指定忽略的键列表
            image_video_weights=[1, 1],  # 图像和视频的权重设置
            only_train_decoder=False,  # 仅训练解码器的标志
            context_parallel_size=0,  # 上下文并行的大小
            **kwargs,  # 其他额外参数
        ):
            super().__init__(**kwargs)  # 调用父类的初始化方法
            self.context_parallel_size = context_parallel_size  # 保存上下文并行大小
            if ckpt_path is not None:  # 如果提供了检查点路径
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)  # 从检查点初始化模型
    
        # 日志记录视频的方法，接受一个批次和额外的日志参数
        def log_videos(self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs) -> dict:
            return self.log_images(batch, additional_log_kwargs, **kwargs)  # 调用 log_images 方法记录视频
    
        # 获取输入数据的方法
        def get_input(self, batch: dict) -> torch.Tensor:
            if self.context_parallel_size > 0:  # 如果上下文并行大小大于0
                if not is_context_parallel_initialized():  # 检查上下文并行是否已初始化
                    initialize_context_parallel(self.context_parallel_size)  # 初始化上下文并行
    
                batch = batch[self.input_key]  # 获取指定键的批次数据
    
                global_src_rank = get_context_parallel_group_rank() * self.context_parallel_size  # 计算全局源排名
                torch.distributed.broadcast(batch, src=global_src_rank, group=get_context_parallel_group())  # 广播批次数据
    
                batch = _conv_split(batch, dim=2, kernel_size=1)  # 在指定维度上分割批次数据
                return batch  # 返回处理后的批次数据
    
            return batch[self.input_key]  # 返回指定键的批次数据
    
        # 应用检查点的方法
        def apply_ckpt(self, ckpt: Union[None, str, dict]):
            if ckpt is None:  # 如果检查点为 None
                return  # 直接返回
            self.init_from_ckpt(ckpt)  # 从检查点初始化模型
    
        # 从检查点初始化模型的方法
        def init_from_ckpt(self, path, ignore_keys=list()):
            sd = torch.load(path, map_location="cpu")["state_dict"]  # 加载检查点的状态字典
            keys = list(sd.keys())  # 获取状态字典中的所有键
            for k in keys:  # 遍历所有键
                for ik in ignore_keys:  # 遍历忽略键列表
                    if k.startswith(ik):  # 如果键以忽略键开头
                        del sd[k]  # 删除该键
    
            missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)  # 加载状态字典，允许不严格匹配
            print("Missing keys: ", missing_keys)  # 打印缺失的键
            print("Unexpected keys: ", unexpected_keys)  # 打印意外的键
            print(f"Restored from {path}")  # 打印恢复信息
```

# `.\cogvideo-finetune\sat\sgm\models\__init__.py`

```py
# 从当前包中导入 AutoencodingEngine 类
from .autoencoder import AutoencodingEngine
```

# `.\cogvideo-finetune\sat\sgm\modules\attention.py`

```py
# 导入数学库
import math
# 从 inspect 模块导入 isfunction 函数，用于检查对象是否为函数
from inspect import isfunction
# 导入 Any 和 Optional 类型注解
from typing import Any, Optional

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能模块
import torch.nn.functional as F
# 从 einops 导入 rearrange 和 repeat 函数，用于张量操作
from einops import rearrange, repeat
# 导入版本管理模块
from packaging import version
# 导入 PyTorch 的神经网络模块
from torch import nn

# 检查 PyTorch 版本是否大于等于 2.0.0
if version.parse(torch.__version__) >= version.parse("2.0.0"):
    # 如果版本合适，设置 SDP 可用为 True
    SDP_IS_AVAILABLE = True
    # 从 CUDA 后端导入 SDPBackend 和 sdp_kernel
    from torch.backends.cuda import SDPBackend, sdp_kernel

    # 定义后端配置映射
    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
# 如果版本低于 2.0.0
else:
    # 从上下文管理器导入 nullcontext
    from contextlib import nullcontext

    # 设置 SDP 可用为 False
    SDP_IS_AVAILABLE = False
    # 设置 sdp_kernel 为 nullcontext
    sdp_kernel = nullcontext
    # 打印提示信息，建议升级 PyTorch
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )

# 尝试导入 xformers 和其操作模块
try:
    import xformers
    import xformers.ops

    # 如果导入成功，设置 XFORMERS 可用为 True
    XFORMERS_IS_AVAILABLE = True
# 如果导入失败
except:
    # 设置 XFORMERS 可用为 False
    XFORMERS_IS_AVAILABLE = False
    # 打印提示信息，说明处理将不使用 xformers
    print("no module 'xformers'. Processing without...")

# 从本地模块导入 checkpoint 函数
from .diffusionmodules.util import checkpoint


# 定义 exists 函数，检查值是否存在
def exists(val):
    return val is not None


# 定义 uniq 函数，返回数组中的唯一元素
def uniq(arr):
    return {el: True for el in arr}.keys()


# 定义 default 函数，返回值或默认值
def default(val, d):
    if exists(val):
        return val
    # 如果默认值是函数，调用它并返回结果，否则返回默认值
    return d() if isfunction(d) else d


# 定义 max_neg_value 函数，返回张量类型的最大负值
def max_neg_value(t):
    return -torch.finfo(t.dtype).max


# 定义 init_ 函数，用于初始化张量
def init_(tensor):
    dim = tensor.shape[-1]  # 获取张量的最后一维大小
    std = 1 / math.sqrt(dim)  # 计算标准差
    tensor.uniform_(-std, std)  # 用均匀分布初始化张量
    return tensor


# 定义 GEGLU 类，继承自 nn.Module
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()  # 调用父类构造函数
        # 定义一个线性层，将输入维度映射到输出维度的两倍
        self.proj = nn.Linear(dim_in, dim_out * 2)

    # 定义前向传播方法
    def forward(self, x):
        # 将输入通过线性层，并将输出分为两个部分
        x, gate = self.proj(x).chunk(2, dim=-1)
        # 返回 x 与 gate 的 GELU 激活函数的乘积
        return x * F.gelu(gate)


# 定义 FeedForward 类，继承自 nn.Module
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()  # 调用父类构造函数
        inner_dim = int(dim * mult)  # 计算内部维度
        dim_out = default(dim_out, dim)  # 获取输出维度，若未提供则使用输入维度
        # 根据是否使用 GLU 选择不同的输入项目
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        # 定义完整的网络结构
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    # 定义前向传播方法
    def forward(self, x):
        return self.net(x)  # 返回网络的输出


# 定义 zero_module 函数，将模块的参数归零并返回模块
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    # 遍历模块的参数并将其归零
    for p in module.parameters():
        p.detach().zero_()
    return module  # 返回归零后的模块


# 定义 Normalize 函数，返回一个 GroupNorm 层
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# 定义 LinearAttention 类，继承自 nn.Module
class LinearAttention(nn.Module):
    # 初始化方法，设置维度、头数和每个头的维度
        def __init__(self, dim, heads=4, dim_head=32):
            # 调用父类初始化方法
            super().__init__()
            # 保存头数
            self.heads = heads
            # 计算隐藏层维度，等于每个头的维度乘以头数
            hidden_dim = dim_head * heads
            # 定义一个卷积层，将输入维度转换为三倍的隐藏层维度，不使用偏置
            self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
            # 定义输出卷积层，将隐藏层维度转换为原始输入维度
            self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    
        # 前向传播方法
        def forward(self, x):
            # 获取输入的批量大小、通道数、高度和宽度
            b, c, h, w = x.shape
            # 通过卷积层转换输入，得到查询、键、值
            qkv = self.to_qkv(x)
            # 重排张量，使得查询、键、值分别分开并调整维度
            q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
            # 对键进行softmax处理，归一化
            k = k.softmax(dim=-1)
            # 计算上下文，使用爱因斯坦求和约定对键和值进行操作
            context = torch.einsum("bhdn,bhen->bhde", k, v)
            # 结合上下文和查询计算输出
            out = torch.einsum("bhde,bhdn->bhen", context, q)
            # 重排输出以适应原始维度
            out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
            # 通过输出卷积层得到最终结果
            return self.to_out(out)
# 定义一个空间自注意力类，继承自 nn.Module
class SpatialSelfAttention(nn.Module):
    # 初始化函数，接收输入通道数
    def __init__(self, in_channels):
        # 调用父类的初始化函数
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels

        # 创建归一化层
        self.norm = Normalize(in_channels)
        # 创建查询卷积层
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建键卷积层
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建值卷积层
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建输出投影卷积层
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播函数
    def forward(self, x):
        # 将输入赋值给 h_
        h_ = x
        # 对输入进行归一化处理
        h_ = self.norm(h_)
        # 计算查询
        q = self.q(h_)
        # 计算键
        k = self.k(h_)
        # 计算值
        v = self.v(h_)

        # 计算注意力
        b, c, h, w = q.shape
        # 重排查询张量
        q = rearrange(q, "b c h w -> b (h w) c")
        # 重排键张量
        k = rearrange(k, "b c h w -> b c (h w)")
        # 计算注意力权重
        w_ = torch.einsum("bij,bjk->bik", q, k)

        # 缩放权重
        w_ = w_ * (int(c) ** (-0.5))
        # 对权重应用 softmax
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # 关注值
        # 重排值张量
        v = rearrange(v, "b c h w -> b c (h w)")
        # 重排权重张量
        w_ = rearrange(w_, "b i j -> b j i")
        # 计算最终输出
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        # 重排输出张量
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        # 应用输出投影层
        h_ = self.proj_out(h_)

        # 返回原始输入与输出的和
        return x + h_


# 定义一个交叉注意力类，继承自 nn.Module
class CrossAttention(nn.Module):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 计算内部维度
        inner_dim = dim_head * heads
        # 确定上下文维度
        context_dim = default(context_dim, query_dim)

        # 设置缩放因子
        self.scale = dim_head**-0.5
        # 保存头的数量
        self.heads = heads

        # 创建查询线性层
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # 创建键线性层
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # 创建值线性层
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 创建输出层，包含线性层和 dropout
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        # 保存后端配置
        self.backend = backend

    # 前向传播函数
    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    # 函数体开始
        ):
            # 获取当前对象的头部数量
            h = self.heads
    
            # 检查是否有附加令牌
            if additional_tokens is not None:
                # 获取输出序列开始部分的掩码令牌数量
                n_tokens_to_mask = additional_tokens.shape[1]
                # 将附加令牌与输入 x 拼接在一起
                x = torch.cat([additional_tokens, x], dim=1)
    
            # 将输入 x 转换为查询向量
            q = self.to_q(x)
            # 使用默认值确保上下文不为空
            context = default(context, x)
            # 将上下文转换为键向量
            k = self.to_k(context)
            # 将上下文转换为值向量
            v = self.to_v(context)
    
            # 检查是否需要跨帧注意力机制
            if n_times_crossframe_attn_in_self:
                # 按照论文中的方法重新编程跨帧注意力
                assert x.shape[0] % n_times_crossframe_attn_in_self == 0
                # 计算每个交叉帧的数量
                n_cp = x.shape[0] // n_times_crossframe_attn_in_self
                # 重复键向量以适应新的形状
                k = repeat(k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)
                # 重复值向量以适应新的形状
                v = repeat(v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)
    
            # 重新排列查询、键和值向量
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
    
            # 旧的注意力机制代码
            """
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
            del q, k
    
            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)
    
            # 注意力机制
            sim = sim.softmax(dim=-1)
    
            out = einsum('b i j, b j d -> b i d', sim, v)
            """
            # 新的注意力机制实现
            with sdp_kernel(**BACKEND_MAP[self.backend]):
                # 打印当前后端信息及 q/k/v 的形状
                # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
                # 计算缩放点积注意力
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # scale 是 dim_head ** -0.5
    
            # 删除查询、键和值向量以释放内存
            del q, k, v
            # 重新排列输出
            out = rearrange(out, "b h n d -> b n (h d)", h=h)
    
            # 如果有附加令牌，则移除它们
            if additional_tokens is not None:
                out = out[:, n_tokens_to_mask:]
            # 返回最终输出
            return self.to_out(out)
# 定义一个内存高效的交叉注意力层，继承自 nn.Module
class MemoryEfficientCrossAttention(nn.Module):
    # 初始化方法，设置各个参数
    # query_dim: 查询向量的维度
    # context_dim: 上下文向量的维度，默认为 None
    # heads: 注意力头的数量，默认为 8
    # dim_head: 每个注意力头的维度，默认为 64
    # dropout: dropout 概率，默认为 0.0
    # kwargs: 其他额外的关键字参数
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs):
        # 调用父类的初始化方法
        super().__init__()
        # 打印模型的设置情况，包括查询维度、上下文维度、头数和每个头的维度
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        # 计算内部维度，即每个头的维度乘以头的数量
        inner_dim = dim_head * heads
        # 如果 context_dim 为 None，使用 query_dim 作为上下文维度
        context_dim = default(context_dim, query_dim)

        # 保存头的数量和每个头的维度
        self.heads = heads
        self.dim_head = dim_head

        # 定义将查询向量映射到内部维度的线性层，不使用偏置
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # 定义将上下文向量映射到内部维度的线性层，不使用偏置
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # 定义将上下文向量映射到内部维度的线性层，不使用偏置
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 定义输出层，由线性层和 dropout 组成
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        # 初始化注意力操作为 None，稍后可能会被赋值
        self.attention_op: Optional[Any] = None

    # 定义前向传播方法
    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        # 检查是否存在附加的令牌
        if additional_tokens is not None:
            # 获取输出序列开头被掩码的令牌数量
            n_tokens_to_mask = additional_tokens.shape[1]
            # 将附加令牌与输入张量拼接
            x = torch.cat([additional_tokens, x], dim=1)
        # 将输入张量转换为查询向量
        q = self.to_q(x)
        # 默认上下文为输入张量
        context = default(context, x)
        # 将上下文转换为键向量
        k = self.to_k(context)
        # 将上下文转换为值向量
        v = self.to_v(context)

        # 如果需要进行跨帧注意力的次数
        if n_times_crossframe_attn_in_self:
            # 重新编程跨帧注意力，参考文献 https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # 计算每个批次中的重复次数
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            # 以跨帧的步长重复键向量
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            # 以跨帧的步长重复值向量
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        # 获取查询向量的形状信息
        b, _, _ = q.shape
        # 对查询、键和值向量进行转换和重塑
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # 实际计算注意力机制
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        # TODO: 将此直接用于注意力操作，作为偏差
        if exists(mask):
            raise NotImplementedError
        # 对输出进行重塑以适应后续处理
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        # 如果存在附加的令牌，移除它们
        if additional_tokens is not None:
            out = out[:, n_tokens_to_mask:]
        # 返回最终输出
        return self.to_out(out)
# 定义一个基础的变换器块，继承自 nn.Module
class BasicTransformerBlock(nn.Module):
    # 定义可用的注意力模式及其对应的类
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # 标准注意力
        "softmax-xformers": MemoryEfficientCrossAttention,  # 高效的注意力实现
    }

    # 初始化方法，设置基本参数
    def __init__(
        self,
        dim,  # 输入维度
        n_heads,  # 注意力头数
        d_head,  # 每个注意力头的维度
        dropout=0.0,  # dropout 概率
        context_dim=None,  # 上下文维度，可选
        gated_ff=True,  # 是否使用门控前馈网络
        checkpoint=True,  # 是否使用检查点
        disable_self_attn=False,  # 是否禁用自注意力
        attn_mode="softmax",  # 选择的注意力模式
        sdp_backend=None,  # 后端设置
    ):
        # 调用父类构造函数
        super().__init__()
        # 确保选择的注意力模式在可用模式中
        assert attn_mode in self.ATTENTION_MODES
        # 如果选择的模式不是软最大并且 xformers 不可用，回退到标准注意力
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"  # 回退到软最大模式
        # 如果选择的是软最大且不支持，则给出警告并回退
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print("We do not support vanilla attention anymore, as it is too expensive. Sorry.")
            # 如果 xformers 不可用，抛出错误
            if not XFORMERS_IS_AVAILABLE:
                assert False, "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                print("Falling back to xformers efficient attention.")  # 回退到 xformers 的高效注意力
                attn_mode = "softmax-xformers"
        # 根据选择的模式获取注意力类
        attn_cls = self.ATTENTION_MODES[attn_mode]
        # 检查 PyTorch 版本，如果是 2.0 或更高版本，检查 sdp_backend
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None  # 在低版本中 sdp_backend 必须为 None
        self.disable_self_attn = disable_self_attn  # 设置自注意力禁用标志
        # 创建第一个注意力层，如果禁用自注意力，则 context_dim 为 None
        self.attn1 = attn_cls(
            query_dim=dim,  # 查询的维度
            heads=n_heads,  # 注意力头数
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # dropout 概率
            context_dim=context_dim if self.disable_self_attn else None,  # 上下文维度
            backend=sdp_backend,  # 后端设置
        )  # 如果禁用自注意力，则为自注意力
        # 创建前馈网络
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # 创建第二个注意力层
        self.attn2 = attn_cls(
            query_dim=dim,  # 查询的维度
            context_dim=context_dim,  # 上下文维度
            heads=n_heads,  # 注意力头数
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # dropout 概率
            backend=sdp_backend,  # 后端设置
        )  # 如果上下文为 None 则为自注意力
        # 创建三个 LayerNorm 层
        self.norm1 = nn.LayerNorm(dim)  # 第一个归一化层
        self.norm2 = nn.LayerNorm(dim)  # 第二个归一化层
        self.norm3 = nn.LayerNorm(dim)  # 第三个归一化层
        self.checkpoint = checkpoint  # 设置检查点标志
        # 如果启用检查点，打印信息
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")  # 输出当前类使用检查点的提示
    # 定义前向传播方法，接收输入数据及可选的上下文和附加标记
    def forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        # 将输入数据封装到字典中，方便后续处理
        kwargs = {"x": x}
    
        # 如果提供了上下文，则将其添加到字典中
        if context is not None:
            kwargs.update({"context": context})
    
        # 如果提供了附加标记，则将其添加到字典中
        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})
    
        # 如果提供了跨帧自注意力的次数，则将其添加到字典中
        if n_times_crossframe_attn_in_self:
            kwargs.update({"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self})
    
        # 使用检查点机制进行前向传播，并返回结果
        # return mixed_checkpoint(self._forward, kwargs, self.parameters(), self.checkpoint)
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
    
    # 定义实际的前向传播实现，处理输入数据及其他参数
    def _forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        # 通过第一个注意力层处理输入数据，应用归一化，并考虑上下文和其他参数
        x = (
            self.attn1(
                self.norm1(x),  # 对输入进行归一化处理
                context=context if self.disable_self_attn else None,  # 根据条件决定是否使用上下文
                additional_tokens=additional_tokens,  # 传递附加标记
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self if not self.disable_self_attn else 0,  # 根据条件设置参数
            )
            + x  # 将注意力层的输出与原输入相加，进行残差连接
        )
        # 通过第二个注意力层处理数据，并应用上下文和附加标记
        x = self.attn2(self.norm2(x), context=context, additional_tokens=additional_tokens) + x  # 残差连接
        # 通过前馈网络处理数据，并应用归一化
        x = self.ff(self.norm3(x)) + x  # 残差连接
        # 返回最终的处理结果
        return x
# 定义一个基本的单层变换器块，继承自 nn.Module
class BasicTransformerSingleLayerBlock(nn.Module):
    # 定义支持的注意力模式及其对应的实现类
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # 普通的注意力机制
        "softmax-xformers": MemoryEfficientCrossAttention,  # 在 A100 上不如上述版本快
        # (todo 可能依赖于 head_dim，待检查，对于 dim!=[16,32,64,128] 回退到半优化内核)
    }

    # 初始化函数，设置变换器的参数
    def __init__(
        self,
        dim,  # 特征维度
        n_heads,  # 注意力头的数量
        d_head,  # 每个注意力头的维度
        dropout=0.0,  # Dropout 概率
        context_dim=None,  # 上下文的维度（可选）
        gated_ff=True,  # 是否使用门控前馈网络
        checkpoint=True,  # 是否使用检查点以节省内存
        attn_mode="softmax",  # 使用的注意力模式
    ):
        # 调用父类构造函数
        super().__init__()
        # 确保传入的注意力模式是有效的
        assert attn_mode in self.ATTENTION_MODES
        # 根据注意力模式选择对应的注意力类
        attn_cls = self.ATTENTION_MODES[attn_mode]
        # 初始化注意力层
        self.attn1 = attn_cls(
            query_dim=dim,  # 查询的维度
            heads=n_heads,  # 注意力头数量
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # Dropout 概率
            context_dim=context_dim,  # 上下文维度
        )
        # 初始化前馈网络
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # 初始化层归一化
        self.norm1 = nn.LayerNorm(dim)  # 第一个归一化层
        self.norm2 = nn.LayerNorm(dim)  # 第二个归一化层
        # 保存检查点标志
        self.checkpoint = checkpoint

    # 前向传播函数
    def forward(self, x, context=None):
        # 使用检查点机制调用 _forward 函数
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    # 内部前向传播实现
    def _forward(self, x, context=None):
        # 应用注意力层并添加残差连接
        x = self.attn1(self.norm1(x), context=context) + x
        # 应用前馈网络并添加残差连接
        x = self.ff(self.norm2(x)) + x
        # 返回处理后的数据
        return x


# 定义一个空间变换器类，继承自 nn.Module
class SpatialTransformer(nn.Module):
    """
    用于图像数据的变换器块。
    首先，对输入进行投影（即嵌入）
    然后重塑为 b, t, d。
    接着应用标准的变换器操作。
    最后，重塑为图像。
    NEW: 使用线性层以提高效率，而不是 1x1 的卷积层。
    """

    # 初始化函数，设置空间变换器的参数
    def __init__(
        self,
        in_channels,  # 输入通道数
        n_heads,  # 注意力头的数量
        d_head,  # 每个注意力头的维度
        depth=1,  # 变换器的深度
        dropout=0.0,  # Dropout 概率
        context_dim=None,  # 上下文维度（可选）
        disable_self_attn=False,  # 是否禁用自注意力
        use_linear=False,  # 是否使用线性层
        attn_type="softmax",  # 注意力类型
        use_checkpoint=True,  # 是否使用检查点
        # sdp_backend=SDPBackend.FLASH_ATTENTION  # (注释掉的选项) SDP 后端
        sdp_backend=None,  # SDP 后端（可选）
    ):
        # 调用父类的构造函数
        super().__init__()
        # 打印当前类的名称、深度、输入通道数和头数
        print(f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads")
        # 从 omegaconf 导入 ListConfig 类
        from omegaconf import ListConfig

        # 如果 context_dim 存在且不是列表或 ListConfig 类型
        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            # 将 context_dim 转换为包含一个元素的列表
            context_dim = [context_dim]
        # 如果 context_dim 存在且是列表类型
        if exists(context_dim) and isinstance(context_dim, list):
            # 检查 context_dim 的长度是否与深度匹配
            if depth != len(context_dim):
                # 打印警告信息，指出深度与 context_dim 长度不匹配
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # 确保所有 context_dim 元素相同
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                # 将 context_dim 设置为深度数量的列表，元素为 context_dim 的第一个元素
                context_dim = depth * [context_dim[0]]
        # 如果 context_dim 为 None
        elif context_dim is None:
            # 将 context_dim 设置为深度数量的 None 列表
            context_dim = [None] * depth
        # 将输入通道数赋值给实例变量
        self.in_channels = in_channels
        # 计算内部维度
        inner_dim = n_heads * d_head
        # 创建归一化层
        self.norm = Normalize(in_channels)
        # 如果不使用线性投影
        if not use_linear:
            # 创建 2D 卷积层作为输入投影
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            # 创建线性层作为输入投影
            self.proj_in = nn.Linear(in_channels, inner_dim)

        # 创建变换器模块列表
        self.transformer_blocks = nn.ModuleList(
            [
                # 为每个深度创建 BasicTransformerBlock
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                )
                for d in range(depth)  # 遍历深度
            ]
        )
        # 如果不使用线性投影
        if not use_linear:
            # 创建 2D 卷积层作为输出投影，并将其设为零模块
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            # 创建线性层作为输出投影，并将其设为零模块
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        # 将使用线性标志赋值给实例变量
        self.use_linear = use_linear
    # 定义前向传播函数，接受输入 x 和可选的上下文 context
        def forward(self, x, context=None):
            # 注意：如果未提供上下文，则交叉注意力默认为自注意力
            if not isinstance(context, list):
                # 如果上下文不是列表，则将其包装为单元素列表
                context = [context]
            # 获取输入张量 x 的形状参数 b（批量大小）、c（通道数）、h（高度）、w（宽度）
            b, c, h, w = x.shape
            # 保存输入张量以便后续使用
            x_in = x
            # 对输入 x 进行归一化处理
            x = self.norm(x)
            # 如果不使用线性层，则对输入进行投影
            if not self.use_linear:
                x = self.proj_in(x)
            # 重新排列 x 的形状，将其从 (b, c, h, w) 转换为 (b, h*w, c)
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()
            # 如果使用线性层，则对 x 进行投影
            if self.use_linear:
                x = self.proj_in(x)
            # 遍历每个 transformer 块
            for i, block in enumerate(self.transformer_blocks):
                # 如果不是第一个块且上下文只有一个，重置索引为 0，以便在每个块中使用相同的上下文
                if i > 0 and len(context) == 1:
                    i = 0  # 使用相同的上下文
                # 通过 transformer 块处理 x 和相应的上下文
                x = block(x, context=context[i])
            # 如果使用线性层，则对 x 进行最终投影
            if self.use_linear:
                x = self.proj_out(x)
            # 重新排列 x 的形状，将其转换回 (b, c, h, w)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            # 如果不使用线性层，则对输出进行投影
            if not self.use_linear:
                x = self.proj_out(x)
            # 返回处理后的 x 与原输入 x_in 之和
            return x + x_in
```