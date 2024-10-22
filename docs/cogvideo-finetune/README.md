# CogVideo & CogVideoX 微调代码源码解析

[中文阅读](./README_zh.md)

[日本語で読む](./README_ja.md)

<div align="center">
<img src=resources/logo.svg width="50%"/>
</div>
<p align="center">
Experience the CogVideoX-5B model online at <a href="https://huggingface.co/spaces/THUDM/CogVideoX-5B" target="_blank"> 🤗 Huggingface Space</a> or <a href="https://modelscope.cn/studios/ZhipuAI/CogVideoX-5b-demo" target="_blank"> 🤖 ModelScope Space</a>
</p>
<p align="center">
📚 View the <a href="https://arxiv.org/abs/2408.06072" target="_blank">paper</a> and <a href="https://zhipu-ai.feishu.cn/wiki/DHCjw1TrJiTyeukfc9RceoSRnCh" target="_blank">user guide</a>
</p>
<p align="center">
    👋 Join our <a href="resources/WECHAT.md" target="_blank">WeChat</a> and <a href="https://discord.gg/dCGfUsagrD" target="_blank">Discord</a> 
</p>
<p align="center">
📍 Visit <a href="https://chatglm.cn/video?lang=en?fr=osm_cogvideo">QingYing</a> and <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9">API Platform</a> to experience larger-scale commercial video generation models.
</p>

## Project Updates

- 🔥🔥 **News**: ```py/10/13```: A more cost-effective fine-tuning framework for `CogVideoX-5B` that works with a single 4090 GPU, [cogvideox-factory](https://github.com/a-r-r-o-w/cogvideox-factory), has been released. It supports fine-tuning with multiple resolutions. Feel free to use it!
- 🔥 **News**: ```py/10/10```: We have updated our technical report. Please click [here](https://arxiv.org/pdf/2408.06072) to view it. More training details and a demo have been added. To see the demo, click [here](https://yzy-thu.github.io/CogVideoX-demo/).- 🔥 **News**: ```py/10/09```: We have publicly released the [technical documentation](https://zhipu-ai.feishu.cn/wiki/DHCjw1TrJiTyeukfc9RceoSRnCh) for CogVideoX fine-tuning on Feishu, further increasing distribution flexibility. All examples in the public documentation can be fully reproduced.
- 🔥 **News**: ```py/9/19```: We have open-sourced the CogVideoX series image-to-video model **CogVideoX-5B-I2V**.
  This model can take an image as a background input and generate a video combined with prompt words, offering greater
  controllability. With this, the CogVideoX series models now support three tasks: text-to-video generation, video
  continuation, and image-to-video generation. Welcome to try it online
  at [Experience](https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space).
- 🔥 ```py/9/19```: The Caption
  model [CogVLM2-Caption](https://huggingface.co/THUDM/cogvlm2-llama3-caption), used in the training process of
  CogVideoX to convert video data into text descriptions, has been open-sourced. Welcome to download and use it.
- 🔥 ```py/8/27```: We have open-sourced a larger model in the CogVideoX series, **CogVideoX-5B**. We have
  significantly optimized the model's inference performance, greatly lowering the inference threshold. You can run *
  *CogVideoX-2B** on older GPUs like `GTX 1080TI`, and **CogVideoX-5B** on desktop GPUs like `RTX 3060`. Please strictly
  follow the [requirements](requirements.txt) to update and install dependencies, and refer
  to [cli_demo](inference/cli_demo.py) for inference code. Additionally, the open-source license for the **CogVideoX-2B
  ** model has been changed to the **Apache 2.0 License**.
- 🔥 ```py/8/6```: We have open-sourced **3D Causal VAE**, used for **CogVideoX-2B**, which can reconstruct videos with
  almost no loss.
- 🔥 ```py/8/6```: We have open-sourced the first model of the CogVideoX series video generation models, **CogVideoX-2B
  **.
- 🌱 **Source**: ```py/5/19```: We have open-sourced the CogVideo video generation model (now you can see it in
  the `CogVideo` branch). This is the first open-source large Transformer-based text-to-video generation model. You can
  access the [ICLR'23 paper](https://arxiv.org/abs/2205.15868) for technical details.

## Table of Contents

Jump to a specific section:

- [Quick Start](#Quick-Start)
    - [SAT](#sat)
    - [Diffusers](#Diffusers)
- [CogVideoX-2B Video Works](#cogvideox-2b-gallery)
- [Introduction to the CogVideoX Model](#Model-Introduction)
- [Full Project Structure](#project-structure)
    - [Inference](#inference)
    - [SAT](#sat)
    - [Tools](#tools)
- [Introduction to CogVideo(ICLR'23) Model](#cogvideoiclr23)
- [Citations](#Citation)
- [Open Source Project Plan](#Open-Source-Project-Plan)
- [Model License](#Model-License)

## Quick Start

### Prompt Optimization

Before running the model, please refer to [this guide](inference/convert_demo.py) to see how we use large models like
GLM-4 (or other comparable products, such as GPT-4) to optimize the model. This is crucial because the model is trained
with long prompts, and a good prompt directly impacts the quality of the video generation.

### SAT

**Please make sure your Python version is between 3.10 and 3.12, inclusive of both 3.10 and 3.12.**

Follow instructions in [sat_demo](sat/README.md): Contains the inference code and fine-tuning code of SAT weights. It is
recommended to improve based on the CogVideoX model structure. Innovative researchers use this code to better perform
rapid stacking and development.

### Diffusers

**Please make sure your Python version is between 3.10 and 3.12, inclusive of both 3.10 and 3.12.**

```py
pip install -r requirements.txt
```

Then follow [diffusers_demo](inference/cli_demo.py): A more detailed explanation of the inference code, mentioning the
significance of common parameters.

For more details on quantized inference, please refer
to [diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao/). With Diffusers and TorchAO, quantized inference
is also possible leading to memory-efficient inference as well as speedup in some cases when compiled. A full list of
memory and time benchmarks with various settings on A100 and H100 has been published
at [diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao).

## Gallery

### CogVideoX-5B

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/cf5953ea-96d3-48fd-9907-c4708752c714" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fe0a78e6-b669-4800-8cf0-b5f9b5145b52" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/c182f606-8f8c-421d-b414-8487070fcfcb" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7db2bbce-194d-434d-a605-350254b6c298" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/62b01046-8cab-44cc-bd45-4d965bb615ec" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/d78e552a-4b3f-4b81-ac3f-3898079554f6" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/30894f12-c741-44a2-9e6e-ddcacc231e5b" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/926575ca-7150-435b-a0ff-4900a963297b" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### CogVideoX-2B

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ea3af39a-3160-4999-90ec-2f7863c5b0e9" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9de41efd-d4d1-4095-aeda-246dd834e91d" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/941d6661-6a8d-4a1b-b912-59606f0b2841" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/938529c4-91ae-4f60-b96b-3c3947fa63cb" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

To view the corresponding prompt words for the gallery, please click [here](resources/galary_prompt.md)

## Model Introduction

CogVideoX is an open-source version of the video generation model originating
from [QingYing](https://chatglm.cn/video?lang=en?fr=osm_cogvideo). The table below displays the list of video generation
models we currently offer, along with their foundational information.

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">Model Name</th>
    <th style="text-align: center;">CogVideoX-2B</th>
    <th style="text-align: center;">CogVideoX-5B</th>
    <th style="text-align: center;">CogVideoX-5B-I2V</th>
  </tr>
  <tr>
    <td style="text-align: center;">Model Description</td>
    <td style="text-align: center;">Entry-level model, balancing compatibility. Low cost for running and secondary development.</td>
    <td style="text-align: center;">Larger model with higher video generation quality and better visual effects.</td>
    <td style="text-align: center;">CogVideoX-5B image-to-video version.</td>
  </tr>
  <tr>
    <td style="text-align: center;">Inference Precision</td>
    <td style="text-align: center;"><b>FP16*(recommended)</b>, BF16, FP32, FP8*, INT8, not supported: INT4</td>
    <td colspan="2" style="text-align: center;"><b>BF16 (recommended)</b>, FP16, FP32, FP8*, INT8, not supported: INT4</td>
  </tr>
  <tr>
    <td style="text-align: center;">Single GPU Memory Usage<br></td>
    <td style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> FP16: 18GB <br><b>diffusers FP16: from 4GB* </b><br><b>diffusers INT8 (torchao): from 3.6GB*</b></td>
    <td colspan="2" style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> BF16: 26GB <br><b>diffusers BF16: from 5GB* </b><br><b>diffusers INT8 (torchao): from 4.4GB*</b></td>
  </tr>
  <tr>
    <td style="text-align: center;">Multi-GPU Inference Memory Usage</td>
    <td style="text-align: center;"><b>FP16: 10GB* using diffusers</b><br></td>
    <td colspan="2" style="text-align: center;"><b>BF16: 15GB* using diffusers</b><br></td>
  </tr>
  <tr>
    <td style="text-align: center;">Inference Speed<br>(Step = 50, FP/BF16)</td>
    <td style="text-align: center;">Single A100: ~90 seconds<br>Single H100: ~45 seconds</td>
    <td colspan="2" style="text-align: center;">Single A100: ~180 seconds<br>Single H100: ~90 seconds</td>
  </tr>
  <tr>
    <td style="text-align: center;">Fine-tuning Precision</td>
    <td style="text-align: center;"><b>FP16</b></td>
    <td colspan="2" style="text-align: center;"><b>BF16</b></td>
  </tr>
  <tr>
    <td style="text-align: center;">Fine-tuning Memory Usage</td>
    <td style="text-align: center;">47 GB (bs=1, LORA)<br> 61 GB (bs=2, LORA)<br> 62GB (bs=1, SFT)</td>
    <td style="text-align: center;">63 GB (bs=1, LORA)<br> 80 GB (bs=2, LORA)<br> 75GB (bs=1, SFT)<br></td>
    <td style="text-align: center;">78 GB (bs=1, LORA)<br> 75GB (bs=1, SFT, 16GPU)<br></td>
  </tr>
  <tr>
    <td style="text-align: center;">Prompt Language</td>
    <td colspan="3" style="text-align: center;">English*</td>
  </tr>
  <tr>
    <td style="text-align: center;">Maximum Prompt Length</td>
    <td colspan="3" style="text-align: center;">226 Tokens</td>
  </tr>
  <tr>
    <td style="text-align: center;">Video Length</td>
    <td colspan="3" style="text-align: center;">6 Seconds</td>
  </tr>
  <tr>
    <td style="text-align: center;">Frame Rate</td>
    <td colspan="3" style="text-align: center;">8 Frames / Second</td>
  </tr>
  <tr>
    <td style="text-align: center;">Video Resolution</td>
    <td colspan="3" style="text-align: center;">720 x 480, no support for other resolutions (including fine-tuning)</td>
  </tr>
    <tr>
    <td style="text-align: center;">Position Encoding</td>
    <td style="text-align: center;">3d_sincos_pos_embed</td>
    <td style="text-align: center;">3d_sincos_pos_embed</td>
    <td style="text-align: center;">3d_rope_pos_embed + learnable_pos_embed</td>
  </tr>
  <tr>
    <td style="text-align: center;">Download Link (Diffusers)</td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-2b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-2b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-2b">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-5b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-5b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-5b-I2V">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-5b-I2V">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b-I2V">🟣 WiseModel</a></td>
  </tr>
  <tr>
    <td style="text-align: center;">Download Link (SAT)</td>
    <td colspan="3" style="text-align: center;"><a href="./sat/README.md">SAT</a></td>
  </tr>
</table>

**Data Explanation**

+ While testing using the diffusers library, all optimizations included in the diffusers library were enabled. This
  scheme has not been tested for actual memory usage on devices outside of **NVIDIA A100 / H100** architectures.
  Generally, this scheme can be adapted to all **NVIDIA Ampere architecture** and above devices. If optimizations are
  disabled, memory consumption will multiply, with peak memory usage being about 3 times the value in the table.
  However, speed will increase by about 3-4 times. You can selectively disable some optimizations, including:

```py
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

+ For multi-GPU inference, the `enable_sequential_cpu_offload()` optimization needs to be disabled.
+ Using INT8 models will slow down inference, which is done to accommodate lower-memory GPUs while maintaining minimal
  video quality loss, though inference speed will significantly decrease.
+ The CogVideoX-2B model was trained in `FP16` precision, and all CogVideoX-5B models were trained in `BF16` precision.
  We recommend using the precision in which the model was trained for inference.
+ [PytorchAO](https://github.com/pytorch/ao) and [Optimum-quanto](https://github.com/huggingface/optimum-quanto/) can be
  used to quantize the text encoder, transformer, and VAE modules to reduce the memory requirements of CogVideoX. This
  allows the model to run on free T4 Colabs or GPUs with smaller memory! Also, note that TorchAO quantization is fully
  compatible with `torch.compile`, which can significantly improve inference speed. FP8 precision must be used on
  devices with NVIDIA H100 and above, requiring source installation of `torch`, `torchao`, `diffusers`, and `accelerate`
  Python packages. CUDA 12.4 is recommended.
+ The inference speed tests also used the above memory optimization scheme. Without memory optimization, inference speed
  increases by about 10%. Only the `diffusers` version of the model supports quantization.
+ The model only supports English input; other languages can be translated into English for use via large model
  refinement.
+ The memory usage of model fine-tuning is tested in an `8 * H100` environment, and the program automatically
  uses `Zero 2` optimization. If a specific number of GPUs is marked in the table, that number or more GPUs must be used
  for fine-tuning.

## Friendly Links

We highly welcome contributions from the community and actively contribute to the open-source community. The following
works have already been adapted for CogVideoX, and we invite everyone to use them:

+ [CogVideoX-Fun](https://github.com/aigc-apps/CogVideoX-Fun): CogVideoX-Fun is a modified pipeline based on the
  CogVideoX architecture, supporting flexible resolutions and multiple launch methods.
+ [CogStudio](https://github.com/pinokiofactory/cogstudio): A separate repository for CogVideo's Gradio Web UI, which
  supports more functional Web UIs.
+ [Xorbits Inference](https://github.com/xorbitsai/inference): A powerful and comprehensive distributed inference
  framework, allowing you to easily deploy your own models or the latest cutting-edge open-source models with just one
  click.
+ [ComfyUI-CogVideoXWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper) Use the ComfyUI framework to integrate
  CogVideoX into your workflow.
+ [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys): VideoSys provides a user-friendly, high-performance
  infrastructure for video generation, with full pipeline support and continuous integration of the latest models and
  techniques.
+ [AutoDL Space](https://www.codewithgpu.com/i/THUDM/CogVideo/CogVideoX-5b-demo): A one-click deployment Huggingface
  Space image provided by community members.
+ [Interior Design Fine-Tuning Model](https://huggingface.co/collections/bertjiazheng/koolcogvideox-66e4762f53287b7f39f8f3ba):
  is a fine-tuned model based on CogVideoX, specifically designed for interior design.
+ [xDiT](https://github.com/xdit-project/xDiT): xDiT is a scalable inference engine for Diffusion Transformers (DiTs) 
   on multiple GPU Clusters. xDiT supports real-time image and video generations services.
+ [cogvideox-factory](https://github.com/a-r-r-o-w/cogvideox-factory): A cost-effective 
   fine-tuning framework for CogVideoX, compatible with the `diffusers` version model. Supports more resolutions, and fine-tuning CogVideoX-5B can be done with a single 4090 GPU.

## Project Structure

This open-source repository will guide developers to quickly get started with the basic usage and fine-tuning examples
of the **CogVideoX** open-source model.

### Quick Start with Colab

Here provide three projects that can be run directly on free Colab T4 instances:

+ [CogVideoX-5B-T2V-Colab.ipynb](https://colab.research.google.com/drive/1pCe5s0bC_xuXbBlpvIH1z0kfdTLQPzCS?usp=sharing):
  CogVideoX-5B Text-to-Video Colab code.
+ [CogVideoX-5B-T2V-Int8-Colab.ipynb](https://colab.research.google.com/drive/1DUffhcjrU-uz7_cpuJO3E_D4BaJT7OPa?usp=sharing):
  CogVideoX-5B Quantized Text-to-Video Inference Colab code, which takes about 30 minutes per run.
+ [CogVideoX-5B-I2V-Colab.ipynb](https://colab.research.google.com/drive/17CqYCqSwz39nZAX2YyonDxosVKUZGzcX?usp=sharing):
  CogVideoX-5B Image-to-Video Colab code.
+ [CogVideoX-5B-V2V-Colab.ipynb](https://colab.research.google.com/drive/1comfGAUJnChl5NwPuO8Ox5_6WCy4kbNN?usp=sharing):
  CogVideoX-5B Video-to-Video Colab code.

### Inference

+ [dcli_demo](inference/cli_demo.py): A more detailed inference code explanation, including the significance of
  common parameters. All of this is covered here.
+ [cli_demo_quantization](inference/cli_demo_quantization.py):
  Quantized model inference code that can run on devices with lower memory. You can also modify this code to support
  running CogVideoX models in FP8 precision.
+ [diffusers_vae_demo](inference/cli_vae_demo.py): Code for running VAE inference separately.
+ [space demo](inference/gradio_composite_demo): The same GUI code as used in the Huggingface Space, with frame
  interpolation and super-resolution tools integrated.

<div style="text-align: center;">
    <img src="resources/web_demo.png" style="width: 100%; height: auto;" />
</div>

+ [convert_demo](inference/convert_demo.py): How to convert user input into long-form input suitable for CogVideoX.
  Since CogVideoX is trained on long texts, we need to transform the input text distribution to match the training data
  using an LLM. The script defaults to using GLM-4, but it can be replaced with GPT, Gemini, or any other large language
  model.
+ [gradio_web_demo](inference/gradio_composite_demo): A simple Gradio web application demonstrating how to use the
  CogVideoX-2B / 5B model to generate videos. Similar to our Huggingface Space, you can use this script to run a simple
  web application for video generation.

### finetune

+ [finetune_demo](finetune/README.md): Fine-tuning scheme and details of the diffusers version of the CogVideoX model.

### sat

+ [sat_demo](sat/README.md): Contains the inference code and fine-tuning code of SAT weights. It is recommended to
  improve based on the CogVideoX model structure. Innovative researchers use this code to better perform rapid stacking
  and development.

### Tools

This folder contains some tools for model conversion / caption generation, etc.

+ [convert_weight_sat2hf](tools/convert_weight_sat2hf.py): Converts SAT model weights to Huggingface model weights.
+ [caption_demo](tools/caption/README.md): Caption tool, a model that understands videos and outputs descriptions in
  text.
+ [export_sat_lora_weight](tools/export_sat_lora_weight.py): SAT fine-tuning model export tool, exports the SAT Lora
  Adapter in diffusers format.
+ [load_cogvideox_lora](tools/load_cogvideox_lora.py): Tool code for loading the diffusers version of fine-tuned Lora
  Adapter.
+ [llm_flux_cogvideox](tools/llm_flux_cogvideox/llm_flux_cogvideox.py): Automatically generate videos using an
  open-source local large language model + Flux + CogVideoX.
+ [parallel_inference_xdit](tools/parallel_inference/parallel_inference_xdit.py):
Supported by [xDiT](https://github.com/xdit-project/xDiT), parallelize the
  video generation process on multiple GPUs.

## CogVideo(ICLR'23)

The official repo for the
paper: [CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868)
is on the [CogVideo branch](https://github.com/THUDM/CogVideo/tree/CogVideo)

**CogVideo is able to generate relatively high-frame-rate videos.**
A 4-second clip of 32 frames is shown below.

![High-frame-rate sample](https://raw.githubusercontent.com/THUDM/CogVideo/CogVideo/assets/appendix-sample-highframerate.png)

![Intro images](https://raw.githubusercontent.com/THUDM/CogVideo/CogVideo/assets/intro-image.png)
<div align="center">
  <video src="https://github.com/user-attachments/assets/2fa19651-e925-4a2a-b8d6-b3f216d490ba" width="80%" controls autoplay></video>
</div>


The demo for CogVideo is at [https://models.aminer.cn/cogvideo](https://models.aminer.cn/cogvideo/), where you can get
hands-on practice on text-to-video generation. *The original input is in Chinese.*

## Citation

🌟 If you find our work helpful, please leave us a star and cite our paper.

```py
@article{yang2024cogvideox,
  title={CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer},
  author={Yang, Zhuoyi and Teng, Jiayan and Zheng, Wendi and Ding, Ming and Huang, Shiyu and Xu, Jiazheng and Yang, Yuanming and Hong, Wenyi and Zhang, Xiaohan and Feng, Guanyu and others},
  journal={arXiv preprint arXiv:2408.06072},
  year={2024}
}
@article{hong2022cogvideo,
  title={CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers},
  author={Hong, Wenyi and Ding, Ming and Zheng, Wendi and Liu, Xinghan and Tang, Jie},
  journal={arXiv preprint arXiv:2205.15868},
  year={2022}
}
```

We welcome your contributions! You can click [here](resources/contribute.md) for more information.

## License Agreement

The code in this repository is released under the [Apache 2.0 License](LICENSE).

The CogVideoX-2B model (including its corresponding Transformers module and VAE module) is released under
the [Apache 2.0 License](LICENSE).

The CogVideoX-5B model (Transformers module, include I2V and T2V) is released under
the [CogVideoX LICENSE](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE).
