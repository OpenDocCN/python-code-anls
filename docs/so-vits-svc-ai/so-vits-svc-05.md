# SO-VITS-SVC源码解析 5

- Open [onnx_export](onnx_export.py)
- project_name = "dddsp" change "project_name" to your project name
- model_path = f'{project_name}/model_500000.pt' change "model_path" to your model path
- Run

# `diffusion/infer_gt_mel.py`



This is a PyTorch implementation of a class for inferring mel spectrograms. The class takes in audio in the form of a tensor with shape (B, S, H), where B is the batch size, S is the sample rate, and H is the number of mel channels. It also takes in the initial f0 and hubert values, as well as the volume of the audio. The infer method applies the vocoder to the audio and returns the mel spectrogram.

The infer method first converts the audio to a Mel spectrogram format, which is a tensor with shape (B, S, N_MEL, N_MEL). It then starts the infer process by searching for the first start frame in the audio. If there is no start frame, it will start the infer process from the beginning. The infer process returns the mel spectrogram, which is a tensor with shape (B, S, N_MEL).

If the audio is from a single audio source, it will use that source for the entire infer process. If multiple audio sources are provided, it will use the specified weights for each source to compute the mel spectrogram.

Note that this implementation assumes that the input audio is a valid audio signal and that the vocoder has already been trained. Additionally, this implementation may not handle all cases, such as audio samples with no mel information or audio samples outside of the input range.


```py
import torch
import torch.nn.functional as F

from diffusion.unit2mel import load_model_vocoder


class DiffGtMel:
    def __init__(self, project_path=None, device=None):
        self.project_path = project_path
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.vocoder = None
        self.args = None

    def flush_model(self, project_path, ddsp_config=None):
        if (self.model is None) or (project_path != self.project_path):
            model, vocoder, args = load_model_vocoder(project_path, device=self.device)
            if self.check_args(ddsp_config, args):
                self.model = model
                self.vocoder = vocoder
                self.args = args

    def check_args(self, args1, args2):
        if args1.data.block_size != args2.data.block_size:
            raise ValueError("DDSP与DIFF模型的block_size不一致")
        if args1.data.sampling_rate != args2.data.sampling_rate:
            raise ValueError("DDSP与DIFF模型的sampling_rate不一致")
        if args1.data.encoder != args2.data.encoder:
            raise ValueError("DDSP与DIFF模型的encoder不一致")
        return True

    def __call__(self, audio, f0, hubert, volume, acc=1, spk_id=1, k_step=0, method='pndm',
                 spk_mix_dict=None, start_frame=0):
        input_mel = self.vocoder.extract(audio, self.args.data.sampling_rate)
        out_mel = self.model(
            hubert,
            f0,
            volume,
            spk_id=spk_id,
            spk_mix_dict=spk_mix_dict,
            gt_spec=input_mel,
            infer=True,
            infer_speedup=acc,
            method=method,
            k_step=k_step,
            use_tqdm=False)
        if start_frame > 0:
            out_mel = out_mel[:, start_frame:, :]
            f0 = f0[:, start_frame:, :]
        output = self.vocoder.infer(out_mel, f0)
        if start_frame > 0:
            output = F.pad(output, (start_frame * self.vocoder.vocoder_hop_size, 0))
        return output

    def infer(self, audio, f0, hubert, volume, acc=1, spk_id=1, k_step=0, method='pndm', silence_front=0,
              use_silence=False, spk_mix_dict=None):
        start_frame = int(silence_front * self.vocoder.vocoder_sample_rate / self.vocoder.vocoder_hop_size)
        if use_silence:
            audio = audio[:, start_frame * self.vocoder.vocoder_hop_size:]
            f0 = f0[:, start_frame:, :]
            hubert = hubert[:, start_frame:, :]
            volume = volume[:, start_frame:, :]
            _start_frame = 0
        else:
            _start_frame = start_frame
        audio = self.__call__(audio, f0, hubert, volume, acc=acc, spk_id=spk_id, k_step=k_step,
                              method=method, spk_mix_dict=spk_mix_dict, start_frame=_start_frame)
        if use_silence:
            if start_frame > 0:
                audio = F.pad(audio, (start_frame * self.vocoder.vocoder_hop_size, 0))
        return audio

```

# `diffusion/onnx_export.py`

这段代码的作用是定义了一个名为 `DotDict` 的类，类似于Python中的 `dict` 类型，但具有自定义的 `__getattr__` 和 `__setattr__` 方法。这个类的定义是为了在PyTorch中更方便地使用`DotDict`类，类似于PyTorch中的`PyTorch.utils.general.DotDict`类。

具体来说，这段代码定义了 `DotDict` 类，其中包含以下属性：

- `__init__` 方法：初始化所有的键值对都为 `None`。
- `__getattr__` 方法：定义了自定义的 `__getattr__` 方法，用于在PyTorch中更方便地访问 `DotDict` 中的键值对。这个方法与PyTorch中的 `dict` 类型中的 `__getattr__` 方法类似，但可以用于`DotDict` 特有的情况。
- `__setattr__` 方法：定义了自定义的 `__setattr__` 方法，用于在PyTorch中更方便地设置 `DotDict` 中的键值对。这个方法与PyTorch中的 `dict` 类型中的 `__setattr__` 方法类似，但可以用于`DotDict` 特有的情况。
- `__delattr__` 方法：定义了自定义的 `__delattr__` 方法，用于在PyTorch中更方便地删除 `DotDict` 中的键值对。

由于 `DotDict` 类中包含自定义的 `__getattr__` 和 `__setattr__` 方法，因此可以像这样使用它：

```py
# 在PyTorch中更方便地访问DotDict中的键值对
dot_dict = DotDict()
for key in dot_dict.keys():
   print(key)

# 在PyTorch中更方便地设置DotDict中的键值对
dot_dict['c'] = 1.0
```

这段代码定义的 `DotDict` 类可以用于缓存和共享数据，使得在PyTorch中访问和设置`DotDict` 中的数据更加方便。


```py
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from diffusion_onnx import GaussianDiffusion


class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

    
```

这段代码的作用是加载一个预训练好的语音模型，并将其使用指定设备上进行预测。

具体来说，它包括以下几个步骤：

1. 加载预训练模型：使用给定的模型路径和设备（如果指定为CPU，则使用CPU设备）加载预训练模型。
2. 读取配置文件：读取模型的配置文件（通常是一个YAML文件），从中提取关键参数，如编码器输出的通道数、模型参数等。
3. 加载模型及其配置：使用提取的参数加载预训练模型，并将其移动到指定设备上（如果指定了设备为CPU，则将其移动到CPU设备上）。
4. 评估模型：将模型设置为评估模式，以便进行预测。
5. 返回模型和参数：返回加载的模型和提取的参数。


```py
def load_model_vocoder(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load model
    model = Unit2Mel(
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                128,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden,
                args.model.timesteps,
                args.model.k_step_max)
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, args


```

This is a PyTorch implementation of a complete end-to-end text-to-speech model, including the preprocessing and inference parts. The model takes in a text prompt, and outputs a synthesized speech waveform.

The model uses a pre-trained刺客数字模型（ Hubert 模型）作为 encoder，并在其后面加上一个扩散层（Diffusion 层）来提高模型性能。扩散层使用了一个速度up的卷积神经网络来加快推理过程。

在推理过程中，首先对输入文本进行处理，包括预处理、获取条件预测以及设置扩散速度。然后进行前向传播，得到条件预测，并将其输入到扩散层中。接下来，使用K_steps迭代更新扩散层中的参数，并使用条件预测和噪声生成新的K_steps个条件。最后，返回生成的语音波形。

模型支持预处理文本和设置扩散速度。通过设置不同的参数，可以调整模型的性能和效果。


```py
class Unit2Mel(nn.Module):
    def __init__(
            self,
            input_channel,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=20, 
            n_chans=384, 
            n_hidden=256,
            timesteps=1000,
            k_step_max=1000):
        super().__init__()

        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)

        self.timesteps = timesteps if timesteps is not None else 1000
        self.k_step_max = k_step_max if k_step_max is not None and k_step_max>0 and k_step_max<self.timesteps else self.timesteps


        # diffusion
        self.decoder = GaussianDiffusion(out_dims, n_layers, n_chans, n_hidden,self.timesteps,self.k_step_max)
        self.hidden_size = n_hidden
        self.speaker_map = torch.zeros((self.n_spk,1,1,n_hidden))
    
        

    def forward(self, units, mel2ph, f0, volume, g = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        decoder_inp = F.pad(units, [0, 0, 1, 0])
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, units.shape[-1]])
        units = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        x = self.unit_embed(units) + self.f0_embed((1 + f0.unsqueeze(-1) / 700).log()) + self.volume_embed(volume.unsqueeze(-1))

        if self.n_spk is not None and self.n_spk > 1:   # [N, S]  *  [S, B, 1, H]
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
            g = g * self.speaker_map  # [N, S, B, 1, H]
            g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
            x = x.transpose(1, 2) + g
            return x
        else:
            return x.transpose(1, 2)
        

    def init_spkembed(self, units, f0, volume, spk_id = None, spk_mix_dict = None, aug_shift = None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        x = self.unit_embed(units) + self.f0_embed((1+ f0 / 700).log()) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                spk_embed_mix = torch.zeros((1,1,self.hidden_size))
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    spk_embeddd = self.spk_embed(spk_id_torch)
                    self.speaker_map[k] = spk_embeddd
                    spk_embed_mix = spk_embed_mix + v * spk_embeddd
                x = x + spk_embed_mix
            else:
                x = x + self.spk_embed(spk_id - 1)
        self.speaker_map = self.speaker_map.unsqueeze(0)
        self.speaker_map = self.speaker_map.detach()
        return x.transpose(1, 2)

    def OnnxExport(self, project_name=None, init_noise=None, export_encoder=True, export_denoise=True, export_pred=True, export_after=True):
        hubert_hidden_size = 768
        n_frames = 100
        hubert = torch.randn((1, n_frames, hubert_hidden_size))
        mel2ph = torch.arange(end=n_frames).unsqueeze(0).long()
        f0 = torch.randn((1, n_frames))
        volume = torch.randn((1, n_frames))
        spk_mix = []
        spks = {}
        if self.n_spk is not None and self.n_spk > 1:
            for i in range(self.n_spk):
                spk_mix.append(1.0/float(self.n_spk))
                spks.update({i:1.0/float(self.n_spk)})
        spk_mix = torch.tensor(spk_mix)
        spk_mix = spk_mix.repeat(n_frames, 1)
        self.init_spkembed(hubert, f0.unsqueeze(-1), volume.unsqueeze(-1), spk_mix_dict=spks)
        self.forward(hubert, mel2ph, f0, volume, spk_mix)
        if export_encoder:
            torch.onnx.export(
                self,
                (hubert, mel2ph, f0, volume, spk_mix),
                f"{project_name}_encoder.onnx",
                input_names=["hubert", "mel2ph", "f0", "volume", "spk_mix"],
                output_names=["mel_pred"],
                dynamic_axes={
                    "hubert": [1],
                    "f0": [1],
                    "volume": [1],
                    "mel2ph": [1],
                    "spk_mix": [0],
                },
                opset_version=16
            )
        
        self.decoder.OnnxExport(project_name, init_noise=init_noise, export_denoise=export_denoise, export_pred=export_pred, export_after=export_after)

    def ExportOnnx(self, project_name=None):
        hubert_hidden_size = 768
        n_frames = 100
        hubert = torch.randn((1, n_frames, hubert_hidden_size))
        mel2ph = torch.arange(end=n_frames).unsqueeze(0).long()
        f0 = torch.randn((1, n_frames))
        volume = torch.randn((1, n_frames))
        spk_mix = []
        spks = {}
        if self.n_spk is not None and self.n_spk > 1:
            for i in range(self.n_spk):
                spk_mix.append(1.0/float(self.n_spk))
                spks.update({i:1.0/float(self.n_spk)})
        spk_mix = torch.tensor(spk_mix)
        self.orgforward(hubert, f0.unsqueeze(-1), volume.unsqueeze(-1), spk_mix_dict=spks)
        self.forward(hubert, mel2ph, f0, volume, spk_mix)

        torch.onnx.export(
                self,
                (hubert, mel2ph, f0, volume, spk_mix),
                f"{project_name}_encoder.onnx",
                input_names=["hubert", "mel2ph", "f0", "volume", "spk_mix"],
                output_names=["mel_pred"],
                dynamic_axes={
                    "hubert": [1],
                    "f0": [1],
                    "volume": [1],
                    "mel2ph": [1]
                },
                opset_version=16
            )

        condition = torch.randn(1,self.decoder.n_hidden,n_frames)
        noise = torch.randn((1, 1, self.decoder.mel_bins, condition.shape[2]), dtype=torch.float32)
        pndm_speedup = torch.LongTensor([100])
        K_steps = torch.LongTensor([1000])
        self.decoder = torch.jit.script(self.decoder)
        self.decoder(condition, noise, pndm_speedup, K_steps)

        torch.onnx.export(
                self.decoder,
                (condition, noise, pndm_speedup, K_steps),
                f"{project_name}_diffusion.onnx",
                input_names=["condition", "noise", "pndm_speedup", "K_steps"],
                output_names=["mel"],
                dynamic_axes={
                    "condition": [2],
                    "noise": [3],
                },
                opset_version=16
            )


```

这段代码是一个Python脚本，它执行以下操作：

1. 检查是否运行脚本作为主程序。如果是，执行以下操作：

  ```py
  project_name = "dddsp"
  model_path = f'{project_name}/model_500000.pt'

  model, _ = load_model_vocoder(model_path)

  # 分开Diffusion导出（需要使用MoeSS/MoeVoiceStudio或者自己编写Pndm/Dpm采样）
  model.OnnxExport(project_name, export_encoder=True, export_denoise=True, export_pred=True, export_after=True)

  # 合并Diffusion导出（Encoder和Diffusion分开，直接将Encoder的结果和初始噪声输入Diffusion即可）
  model.ExportOnnx(project_name)
  ```

2. 如果不是作为主程序运行，那么执行以下操作：

  ```py
  project_name = "dddsp"
  model_path = f'{project_name}/model_500000.pt'

  # Load the model and the evaluation file
  model, evaluation = load_model_vocoder(model_path)

  # Predict the input audio
  input_audio = prediction_audio()

  # Run the Inference
  output = model.Inference(input_audio)

  # Export the results
  model.OnnxExport(project_name, export_encoder=True, export_denoise=True, export_pred=True, export_after=True)
  model.ExportOnnx(project_name)
  ```

3. 在主程序中，加载预训练的模型，并导出M elif __name__ == "__main__":：

  ```py
  if __name__ == "__main__":
      project_name = "dddsp"
      model_path = f'{project_name}/model_500000.pt'

      model, _ = load_model_vocoder(model_path)

      # Split the model for diffusion
      model.SplitModel(Diffusion=model.model_path)
      model.DiffusionTraining(Training=model.model_path)

      # Run the prediction audio
      input_audio = prediction_audio()
      output = model.Inference(input_audio)

      # Export the results
      model.OnnxExport(project_name, export_encoder=True, export_denoise=True, export_pred=True, export_after=True)
      model.ExportOnnx(project_name)
  ```

4. 加载预训练的模型并导出不同模式（Encoder和Diffusion分离），主要针对Diffusion：

  ```py
  if __name__ == "__main__":
      project_name = "dddsp"
      model_path = f'{project_name}/model_500000.pt'

      model, _ = load_model_vocoder(model_path)

      # Split the model for diffusion
      model.SplitModel(Diffusion=model.model_path)
      model.DiffusionTraining(Training=model.model_path)

      # Encoder
      input_audio = audio_source()
      output_encoder = model.encoder.OnnxExport(project_name, export_encoder=True, export_denoise=True, export_pred=True, export_after=True)

      # Add noise
      input_audio = audio_source()
      output_diffusion = model.diffusion.OnnxExport(project_name, export_denoise=True, export_pred=True, export_after=True)

      # Merge diffusion results
      output = model.merge_diffusion(output_encoder, output_diffusion)

      # Export the results
      model.OnnxExport(project_name, export_encoder=True, export_denoise=True, export_pred=True, export_after=True)
      model.ExportOnnx(project_name)
  ```

5. 在主程序中，导出训练后的预训练模型，以评估怪物能力：

  ```py
  if __name__ == "__main__":
      project_name = "dddsp"
      model_path = f'{project_name}/model_500000.pt'

      model, _ = load_model_vocoder(model_path)

      # Split the model for diffusion
      model.SplitModel(Diffusion=model.model_path)
      model.DiffusionTraining(Training=model.model_path)

      # Run the Inference
      input_audio = audio_source()
      output =
```


```py
if __name__ == "__main__":
    project_name = "dddsp"
    model_path = f'{project_name}/model_500000.pt'

    model, _ = load_model_vocoder(model_path)

    # 分开Diffusion导出（需要使用MoeSS/MoeVoiceStudio或者自己编写Pndm/Dpm采样）
    model.OnnxExport(project_name, export_encoder=True, export_denoise=True, export_pred=True, export_after=True)

    # 合并Diffusion导出（Encoder和Diffusion分开，直接将Encoder的结果和初始噪声输入Diffusion即可）
    # model.ExportOnnx(project_name)


```

# `diffusion/solver.py`

This is a Python implementation of a speech-to-text (infer) model using the GTZAN dataset as a source. The model uses a combination of last-init-weight (LIR) and knowledge-epoch (KE) based methods to select the best feature during training. The LIR method is used for initialization of the model's weights, while the KE method is used for fine-tuning the model during training.

The input to the model is a list of features (e.g., mel-F1), including the speaker name, and the ground-truth audio. The output is the predicted audio, which is generated based on the input features.

The training loop is the main method that runs the entire training process. It initializes the model, sets the number of training epochs, and sets the logging parameters. Then, it loops over the batches of data, performs the inference using the model, and logs the results.

The test loop is responsible for evaluating the model's performance on the test set. It runs the inference for all the test samples, and logs the results. Finally, it汇总es the results and reports the average test loss.

The `词汇表` is a dictionary that maps each英文单词 (包括标点符号) to its corresponding Chinese拼音。


```py
import time

import librosa
import numpy as np
import torch
from torch import autocast
from torch.cuda.amp import GradScaler

from diffusion.logger import utils
from diffusion.logger.saver import Saver


def test(args, model, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()

    # losses
    test_loss = 0.
    
    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0].split("/")[-1]
            speaker = data['name'][0].split("/")[-2]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            mel = model(
                    data['units'], 
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    gt_spec=None if model.k_step_max == model.timesteps else data['mel'],
                    infer=True, 
                    infer_speedup=args.infer.speedup, 
                    method=args.infer.method,
                    k_step=model.k_step_max
                    )
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            for i in range(args.train.batch_size):
                loss = model(
                    data['units'], 
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'], 
                    gt_spec=data['mel'],
                    infer=False,
                    k_step=model.k_step_max)
                test_loss += loss.item()
            
            # log mel
            saver.log_spec(f"{speaker}_{fn}.wav", data['mel'], mel)
            
            # log audi
            path_audio = data['name_ext'][0]
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({f"{speaker}_{fn}_gt.wav": audio,f"{speaker}_{fn}_pred.wav": signal})
    # report
    test_loss /= args.train.batch_size
    test_loss /= num_batches 
    
    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss


```

This is a Python code that uses the PyTorch library to train a neural network during the training process. The code consists of two main components: the training loop and the validation loop.

The training loop is responsible for updating the weights of the neural network based on the input data and the learning rate determined by the optimizer. It is defined as follows:
```pyless
for epoch in range(1, args.num_epochs + 1):
   for batch in enumerate(train_loader):
       inputs, targets = batch
       optimizer.zero_grad()
       outputs = model(inputs)
       loss = criterion(outputs, targets)
       loss.backward()
       optimizer.step()
       saver.log_info(
           'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
               epoch,
               batch_idx,
               num_batches,
               args.env.expdir,
               current_lr,
               loss.item(),
               saver.get_total_time(),
               saver.global_step
           )
       )
       saver.log_value({
           'train/loss': loss.item()
       })
       saver.log_value({
           'train/lr': current_lr
       })
   
   # validation
   if epoch % args.train.interval_val == 0:
       optimizer_save = optimizer if args.train.save_optimal else None
       
       # save latest
       saver.save_model(model, optimizer_save, postfix=f'{epoch}')
       last_val_step = epoch - args.train.interval_val
       if last_val_step % args.train.interval_force_save != 0:
           saver.delete_model(postfix=f'{last_val_step}')
       
       # run testing set
       test_loss = test(args, model, vocoder, loader_test, saver)
       
       # log loss
       saver.log_info(
           ' --- <validation> --- \nloss: {:.3f}. '.format(
               test_loss,
           )
       )
       
       saver.log_value({
           'validation/loss': test_loss
       })
       
       model.train()

   # testing
   model.eval()
   test_loss = test(args, model, vocoder, loader_test, saver)

   # log testing set loss
   saver.log_info(
       ' --- <testing> --- \ntesting loss: {:.3f} '.format(test_loss)
   )
```
The validation loop is responsible for evaluating the performance of the neural network during the validation process. It is defined as follows:
```pypython
   # validation
   if saver.global_step % args.train.interval_val == 0:
       optimizer_save = optimizer if args.train.save_optimal else None
       
       # save latest
       saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
       last_val_step = saver.global_step - args.train.interval_val
       if last_val_step % args.train.interval_force_save != 0:
           saver.delete_model(postfix=f'{last_val_step}')
       
       # run testing set
       test_loss = test(args, model, vocoder, loader_test, saver)
       
       # log loss
       saver.log_info(
           ' --- <validation> --- \nloss: {:.3f}. '.format(
               test_loss,
           )
       )
       
       saver.log_value({
           'validation/loss': test_loss
       })
       
       model.train()
```
Finally, during the testing phase, the code runs the neural network on the testing set and logs the testing loss.


```py
def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    # run
    num_batches = len(loader_train)
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    saver.log_info("epoch|batch_idx/num_batches|output_dir|batch/s|lr|time|step")
    for epoch in range(args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            
            # forward
            if dtype == torch.float32:
                loss = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'], 
                                aug_shift = data['aug_shift'], gt_spec=data['mel'].float(), infer=False, k_step=model.k_step_max)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    loss = model(data['units'], data['f0'], data['volume'], data['spk_id'], 
                                    aug_shift = data['aug_shift'], gt_spec=data['mel'], infer=False, k_step=model.k_step_max)
            
            # handle nan loss
            if torch.isnan(loss):
                raise ValueError(' [x] nan loss ')
            else:
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item()
                })
                
                saver.log_value({
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_loss = test(args, model, vocoder, loader_test, saver)
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss
                })
                
                model.train()

                          

```

# `diffusion/unit2mel.py`

这段代码是一个Python脚本，它实现了几个PyTorch中的数据结构的创建和初始化。

具体来说，它实现了以下几个功能：

1. 导入os模块，用于在运行时导入操作系统中的文件和目录。
2. 导入NumPy和PyTorch库，用于实现数值计算和深度学习任务。
3. 导入PyTorch中的nn模块，用于实现神经网络。
4. 导入yaml库，用于解析和生成YAML格式的配置文件。
5. 实现了一个名为DotDict的类，用于实现PyTorch中的`DotDict`数据类型，即PyTorch中的`Googledoc`数据类型。
6. 实现了一个名为GaussianDiffusion的类，用于实现Gaussian扩散降噪的神经网络。
7. 实现了一个名为Vocoder的类，用于实现语音语调的提升和降噪。
8. 实现了一个名为WaveNet的类，用于实现基于WaveNet的语音模型。


```py
import os

import numpy as np
import torch
import torch.nn as nn
import yaml

from .diffusion import GaussianDiffusion
from .vocoder import Vocoder
from .wavenet import WaveNet


class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

    
```

这段代码的作用是加载一个预训练的语音模型，其中包括语音语调变化信息的处理。它可以让用户使用预训练的模型进行语音识别，即听取语音并将其转换为文本。用户还可以在训练时使用加速技术，包括使用CUDA（GPU）进行计算时的速度提升。


```py
def load_model_vocoder(
        model_path,
        device='cpu',
        config_path = None
        ):
    if config_path is None:
        config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    else:
        config_file = config_path

    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    
    # load model
    model = Unit2Mel(
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden,
                args.model.timesteps,
                args.model.k_step_max
                )
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded diffusion model, sampler is {args.infer.method}, speedup: {args.infer.speedup} ')
    return model, vocoder, args


```

This is a PyTorch implementation of a general depth-wise separable convolutional neural network (CNN) model that can be used for various tasks, including shallow diffusion and inference. The model has an input of type `dict`, which contains the input units, initialized spseek data, and the initial mixed data. It also has an output of type `dict`, which contains the output data and the augmented data for each input unit.

The depth-wise separable CNN model takes in an input tensor, and performs a depth-wise separable convolution operation. The separable convolution operation is defined as follows: `x = self.depthwise_ separable_ convolution(x, gt_spec, infer, method, k_step)`, where `x` is the input tensor, `gt_spec` is the ground-truth feature map, `infer` is whether to perform inference, `method` is the method used for the convolution operation, and `k_step` is the maximum number of steps to perform in the convolution operation.

The depth-wise separable CNN model also performs a normalization step, where the input tensor is divided by a squashing function. This normalization step is defined as `x = self.norm_activation(x)`, where `x` is the input tensor and `norm_activation` is a function that normalizes the input tensor.

The output of the depth-wise separable CNN model is a tensor of the same type as the input tensor.


```py
class Unit2Mel(nn.Module):
    def __init__(
            self,
            input_channel,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=20, 
            n_chans=384, 
            n_hidden=256,
            timesteps=1000,
            k_step_max=1000
            ):
        super().__init__()
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
        
        self.timesteps = timesteps if timesteps is not None else 1000
        self.k_step_max = k_step_max if k_step_max is not None and k_step_max>0 and k_step_max<self.timesteps else self.timesteps

        self.n_hidden = n_hidden
        # diffusion
        self.decoder = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden),timesteps=self.timesteps,k_step=self.k_step_max, out_dims=out_dims)
        self.input_channel = input_channel
    
    def init_spkembed(self, units, f0, volume, spk_id = None, spk_mix_dict = None, aug_shift = None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        x = self.unit_embed(units) + self.f0_embed((1+ f0 / 700).log()) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                spk_embed_mix = torch.zeros((1,1,self.hidden_size))
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    spk_embeddd = self.spk_embed(spk_id_torch)
                    self.speaker_map[k] = spk_embeddd
                    spk_embed_mix = spk_embed_mix + v * spk_embeddd
                x = x + spk_embed_mix
            else:
                x = x + self.spk_embed(spk_id - 1)
        self.speaker_map = self.speaker_map.unsqueeze(0)
        self.speaker_map = self.speaker_map.detach()
        return x.transpose(1, 2)

    def init_spkmix(self, n_spk):
        self.speaker_map = torch.zeros((n_spk,1,1,self.n_hidden))
        hubert_hidden_size = self.input_channel
        n_frames = 10
        hubert = torch.randn((1, n_frames, hubert_hidden_size))
        f0 = torch.randn((1, n_frames))
        volume = torch.randn((1, n_frames))
        spks = {}
        for i in range(n_spk):
            spks.update({i:1.0/float(self.n_spk)})
        self.init_spkembed(hubert, f0.unsqueeze(-1), volume.unsqueeze(-1), spk_mix_dict=spks)

    def forward(self, units, f0, volume, spk_id = None, spk_mix_dict = None, aug_shift = None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        if not self.training and gt_spec is not None and k_step>self.k_step_max:
            raise Exception("The shallow diffusion k_step is greater than the maximum diffusion k_step(k_step_max)!")

        if not self.training and gt_spec is None and self.k_step_max!=self.timesteps:
            raise Exception("This model can only be used for shallow diffusion and can not infer alone!")

        x = self.unit_embed(units) + self.f0_embed((1+ f0 / 700).log()) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    x = x + v * self.spk_embed(spk_id_torch)
            else:
                if spk_id.shape[1] > 1:
                    g = spk_id.reshape((spk_id.shape[0], spk_id.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
                    g = g * self.speaker_map  # [N, S, B, 1, H]
                    g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
                    g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
                    x = x + g
                else:
                    x = x + self.spk_embed(spk_id)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5) 
        x = self.decoder(x, gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
    
        return x


```

# `diffusion/uni_pc.py`

The `InverseLogit` class computes the inverse logit for a given continuous-time label `t` in the range [0, T]. It uses the `MarginalLogMean` and `Beta分布` components of the PyTorch `Distribution` class to estimate the probability distribution over the labels. The inverse logit is computed based on the log mean and standard deviation of the probability distribution.

The `inverse_lambda` method of the `InverseLogit` class computes the continuous-time label for a given half-log SNR value `lambda_t`. It returns the label t in the range [0, T] corresponding to the inverse logit.

The `schedule` parameter determines the type of update strategy to use. If it is set to 'linear', the update is performed linearly. If it is set to 'discrete', the update is performed using the `interpolate_fn` function.

The `beta_1` and `beta_0` parameters are used to control the weight of the softmax function in the linear update strategy. The `log_mean_coeff` is computed as the log mean of the softmax function with `num_classes` many positive labels.

The `beta_0` is a scaling factor for the logarithmic update strategy.

The `t_array` is the array of discrete-time labels in the range [0, T].

The `MarginalLogMean` is a method to compute the log mean of a probability distribution.

The `Beta分布` is a method to compute the probability distribution from a set of beta parameters.


```py
import math

import torch


class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
        ):
        """Create a wrapper class for the forward SDE (VP type).
        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***
        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:
            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)
        Moreover, as lambda(t) is an invertible function, we also support its inverse function:
            t = self.inverse_lambda(lambda_t)
        ===============================================================
        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).
        1. For discrete-time DPMs:
            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.
            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)
            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.
            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).
        2. For continuous-time DPMs:
            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:
            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.
        ===============================================================
        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).
        
        ===============================================================
        Example:
        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)
        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)
        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)
        """

        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.log_alpha_array = log_alphas.reshape((1, -1,)).to(dtype=dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            self.schedule = schedule
            if schedule == 'cosine':
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            def log_alpha_fn(s):
                return torch.log(torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            def t_fn(log_alpha_t):
                return torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2.0 * (1.0 + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t


```

This is a Rust implementation of a neural network classifier that uses the DPM-Solver for stochastic gradient descent. The code includes three functions: `model_fn`, `noise_pred_fn`, and `cond_grad_fn`.

The `model_fn` function takes in a single input `x`, which is the input to the classifier. This function is used for both the training and the prediction phase. It returns the noise prediction for the given input.

The `noise_pred_fn` function is used for the noise prediction. It takes in the input `x` and the current time step `t`. It returns the predicted noise value for the given input and time step.

The `cond_grad_fn` function is used for the gradient computation of the log loss over the conditional distribution. It takes in the input `x` and the current time step `t`. It returns the gradient of the log loss with respect to the input in the given time step.

The `model_fn` function is the main function that defines the neural network model. It takes in the input `x` and the current time step `t`. It returns the noise prediction for the given input and time step.

The neural network classifier can be instantiated by passing an appropriate number of arguments to the constructor.


```py
def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
):
    """Create a wrapper function for the noise prediction model.
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * noise_schedule.total_N
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * sigma_t * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


```

This is a function definition that outlines the multi-step time-步 hornwork initializer for a neural network. The hornwork initializer is a technique used to improve the performance of neural networks by initializing the weights of the network with random values, which can help to避免预设梯度这一问题.

The multi-step time-step hornwork initializer takes in an array of weights `weights` and an array of datetime indices `time_steps`, which indicates the number of time-steps in each hornwork. The function returns the updated weights, as well as a list of intermediate values that were used during the initialization process.

The initializer works by first initializing the weights with random values. Next, it iterates through each time-step and uses the hornwork initializer for each time-step. This allows the initializer to provide a more complex and robust initialization for the neural network.

The hornwork initializer also supports the option of using a corrector to improve the quality of the initialization. If the `use_corrector` parameter is set to `True`, the weights are instead initialized using a corrector function, which adds some extra randomness to the initialization process.

Note that this is just an example implementation and should not be used as is in a production environment. It is recommended to carefully read the documentation for this tool and follow best practices for initializing neural network weights.


```py
class UniPC:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="data_prediction",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
        variant='bh1'
    ):
        """Construct a UniPC. 

        We support both data_prediction and noise_prediction.
        """
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["data_prediction", "noise_prediction"]
        
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
            
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val
        
        self.variant = variant
        self.predict_x0 = algorithm_type == "data_prediction"

    def dynamic_thresholding_fn(self, x0, t=None):
        """
        The dynamic thresholding method. 
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            else:
                orders = [3,] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        elif order == 1:
            K = steps
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization. 
        """
        return self.data_prediction_fn(x, s)

    def multistep_uni_pc_update(self, x, model_prev_list, t_prev_list, t, order, **kwargs):
        if len(t.shape) == 0:
            t = t.view(-1)
        if 'bh' in self.variant:
            return self.multistep_uni_pc_bh_update(x, model_prev_list, t_prev_list, t, order, **kwargs)
        else:
            assert self.variant == 'vary_coeff'
            return self.multistep_uni_pc_vary_update(x, model_prev_list, t_prev_list, t, order, **kwargs)

    def multistep_uni_pc_vary_update(self, x, model_prev_list, t_prev_list, t, order, use_corrector=True):
        #print(f'using unified predictor-corrector with order {order} (solver type: vary coeff)')
        ns = self.noise_schedule
        assert order <= len(model_prev_list)

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = (lambda_prev_i - lambda_prev_0) / h
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        K = len(rks)
        # build C matrix
        C = []

        col = torch.ones_like(rks)
        for k in range(1, K + 1):
            C.append(col)
            col = col * rks / (k + 1) 
        C = torch.stack(C, dim=1)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1) # (B, K)
            C_inv_p = torch.linalg.inv(C[:-1, :-1])
            A_p = C_inv_p

        if use_corrector:
            #print('using corrector')
            C_inv = torch.linalg.inv(C)
            A_c = C_inv

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_ks = []
        factorial_k = 1
        h_phi_k = h_phi_1
        for k in range(1, K + 2):
            h_phi_ks.append(h_phi_k)
            h_phi_k = h_phi_k / hh - 1 / factorial_k
            factorial_k *= (k + 1)

        model_t = None
        if self.predict_x0:
            x_t_ = (
                sigma_t / sigma_prev_0 * x
                - alpha_t * h_phi_1 * model_prev_0
            )
            # now predictor
            x_t = x_t_
            if len(D1s) > 0:
                # compute the residuals for predictor
                for k in range(K - 1):
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * torch.einsum('bkchw,k->bchw', D1s, A_p[k])
            # now corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = (model_t - model_prev_0)
                x_t = x_t_
                k = 0
                for k in range(K - 1):
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * torch.einsum('bkchw,k->bchw', D1s, A_c[k][:-1])
                x_t = x_t - alpha_t * h_phi_ks[K] * (D1_t * A_c[k][-1])
        else:
            log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
            x_t_ = (
                (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                - (sigma_t * h_phi_1) * model_prev_0
            )
            # now predictor
            x_t = x_t_
            if len(D1s) > 0:
                # compute the residuals for predictor
                for k in range(K - 1):
                    x_t = x_t - sigma_t * h_phi_ks[k + 1] * torch.einsum('bkchw,k->bchw', D1s, A_p[k])
            # now corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = (model_t - model_prev_0)
                x_t = x_t_
                k = 0
                for k in range(K - 1):
                    x_t = x_t - sigma_t * h_phi_ks[k + 1] * torch.einsum('bkchw,k->bchw', D1s, A_c[k][:-1])
                x_t = x_t - sigma_t * h_phi_ks[K] * (D1_t * A_c[k][-1])
        return x_t, model_t

    def multistep_uni_pc_bh_update(self, x, model_prev_list, t_prev_list, t, order, x_t=None, use_corrector=True):
        #print(f'using unified predictor-corrector with order {order} (solver type: B(h))')
        ns = self.noise_schedule
        assert order <= len(model_prev_list)

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = (lambda_prev_i - lambda_prev_0) / h
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh) # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == 'bh1':
            B_h = hh
        elif self.variant == 'bh2':
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()
            
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i 

        R = torch.stack(R)
        b = torch.cat(b)

        # now predictor
        use_predictor = len(D1s) > 0 and x_t is None
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1) # (B, K)
            if x_t is None:
                # for order 2, we use a simplified version
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if use_corrector:
            #print('using corrector')
            # for order 1, we use a simplified version
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)

        model_t = None
        if self.predict_x0:
            x_t_ = (
                sigma_t / sigma_prev_0 * x
                - alpha_t * h_phi_1 * model_prev_0
            )

            if x_t is None:
                if use_predictor:
                    pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
                else:
                    pred_res = 0
                x_t = x_t_ - alpha_t * B_h * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:
                    corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
                else:
                    corr_res = 0
                D1_t = (model_t - model_prev_0)
                x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = (
                torch.exp(log_alpha_t - log_alpha_prev_0) * x
                - sigma_t * h_phi_1 * model_prev_0
            )
            if x_t is None:
                if use_predictor:
                    pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
                else:
                    pred_res = 0
                x_t = x_t_ - sigma_t * B_h * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:
                    corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
                else:
                    corr_res = 0
                D1_t = (model_t - model_prev_0)
                x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        return x_t, model_t

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Compute the sample at time `t_end` by UniPC, given the initial `x` at time `t_start`.
        """
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method == 'multistep':
                assert steps >= order
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                
                # Init the first `order` values by lower order multistep UniPC.
                for step in range(1, order):
                    t = timesteps[step]
                    x, model_x = self.multistep_uni_pc_update(x, model_prev_list, t_prev_list, t, step, use_corrector=True)
                    if model_x is None:
                        model_x = self.model_fn(x, t)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(model_x)
                    
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    if lower_order_final:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    if step == steps:
                        #print('do not run corrector at the last step')
                        use_corrector = False
                    else:
                        use_corrector = True
                    x, model_x = self.multistep_uni_pc_update(x, model_prev_list, t_prev_list, t, step_order, use_corrector=use_corrector)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        if model_x is None:
                            model_x = self.model_fn(x, t)
                        model_prev_list[-1] = model_x
            else:
                raise ValueError("Got wrong method {}".format(method))
            
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x


```

This is a function that takes in two tensors `sorted_all_x` and `y_positions_expanded`, and returns a tensor `cand` that represents the candidate locations for a data point in the data space.

The function first sorts the elements of `sorted_all_x` along the second dimension (`dim=2`) using the `torch.sort` method. It then uses the `argmin` method along the same dimension to get the index of the first non-zero element in each sorted element, which is used to determine the start position of the data point in `y_positions_expanded`.

The function then iterates through the sorted elements and uses the `where` clause to select the appropriate candidate position based on the index and the value of `K`. The final result is the tensor `cand`, which represents the candidate locations for each data point in `y_positions_expanded`.



```py
#############################################################
# other utility functions
#############################################################

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


```

这段代码定义了一个名为 `expand_dims` 的函数，它接受一个 PyTorch 张量 `v` 和一个维度 `dims` 作为参数。函数的作用是返回一个与 `v` 具有相同形状，但尺寸为 `(N, 1, 1, ..., 1)` 的张量，其中 `N` 是输入张量的形状，`1` 是维度扩展的维度，`dims` 是输入张量所拥有的维度。

具体来说，函数首先通过 `沿轴` 操作将输入张量 `v` 的形状扩展至 `(N, 1, 1, ..., 1)`，然后通过 `维度` 操作将其扩展到了 `dims` 维度。最后，函数返回一个新的张量，它的形状与 `v` 相同，但尺寸为 `(N, 1, 1, ..., 1)` 并且包含了一个维度为 `dims` 的维度。


```py
def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]
```