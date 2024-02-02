# `so-vits-svc\onnx_export.py`

```py
# 导入必要的库
import argparse
import json
import torch
import utils
from onnxexport.model_onnx_speaker_mix import SynthesizerTrn

# 创建解析器对象
parser = argparse.ArgumentParser(description='SoVitsSvc OnnxExport')

# 定义函数，用于导出模型到ONNX格式
def OnnxExport(path=None):
    # 设置设备为CPU
    device = torch.device("cpu")
    # 从配置文件中获取超参数
    hps = utils.get_hparams_from_file(f"checkpoints/{path}/config.json")
    # 创建SynthesizerTrn对象
    SVCVITS = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    # 加载模型的参数
    _ = utils.load_checkpoint(f"checkpoints/{path}/model.pth", SVCVITS, None)
    # 将模型设置为评估模式，并移动到指定设备
    _ = SVCVITS.eval().to(device)
    # 冻结模型参数，不进行梯度更新
    for i in SVCVITS.parameters():
        i.requires_grad = False
    
    # 设置帧数
    num_frames = 200

    # 创建测试用的隐层单元、音高、音量、mel2ph、uv、噪声、说话人ID
    test_hidden_unit = torch.rand(1, num_frames, SVCVITS.gin_channels)
    test_pitch = torch.rand(1, num_frames)
    test_vol = torch.rand(1, num_frames)
    test_mel2ph = torch.LongTensor(torch.arange(0, num_frames)).unsqueeze(0)
    test_uv = torch.ones(1, num_frames, dtype=torch.float32)
    test_noise = torch.randn(1, 192, num_frames)
    test_sid = torch.LongTensor([0])
    export_mix = True
    # 如果说话人数量小于2，则不导出混合模型
    if len(hps.spk) < 2:
        export_mix = False
    
    # 如果需要导出混合模型
    if export_mix:
        spk_mix = []
        n_spk = len(hps.spk)
        for i in range(n_spk):
            spk_mix.append(1.0/float(n_spk))
        test_sid = torch.tensor(spk_mix)
        SVCVITS.export_chara_mix(hps.spk)
        test_sid = test_sid.unsqueeze(0)
        test_sid = test_sid.repeat(num_frames, 1)
    
    # 将模型设置为评估模式
    SVCVITS.eval()

    # 根据是否导出混合模型，设置数据轴的字典
    if export_mix:
        daxes = {
            "c": [0, 1],
            "f0": [1],
            "mel2ph": [1],
            "uv": [1],
            "noise": [2],
            "sid":[0]
        }
    else:
        daxes = {
            "c": [0, 1],
            "f0": [1],
            "mel2ph": [1],
            "uv": [1],
            "noise": [2]
        }
    
    # 设置输入和输出的名称
    input_names = ["c", "f0", "mel2ph", "uv", "noise", "sid"]
    output_names = ["audio", ]
    # 如果存在音量嵌入，则将"vol"添加到输入名称列表中
    if SVCVITS.vol_embedding:
        input_names.append("vol")
        # 创建包含"vol"键和值为[1]的字典
        vol_dadict = {"vol" : [1]}
        # 更新daxes字典，添加vol_dadict的内容
        daxes.update(vol_dadict)
        # 将测试输入转换为元组，如果存在音量嵌入，则包含test_vol.to(device)
        test_inputs = (
            test_hidden_unit.to(device),
            test_pitch.to(device),
            test_mel2ph.to(device),
            test_uv.to(device),
            test_noise.to(device),
            test_sid.to(device),
            test_vol.to(device)
        )
    else:
        # 将测试输入转换为元组，如果不存在音量嵌入，则不包含test_vol.to(device)
        test_inputs = (
            test_hidden_unit.to(device),
            test_pitch.to(device),
            test_mel2ph.to(device),
            test_uv.to(device),
            test_noise.to(device),
            test_sid.to(device)
        )

    # 将SVCVITS模型转换为Torch脚本
    # SVCVITS = torch.jit.script(SVCVITS)
    # 将测试输入传递给SVCVITS模型，并将结果存储在输出中
    SVCVITS(test_hidden_unit.to(device),
            test_pitch.to(device),
            test_mel2ph.to(device),
            test_uv.to(device),
            test_noise.to(device),
            test_sid.to(device),
            test_vol.to(device))
    
    # 调用SVCVITS模型的dec属性的OnnxExport方法
    SVCVITS.dec.OnnxExport()

    # 将SVCVITS模型导出为ONNX格式
    torch.onnx.export(
        SVCVITS,
        test_inputs,
        f"checkpoints/{path}/{path}_SoVits.onnx",
        dynamic_axes=daxes,
        do_constant_folding=False,
        opset_version=16,
        verbose=False,
        input_names=input_names,
        output_names=output_names
    )

    # 根据SVCVITS模型的gin_channels属性确定vec_lay的值
    vec_lay = "layer-12" if SVCVITS.gin_channels == 768 else "layer-9"
    # 创建一个空列表用于存储hps.spk字典的键
    spklist = []
    # 遍历hps.spk字典的键，并将其添加到spklist列表中
    for key in hps.spk.keys():
        spklist.append(key)

    # 创建MoeVSConf字典，包含各种配置信息
    MoeVSConf = {
        "Folder" : f"{path}",
        "Name" : f"{path}",
        "Type" : "SoVits",
        "Rate" : hps.data.sampling_rate,
        "Hop" : hps.data.hop_length,
        "Hubert": f"vec-{SVCVITS.gin_channels}-{vec_lay}",
        "SoVits4": True,
        "SoVits3": False,
        "CharaMix": export_mix,
        "Volume": SVCVITS.vol_embedding,
        "HiddenSize": SVCVITS.gin_channels,
        "Characters": spklist,
        "Cluster": ""
    }
    # 使用写入模式打开指定路径下的 JSON 文件，如果文件不存在则创建
    with open(f"checkpoints/{path}.json", 'w') as MoeVsConfFile:
        # 将 MoeVSConf 对象以 JSON 格式写入到文件中，缩进为4个空格
        json.dump(MoeVSConf, MoeVsConfFile, indent = 4)
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 添加一个命令行参数，用于指定模型文件夹名，默认为"TransformerFlow"
    parser.add_argument('-n', '--model_name', type=str, default="TransformerFlow", help='模型文件夹名（根目录下新建ckeckpoints文件夹，在此文件夹下建立一个新的文件夹，放置模型，该文件夹名即为此项）')
    # 解析命令行参数
    args = parser.parse_args()
    # 获取模型文件夹名
    path = args.model_name
    # 调用OnnxExport函数，传入模型文件夹名作为参数
    OnnxExport(path)
```