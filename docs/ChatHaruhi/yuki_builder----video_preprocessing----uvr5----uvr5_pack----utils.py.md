# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\utils.py`

```py
import torch
import numpy as np
from tqdm import tqdm
import json


# 从指定的 JSON 文件中加载数据并返回
def load_data(file_name: str = "./uvr5_pack/name_params.json") -> dict:
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


# 计算填充值和裁剪尺寸，以及偏移量
def make_padding(width, cropsize, offset):
    left = offset
    roi_size = cropsize - left * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left
    return left, right, roi_size


# 推断函数，根据输入数据和模型进行推断
def inference(X_spec, device, model, aggressiveness, data):
    """
    data ： dic configs
    """

    # 内部执行函数，处理具体推断过程
    def _execute(
        X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half=True
    ):
        model.eval()
        with torch.no_grad():
            preds = []

            iterations = [n_window]

            total_iterations = sum(iterations)
            for i in tqdm(range(n_window)):
                start = i * roi_size
                X_mag_window = X_mag_pad[
                    None, :, :, start : start + data["window_size"]
                ]
                X_mag_window = torch.from_numpy(X_mag_window)
                if is_half:
                    X_mag_window = X_mag_window.half()
                X_mag_window = X_mag_window.to(device)

                pred = model.predict(X_mag_window, aggressiveness)

                pred = pred.detach().cpu().numpy()
                preds.append(pred[0])

            pred = np.concatenate(preds, axis=2)
        return pred

    # 预处理函数，计算幅度和相位
    def preprocess(X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)
        return X_mag, X_phase

    # 对输入的频谱数据进行预处理
    X_mag, X_phase = preprocess(X_spec)

    # 根据最大幅度归一化输入的幅度数据
    coef = X_mag.max()
    X_mag_pre = X_mag / coef

    n_frame = X_mag_pre.shape[2]

    # 计算填充值、裁剪尺寸和窗口数
    pad_l, pad_r, roi_size = make_padding(n_frame, data["window_size"], model.offset)
    n_window = int(np.ceil(n_frame / roi_size))

    # 对幅度数据进行填充
    X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

    # 判断模型参数类型是否为 torch.float16
    if list(model.state_dict().values())[0].dtype == torch.float16:
        is_half = True
    else:
        is_half = False

    # 执行推断过程
    pred = _execute(
        X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half
    )
    pred = pred[:, :, :n_frame]

    # 如果开启了测试时间增强 (TTA)
    if data["tta"]:
        pad_l += roi_size // 2
        pad_r += roi_size // 2
        n_window += 1

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

        pred_tta = _execute(
            X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half
        )
        pred_tta = pred_tta[:, :, roi_size // 2 :]
        pred_tta = pred_tta[:, :, :n_frame]

        return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.0j * X_phase)
    else:
        return pred * coef, X_mag, np.exp(1.0j * X_phase)


# 获取模型路径和哈希值相关的名称参数
def _get_name_params(model_path, model_hash):
    data = load_data()
    flag = False
    ModelName = model_path
    # 遍历数据中的每种类型
    for type in list(data):
        # 遍历每种类型中的第一个模型
        for model in list(data[type][0]):
            # 遍历当前模型的所有索引
            for i in range(len(data[type][0][model])):
                # 检查当前模型的哈希名是否与给定的模型哈希名匹配
                if str(data[type][0][model][i]["hash_name"]) == model_hash:
                    # 如果匹配，设置标志为True
                    flag = True
                # 如果当前模型的哈希名在ModelName中，也设置标志为True
                elif str(data[type][0][model][i]["hash_name"]) in ModelName:
                    flag = True

                # 如果标志为True，获取当前模型的参数和名称
                if flag:
                    model_params_auto = data[type][0][model][i]["model_params"]
                    param_name_auto = data[type][0][model][i]["param_name"]
                    # 如果类型为"equivalent"，直接返回参数名称和模型参数
                    if type == "equivalent":
                        return param_name_auto, model_params_auto
                    else:
                        # 否则，重置标志为False，继续遍历
                        flag = False
    # 最后返回最后一个遍历到的模型的参数名称和模型参数（最后一个匹配到的值）
    return param_name_auto, model_params_auto
```