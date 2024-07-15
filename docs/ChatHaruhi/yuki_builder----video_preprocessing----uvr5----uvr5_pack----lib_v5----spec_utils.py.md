# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\spec_utils.py`

```py
# 导入所需的模块和库
import os, librosa  # 导入操作系统和音频处理库
import numpy as np  # 导入数值计算库
import soundfile as sf  # 导入音频文件读写库
from tqdm import tqdm  # 导入进度条显示库
import json, math, hashlib  # 导入 JSON、数学和哈希库

# 定义函数：从中心裁剪音频数据
def crop_center(h1, h2):
    h1_shape = h1.size()  # 获取 h1 的尺寸
    h2_shape = h2.size()  # 获取 h2 的尺寸

    if h1_shape[3] == h2_shape[3]:  # 检查 h1 和 h2 的时间维度是否相同
        return h1  # 如果相同，直接返回 h1
    elif h1_shape[3] < h2_shape[3]:  # 如果 h1 的时间维度小于 h2 的时间维度
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")  # 抛出数值错误异常

    # 计算裁剪起始和结束时间
    s_time = (h1_shape[3] - h2_shape[3]) // 2  # 计算开始时间
    e_time = s_time + h2_shape[3]  # 计算结束时间
    h1 = h1[:, :, :, s_time:e_time]  # 对 h1 进行时间维度裁剪

    return h1  # 返回裁剪后的 h1

# 定义函数：将波形转换为频谱图
def wave_to_spectrogram(
    wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False
):
    if reverse:
        wave_left = np.flip(np.asfortranarray(wave[0]))  # 若反转标志为真，则反转左声道波形
        wave_right = np.flip(np.asfortranarray(wave[1]))  # 若反转标志为真，则反转右声道波形
    elif mid_side:
        wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)  # 若 mid_side 标志为真，则左声道为两声道平均值
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))  # 若 mid_side 标志为真，则右声道为两声道差值
    elif mid_side_b2:
        wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))  # 若 mid_side_b2 标志为真，则左声道为右声道加上左声道一半
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))  # 若 mid_side_b2 标志为真，则右声道为左声道减去右声道一半
    else:
        wave_left = np.asfortranarray(wave[0])  # 否则左声道为原始左声道
        wave_right = np.asfortranarray(wave[1])  # 否则右声道为原始右声道

    # 使用 librosa 库计算左声道和右声道的短时傅里叶变换
    spec_left = librosa.stft(wave_left, n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft, hop_length=hop_length)

    spec = np.asfortranarray([spec_left, spec_right])  # 将左右声道的频谱图组合成复数数组形式

    return spec  # 返回组合后的频谱图

# 定义多线程版本的函数：将波形转换为频谱图
def wave_to_spectrogram_mt(
    wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False
):
    import threading  # 导入线程模块

    if reverse:
        wave_left = np.flip(np.asfortranarray(wave[0]))  # 若反转标志为真，则反转左声道波形
        wave_right = np.flip(np.asfortranarray(wave[1]))  # 若反转标志为真，则反转右声道波形
    elif mid_side:
        wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)  # 若 mid_side 标志为真，则左声道为两声道平均值
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))  # 若 mid_side 标志为真，则右声道为两声道差值
    elif mid_side_b2:
        wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))  # 若 mid_side_b2 标志为真，则左声道为右声道加上左声道一半
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))  # 若 mid_side_b2 标志为真，则右声道为左声道减去右声道一半
    else:
        wave_left = np.asfortranarray(wave[0])  # 否则左声道为原始左声道
        wave_right = np.asfortranarray(wave[1])  # 否则右声道为原始右声道

    def run_thread(**kwargs):
        global spec_left  # 声明全局变量：左声道频谱图
        spec_left = librosa.stft(**kwargs)  # 计算左声道的短时傅里叶变换

    # 创建线程对象，计算左声道的短时傅里叶变换
    thread = threading.Thread(
        target=run_thread,
        kwargs={"y": wave_left, "n_fft": n_fft, "hop_length": hop_length},
    )
    thread.start()  # 启动线程
    spec_right = librosa.stft(y=wave_right, n_fft=n_fft, hop_length=hop_length)  # 计算右声道的短时傅里叶变换
    thread.join()  # 等待线程执行完成

    spec = np.asfortranarray([spec_left, spec_right])  # 将左右声道的频谱图组合成复数数组形式

    return spec  # 返回组合后的频谱图

# 定义函数：将多个频谱图合并为一个
def combine_spectrograms(specs, mp):
    l = min([specs[i].shape[2] for i in specs])  # 计算最小的频谱长度
    spec_c = np.zeros(shape=(2, mp.param["bins"] + 1, l), dtype=np.complex64)  # 创建复数数组用于存储合并后的频谱图
    offset = 0  # 偏移量初始化为0
    bands_n = len(mp.param["band"])  # 获取频谱带宽数量
    # 遍历每个波段的数据，其中 d 从 1 到 bands_n（波段数）
    for d in range(1, bands_n + 1):
        # 计算当前波段 d 的裁剪范围并计算裁剪后的高度 h
        h = mp.param["band"][d]["crop_stop"] - mp.param["band"][d]["crop_start"]
        # 将波段 d 的裁剪后的数据复制到 spec_c 中的指定位置
        spec_c[:, offset : offset + h, :l] = specs[d][
            :, mp.param["band"][d]["crop_start"] : mp.param["band"][d]["crop_stop"], :l
        ]
        # 更新 offset，以便下一个波段的数据可以正确放置
        offset += h

    # 检查 offset 是否超过了 mp.param["bins"]，如果是则抛出 ValueError 异常
    if offset > mp.param["bins"]:
        raise ValueError("Too much bins")

    # 低通滤波器
    if (
        mp.param["pre_filter_start"] > 0
    ):  # 如果预处理滤波的起始位置大于 0
        # 如果只有一个波段
        if bands_n == 1:
            # 对 spec_c 应用低通滤波器，过滤范围为 mp.param["pre_filter_start"] 到 mp.param["pre_filter_stop"]
            spec_c = fft_lp_filter(
                spec_c, mp.param["pre_filter_start"], mp.param["pre_filter_stop"]
            )
        else:
            gp = 1
            # 遍历预处理滤波的范围，对每个频率应用衰减系数
            for b in range(
                mp.param["pre_filter_start"] + 1, mp.param["pre_filter_stop"]
            ):
                # 计算当前频率的衰减系数 g
                g = math.pow(
                    10, -(b - mp.param["pre_filter_start"]) * (3.5 - gp) / 20.0
                )
                # 更新 gp 为当前的衰减系数
                gp = g
                # 对 spec_c 的第 b 列数据应用衰减系数 g
                spec_c[:, b, :] *= g

    # 返回以 Fortran 风格存储的 spec_c 数组
    return np.asfortranarray(spec_c)
# 将频谱转换为图像
def spectrogram_to_image(spec, mode="magnitude"):
    # 根据模式选择处理方式：幅度模式
    if mode == "magnitude":
        # 如果频谱是复数类型，取其幅度；否则直接使用频谱
        if np.iscomplexobj(spec):
            y = np.abs(spec)
        else:
            y = spec
        # 对幅度进行对数变换，避免数值过大
        y = np.log10(y**2 + 1e-8)
    # 相位模式
    elif mode == "phase":
        # 如果频谱是复数类型，取其相位；否则直接使用频谱
        if np.iscomplexobj(spec):
            y = np.angle(spec)
        else:
            y = spec

    # 将数据归一化到0-255范围内
    y -= y.min()
    y *= 255 / y.max()
    # 将数据类型转换为无符号8位整数
    img = np.uint8(y)

    # 如果数据是三维的，调整维度顺序并添加通道信息
    if y.ndim == 3:
        img = img.transpose(1, 2, 0)
        img = np.concatenate([np.max(img, axis=2, keepdims=True), img], axis=2)

    return img


# 强烈减少声音的函数
def reduce_vocal_aggressively(X, y, softmask):
    # 计算伴奏信号
    v = X - y
    # 计算声音信号的幅度
    y_mag_tmp = np.abs(y)
    # 计算伴奏信号的幅度
    v_mag_tmp = np.abs(v)

    # 创建用于削减声音的掩码
    v_mask = v_mag_tmp > y_mag_tmp
    # 削减声音信号的幅度
    y_mag = np.clip(y_mag_tmp - v_mag_tmp * v_mask * softmask, 0, np.inf)

    # 返回削减后的声音信号
    return y_mag * np.exp(1.0j * np.angle(y))


# 屏蔽静音部分的函数
def mask_silence(mag, ref, thres=0.2, min_range=64, fade_size=32):
    # 检查参数有效性
    if min_range < fade_size * 2:
        raise ValueError("min_range must be >= fade_area * 2")

    # 复制幅度信息
    mag = mag.copy()

    # 找到参考信号中低于阈值的索引
    idx = np.where(ref.mean(axis=(0, 1)) < thres)[0]
    # 分割静音部分的起始点和终止点
    starts = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
    ends = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
    # 筛选出长度大于最小范围的静音部分
    uninformative = np.where(ends - starts > min_range)[0]
    if len(uninformative) > 0:
        starts = starts[uninformative]
        ends = ends[uninformative]
        old_e = None
        # 处理静音部分的重叠和衰减
        for s, e in zip(starts, ends):
            if old_e is not None and s - old_e < fade_size:
                s = old_e - fade_size * 2

            if s != 0:
                weight = np.linspace(0, 1, fade_size)
                mag[:, :, s : s + fade_size] += weight * ref[:, :, s : s + fade_size]
            else:
                s -= fade_size

            if e != mag.shape[2]:
                weight = np.linspace(1, 0, fade_size)
                mag[:, :, e - fade_size : e] += weight * ref[:, :, e - fade_size : e]
            else:
                e += fade_size

            mag[:, :, s + fade_size : e - fade_size] += ref[
                :, :, s + fade_size : e - fade_size
            ]
            old_e = e

    # 返回屏蔽静音后的幅度信息
    return mag


# 对齐波形头尾的函数
def align_wave_head_and_tail(a, b):
    # 计算长度较短的一方
    l = min([a[0].size, b[0].size])

    # 返回截取后长度一致的波形
    return a[:l, :l], b[:l, :l]


# 缓存或加载函数
def cache_or_load(mix_path, inst_path, mp):
    # 获取混音文件和乐器文件的基本名称
    mix_basename = os.path.splitext(os.path.basename(mix_path))[0]
    inst_basename = os.path.splitext(os.path.basename(inst_path))[0]

    # 创建缓存目录
    cache_dir = "mph{}".format(
        hashlib.sha1(json.dumps(mp.param, sort_keys=True).encode("utf-8")).hexdigest()
    )
    mix_cache_dir = os.path.join("cache", cache_dir)
    inst_cache_dir = os.path.join("cache", cache_dir)

    os.makedirs(mix_cache_dir, exist_ok=True)
    os.makedirs(inst_cache_dir, exist_ok=True)

    # 混音文件和乐器文件的缓存路径
    mix_cache_path = os.path.join(mix_cache_dir, mix_basename + ".npy")
    inst_cache_path = os.path.join(inst_cache_dir, inst_basename + ".npy")
    # 检查混合音频缓存路径和乐器音频缓存路径是否存在
    if os.path.exists(mix_cache_path) and os.path.exists(inst_cache_path):
        # 如果存在缓存文件，从缓存文件中加载混合音频的频谱数据和乐器音频的频谱数据
        X_spec_m = np.load(mix_cache_path)
        y_spec_m = np.load(inst_cache_path)
    else:
        # 如果缓存文件不存在，则开始处理音频数据

        # 初始化空字典用于存储每个频段的波形和频谱
        X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}

        # 逆向遍历频段参数列表
        for d in range(len(mp.param["band"]), 0, -1):
            bp = mp.param["band"][d]

            if d == len(mp.param["band"]):  # 高频段
                # 加载混合音频和乐器音频的波形数据
                X_wave[d], _ = librosa.load(
                    mix_path, bp["sr"], False, dtype=np.float32, res_type=bp["res_type"]
                )
                y_wave[d], _ = librosa.load(
                    inst_path,
                    bp["sr"],
                    False,
                    dtype=np.float32,
                    res_type=bp["res_type"],
                )
            else:  # 低频段
                # 对上一频段的波形数据进行重采样
                X_wave[d] = librosa.resample(
                    X_wave[d + 1],
                    mp.param["band"][d + 1]["sr"],
                    bp["sr"],
                    res_type=bp["res_type"],
                )
                y_wave[d] = librosa.resample(
                    y_wave[d + 1],
                    mp.param["band"][d + 1]["sr"],
                    bp["sr"],
                    res_type=bp["res_type"],
                )

            # 对波形数据进行头尾对齐处理
            X_wave[d], y_wave[d] = align_wave_head_and_tail(X_wave[d], y_wave[d])

            # 将波形数据转换为频谱数据
            X_spec_s[d] = wave_to_spectrogram(
                X_wave[d],
                bp["hl"],
                bp["n_fft"],
                mp.param["mid_side"],
                mp.param["mid_side_b2"],
                mp.param["reverse"],
            )
            y_spec_s[d] = wave_to_spectrogram(
                y_wave[d],
                bp["hl"],
                bp["n_fft"],
                mp.param["mid_side"],
                mp.param["mid_side_b2"],
                mp.param["reverse"],
            )

        # 释放不再需要的波形数据字典内存
        del X_wave, y_wave

        # 组合所有频段的混合音频和乐器音频的频谱数据
        X_spec_m = combine_spectrograms(X_spec_s, mp)
        y_spec_m = combine_spectrograms(y_spec_s, mp)

        # 检查混合音频和乐器音频的频谱数据形状是否相同
        if X_spec_m.shape != y_spec_m.shape:
            raise ValueError("The combined spectrograms are different: " + mix_path)

        # 获取混合音频文件的扩展名
        _, ext = os.path.splitext(mix_path)

        # 将混合音频的频谱数据和乐器音频的频谱数据保存到缓存文件中
        np.save(mix_cache_path, X_spec_m)
        np.save(inst_cache_path, y_spec_m)

    # 返回处理后的混合音频的频谱数据和乐器音频的频谱数据
    return X_spec_m, y_spec_m
# 将左声道的频谱转换为 Fortran 风格的数组
spec_left = np.asfortranarray(spec[0])
# 将右声道的频谱转换为 Fortran 风格的数组
spec_right = np.asfortranarray(spec[1])

# 根据左声道的频谱重建波形
wave_left = librosa.istft(spec_left, hop_length=hop_length)
# 根据右声道的频谱重建波形
wave_right = librosa.istft(spec_right, hop_length=hop_length)

# 如果需要反转波形
if reverse:
    # 返回左右声道波形的反转版本
    return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
# 如果是 mid-side 编码
elif mid_side:
    # 返回经过 mid-side 解码后的左右声道波形
    return np.asfortranarray(
        [np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)]
    )
# 如果是 mid-side_b2 编码
elif mid_side_b2:
    # 返回经过 mid-side_b2 解码后的左右声道波形
    return np.asfortranarray(
        [
            np.add(wave_right / 1.25, 0.4 * wave_left),
            np.subtract(wave_left / 1.25, 0.4 * wave_right),
        ]
    )
else:
    # 返回未编码的原始左右声道波形
    return np.asfortranarray([wave_left, wave_right])


def spectrogram_to_wave_mt(spec, hop_length, mid_side, reverse, mid_side_b2):
    import threading

    # 将左声道的频谱转换为 Fortran 风格的数组
    spec_left = np.asfortranarray(spec[0])
    # 将右声道的频谱转换为 Fortran 风格的数组
    spec_right = np.asfortranarray(spec[1])

    # 定义一个在新线程中执行的函数，重建左声道波形
    def run_thread(**kwargs):
        global wave_left
        wave_left = librosa.istft(**kwargs)

    # 创建一个新线程来执行波形重建任务
    thread = threading.Thread(
        target=run_thread, kwargs={"stft_matrix": spec_left, "hop_length": hop_length}
    )
    thread.start()
    
    # 在当前线程中重建右声道的波形
    wave_right = librosa.istft(spec_right, hop_length=hop_length)
    
    # 等待新线程执行完成
    thread.join()

    # 如果需要反转波形
    if reverse:
        # 返回左右声道波形的反转版本
        return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
    # 如果是 mid-side 编码
    elif mid_side:
        # 返回经过 mid-side 解码后的左右声道波形
        return np.asfortranarray(
            [np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)]
        )
    # 如果是 mid-side_b2 编码
    elif mid_side_b2:
        # 返回经过 mid-side_b2 解码后的左右声道波形
        return np.asfortranarray(
            [
                np.add(wave_right / 1.25, 0.4 * wave_left),
                np.subtract(wave_left / 1.25, 0.4 * wave_right),
            ]
        )
    else:
        # 返回未编码的原始左右声道波形
        return np.asfortranarray([wave_left, wave_right])


def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None):
    # 创建一个空字典，用于存储不同频段的波形数据
    wave_band = {}
    # 获取频段数量
    bands_n = len(mp.param["band"])
    # 设置偏移量初始值为 0
    offset = 0
    # 对每个频带进行循环处理，范围是从1到bands_n + 1
    for d in range(1, bands_n + 1):
        # 从参数字典中获取当前频带的参数
        bp = mp.param["band"][d]
        
        # 创建一个复数类型的三维数组，用于存储频谱信息
        spec_s = np.ndarray(
            shape=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex
        )
        
        # 计算当前频带的裁剪范围
        h = bp["crop_stop"] - bp["crop_start"]
        
        # 将原始频谱数据中指定范围的数据复制到 spec_s 中对应的位置
        spec_s[:, bp["crop_start"] : bp["crop_stop"], :] = spec_m[
            :, offset : offset + h, :
        ]
        
        # 更新偏移量，以便处理下一个频带
        offset += h
        
        # 如果当前频带是最高频带
        if d == bands_n:
            # 如果有额外的高频数据需要处理
            if extra_bins_h:
                # 获取最大频率 bin 的索引
                max_bin = bp["n_fft"] // 2
                # 将额外的高频数据复制到 spec_s 中指定位置
                spec_s[:, max_bin - extra_bins_h : max_bin, :] = extra_bins[
                    :, :extra_bins_h, :
                ]
            
            # 如果有高通滤波器的起始频率大于 0，则对 spec_s 进行高通滤波处理
            if bp["hpf_start"] > 0:
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
            
            # 如果只有一个频带，则将 spec_s 转换为音频波形数据
            if bands_n == 1:
                wave = spectrogram_to_wave(
                    spec_s,
                    bp["hl"],
                    mp.param["mid_side"],
                    mp.param["mid_side_b2"],
                    mp.param["reverse"],
                )
            else:
                # 如果有多个频带，则将当前频带的波形数据添加到总波形数据中
                wave = np.add(
                    wave,
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                )
        
        # 如果当前频带不是最高频带
        else:
            # 获取下一个频带的采样率
            sr = mp.param["band"][d + 1]["sr"]
            
            # 如果当前频带是第一个频带（最低频带），则进行低通滤波和重采样处理
            if d == 1:
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave = librosa.resample(
                    y=spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                    orig_sr=bp["sr"],
                    target_sr=sr,
                    res_type="sinc_fastest",
                )
            
            # 如果当前频带是中间频带，则先进行高通滤波和低通滤波处理，然后添加到 wave 中
            else:
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave2 = np.add(
                    wave,
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                )
                # 使用 scipy 方法对波形数据进行重采样
                wave = librosa.core.resample(y=wave2, orig_sr=bp["sr"], target_sr=sr, res_type="scipy")
    
    # 返回转置后的波形数据
    return wave.T
# 如果是低通滤波器，根据指定的频率范围对频谱进行低通滤波处理
def fft_lp_filter(spec, bin_start, bin_stop):
    # 初始化增益参数为1.0
    g = 1.0
    # 遍历从 bin_start 到 bin_stop 的频率区间
    for b in range(bin_start, bin_stop):
        # 计算每个频率 bin 的增益减少量
        g -= 1 / (bin_stop - bin_start)
        # 对频谱的第二维度（频率维度）进行低通滤波处理
        spec[:, b, :] = g * spec[:, b, :]

    # 将 bin_stop 之后的频率区域置零
    spec[:, bin_stop:, :] *= 0

    # 返回处理后的频谱
    return spec


# 如果是高通滤波器，根据指定的频率范围对频谱进行高通滤波处理
def fft_hp_filter(spec, bin_start, bin_stop):
    # 初始化增益参数为1.0
    g = 1.0
    # 遍历从 bin_start 到 bin_stop 的频率区间（反向）
    for b in range(bin_start, bin_stop, -1):
        # 计算每个频率 bin 的增益减少量
        g -= 1 / (bin_start - bin_stop)
        # 对频谱的第二维度（频率维度）进行高通滤波处理
        spec[:, b, :] = g * spec[:, b, :]

    # 将从频率 bin_stop 到 0 的频率区域置零
    spec[:, 0 : bin_stop + 1, :] *= 0

    # 返回处理后的频谱
    return spec


# 根据指定的条件进行镜像处理
def mirroring(a, spec_m, input_high_end, mp):
    if "mirroring" == a:
        # 对输入的高端数据进行镜像处理，返回处理后的数据
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"]
                    - 10
                    - input_high_end.shape[1] : mp.param["pre_filter_start"]
                    - 10,
                    :,
                ]
            ),
            1,
        )
        mirror = mirror * np.exp(1.0j * np.angle(input_high_end))

        return np.where(
            np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror
        )

    if "mirroring2" == a:
        # 对输入的高端数据进行镜像处理，返回处理后的数据
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"]
                    - 10
                    - input_high_end.shape[1] : mp.param["pre_filter_start"]
                    - 10,
                    :,
                ]
            ),
            1,
        )
        mi = np.multiply(mirror, input_high_end * 1.7)

        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)


# 将多个频谱合并，根据指定的方法（最小值或最大值）进行合并
def ensembling(a, specs):
    # 初始化第一个频谱为 specs 中的第一个元素
    for i in range(1, len(specs)):
        if i == 1:
            spec = specs[0]

        # 获取两个频谱的最小长度
        ln = min([spec.shape[2], specs[i].shape[2]])
        spec = spec[:, :, :ln]
        specs[i] = specs[i][:, :, :ln]

        # 根据指定的方法（最小值或最大值）对频谱进行合并
        if "min_mag" == a:
            spec = np.where(np.abs(specs[i]) <= np.abs(spec), specs[i], spec)
        if "max_mag" == a:
            spec = np.where(np.abs(specs[i]) >= np.abs(spec), specs[i], spec)

    # 返回合并后的频谱
    return spec


# 对输入波形进行短时傅里叶变换（STFT）
def stft(wave, nfft, hl):
    # 将左右声道的波形转换为 Fortran 风格数组
    wave_left = np.asfortranarray(wave[0])
    wave_right = np.asfortranarray(wave[1])
    # 对左右声道分别进行短时傅里叶变换
    spec_left = librosa.stft(wave_left, nfft, hop_length=hl)
    spec_right = librosa.stft(wave_right, nfft, hop_length=hl)
    # 将变换后的频谱重新组合成 Fortran 风格的数组
    spec = np.asfortranarray([spec_left, spec_right])

    # 返回变换后的频谱
    return spec


# 对输入的频谱进行逆短时傅里叶变换（ISTFT）
def istft(spec, hl):
    # 将左右声道的频谱转换为 Fortran 风格数组
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])
    # 对左右声道分别进行逆短时傅里叶变换
    wave_left = librosa.istft(spec_left, hop_length=hl)
    wave_right = librosa.istft(spec_right, hop_length=hl)
    # 将逆变换后的波形重新组合成 Fortran 风格的数组
    wave = np.asfortranarray([wave_left, wave_right])

    # 未返回波形数据，需要添加 return 语句
    return wave


if __name__ == "__main__":
    import cv2
    import sys
    import time
    import argparse
    from model_param_init import ModelParameters

    p = argparse.ArgumentParser()
    # 添加一个命令行参数，用于指定算法类型，支持的选项有：invert, invert_p, min_mag, max_mag, deep, align，默认为 min_mag
    p.add_argument(
        "--algorithm",
        "-a",
        type=str,
        choices=["invert", "invert_p", "min_mag", "max_mag", "deep", "align"],
        default="min_mag",
    )
    # 添加一个命令行参数，用于指定模型参数的文件路径，默认为 "modelparams/1band_sr44100_hl512.json"
    p.add_argument(
        "--model_params",
        "-m",
        type=str,
        default=os.path.join("modelparams", "1band_sr44100_hl512.json"),
    )
    # 添加一个命令行参数，用于指定输出文件名，默认为 "output"
    p.add_argument("--output_name", "-o", type=str, default="output")
    # 添加一个命令行参数，如果设置，则仅提取人声部分，默认为 False
    p.add_argument("--vocals_only", "-v", action="store_true")
    # 添加一个位置参数，用于接收输入文件的路径，支持多个文件输入
    p.add_argument("input", nargs="+")
    # 解析命令行参数，返回一个包含所有参数信息的对象
    args = p.parse_args()

    # 记录程序开始运行的时间
    start_time = time.time()

    # 如果算法以 "invert" 开头并且输入文件数不等于 2，则抛出 ValueError 异常
    if args.algorithm.startswith("invert") and len(args.input) != 2:
        raise ValueError("There should be two input files.")

    # 如果算法不以 "invert" 开头并且输入文件数小于 2，则抛出 ValueError 异常
    if not args.algorithm.startswith("invert") and len(args.input) < 2:
        raise ValueError("There must be at least two input files.")

    # 初始化波形和频谱字典
    wave, specs = {}, {}
    # 根据模型参数文件路径创建模型参数对象
    mp = ModelParameters(args.model_params)

    # 遍历输入文件列表
    for i in range(len(args.input)):
        # 初始化频谱字典
        spec = {}

        # 遍历模型参数中的各个频段
        for d in range(len(mp.param["band"]), 0, -1):
            # 获取当前频段的参数
            bp = mp.param["band"][d]

            # 如果当前频段是最高频段
            if d == len(mp.param["band"]):
                # 加载音频文件并转换为波形数据
                wave[d], _ = librosa.load(
                    args.input[i],
                    bp["sr"],
                    False,
                    dtype=np.float32,
                    res_type=bp["res_type"],
                )

                # 如果波形数据是单声道，则复制成立体声数据
                if len(wave[d].shape) == 1:
                    wave[d] = np.array([wave[d], wave[d]])
            else:
                # 对低频段进行重采样
                wave[d] = librosa.resample(
                    wave[d + 1],
                    mp.param["band"][d + 1]["sr"],
                    bp["sr"],
                    res_type=bp["res_type"],
                )

            # 将波形数据转换为频谱数据
            spec[d] = wave_to_spectrogram(
                wave[d],
                bp["hl"],
                bp["n_fft"],
                mp.param["mid_side"],
                mp.param["mid_side_b2"],
                mp.param["reverse"],
            )

        # 将各个频段的频谱数据合并成一个整体频谱数据
        specs[i] = combine_spectrograms(spec, mp)

    # 释放波形数据的内存占用
    del wave

    # 如果算法是 "deep"，执行深度处理操作
    if args.algorithm == "deep":
        # 生成深度处理后的频谱数据
        d_spec = np.where(np.abs(specs[0]) <= np.abs(specs[1]), specs[0], specs[1])
        # 计算处理后的频谱数据与原始频谱数据的差值
        v_spec = d_spec - specs[1]
        # 将混合后的频谱数据转换为音频文件并保存
        sf.write(
            os.path.join("{}.wav".format(args.output_name)),
            cmb_spectrogram_to_wave(v_spec, mp),
            mp.param["sr"],
        )
    # 检查算法参数是否以 "invert" 开头
    if args.algorithm.startswith("invert"):
        # 计算两个频谱的最小长度
        ln = min([specs[0].shape[2], specs[1].shape[2]])
        # 对两个频谱进行截取，保留最小长度部分
        specs[0] = specs[0][:, :, :ln]
        specs[1] = specs[1][:, :, :ln]

        # 如果算法参数为 "invert_p"
        if "invert_p" == args.algorithm:
            # 计算第一个频谱的幅度
            X_mag = np.abs(specs[0])
            # 计算第二个频谱的幅度
            y_mag = np.abs(specs[1])
            # 计算两者幅度的最大值
            max_mag = np.where(X_mag >= y_mag, X_mag, y_mag)
            # 对第二个频谱进行相位反转
            v_spec = specs[1] - max_mag * np.exp(1.0j * np.angle(specs[0]))
        else:
            # 对第二个频谱进行更激进的降噪处理
            specs[1] = reduce_vocal_aggressively(specs[0], specs[1], 0.2)
            # 计算声音部分的频谱
            v_spec = specs[0] - specs[1]

            # 如果不仅需要声音部分
            if not args.vocals_only:
                # 计算第一个频谱的幅度
                X_mag = np.abs(specs[0])
                # 计算第二个频谱的幅度
                y_mag = np.abs(specs[1])
                # 计算声音部分的幅度
                v_mag = np.abs(v_spec)

                # 将幅度转换为图像格式
                X_image = spectrogram_to_image(X_mag)
                y_image = spectrogram_to_image(y_mag)
                v_image = spectrogram_to_image(v_mag)

                # 将图像保存为 PNG 文件
                cv2.imwrite("{}_X.png".format(args.output_name), X_image)
                cv2.imwrite("{}_y.png".format(args.output_name), y_image)
                cv2.imwrite("{}_v.png".format(args.output_name), v_image)

                # 将频谱转换为波形并保存为 WAV 文件
                sf.write(
                    "{}_X.wav".format(args.output_name),
                    cmb_spectrogram_to_wave(specs[0], mp),
                    mp.param["sr"],
                )
                sf.write(
                    "{}_y.wav".format(args.output_name),
                    cmb_spectrogram_to_wave(specs[1], mp),
                    mp.param["sr"],
                )

        # 将声音部分的频谱转换为 WAV 文件
        sf.write(
            "{}_v.wav".format(args.output_name),
            cmb_spectrogram_to_wave(v_spec, mp),
            mp.param["sr"],
        )
    else:
        # 如果算法不是以 "deep" 开头，则进行集成处理
        if not args.algorithm == "deep":
            # 将集成后的频谱转换为 WAV 文件并保存
            sf.write(
                os.path.join("ensembled", "{}.wav".format(args.output_name)),
                cmb_spectrogram_to_wave(ensembling(args.algorithm, specs), mp),
                mp.param["sr"],
            )

    # 如果算法参数为 "align"，执行音轨对齐操作
    if args.algorithm == "align":
        # 准备音轨对齐所需的参数列表
        trackalignment = [
            {
                "file1": '"{}"'.format(args.input[0]),
                "file2": '"{}"'.format(args.input[1]),
            }
        ]

        # 对每对音轨执行对齐操作，并显示进度条
        for i, e in tqdm(enumerate(trackalignment), desc="Performing Alignment..."):
            os.system(f"python lib/align_tracks.py {e['file1']} {e['file2']}")

    # 打印总运行时间（已注释掉的打印语句）
    # print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))
```