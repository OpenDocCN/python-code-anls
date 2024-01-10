# `so-vits-svc\inference_main.py`

```
import logging  # 导入日志模块
import soundfile  # 导入音频文件处理模块
from inference import infer_tool  # 从推断模块中导入推断工具
from inference.infer_tool import Svc  # 从推断工具中导入 Svc 类
from spkmix import spk_mix_map  # 从 spkmix 模块中导入 spk_mix_map

logging.getLogger('numba').setLevel(logging.WARNING)  # 设置 numba 模块的日志级别为 WARNING
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")  # 读取推断工具中的临时数据文件

def main():  # 定义主函数
    import argparse  # 导入命令行参数解析模块

    parser = argparse.ArgumentParser(description='sovits4 inference')  # 创建命令行参数解析器，设置描述信息

    # 一定要设置的部分
    parser.add_argument('-m', '--model_path', type=str, default="logs/44k/G_37600.pth", help='模型路径')  # 添加模型路径参数
    parser.add_argument('-c', '--config_path', type=str, default="logs/44k/config.json", help='配置文件路径')  # 添加配置文件路径参数
    parser.add_argument('-cl', '--clip', type=float, default=0, help='音频强制切片，默认0为自动切片，单位为秒/s')  # 添加音频切片参数
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["君の知らない物語-src.wav"], help='wav文件名列表，放在raw文件夹下')  # 添加清洁文件名列表参数
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0], help='音高调整，支持正负（半音）')  # 添加音高调整参数
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['buyizi'], help='合成目标说话人名称')  # 添加合成目标说话人列表参数
    
    # 可选项部分
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False, help='语音转换自动预测音高，转换歌声时不要打开这个会严重跑调')  # 添加自动预测音高参数
    parser.add_argument('-cm', '--cluster_model_path', type=str, default="", help='聚类模型或特征检索索引路径，留空则自动设为各方案模型的默认路径，如果没有训练聚类或特征检索则随便填')  # 添加聚类模型或特征检索索引路径参数
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=0, help='聚类方案或特征检索占比，范围0-1，若没有训练聚类模型或特征检索则默认0即可')  # 添加聚类方案或特征检索占比参数
    parser.add_argument('-lg', '--linear_gradient', type=float, default=0, help='两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，单位为秒')  # 添加线性渐变参数
    parser.add_argument('-f0p', '--f0_predictor', type=str, default="pm", help='选择F0预测器,可选择crepe,pm,dio,harvest,rmvpe,fcpe默认为pm(注意：crepe为原F0使用均值滤波器)')  # 添加F0预测器选择参数
    parser.add_argument('-eh', '--enhance', action='store_true', default=False, help='是否使用NSF_HIFIGAN增强器,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭')  # 添加是否使用 NSF_HIFIGAN 增强器参数
    parser.add_argument('-shd', '--shallow_diffusion', action='store_true', default=False, help='是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN增强器将会被禁止')  # 添加是否使用浅层扩散参数
    # 添加一个布尔类型的命令行参数，用于指示是否使用角色融合
    parser.add_argument('-usm', '--use_spk_mix', action='store_true', default=False, help='是否使用角色融合')
    # 添加一个浮点类型的命令行参数，用于调整输入源响度包络替换输出响度包络融合比例
    parser.add_argument('-lea', '--loudness_envelope_adjustment', type=float, default=1, help='输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络')
    # 添加一个布尔类型的命令行参数，用于指示是否使用特征检索
    parser.add_argument('-fr', '--feature_retrieval', action='store_true', default=False, help='是否使用特征检索，如果使用聚类模型将被禁用，且cm与cr参数将会变成特征检索的索引路径与混合比例')

    # 浅扩散设置
    # 添加一个字符串类型的命令行参数，用于指定扩散模型路径
    parser.add_argument('-dm', '--diffusion_model_path', type=str, default="logs/44k/diffusion/model_0.pt", help='扩散模型路径')
    # 添加一个字符串类型的命令行参数，用于指定扩散模型配置文件路径
    parser.add_argument('-dc', '--diffusion_config_path', type=str, default="logs/44k/diffusion/config.yaml", help='扩散模型配置文件路径')
    # 添加一个整数类型的命令行参数，用于指定扩散步数
    parser.add_argument('-ks', '--k_step', type=int, default=100, help='扩散步数，越大越接近扩散模型的结果，默认100')
    # 添加一个布尔类型的命令行参数，用于指示是否进行二次编码
    parser.add_argument('-se', '--second_encoding', action='store_true', default=False, help='二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差')
    # 添加一个布尔类型的命令行参数，用于指示是否使用纯扩散模式
    parser.add_argument('-od', '--only_diffusion', action='store_true', default=False, help='纯扩散模式，该模式不会加载sovits模型，以扩散模型推理')

    # 不用动的部分
    # 添加一个整数类型的命令行参数，用于指定默认的噪音级别
    parser.add_argument('-sd', '--slice_db', type=int, default=-40, help='默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50')
    # 添加一个字符串类型的命令行参数，用于指定推理设备
    parser.add_argument('-d', '--device', type=str, default=None, help='推理设备，None则为自动选择cpu和gpu')
    # 添加一个浮点类型的命令行参数，用于指定噪音级别
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4, help='噪音级别，会影响咬字和音质，较为玄学')
    # 添加一个浮点类型的命令行参数，用于指定推理音频的pad秒数
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5, help='推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现')
    # 添加一个字符串类型的命令行参数，用于指定音频输出格式
    parser.add_argument('-wf', '--wav_format', type=str, default='flac', help='音频输出格式')
    # 添加一个浮点类型的命令行参数，用于指定自动音频切片后，需要舍弃每段切片的头尾的比例
    parser.add_argument('-lgr', '--linear_gradient_retain', type=float, default=0.75, help='自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭')
    # 添加一个整数类型的命令行参数，用于使增强器适应更高的音域
    parser.add_argument('-eak', '--enhancer_adaptive_key', type=int, default=0, help='使增强器适应更高的音域(单位为半音数)|默认为0')
    # 添加一个浮点类型的命令行参数，用于指定F0过滤阈值
    parser.add_argument('-ft', '--f0_filter_threshold', type=float, default=0.05,help='F0过滤阈值，只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音')

    # 解析命令行参数
    args = parser.parse_args()

    # 获取命令行参数中的clean_names值
    clean_names = args.clean_names
    # 从参数中获取转换器
    trans = args.trans
    # 从参数中获取说话人列表
    spk_list = args.spk_list
    # 从参数中获取切片数据库
    slice_db = args.slice_db
    # 从参数中获取音频格式
    wav_format = args.wav_format
    # 从参数中获取是否自动预测基频
    auto_predict_f0 = args.auto_predict_f0
    # 从参数中获取聚类推断比例
    cluster_infer_ratio = args.cluster_infer_ratio
    # 从参数中获取噪声比例
    noice_scale = args.noice_scale
    # 从参数中获取填充秒数
    pad_seconds = args.pad_seconds
    # 从参数中获取剪辑
    clip = args.clip
    # 从参数中获取线性渐变
    lg = args.linear_gradient
    # 从参数中获取保留线性渐变
    lgr = args.linear_gradient_retain
    # 从参数中获取基频预测器
    f0p = args.f0_predictor
    # 从参数中获取增强
    enhance = args.enhance
    # 从参数中获取增强器自适应键
    enhancer_adaptive_key = args.enhancer_adaptive_key
    # 从参数中获取基频过滤阈值
    cr_threshold = args.f0_filter_threshold
    # 从参数中获取扩散模型路径
    diffusion_model_path = args.diffusion_model_path
    # 从参数中获取扩散配置路径
    diffusion_config_path = args.diffusion_config_path
    # 从参数中获取K步
    k_step = args.k_step
    # 从参数中获取仅扩散
    only_diffusion = args.only_diffusion
    # 从参数中获取浅扩散
    shallow_diffusion = args.shallow_diffusion
    # 从参数中获取使用说话人混合
    use_spk_mix = args.use_spk_mix
    # 从参数中获取第二编码
    second_encoding = args.second_encoding
    # 从参数中获取响度包络调整
    loudness_envelope_adjustment = args.loudness_envelope_adjustment
    
    # 如果聚类推断比例不为0
    if cluster_infer_ratio != 0:
        # 如果聚类模型路径为空
        if args.cluster_model_path == "":
            # 如果指定了特征检索，则使用默认的模型路径
            if args.feature_retrieval:
                args.cluster_model_path = "logs/44k/feature_and_index.pkl"
            # 否则使用默认的K均值模型路径
            else:
                args.cluster_model_path = "logs/44k/kmeans_10000.pt"
    # 如果聚类推断比例为0
    else:
        # 将聚类模型路径置空
        args.cluster_model_path = ""
    
    # 创建支持向量分类器模型
    svc_model = Svc(args.model_path,
                    args.config_path,
                    args.device,
                    args.cluster_model_path,
                    enhance,
                    diffusion_model_path,
                    diffusion_config_path,
                    shallow_diffusion,
                    only_diffusion,
                    use_spk_mix,
                    args.feature_retrieval)
    
    # 创建目录
    infer_tool.mkdir(["raw", "results"])
    
    # 如果说话人混合映射数量小于等于1
    if len(spk_mix_map)<=1:
        # 不使用说话人混合
        use_spk_mix = False
    # 如果使用说话人混合
    if use_spk_mix:
        # 将说话人列表设置为说话人混合映射
        spk_list = [spk_mix_map]
    
    # 填充A到B
    infer_tool.fill_a_to_b(trans, clean_names)
    # 遍历 clean_names 和 trans 列表，同时取出对应的值
    for clean_name, tran in zip(clean_names, trans):
        # 根据 clean_name 构建原始音频路径
        raw_audio_path = f"raw/{clean_name}"
        # 如果原始音频路径中不包含 .wav 后缀，则添加 .wav 后缀
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        # 使用 infer_tool 对原始音频进行格式化处理
        infer_tool.format_wav(raw_audio_path)
        # 遍历 spk_list 列表，同时取出对应的值
        for spk in spk_list:
            # 构建参数字典 kwarg
            kwarg = {
                "raw_audio_path" : raw_audio_path,  # 原始音频路径
                "spk" : spk,  # 说话人
                "tran" : tran,  # 转录
                "slice_db" : slice_db,  # 切片数据库
                "cluster_infer_ratio" : cluster_infer_ratio,  # 聚类推断比例
                "auto_predict_f0" : auto_predict_f0,  # 是否自动预测 f0
                "noice_scale" : noice_scale,  # 噪声比例
                "pad_seconds" : pad_seconds,  # 填充秒数
                "clip_seconds" : clip,  # 剪辑秒数
                "lg_num": lg,  # lg 数
                "lgr_num" : lgr,  # lgr 数
                "f0_predictor" : f0p,  # f0 预测器
                "enhancer_adaptive_key" : enhancer_adaptive_key,  # 增强自适应密钥
                "cr_threshold" : cr_threshold,  # cr 阈值
                "k_step":k_step,  # k 步长
                "use_spk_mix":use_spk_mix,  # 是否使用说话人混合
                "second_encoding":second_encoding,  # 第二编码
                "loudness_envelope_adjustment":loudness_envelope_adjustment  # 响度包络调整
            }
            # 使用 svc_model 进行切片推断，传入参数字典 kwarg
            audio = svc_model.slice_inference(**kwarg)
            # 根据 auto_predict_f0 的值确定 key 的值
            key = "auto" if auto_predict_f0 else f"{tran}key"
            # 根据 cluster_infer_ratio 的值确定 cluster_name 的值
            cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            # 初始化 isdiffusion 变量
            isdiffusion = "sovits"
            # 如果 shallow_diffusion 为真，则 isdiffusion 为 "sovdiff"
            if shallow_diffusion :
                isdiffusion = "sovdiff"
            # 如果 only_diffusion 为真，则 isdiffusion 为 "diff"
            if only_diffusion :
                isdiffusion = "diff"
            # 如果 use_spk_mix 为真，则 spk 为 "spk_mix"
            if use_spk_mix:
                spk = "spk_mix"
            # 构建结果路径 res_path
            res_path = f'results/{clean_name}_{key}_{spk}{cluster_name}_{isdiffusion}_{f0p}.{wav_format}'
            # 将音频数据写入结果路径 res_path
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            # 清除 svc_model 中的空值
            svc_model.clear_empty()
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == '__main__':
    main()
```