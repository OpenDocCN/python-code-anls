# `D:\src\scipysrc\scipy\scipy\io\wavfile.py`

```
"""
Module to read / write wav files using NumPy arrays

Functions
---------
`read`: Return the sample rate (in samples/sec) and data from a WAV file.

`write`: Write a NumPy array as a WAV file.

"""
import io                      # 导入io模块，用于处理字节流和文件流
import sys                     # 导入sys模块，提供系统相关的功能
import numpy as np             # 导入NumPy库，用于处理数值数据
import struct                  # 导入struct模块，用于处理二进制数据的打包和解包
import warnings                # 导入warnings模块，用于处理警告信息
from enum import IntEnum       # 从enum模块导入IntEnum类，用于创建枚举类型


__all__ = [                    # 定义公开的模块成员列表
    'WavFileWarning',          # 公开警告类WavFileWarning
    'read',                    # 公开函数read
    'write'                    # 公开函数write
]


class WavFileWarning(UserWarning):
    pass                        # 定义一个警告类WavFileWarning，继承自UserWarning


class WAVE_FORMAT(IntEnum):
    """
    WAVE form wFormatTag IDs

    Complete list is in mmreg.h in Windows 10 SDK.  ALAC and OPUS are the
    newest additions, in v10.0.14393 2016-07
    """
    UNKNOWN = 0x0000            # 未知格式
    PCM = 0x0001                # PCM格式
    ADPCM = 0x0002              # ADPCM格式
    IEEE_FLOAT = 0x0003         # IEEE浮点格式
    VSELP = 0x0004              # VSELP格式
    IBM_CVSD = 0x0005           # IBM CVSD格式
    ALAW = 0x0006               # A-Law格式
    MULAW = 0x0007              # Mu-Law格式
    DTS = 0x0008                # DTS格式
    DRM = 0x0009                # DRM格式
    WMAVOICE9 = 0x000A          # WMA Voice 9格式
    WMAVOICE10 = 0x000B         # WMA Voice 10格式
    OKI_ADPCM = 0x0010          # OKI ADPCM格式
    DVI_ADPCM = 0x0011          # DVI ADPCM格式
    IMA_ADPCM = 0x0011          # IMA ADPCM格式（与DVI ADPCM重复）
    MEDIASPACE_ADPCM = 0x0012   # Media Space ADPCM格式
    SIERRA_ADPCM = 0x0013       # Sierra ADPCM格式
    G723_ADPCM = 0x0014         # G.723 ADPCM格式
    DIGISTD = 0x0015            # Digistd格式
    DIGIFIX = 0x0016            # Digifix格式
    DIALOGIC_OKI_ADPCM = 0x0017 # Dialogic OKI ADPCM格式
    MEDIAVISION_ADPCM = 0x0018  # Media Vision ADPCM格式
    CU_CODEC = 0x0019           # CU CODEC格式
    HP_DYN_VOICE = 0x001A       # HP Dynamic Voice格式
    YAMAHA_ADPCM = 0x0020       # Yamaha ADPCM格式
    SONARC = 0x0021             # Sonarc格式
    DSPGROUP_TRUESPEECH = 0x0022# DSP Group TrueSpeech格式
    ECHOSC1 = 0x0023            # EchoSC1格式
    AUDIOFILE_AF36 = 0x0024     # AudioFile AF36格式
    APTX = 0x0025               # AptX格式
    AUDIOFILE_AF10 = 0x0026     # AudioFile AF10格式
    PROSODY_1612 = 0x0027       # Prosody 1612格式
    LRC = 0x0028                # LRC格式
    DOLBY_AC2 = 0x0030          # Dolby AC2格式
    GSM610 = 0x0031             # GSM610格式
    MSNAUDIO = 0x0032           # MSN Audio格式
    ANTEX_ADPCME = 0x0033       # Antex ADPCME格式
    CONTROL_RES_VQLPC = 0x0034  # Control Res VQLPC格式
    DIGIREAL = 0x0035           # DigiReal格式
    DIGIADPCM = 0x0036          # DigiADPCM格式
    CONTROL_RES_CR10 = 0x0037   # Control Res CR10格式
    NMS_VBXADPCM = 0x0038       # NMS VBXADPCM格式
    CS_IMAADPCM = 0x0039        # CS IMAADPCM格式
    ECHOSC3 = 0x003A            # EchoSC3格式
    ROCKWELL_ADPCM = 0x003B     # Rockwell ADPCM格式
    ROCKWELL_DIGITALK = 0x003C  # Rockwell DigitaLK格式
    XEBEC = 0x003D              # Xebec格式
    G721_ADPCM = 0x0040         # G.721 ADPCM格式
    G728_CELP = 0x0041          # G.728 CELP格式
    MSG723 = 0x0042             # Microsoft G.723格式
    INTEL_G723_1 = 0x0043       # Intel G.723.1格式
    INTEL_G729 = 0x0044         # Intel G.729格式
    SHARP_G726 = 0x0045         # Sharp G.726格式
    MPEG = 0x0050                # MPEG格式
    RT24 = 0x0052               # RT24格式
    PAC = 0x0053                # PAC格式
    MPEGLAYER3 = 0x0055         # MPEG Layer 3格式
    LUCENT_G723 = 0x0059        # Lucent G.723格式
    CIRRUS = 0x0060             # Cirrus格式
    ESPCM = 0x0061              # ESPCM格式
    VOXWARE = 0x0062            # Voxware格式
    CANOPUS_ATRAC = 0x0063      # Canopus ATRAC格式
    G726_ADPCM = 0x0064         # G.726 ADPCM格式
    G722_ADPCM = 0x0065         # G.722 ADPCM格式
    DSAT = 0x0066               # DSAT格式
    DSAT_DISPLAY = 0x0067       # DSAT Display格式
    VOXWARE_BYTE_ALIGNED = 0x0069# Voxware Byte Aligned格式
    VOXWARE_AC8 = 0x0070        # Voxware AC8格式
    VOXWARE_AC10 = 0x0071       # Voxware AC10格式
    VOXWARE_AC16 = 0x0072       # Voxware AC16格式
    VOXWARE_AC20 = 0x0073       # Voxware AC20格式
    VOXWARE_RT24 = 0x0074       # Voxware RT24格式
    VOXWARE_RT29 = 0x0075       # Voxware RT29格式
    VOXWARE_RT29HW = 0x0076     # Voxware RT29HW格式
    VOXWARE_VR12 = 0x0077       # Voxware VR12格式
    VOXWARE_VR18 = 0x0078       # Voxware VR18格式
    VOXWARE_TQ40 = 0x0079       # Voxware TQ40格式
    VOXWARE_SC3 = 0x007A        # Voxware SC3格式
    VOXWARE_SC3_1 = 0x007B      # Voxware SC3-1格式
    SOFTSOUND = 0x0080          # Softsound格式
    VOXWARE_TQ60 = 0x0081       # Voxware TQ60格式
    MSRT24 = 0x0082             # MSRT24格式
    G729A = 0x0083              # G.729A格式
    MVI_MVI2 = 0x0084           # MVI/MVI2格式
    DF_G726 = 0x0085            # DF G.726格式
    DF_GSM610 = 0x0086          # DF GSM610格式
    ISIAUDIO = 0x0088           # ISIAudio格式
    ONLIVE = 0x0089             # Onlive格式
    MULTITUDE_FT_SX20 = 0x008A  # Multitude FT SX20格式
    INFOCOM_ITS_G721_ADPCM = 0x008B# Infocom ITS G.721 ADPCM格式
    CONVEDIA_G729 = 0x008C      #
    MALDEN_PHONYTALK = 0x00A0
    # 定义常量 MALDEN_PHONYTALK，其十六进制值为 0x00A0

    RACAL_RECORDER_GSM = 0x00A1
    # 定义常量 RACAL_RECORDER_GSM，其十六进制值为 0x00A1

    RACAL_RECORDER_G720_A = 0x00A2
    # 定义常量 RACAL_RECORDER_G720_A，其十六进制值为 0x00A2

    RACAL_RECORDER_G723_1 = 0x00A3
    # 定义常量 RACAL_RECORDER_G723_1，其十六进制值为 0x00A3

    RACAL_RECORDER_TETRA_ACELP = 0x00A4
    # 定义常量 RACAL_RECORDER_TETRA_ACELP，其十六进制值为 0x00A4

    NEC_AAC = 0x00B0
    # 定义常量 NEC_AAC，其十六进制值为 0x00B0

    RAW_AAC1 = 0x00FF
    # 定义常量 RAW_AAC1，其十六进制值为 0x00FF

    RHETOREX_ADPCM = 0x0100
    # 定义常量 RHETOREX_ADPCM，其十六进制值为 0x0100

    IRAT = 0x0101
    # 定义常量 IRAT，其十六进制值为 0x0101

    VIVO_G723 = 0x0111
    # 定义常量 VIVO_G723，其十六进制值为 0x0111

    VIVO_SIREN = 0x0112
    # 定义常量 VIVO_SIREN，其十六进制值为 0x0112

    PHILIPS_CELP = 0x0120
    # 定义常量 PHILIPS_CELP，其十六进制值为 0x0120

    PHILIPS_GRUNDIG = 0x0121
    # 定义常量 PHILIPS_GRUNDIG，其十六进制值为 0x0121

    DIGITAL_G723 = 0x0123
    # 定义常量 DIGITAL_G723，其十六进制值为 0x0123

    SANYO_LD_ADPCM = 0x0125
    # 定义常量 SANYO_LD_ADPCM，其十六进制值为 0x0125

    SIPROLAB_ACEPLNET = 0x0130
    # 定义常量 SIPROLAB_ACEPLNET，其十六进制值为 0x0130

    SIPROLAB_ACELP4800 = 0x0131
    # 定义常量 SIPROLAB_ACELP4800，其十六进制值为 0x0131

    SIPROLAB_ACELP8V3 = 0x0132
    # 定义常量 SIPROLAB_ACELP8V3，其十六进制值为 0x0132

    SIPROLAB_G729 = 0x0133
    # 定义常量 SIPROLAB_G729，其十六进制值为 0x0133

    SIPROLAB_G729A = 0x0134
    # 定义常量 SIPROLAB_G729A，其十六进制值为 0x0134

    SIPROLAB_KELVIN = 0x0135
    # 定义常量 SIPROLAB_KELVIN，其十六进制值为 0x0135

    VOICEAGE_AMR = 0x0136
    # 定义常量 VOICEAGE_AMR，其十六进制值为 0x0136

    G726ADPCM = 0x0140
    # 定义常量 G726ADPCM，其十六进制值为 0x0140

    DICTAPHONE_CELP68 = 0x0141
    # 定义常量 DICTAPHONE_CELP68，其十六进制值为 0x0141

    DICTAPHONE_CELP54 = 0x0142
    # 定义常量 DICTAPHONE_CELP54，其十六进制值为 0x0142

    QUALCOMM_PUREVOICE = 0x0150
    # 定义常量 QUALCOMM_PUREVOICE，其十六进制值为 0x0150

    QUALCOMM_HALFRATE = 0x0151
    # 定义常量 QUALCOMM_HALFRATE，其十六进制值为 0x0151

    TUBGSM = 0x0155
    # 定义常量 TUBGSM，其十六进制值为 0x0155

    MSAUDIO1 = 0x0160
    # 定义常量 MSAUDIO1，其十六进制值为 0x0160

    WMAUDIO2 = 0x0161
    # 定义常量 WMAUDIO2，其十六进制值为 0x0161

    WMAUDIO3 = 0x0162
    # 定义常量 WMAUDIO3，其十六进制值为 0x0162

    WMAUDIO_LOSSLESS = 0x0163
    # 定义常量 WMAUDIO_LOSSLESS，其十六进制值为 0x0163

    WMASPDIF = 0x0164
    # 定义常量 WMASPDIF，其十六进制值为 0x0164

    UNISYS_NAP_ADPCM = 0x0170
    # 定义常量 UNISYS_NAP_ADPCM，其十六进制值为 0x0170

    UNISYS_NAP_ULAW = 0x0171
    # 定义常量 UNISYS_NAP_ULAW，其十六进制值为 0x0171

    UNISYS_NAP_ALAW = 0x0172
    # 定义常量 UNISYS_NAP_ALAW，其十六进制值为 0x0172

    UNISYS_NAP_16K = 0x0173
    # 定义常量 UNISYS_NAP_16K，其十六进制值为 0x0173

    SYCOM_ACM_SYC008 = 0x0174
    # 定义常量 SYCOM_ACM_SYC008，其十六进制值为 0x0174

    SYCOM_ACM_SYC701_G726L = 0x0175
    # 定义常量 SYCOM_ACM_SYC701_G726L，其十六进制值为 0x0175

    SYCOM_ACM_SYC701_CELP54 = 0x0176
    # 定义常量 SYCOM_ACM_SYC701_CELP54，其十六进制值为 0x0176

    SYCOM_ACM_SYC701_CELP68 = 0x0177
    # 定义常量 SYCOM_ACM_SYC701_CELP68，其十六进制值为 0x0177

    KNOWLEDGE_ADVENTURE_ADPCM = 0x0178
    # 定义常量 KNOWLEDGE_ADVENTURE_ADPCM，其十六进制值为 0x0178

    FRAUNHOFER_IIS_MPEG2_AAC = 0x0180
    # 定义常量 FRAUNHOFER_IIS_MPEG2_AAC，其十六进制值为 0x0180

    DTS_DS = 0x0190
    # 定义常量 DTS_DS，其十六进制值为 0x0190

    CREATIVE_ADPCM = 0x0200
    # 定义常量 CREATIVE_ADPCM，其十六进制值为 0x0200

    CREATIVE_FASTSPEECH8 = 0x0202
    # 定义常量 CREATIVE_FASTSPEECH8，其十六进制值为 0x0202

    CREATIVE_FASTSPEECH10 = 0x0203
    # 定义常量 CREATIVE_FASTSPEECH10，其十六进制值为 0x0203

    UHER_ADPCM = 0x0210
    # 定义常量 UHER_ADPCM，其
    # 定义一系列音频编解码器的标识符，采用十六进制表示
    
    NOKIA_ADAPTIVE_MULTIRATE = 0x4201           # Nokia 自适应多速率编解码器
    DIVIO_G726 = 0x4243                          # Divio G.726 编解码器
    LEAD_SPEECH = 0x434C                         # Lead Speech 编解码器
    LEAD_VORBIS = 0x564C                         # Lead Vorbis 编解码器
    WAVPACK_AUDIO = 0x5756                       # WavPack 音频格式
    OGG_VORBIS_MODE_1 = 0x674F                   # Ogg Vorbis 编解码器 Mode 1
    OGG_VORBIS_MODE_2 = 0x6750                   # Ogg Vorbis 编解码器 Mode 2
    OGG_VORBIS_MODE_3 = 0x6751                   # Ogg Vorbis 编解码器 Mode 3
    OGG_VORBIS_MODE_1_PLUS = 0x676F              # Ogg Vorbis 编解码器 Mode 1 Plus
    OGG_VORBIS_MODE_2_PLUS = 0x6770              # Ogg Vorbis 编解码器 Mode 2 Plus
    OGG_VORBIS_MODE_3_PLUS = 0x6771              # Ogg Vorbis 编解码器 Mode 3 Plus
    ALAC = 0x6C61                                # Apple Lossless Audio Codec (ALAC)
    _3COM_NBX = 0x7000                           # 3Com NBX 电话系统音频
    OPUS = 0x704F                                # Opus 音频编解码器
    FAAD_AAC = 0x706D                            # FAAD AAC 编解码器
    AMR_NB = 0x7361                              # AMR-NB (Adaptive Multi-Rate Narrowband)
    AMR_WB = 0x7362                              # AMR-WB (Adaptive Multi-Rate Wideband)
    AMR_WP = 0x7363                              # AMR-WP (Adaptive Multi-Rate Wideband Plus)
    GSM_AMR_CBR = 0x7A21                         # GSM AMR 编码固定比特率
    GSM_AMR_VBR_SID = 0x7A22                     # GSM AMR 编码可变比特率 SID
    COMVERSE_INFOSYS_G723_1 = 0xA100             # Comverse Infosys G.723.1 编解码器
    COMVERSE_INFOSYS_AVQSBC = 0xA101             # Comverse Infosys AVQSBC 编解码器
    COMVERSE_INFOSYS_SBC = 0xA102                # Comverse Infosys SBC 编解码器
    SYMBOL_G729_A = 0xA103                       # Symbol Technologies G.729A 编解码器
    VOICEAGE_AMR_WB = 0xA104                     # VoiceAge AMR-WB 编解码器
    INGENIENT_G726 = 0xA105                      # Ingenient G.726 编解码器
    MPEG4_AAC = 0xA106                           # MPEG-4 AAC 编解码器
    ENCORE_G726 = 0xA107                         # Encore G.726 编解码器
    ZOLL_ASAO = 0xA108                           # ZOLL ASAO 编解码器
    SPEEX_VOICE = 0xA109                         # Speex 语音编解码器
    VIANIX_MASC = 0xA10A                         # Vianix MASC 编解码器
    WM9_SPECTRUM_ANALYZER = 0xA10B               # Windows Media 9 Spectrum Analyzer
    WMF_SPECTRUM_ANAYZER = 0xA10C                # Windows Media Format Spectrum Analyzer
    GSM_610 = 0xA10D                             # GSM 610 编解码器
    GSM_620 = 0xA10E                             # GSM 620 编解码器
    GSM_660 = 0xA10F                             # GSM 660 编解码器
    GSM_690 = 0xA110                             # GSM 690 编解码器
    GSM_ADAPTIVE_MULTIRATE_WB = 0xA111           # GSM 自适应多速率宽带编解码器
    POLYCOM_G722 = 0xA112                        # Polycom G.722 编解码器
    POLYCOM_G728 = 0xA113                        # Polycom G.728 编解码器
    POLYCOM_G729_A = 0xA114                      # Polycom G.729A 编解码器
    POLYCOM_SIREN = 0xA115                       # Polycom Siren 编解码器
    GLOBAL_IP_ILBC = 0xA116                      # Global IP iLBC 编解码器
    RADIOTIME_TIME_SHIFT_RADIO = 0xA117          # RadioTime Time Shift Radio 编解码器
    NICE_ACA = 0xA118                            # NICE ACA 编解码器
    NICE_ADPCM = 0xA119                          # NICE ADPCM 编解码器
    VOCORD_G721 = 0xA11A                         # Vocord G.721 编解码器
    VOCORD_G726 = 0xA11B                         # Vocord G.726 编解码器
    VOCORD_G722_1 = 0xA11C                       # Vocord G.722.1 编解码器
    VOCORD_G728 = 0xA11D                         # Vocord G.728 编解码器
    VOCORD_G729 = 0xA11E                         # Vocord G.729 编解码器
    VOCORD_G729_A = 0xA11F                       # Vocord G.729A 编解码器
    VOCORD_G723_1 = 0xA120                       # Vocord G.723.1 编解码器
    VOCORD_LBC = 0xA121                          # Vocord LBC 编解码器
    NICE_G728 = 0xA122                           # NICE G.728 编解码器
    FRACE_TELECOM_G729 = 0xA123                  # France Telecom G.729 编解码器
    CODIAN = 0xA124                              # Codian 编解码器
    FLAC = 0xF1AC                                # FLAC 音频编解码器
    EXTENSIBLE = 0xFFFE                          # 可扩展音频编解码器
    DEVELOPMENT = 0xFFFF                         # 开发和测试用途保留的标识
# 已知的波形格式集合，包括 PCM 和 IEEE_FLOAT
KNOWN_WAVE_FORMATS = {WAVE_FORMAT.PCM, WAVE_FORMAT.IEEE_FLOAT}


def _raise_bad_format(format_tag):
    try:
        # 尝试获取对应格式标签的名称
        format_name = WAVE_FORMAT(format_tag).name
    except ValueError:
        # 如果未知格式标签，以十六进制形式表示
        format_name = f'{format_tag:#06x}'
    # 抛出值错误，指示未知的波形文件格式
    raise ValueError(f"Unknown wave file format: {format_name}. Supported "
                     "formats: " +
                     ', '.join(x.name for x in KNOWN_WAVE_FORMATS))


def _read_fmt_chunk(fid, is_big_endian):
    """
    Returns
    -------
    size : int
        format subchunk的大小（减去 "fmt " 和它本身的8个字节）
    format_tag : int
        PCM、float或压缩格式
    channels : int
        通道数
    fs : int
        采样频率，每秒样本数
    bytes_per_second : int
        文件的总字节速率
    block_align : int
        每个样本的字节数，包括所有通道
    bit_depth : int
        每个样本的位深度

    Notes
    -----
    假设文件指针位于 'fmt ' 标识之后
    """
    if is_big_endian:
        fmt = '>'
    else:
        fmt = '<'

    # 解析格式子块的大小
    size = struct.unpack(fmt+'I', fid.read(4))[0]

    if size < 16:
        # 如果子块大小小于16字节，抛出值错误
        raise ValueError("Binary structure of wave file is not compliant")

    # 解析格式子块的其他字段
    res = struct.unpack(fmt+'HHIIHH', fid.read(16))
    bytes_read = 16

    format_tag, channels, fs, bytes_per_second, block_align, bit_depth = res

    if format_tag == WAVE_FORMAT.EXTENSIBLE and size >= (16+2):
        # 对于扩展格式，解析扩展块大小和数据
        ext_chunk_size = struct.unpack(fmt+'H', fid.read(2))[0]
        bytes_read += 2
        if ext_chunk_size >= 22:
            extensible_chunk_data = fid.read(22)
            bytes_read += 22
            raw_guid = extensible_chunk_data[2+4:2+4+16]
            # 检查 GUID 是否符合波形扩展格式的标准
            if is_big_endian:
                tail = b'\x00\x00\x00\x10\x80\x00\x00\xAA\x00\x38\x9B\x71'
            else:
                tail = b'\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71'
            if raw_guid.endswith(tail):
                format_tag = struct.unpack(fmt+'I', raw_guid[:4])[0]
        else:
            raise ValueError("Binary structure of wave file is not compliant")

    if format_tag not in KNOWN_WAVE_FORMATS:
        # 如果格式标签不在已知的波形格式集合中，抛出错误
        _raise_bad_format(format_tag)

    # 将文件指针移动到下一个块
    if size > bytes_read:
        fid.read(size - bytes_read)

    # 格式子块应该始终为16、18或40字节，但为了安全起见进行处理
    _handle_pad_byte(fid, size)
    # 如果音频格式标签为 PCM
    if format_tag == WAVE_FORMAT.PCM:
        # 检查每秒字节数是否等于采样率乘以块对齐数
        if bytes_per_second != fs * block_align:
            # 如果不相等，抛出数值错误异常，说明 WAV 头部信息无效
            raise ValueError("WAV header is invalid: nAvgBytesPerSec must"
                             " equal product of nSamplesPerSec and"
                             " nBlockAlign, but file has nSamplesPerSec ="
                             f" {fs}, nBlockAlign = {block_align}, and"
                             f" nAvgBytesPerSec = {bytes_per_second}")

    # 返回包含音频信息的元组
    return (size, format_tag, channels, fs, bytes_per_second, block_align,
            bit_depth)
# 定义一个函数用于从音频文件中读取数据块
def _read_data_chunk(fid, format_tag, channels, bit_depth, is_big_endian, is_rf64,
                     block_align, mmap=False):
    """
    Notes
    -----
    假设文件指针立即在 'data' 标识后面

    在一个容器中可能没有使用所有可用的位数，或者将样本存储在比必要更大的容器中，
    因此 bytes_per_sample 使用实际报告的容器大小 (nBlockAlign / nChannels)。实际案例包括：

    Adobe Audition 的 "24-bit packed int (type 1, 20-bit)"：
        nChannels = 2, nBlockAlign = 6, wBitsPerSample = 20

    http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Samples/AFsp/M1F1-int12-AFsp.wav
    是：
        nChannels = 2, nBlockAlign = 4, wBitsPerSample = 12

    http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Docs/multichaudP.pdf
    提供了一个例子：
        nChannels = 2, nBlockAlign = 8, wBitsPerSample = 20
    """

    # 根据大端或小端确定格式
    if is_big_endian:
        fmt = '>'
    else:
        fmt = '<'

    # 数据子块的大小（以字节为单位）
    if not is_rf64:
        size = struct.unpack(fmt+'I', fid.read(4))[0]
    else:
        pos = fid.tell()
        # 对于 RF64，块大小存储在全局文件头中
        fid.seek(28)
        size = struct.unpack('<Q', fid.read(8))[0]
        fid.seek(pos)
        # 跳过数据块大小，因为它是 0xFFFFFFF
        fid.read(4)

    # 每个样本的字节数（样本容器大小）
    bytes_per_sample = block_align // channels
    # 根据数据大小计算样本数
    n_samples = size // bytes_per_sample

    # 如果是 PCM 格式
    if format_tag == WAVE_FORMAT.PCM:
        if 1 <= bit_depth <= 8:
            dtype = 'u1'  # 8 位整数或更少的 WAV 文件是无符号的
        elif bytes_per_sample in {3, 5, 6, 7}:
            # 没有兼容的 dtype。稍后用原始字节加载以进行重塑。
            dtype = 'V1'
        elif bit_depth <= 64:
            # 剩余位深度可以直接映射到有符号的 numpy dtypes
            dtype = f'{fmt}i{bytes_per_sample}'
        else:
            raise ValueError("Unsupported bit depth: the WAV file "
                             f"has {bit_depth}-bit integer data.")
    # 如果是 IEEE_FLOAT 格式
    elif format_tag == WAVE_FORMAT.IEEE_FLOAT:
        if bit_depth in {32, 64}:
            dtype = f'{fmt}f{bytes_per_sample}'
        else:
            raise ValueError("Unsupported bit depth: the WAV file "
                             f"has {bit_depth}-bit floating-point data.")
    else:
        _raise_bad_format(format_tag)  # 抛出错误，不支持的音频格式

    start = fid.tell()  # 记录当前文件指针位置
    # 如果不使用内存映射（mmap=False）
    if not mmap:
        try:
            # 根据指定的数据类型从文件中读取数据
            count = size if dtype == 'V1' else n_samples
            data = np.fromfile(fid, dtype=dtype, count=count)
        except io.UnsupportedOperation:  # 如果文件不像 C 语言风格的文件
            fid.seek(start, 0)  # 即使不应该进行 seek，也进行一次
            # 从文件中读取指定大小的数据到缓冲区，并按指定数据类型转换
            data = np.frombuffer(fid.read(size), dtype=dtype)

        if dtype == 'V1':
            # 将原始字节重新排列成最小兼容的 numpy 数据类型
            dt = f'{fmt}i4' if bytes_per_sample == 3 else f'{fmt}i8'
            a = np.zeros((len(data) // bytes_per_sample, np.dtype(dt).itemsize),
                         dtype='V1')
            if is_big_endian:
                # 如果是大端序，按字节顺序填充数组
                a[:, :bytes_per_sample] = data.reshape((-1, bytes_per_sample))
            else:
                # 如果是小端序，按逆序填充数组
                a[:, -bytes_per_sample:] = data.reshape((-1, bytes_per_sample))
            # 将数组视图转换为指定数据类型，然后重新形状为原始形状
            data = a.view(dt).reshape(a.shape[:-1])
    else:
        # 如果使用内存映射（mmap=True）
        if bytes_per_sample in {1, 2, 4, 8}:
            start = fid.tell()
            # 创建一个内存映射，从文件中读取指定的数据类型和形状
            data = np.memmap(fid, dtype=dtype, mode='c', offset=start,
                             shape=(n_samples,))
            fid.seek(start + size)
        else:
            # 抛出异常，因为指定的字节大小不兼容内存映射
            raise ValueError("mmap=True not compatible with "
                             f"{bytes_per_sample}-byte container size.")

    # 处理填充字节
    _handle_pad_byte(fid, size)

    # 如果有多个通道，则重新形状数据以匹配通道数
    if channels > 1:
        data = data.reshape(-1, channels)
    # 返回处理后的数据
    return data
def _skip_unknown_chunk(fid, is_big_endian):
    # 根据是否大端序选择格式
    if is_big_endian:
        fmt = '>I'
    else:
        fmt = '<I'

    # 读取四个字节数据
    data = fid.read(4)
    # 只有在真正读取到数据时调用 unpack() 和 seek()
    # 否则在 unpack() 调用时会触发不必要的异常，因为文件末尾的空读取
    # 如果 data 为 0，也无需调用 seek()
    if data:
        # 根据格式解包数据大小
        size = struct.unpack(fmt, data)[0]
        # 跳过指定大小的数据块
        fid.seek(size, 1)
        # 处理可能的填充字节
        _handle_pad_byte(fid, size)


def _read_riff_chunk(fid):
    # 读取四个字节的文件签名
    str1 = fid.read(4)  # 文件签名
    if str1 == b'RIFF':
        is_rf64 = False
        is_big_endian = False
        fmt = '<I'
    elif str1 == b'RIFX':
        is_rf64 = False
        is_big_endian = True
        fmt = '>I'
    elif str1 == b'RF64':
        is_rf64 = True
        is_big_endian = False
        fmt = '<Q'
    else:
        # 抛出错误，不支持的文件格式
        raise ValueError(f"File format {repr(str1)} not understood. Only "
                         "'RIFF', 'RIFX', and 'RF64' supported.")
    # 获取整个文件大小
    if not is_rf64:
        file_size = struct.unpack(fmt, fid.read(4))[0] + 8
        str2 = fid.read(4)
    else:
        # 跳过 0xFFFFFFFF (-1) 字节
        fid.read(4)
        str2 = fid.read(4)
        str3 = fid.read(4)
        if str3 != b'ds64':
            raise ValueError("Invalid RF64 file: ds64 chunk not found.")
        ds64_size = struct.unpack("<I", fid.read(4))[0]
        file_size = struct.unpack(fmt, fid.read(8))[0] + 8
        # 忽略 ds64 块的附加属性，如采样计数、表格等，直接跳到下一个块
        fid.seek(ds64_size - 8, 1)

    # 检查文件类型是否为 WAVE
    if str2 != b'WAVE':
        raise ValueError(f"Not a WAV file. RIFF form type is {repr(str2)}.")

    return file_size, is_big_endian, is_rf64


def _handle_pad_byte(fid, size):
    # "如果块大小为奇数个字节，则在 ckData 后写入值为零的填充字节。"
    # 因此在每个块后我们需要跳过这些字节。
    if size % 2:
        fid.seek(1, 1)


def read(filename, mmap=False):
    """
    打开 WAV 文件。

    返回 LPCM WAV 文件的采样率（每秒样本数）和数据。

    参数
    ----------
    filename : string 或打开的文件句柄
        输入的 WAV 文件。
    mmap : bool, optional
        是否以内存映射方式读取数据（默认为 False）。
        不兼容某些比特深度；请参阅注释。仅适用于真实文件。

        .. versionadded:: 0.12.0

    返回
    -------
    rate : int
        WAV 文件的采样率。
    data : numpy 数组
        从 WAV 文件读取的数据。数据类型根据文件确定；请参阅注释。
        对于单声道 WAV，数据是一维的；对于多声道，形状是 (Nsamples, Nchannels)。
        如果传入类似文件但没有 C 类文件描述符的输入（例如 :class:`python:io.BytesIO`），则不可写。

    注释
    """
    -----
    Common data types: [1]_

    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit integer PCM     -2147483648  +2147483647  int32
    24-bit integer PCM     -2147483648  +2147483392  int32
    16-bit integer PCM     -32768       +32767       int16
    8-bit integer PCM      0            255          uint8
    =====================  ===========  ===========  =============

    WAV files can specify arbitrary bit depth, and this function supports
    reading any integer PCM depth from 1 to 64 bits.  Data is returned in the
    smallest compatible numpy int type, in left-justified format.  8-bit and
    lower is unsigned, while 9-bit and higher is signed.

    For example, 24-bit data will be stored as int32, with the MSB of the
    24-bit data stored at the MSB of the int32, and typically the least
    significant byte is 0x00.  (However, if a file actually contains data past
    its specified bit depth, those bits will be read and output, too. [2]_)

    This bit justification and sign matches WAV's native internal format, which
    allows memory mapping of WAV files that use 1, 2, 4, or 8 bytes per sample
    (so 24-bit files cannot be memory-mapped, but 32-bit can).

    IEEE float PCM in 32- or 64-bit format is supported, with or without mmap.
    Values exceeding [-1, +1] are not clipped.

    Non-linear PCM (mu-law, A-law) is not supported.

    References
    ----------
    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming
       Interface and Data Specifications 1.0", section "Data Format of the
       Samples", August 1991
       http://www.tactilemedia.com/info/MCI_Control_Info.html
    .. [2] Adobe Systems Incorporated, "Adobe Audition 3 User Guide", section
       "Audio file formats: 24-bit Packed Int (type 1, 20-bit)", 2007

    Examples
    --------
    >>> from os.path import dirname, join as pjoin
    >>> from scipy.io import wavfile
    >>> import scipy.io

    Get the filename for an example .wav file from the tests/data directory.

    >>> data_dir = pjoin(dirname(scipy.io.__file__), 'tests', 'data')
    # 拼接出示例 WAV 文件的完整路径
    >>> wav_fname = pjoin(data_dir, 'test-44100Hz-2ch-32bit-float-be.wav')

    Load the .wav file contents.

    >>> samplerate, data = wavfile.read(wav_fname)
    # 调用 `wavfile.read` 函数读取 WAV 文件，返回采样率和音频数据
    >>> print(f"number of channels = {data.shape[1]}")
    # 打印音频数据的通道数
    number of channels = 2
    >>> length = data.shape[0] / samplerate
    # 计算音频时长
    >>> print(f"length = {length}s")
    # 打印音频时长
    length = 0.01s

    Plot the waveform.

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time = np.linspace(0., length, data.shape[0])
    # 生成时间轴
    >>> plt.plot(time, data[:, 0], label="Left channel")
    # 绘制左声道波形
    >>> plt.plot(time, data[:, 1], label="Right channel")
    # 绘制右声道波形
    >>> plt.legend()
    # 显示图例
    >>> plt.xlabel("Time [s]")
    # 设置 x 轴标签
    >>> plt.ylabel("Amplitude")
    # 设置 y 轴标签
    >>> plt.show()

    """
    # 如果 filename 具有 'read' 属性，则将 fid 设为 filename，且不使用内存映射
    if hasattr(filename, 'read'):
        fid = filename
        mmap = False
    else:
        # 否则，以二进制只读方式打开 filename 并将 fid 设为打开的文件对象
        fid = open(filename, 'rb')

    try:
        # 读取 RIFF 头部信息并返回文件大小、大小端信息、是否为 RF64 格式
        file_size, is_big_endian, is_rf64 = _read_riff_chunk(fid)
        fmt_chunk_received = False
        data_chunk_received = False
        # 在文件指针未达到文件末尾之前，循环读取下一个 chunk
        while fid.tell() < file_size:
            # 读取下一个 chunk 的 chunk ID
            chunk_id = fid.read(4)

            if not chunk_id:
                # 如果未读取到 chunk ID
                if data_chunk_received:
                    # 文件结束但成功读取数据时发出警告
                    warnings.warn(
                        f"Reached EOF prematurely; finished at {fid.tell():d} bytes, "
                        f"expected {file_size:d} bytes from header.",
                        WavFileWarning, stacklevel=2)
                    break
                else:
                    # 否则，抛出数值错误
                    raise ValueError("Unexpected end of file.")
            elif len(chunk_id) < 4:
                # 如果 chunk ID 不完整，发出警告并忽略之
                msg = f"Incomplete chunk ID: {repr(chunk_id)}"
                if fmt_chunk_received and data_chunk_received:
                    # 如果已经收到格式和数据 chunk，则忽略损坏的 chunk
                    warnings.warn(msg + ", ignoring it.", WavFileWarning,
                                  stacklevel=2)
                else:
                    # 否则，抛出数值错误
                    raise ValueError(msg)

            if chunk_id == b'fmt ':
                # 如果是 fmt chunk，标记已接收 fmt chunk，读取 fmt chunk 的详细信息
                fmt_chunk_received = True
                fmt_chunk = _read_fmt_chunk(fid, is_big_endian)
                format_tag, channels, fs = fmt_chunk[1:4]
                bit_depth = fmt_chunk[6]
                block_align = fmt_chunk[5]
            elif chunk_id == b'fact':
                # 如果是 fact chunk，跳过该 chunk
                _skip_unknown_chunk(fid, is_big_endian)
            elif chunk_id == b'data':
                # 如果是 data chunk，标记已接收 data chunk，检查是否已接收 fmt chunk，读取数据 chunk 的内容
                data_chunk_received = True
                if not fmt_chunk_received:
                    raise ValueError("No fmt chunk before data")
                data = _read_data_chunk(fid, format_tag, channels, bit_depth,
                                        is_big_endian, is_rf64, block_align, mmap)
            elif chunk_id == b'LIST':
                # 如果是 LIST chunk，目前跳过处理
                _skip_unknown_chunk(fid, is_big_endian)
            elif chunk_id in {b'JUNK', b'Fake'}:
                # 如果是 JUNK 或 Fake chunk，跳过对齐 chunk 而不发出警告
                _skip_unknown_chunk(fid, is_big_endian)
            else:
                # 对于未理解的非数据 chunk，发出警告并跳过处理
                warnings.warn("Chunk (non-data) not understood, skipping it.",
                              WavFileWarning, stacklevel=2)
                _skip_unknown_chunk(fid, is_big_endian)
    finally:
        # 最终处理：如果 filename 没有 'read' 属性，则关闭 fid；否则，将文件指针移至开头
        if not hasattr(filename, 'read'):
            fid.close()
        else:
            fid.seek(0)

    # 返回采样率和读取的音频数据
    return fs, data
# 定义一个函数，将 NumPy 数组写入 WAV 文件
def write(filename, rate, data):
    """
    Write a NumPy array as a WAV file.

    Parameters
    ----------
    filename : string or open file handle
        Output wav file. 输出的 WAV 文件名或打开的文件句柄
    rate : int
        The sample rate (in samples/sec). 采样率（每秒的样本数）
    data : ndarray
        A 1-D or 2-D NumPy array of either integer or float data-type.
        NumPy 数组，可以是 1-D 或 2-D，整数或浮点数数据类型

    Notes
    -----
    * Writes a simple uncompressed WAV file. 写入一个简单的无压缩 WAV 文件
    * To write multiple-channels, use a 2-D array of shape (Nsamples, Nchannels).
      要写入多个通道，使用形状为 (Nsamples, Nchannels) 的 2-D 数组
    * The bits-per-sample and PCM/float will be determined by the data-type.
      位深度和 PCM/浮点数格式将由数据类型确定

    Common data types: [1]_

    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============

    Note that 8-bit PCM is unsigned. 注意 8-bit PCM 是无符号的

    References
    ----------
    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming
       Interface and Data Specifications 1.0", section "Data Format of the
       Samples", August 1991
       http://www.tactilemedia.com/info/MCI_Control_Info.html

    Examples
    --------
    Create a 100Hz sine wave, sampled at 44100Hz.
    Write to 16-bit PCM, Mono.

    >>> from scipy.io.wavfile import write
    >>> import numpy as np
    >>> samplerate = 44100; fs = 100
    >>> t = np.linspace(0., 1., samplerate)
    >>> amplitude = np.iinfo(np.int16).max
    >>> data = amplitude * np.sin(2. * np.pi * fs * t)
    >>> write("example.wav", samplerate, data.astype(np.int16))

    """
    # 如果 filename 是文件句柄，则直接使用；否则以写二进制方式打开 filename
    if hasattr(filename, 'write'):
        fid = filename
    else:
        fid = open(filename, 'wb')

    # 设置采样率为 rate
    fs = rate

    # 如果 filename 不是文件句柄，则关闭 fid；否则将文件指针移动到文件开头
    finally:
        if not hasattr(filename, 'write'):
            fid.close()
        else:
            fid.seek(0)


def _array_tofile(fid, data):
    # ravel 方法返回一个 C 风格连续的缓冲区
    fid.write(data.ravel().view('b').data)
```