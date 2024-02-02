# `ChatRWKV\music\run.py`

```py
# The RWKV Language Model - https://github.com/BlinkDL/RWKV
# 导入所需的库
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# 设置环境变量
os.environ['RWKV_JIT_ON'] = '1' # 在导入 RWKV 之前设置这些
os.environ["RWKV_CUDA_ON"] = '0'
os.environ["RWKV_RESCALE_LAYER"] = '999' # 必须为 RWKV-music 模型设置此项，并且升级到 v0.8.12+

# 导入 RWKV 模型和管道
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

# 设置模型文件路径
# MODEL_FILE = '/fsx/BlinkDL/HF-MODEL/rwkv-4-music/RWKV-4-MIDI-120M-v1-20230714-ctx4096' # MIDI model
MODEL_FILE = '/fsx/BlinkDL/HF-MODEL/rwkv-5-music/RWKV-5-ABC-82M-v1-20230901-ctx1024' # ABC model (see https://abc.rectanglered.com)

# 检查模型类型
ABC_MODE = ('-ABC-' in MODEL_FILE)
MIDI_MODE = ('-MIDI-' in MODEL_FILE)

# 创建 RWKV 模型对象
model = RWKV(model=MODEL_FILE, strategy='cpu fp32')
pipeline = PIPELINE(model, "tokenizer-midi.json")

# 根据模型类型选择相应的 tokenizer
if ABC_MODE:
    class ABCTokenizer():
        def __init__(self):
            self.pad_token_id = 0
            self.bos_token_id = 2
            self.eos_token_id = 3
        def encode(self, text):
            ids = [ord(c) for c in text]
            return ids
        def decode(self, ids):
            txt = ''.join(chr(idx) if idx > self.eos_token_id else '' for idx in ids if idx != self.eos_token_id)
            return txt
    tokenizer = ABCTokenizer()
    EOS_ID = tokenizer.eos_token_id
    TOKEN_SEP = ''
elif MIDI_MODE:
    tokenizer = pipeline
    EOS_ID = 0
    TOKEN_SEP = ' '

# MIDI 模型的附加信息
# MIDI model:
# Use https://github.com/briansemrau/MIDI-LLM-tokenizer/blob/main/str_to_midi.py to convert output to MIDI
# Use https://midiplayer.ehubsoft.net/ and select Full MIDI Player (30M) to play MIDI
# 循环10次，每次输出当前的循环次数
for TRIAL in range(10):
    print(TRIAL)

    # 如果启用了 ABC 模式
    if ABC_MODE:
        # 设置 ccc_output 变量为 'S:2'，表示有2个段落（也可以尝试1/2/3/4）
        ccc_output = 'S:2' 

        # 另一个例子
        # 设置 ccc_output 变量为一段 ABC 音乐谱的字符串
        ccc_output = '''S:3
B:9
E:4
B:9
E:4
E:4
B:9
L:1/8
M:3/4
K:D
 Bc |"G" d2 cB"A" A2 FE |"Bm" F2 B4 F^G |'''
        # 在 ccc_output 字符串前加上特殊字符，表示开始
        ccc = chr(tokenizer.bos_token_id) + ccc_output

        # 打开一个文件，文件名为 abc_当前循环次数.txt，以写入模式
        fout = open(f"abc_{TRIAL}.txt", "w")
    # 如果 MIDI_MODE 为真，则执行以下代码块
    elif MIDI_MODE:
        # 设置 ccc 变量为 '<pad>'，用作 MIDI 模型的分隔符
        ccc = '<pad>'
        # 设置 ccc_output 变量为 '<start>'，用作 str_to_midi.py 的分隔符
        ccc_output = '<start>'
        
        # 如果需要继续一个旋律，则取消下面的注释，并设置相应的旋律内容
        # ccc = "v:5b:3 v:5b:2 t125 t125 t125 t106 pi:43:5 t24 pi:4a:7 t15 pi:4f:7 t17 pi:56:7 t18 pi:54:7 t125 t49 pi:51:7 t117 pi:4d:7 t125 t125 t111 pi:37:7 t14 pi:3e:6 t15 pi:43:6 t12 pi:4a:7 t17 pi:48:7 t125 t60 pi:45:7 t121 pi:41:7 t125 t117 s:46:5 s:52:5 f:46:5 f:52:5 t121 s:45:5 s:46:0 s:51:5 s:52:0 f:45:5 f:46:0 f:51:5 f:52:0 t121 s:41:5 s:45:0 s:4d:5 s:51:0 f:41:5 f:45:0 f:4d:5 f:51:0 t102 pi:37:0 pi:3e:0 pi:41:0 pi:43:0 pi:45:0 pi:48:0 pi:4a:0 pi:4d:0 pi:4f:0 pi:51:0 pi:54:0 pi:56:0 t19 s:3e:5 s:41:0 s:4a:5 s:4d:0 f:3e:5 f:41:0 f:4a:5 f:4d:0 t121 v:3a:5 t121 v:39:7 t15 v:3a:0 t106 v:35:8 t10 v:39:0 t111 v:30:8 v:35:0 t125 t117 v:32:8 t10 v:30:0 t125 t125 t103 v:5b:0 v:5b:0 t9 pi:4a:7"
        # ccc = '<pad> ' + ccc
        # ccc_output = '<start> pi:4a:7'
        
        # 打开一个名为 midi_{TRIAL}.txt 的文件，以写入模式
        fout = open(f"midi_{TRIAL}.txt", "w")
        
    # 将 ccc_output 写入文件
    fout.write(ccc_output)

    # 创建一个空字典 occurrence
    occurrence = {}
    # 设置 state 变量为 None
    state = None
    # 循环4096次，仅使用ctx4096进行训练（很快会变长）
    for i in range(4096): 
        
        # 如果是第一次循环，使用模型前向传播计算输出和状态
        if i == 0:
            out, state = model.forward(tokenizer.encode(ccc), state)
        else:
            out, state = model.forward([token], state)

        # 如果是MIDI模式，对输出进行特定处理
        if MIDI_MODE: 
            # 对出现的音符进行惩罚
            for n in occurrence:
                out[n] -= (0 + occurrence[n] * 0.5)
            
            # 调整输出的特定位置的值
            out[0] += (i - 2000) / 500 # 不要太短，也不要太长
            out[127] -= 1 # 避免 "t125"

            # 取消注释以启用仅钢琴模式
            # out[128:12416] -= 1e10
            # out[13952:20096] -= 1e10
        
        # 根据个人口味找到最佳的采样方式
        token = pipeline.sample_logits(out, temperature=1.0, top_k=8, top_p=0.8)
        # token = pipeline.sample_logits(out, temperature=1.0, top_p=0.7)
        # token = pipeline.sample_logits(out, temperature=1.0, top_p=0.5)

        # 如果token为结束符号，跳出循环
        if token == EOS_ID: break
        
        # 如果是MIDI模式，对输出进行特定处理
        if MIDI_MODE: 
            # 减少重复惩罚
            for n in occurrence: occurrence[n] *= 0.997 
            # 如果token大于等于128或者等于127，对出现的音符进行处理
            if token >= 128 or token == 127:
                occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
            else:
                occurrence[token] = 0.3 + (occurrence[token] if token in occurrence else 0)
        
        # 将token解码后写入文件
        fout.write(TOKEN_SEP + tokenizer.decode([token]))
        fout.flush()

    # 如果是MIDI模式，写入结束标记
    if MIDI_MODE:
        fout.write(' <end>')
    fout.close()
```