# `ChatRWKV\chat.py`

```py
# 导入所需的库
import os, copy, types, gc, sys
import numpy as np
from prompt_toolkit import prompt

# 尝试设置环境变量 CUDA_VISIBLE_DEVICES 为命令行参数中的第一个参数
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass

# 设置 numpy 打印选项
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# 创建一个简单的命名空间对象
args = types.SimpleNamespace()

# 打印项目链接
print('\n\nChatRWKV project: https://github.com/BlinkDL/ChatRWKV')

# 打印提示信息
for i in range(10):
    print('NOTE: This code is v1 and only for reference. Use v2 instead.')

# 导入 torch 库，并设置一些相关的参数
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# 下面是一些需要调整的参数，用于找到最快的设置
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

# 设置运行设备为 cuda 或 cpu
args.RUN_DEVICE = "cuda"  # cuda // cpu
# 设置浮点数模式为 fp16（适用于 GPU，不支持 CPU）// fp32（适用于 CPU）// bf16（精度较差，支持 CPU）
args.FLOAT_MODE = "fp16"

# 设置环境变量 RWKV_JIT_ON 为 '1'，请使用 torch 1.13+ 并进行性能测试
os.environ["RWKV_JIT_ON"] = '1'

# 设置聊天语言为英文
CHAT_LANG = 'English' # English // Chinese // more to come

# 设置问答提示为 False（用户和机器人交互）或 True（问答交互）
QA_PROMPT = False # True: Q & A prompt // False: User & Bot prompt
# 中文问答设置QA_PROMPT=True（只能问答，问答效果更好，但不能闲聊） 中文聊天设置QA_PROMPT=False（可以闲聊，但需要大模型才适合闲聊）

# 从 https://huggingface.co/BlinkDL 下载 RWKV-4 模型（不要使用 Instruct-test 模型，除非使用它们的提示模板）
    # 设置模型名称为指定路径
    args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230228-ctx4096-test663'
    # 以下是备选的模型名称，可以根据需要注释掉或者修改
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20221115-8047'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/7-run1z/rwkv-340'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/14b-run1/rwkv-6210'
# 如果聊天语言是中文，设置模型名称为指定的中文测试小说模型
elif CHAT_LANG == 'Chinese': # testNovel系列是网文模型，请只用 +gen 指令续写。test4 系列可以问答（只用了小中文语料微调，纯属娱乐）
    args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-EngChn-testNovel-441-ctx2048-20230217'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-EngChn-testNovel-711-ctx2048-20230216'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-testNovel-671-ctx2048-20230216'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/7-run1z/rwkv-973'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/3-run1z/rwkv-711'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/1.5-run1z/rwkv-671'

# 设置上下文长度为1024
args.ctx_len = 1024

# 设置短对话长度为40
CHAT_LEN_SHORT = 40
# 设置长对话长度为150
CHAT_LEN_LONG = 150
# 设置自由生成长度为200
FREE_GEN_LEN = 200

# 设置生成温度为1.0
GEN_TEMP = 1.0
# 设置生成的top-p值为0.85
GEN_TOP_P = 0.85

# 设置要避免的重复字符
AVOID_REPEAT = '，。：？！'

########################################################################################################

# 设置环境变量RWKV_RUN_DEVICE为args.RUN_DEVICE
os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE
# 打印加载的ChatRWKV模型信息
print(f'\nLoading ChatRWKV - {CHAT_LANG} - {args.RUN_DEVICE} - {args.FLOAT_MODE} - QA_PROMPT {QA_PROMPT}')

# 从src.model_run导入RWKV_RNN
from src.model_run import RWKV_RNN
# 从src.utils导入TOKENIZER
from src.utils import TOKENIZER
# 使用"20B_tokenizer.json"创建tokenizer
tokenizer = TOKENIZER("20B_tokenizer.json")

# 设置词汇表大小为50277
args.vocab_size = 50277
# 设置头部qk为0
args.head_qk = 0
# 设置预处理ffn为0
args.pre_ffn = 0
# 设置梯度检查点为0
args.grad_cp = 0
# 设置自定义位置嵌入为0
args.my_pos_emb = 0
# 设置模型名称为args.MODEL_NAME
MODEL_NAME = args.MODEL_NAME

# 如果聊天语言是英文
if CHAT_LANG == 'English':
    # 设置界面为冒号
    interface = ":"

    # 如果QA_PROMPT为真
    if QA_PROMPT:
        # 设置用户为"User"
        user = "User"
        # 设置机器人为"Bot"
        bot = "Bot" # Or: 'The following is a verbose and detailed Q & A conversation of factual information.'
        # 设置初始提示
        init_prompt = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} french revolution what year

{bot}{interface} The French Revolution started in 1789, and lasted 10 years until 1799.

{user}{interface} 3+5=?

{bot}{interface} The answer is 8.

{user}{interface} guess i marry who ?
# 用户和机器人之间的对话
{bot}{interface} Only if you tell me more about yourself - what are your interests?

{user}{interface} solve for a: 9-a=2

{bot}{interface} The answer is a = 7, because 9 - 7 = 2.

{user}{interface} wat is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

# 如果语言是英语
else:
    user = "Bob"
    bot = "Alice"
    init_prompt = f'''
The following is a verbose detailed conversation between {user} and a young girl {bot}. {bot} is intelligent, friendly and cute. {bot} is unlikely to disagree with {user}.

{user}{interface} Hello {bot}, how are you doing?

{bot}{interface} Hi {user}! Thanks, I'm fine. What about you?

{user}{interface} I am very good! It's nice to see you. Would you mind me chatting with you for a while?

{bot}{interface} Not at all! I'm listening.

'''

# 如果语言是中文
elif CHAT_LANG == 'Chinese':
    interface = ":"
    if QA_PROMPT:
        user = "Q"
        bot = "A"
        init_prompt = f'''
Expert Questions & Helpful Answers

Ask Research Experts

'''
    else:
        user = "User"
        bot = "Bot"
        init_prompt = f'''
# 定义一个详细的对话，包括一个名为{bot}的AI助手和一个名为{user}的人类用户
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

# 用户询问关于LHC的问题
{user}{interface} wat is lhc

# 机器人回答关于LHC的问题
{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

# 用户询问企鹅是否会飞
{user}{interface} 企鹅会飞吗

# 机器人回答企鹅不会飞，并解释了企鹅的翅膀主要用于游泳和平衡
{bot}{interface} 企鹅是不会飞的。它们的翅膀主要用于游泳和平衡，而不是飞行。

# 定义帮助信息
HELP_MSG = f'''指令:

# 用户可以直接输入内容与机器人聊天，用\\n代表换行
直接输入内容 --> 和机器人聊天（建议问机器人问题），用\\n代表换行

# 用户可以使用+号让机器人换个回答
+ --> 让机器人换个回答

# 用户可以使用+reset重置对话
+reset --> 重置对话

# 用户可以使用+gen续写任何中英文内容，用\\n代表换行
+gen 某某内容 --> 续写任何中英文内容，用\\n代表换行

# 用户可以使用+qa问独立的问题（忽略上下文），用\\n代表换行
+qa 某某问题 --> 问独立的问题（忽略上下文），用\\n代表换行

# 用户可以使用+qq问独立的问题（忽略上下文），且敞开想象力，用\\n代表换行
+qq 某某问题 --> 问独立的问题（忽略上下文），且敞开想象力，用\\n代表换行

# 用户可以使用+++继续 +gen / +qa / +qq 的回答
+++ --> 继续 +gen / +qa / +qq 的回答

# 用户可以使用++换个 +gen / +qa / +qq 的回答
++ --> 换个 +gen / +qa / +qq 的回答

# 作者信息
作者：彭博 请关注我的知乎: https://zhuanlan.zhihu.com/p/603840957

# 推广信息
如果喜欢，请看我们的优质护眼灯: https://withablink.taobao.com

# 提示用户可以和机器人聊天，并说明机器人更擅长英文
现在可以输入内容和机器人聊天（注意它不大懂中文，它更懂英文）。请经常使用 +reset 重置机器人记忆。

# 提示用户没有“重复惩罚”，并告知机器人有时会重复，需要使用+换成正常回答
目前没有“重复惩罚”，所以机器人有时会重复，此时必须使用 + 换成正常回答，以免污染电脑记忆。

# 提示用户和上下文无关的独立问题，必须用 +qa 或 +qq 问，以免污染电脑记忆
注意：和上下文无关的独立问题，必须用 +qa 或 +qq 问，以免污染电脑记忆。

# 提示用户试用咒语，并强调咒语至关重要
请先试下列咒语，理解咒语的写法。咒语至关重要。

# 提示用户可以使用中文网文【testNovel】模型进行测试
中文网文【testNovel】模型，试下面这些，注意，必须是【testNovel】模型：
+gen 这是一颗
+gen 以下是不朽的科幻史诗长篇巨著，描写细腻，刻画了数百位个性鲜明的英雄和宏大的星际文明战争。\\n第一章
+gen 这是一个修真世界，详细世界设定如下：\\n1.

# 提示用户可以使用中文问答【test数字】模型进行测试
中文问答【test数字】模型，试下面这些，注意，必须是【test数字】模型：
+gen \\n活动出席发言稿：\\n大家好，
+gen \\n怎样创立一家快速盈利的AI公司：\\n1.
+gen \\nimport torch
+qq 请以《我的驴》为题写一篇作文
+qq 请以《企鹅》为题写一首诗歌
+qq 请设定一个奇幻世界，告诉我详细的世界设定。

# 加载模型
print(f'Loading model - {MODEL_NAME}')
model = RWKV_RNN(args)

# 初始化模型tokens和状态
model_tokens = []
model_state = None

# 初始化避免重复的tokens列表
AVOID_REPEAT_TOKENS = []

# 将避免重复的词转换为tokens，并添加到AVOID_REPEAT_TOKENS列表中
for i in AVOID_REPEAT:
    dd = tokenizer.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd

# 定义运行RNN的函数
def run_rnn(tokens, newline_adj = 0):
    global model_tokens, model_state

    # 将tokens转换为整数列表
    tokens = [int(x) for x in tokens]
    # 将tokens添加到模型tokens中
    model_tokens += tokens
    # 运行模型，获取输出和模型状态
    out, model_state = model.forward(tokens, model_state)

    # 禁用
    # 将out[187]增加newline_adj的值，用于调整换行符的概率
    out[187] += newline_adj # adjust \n probability
    # 如果newline_adj大于0，则将out[15]增加newline_adj的一半，用于调整句号的概率
    # out[15] += newline_adj / 2 # '.'
    # 如果model_tokens列表中的最后一个元素在AVOID_REPEAT_TOKENS中，将其对应的值设为-999999999
    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    # 返回修改后的out字典
    return out
# 创建一个空的字典用于存储所有状态
all_state = {}

# 保存所有状态
def save_all_stat(srv, name, last_out):
    # 根据服务器和名称创建一个唯一的键
    n = f'{name}_{srv}'
    # 为该键创建一个空字典
    all_state[n] = {}
    # 将最后的输出保存到对应键的字典中
    all_state[n]['out'] = last_out
    # 深拷贝模型状态并保存到对应键的字典中
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    # 深拷贝模型标记并保存到对应键的字典中
    all_state[n]['token'] = copy.deepcopy(model_tokens)

# 加载所有状态
def load_all_stat(srv, name):
    global model_tokens, model_state
    # 根据服务器和名称获取对应的键
    n = f'{name}_{srv}'
    # 从对应键的字典中加载模型状态和标记
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    # 返回对应键的最后输出
    return all_state[n]['out']

# 运行推理
print(f'\nRun prompt...')

# 运行 RNN 模型并保存状态
out = run_rnn(tokenizer.encode(init_prompt))
save_all_stat('', 'chat_init', out)
# 执行垃圾回收
gc.collect()
# 清空 CUDA 缓存
torch.cuda.empty_cache()

# 服务器列表
srv_list = ['dummy_server']
for s in srv_list:
    # 保存状态
    save_all_stat(s, 'chat', out)

# 回复消息
def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')

# 处理消息
def on_message(message):
    global model_tokens, model_state

    srv = 'dummy_server'

    msg = message.replace('\\n','\n').strip()
    # 如果消息长度超过1000，则回复消息并返回
    # if len(msg) > 1000:
    #     reply_msg('your message is too long (max 1000 tokens)')
    #     return

    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P
    # 如果消息中包含 "-temp="，则更新温度值并从消息中移除
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    # 如果消息中包含 "-top_p="，则更新 top_p 值并从消息中移除
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    # 确保温度值在合理范围内
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    # 确保 top_p 值在合理范围内
    if x_top_p <= 0:
        x_top_p = 0
    
    # 如果消息为 '+reset'，则加载初始状态并保存状态，然后回复消息并返回
    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return

# 打印帮助消息
print(HELP_MSG)
# 打印准备就绪消息
print(f'Ready - {CHAT_LANG} {args.RUN_DEVICE} {args.FLOAT_MODE} QA_PROMPT={QA_PROMPT} {args.MODEL_NAME}')
# 打印模型生成的文本，并替换其中的指定字符串，然后以空字符结尾
print(f'{tokenizer.decode(model_tokens)}'.replace(f'\n\n{bot}',f'\n{bot}'), end='')

# 进入无限循环，等待用户输入消息
while True:
    # 从用户输入中获取消息
    msg = prompt(f'{user}{interface} ')
    # 如果消息长度大于0，则调用on_message函数处理消息
    if len(msg.strip()) > 0:
        on_message(msg)
    # 如果消息长度为0，则打印错误提示
    else:
        print('Error: please say something')
```