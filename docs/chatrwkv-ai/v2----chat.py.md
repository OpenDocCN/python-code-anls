# `ChatRWKV\v2\chat.py`

```
# 导入必要的库和模块
import os, copy, types, gc, sys
# 获取当前文件所在路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 将上级目录的 src 目录添加到系统路径中
sys.path.append(f'{current_path}/../rwkv_pip_package/src')

# 导入 numpy 库
import numpy as np
# 导入 prompt_toolkit 库中的 prompt 函数
from prompt_toolkit import prompt
# 尝试设置环境变量 CUDA_VISIBLE_DEVICES 为命令行参数中的第一个参数
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
# 设置 numpy 打印选项
np.set_printoptions(precision=4, suppress=True, linewidth=200)
# 创建一个简单的命名空间对象 args
args = types.SimpleNamespace()

# 打印提示信息
print('\n\nChatRWKV v2 https://github.com/BlinkDL/ChatRWKV')

# 导入 torch 库
import torch
# 设置 cudnn 的 benchmark 为 True
torch.backends.cudnn.benchmark = True
# 设置 cudnn 的 allow_tf32 为 True
torch.backends.cudnn.allow_tf32 = True
# 设置 cuda 的 matmul 的 allow_tf32 为 True

# 下面是一系列的设置，用于调优性能，可以尝试不同的 True/False 组合来找到最快的设置
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

# 设置策略说明，可以参考 https://pypi.org/project/rwkv/ 中的策略指南
# args.strategy = 'cpu fp32'
args.strategy = 'cuda fp16'
# args.strategy = 'cuda:0 fp16 -> cuda:1 fp16'
# args.strategy = 'cuda fp16i8 *10 -> cuda fp16'
# args.strategy = 'cuda fp16i8'
# args.strategy = 'cuda fp16i8 -> cpu fp32 *10'
# args.strategy = 'cuda fp16i8 *10+'
# 设置环境变量，用于控制是否启用即时编译
os.environ["RWKV_JIT_ON"] = '1' # '1' or '0', please use torch 1.13+ and benchmark speed
# 设置环境变量，用于控制是否启用CUDA编译
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

# 设置聊天语言为英语
CHAT_LANG = 'English' # English // Chinese // more to come

# 从https://huggingface.co/BlinkDL下载RWKV模型
# 在模型路径中使用'/'而不是'\'
# 使用convert_model.py将模型转换为一种策略，以加快加载速度并节省CPU内存
if CHAT_LANG == 'English':
    args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230313-ctx8192-test1050'

elif CHAT_LANG == 'Chinese': # Raven系列可以对话和 +i 问答。Novel系列是小说模型，请只用 +gen 指令续写。
    args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-world/RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-novel/RWKV-4-Novel-7B-v1-ChnEng-20230426-ctx8192'

elif CHAT_LANG == 'Japanese':
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-14B-v8-EngAndMore-20230408-ctx4096'
    args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v10-Eng89%-Jpn10%-Other1%-20230420-ctx4096'

# 为[用户和机器人]（问答）提示使用-1.py
# 为[Bob和Alice]（聊天）提示使用-2.py
PROMPT_FILE = f'{current_path}/prompt/default/{CHAT_LANG}-2.py'

# 短聊天长度
CHAT_LEN_SHORT = 40
# 长聊天长度
CHAT_LEN_LONG = 150
# 自由生成长度
FREE_GEN_LEN = 256

# 为了更好的聊天和问答质量：降低温度，降低top-p，增加重复惩罚
# 解释：https://platform.openai.com/docs/api-reference/parameter-details
GEN_TEMP = 1.2 # It could be a good idea to increase temp when top_p is low
# 设置生成文本时的 top_p 参数，减少其值可以提高问答的准确性（0.5, 0.2, 0.1 等）
GEN_TOP_P = 0.5 
# 存在惩罚参数
GEN_alpha_presence = 0.4 
# 频率惩罚参数
GEN_alpha_frequency = 0.4 
# 惩罚衰减
GEN_penalty_decay = 0.996
# 避免重复的标点符号
AVOID_REPEAT = '，：？！'

# 将输入分成块以节省显存（长度越短 -> 速度越慢）
CHUNK_LEN = 256 

# 如果 MODEL_NAME 以 '/' 结尾，则进行以下操作
if args.MODEL_NAME.endswith('/'): 
    # 如果目录下存在 'rwkv-final.pth' 文件，则将 MODEL_NAME 设置为该文件的路径
    if 'rwkv-final.pth' in os.listdir(args.MODEL_NAME):
        args.MODEL_NAME = args.MODEL_NAME + 'rwkv-final.pth'
    # 否则，找到最新的以 '.pth' 结尾的文件，并将 MODEL_NAME 设置为该文件的路径
    else:
        latest_file = sorted([x for x in os.listdir(args.MODEL_NAME) if x.endswith('.pth')], key=lambda x: os.path.getctime(os.path.join(args.MODEL_NAME, x)))[-1]
        args.MODEL_NAME = args.MODEL_NAME + latest_file

########################################################################################################

# 打印提示信息
print(f'\n{CHAT_LANG} - {args.strategy} - {PROMPT_FILE}')
# 导入 RWKV 模型和 PIPELINE 工具
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

# 加载提示文件
def load_prompt(PROMPT_FILE):
    variables = {}
    # 以二进制形式打开提示文件
    with open(PROMPT_FILE, 'rb') as file:
        # 执行提示文件中的代码，并将结果存储在 variables 中
        exec(compile(file.read(), PROMPT_FILE, 'exec'), variables)
    # 从 variables 中获取 user, bot, interface, init_prompt 变量
    user, bot, interface, init_prompt = variables['user'], variables['bot'], variables['interface'], variables['init_prompt']
    # 对 init_prompt 进行处理，去除空格和换行符
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
        init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n' + ('\n'.join(init_prompt)).strip() + '\n\n'
    # 返回处理后的 user, bot, interface, init_prompt
    return user, bot, interface, init_prompt

# 打印加载模型的信息
print(f'Loading model - {args.MODEL_NAME}')
# 使用 RWKV 模型和指定的策略加载模型
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
# 如果 MODEL_NAME 中包含 'world/' 或 '-World-'，则执行以下操作
if 'world/' in args.MODEL_NAME or '-World-' in args.MODEL_NAME:
    # 创建一个管道对象，使用指定的模型和词汇表版本
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    # 定义表示文本结束的标记
    END_OF_TEXT = 0
    # 定义表示行结束的标记
    END_OF_LINE = 11
else:
    # 使用模型和指定的tokenizer文件创建pipeline对象
    pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")
    # 定义特殊标记
    END_OF_TEXT = 0
    END_OF_LINE = 187
    END_OF_LINE_DOUBLE = 535
# pipeline = PIPELINE(model, "cl100k_base")
# END_OF_TEXT = 100257
# END_OF_LINE = 198

# 初始化模型tokens和状态
model_tokens = []
model_state = None

# 定义需要避免重复的tokens列表
AVOID_REPEAT_TOKENS = []
# 遍历AVOID_REPEAT列表，使用pipeline对象对每个元素进行编码，并将结果添加到AVOID_REPEAT_TOKENS列表中
for i in AVOID_REPEAT:
    dd = pipeline.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd

########################################################################################################

# 定义运行RNN的函数
def run_rnn(tokens, newline_adj = 0):
    global model_tokens, model_state

    # 将tokens转换为整数列表
    tokens = [int(x) for x in tokens]
    # 将tokens添加到model_tokens中
    model_tokens += tokens
    # 在控制台打印模型tokens和对应的文本
    # print(f'### model ###\n{tokens}\n[{pipeline.decode(model_tokens)}]')

    # 当tokens列表长度大于0时，循环执行模型的前向传播
    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    # 调整换行符的概率
    out[END_OF_LINE] += newline_adj

    # 如果model_tokens中的最后一个token在AVOID_REPEAT_TOKENS中，则将其概率设置为极小值
    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return out

# 定义保存所有状态的函数
all_state = {}
def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)

# 定义加载所有状态的函数
def load_all_stat(srv, name):
    global model_tokens, model_state
    n = f'{name}_{srv}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']

# 修正tokens的函数
# 在特定条件下，对tokens进行修正
def fix_tokens(tokens):
    if 'world/' in args.MODEL_NAME or '-World-' in args.MODEL_NAME:
        return tokens
    if len(tokens) > 0 and tokens[-1] == END_OF_LINE_DOUBLE:
        tokens = tokens[:-1] + [END_OF_LINE, END_OF_LINE]
    return tokens

########################################################################################################

# 运行推断
# 打印提示信息
print(f'\nRun prompt...')

# 从文件中加载用户、机器人、接口和初始提示信息
user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)

# 运行 RNN 模型，对初始提示信息进行编码
out = run_rnn(fix_tokens(pipeline.encode(init_prompt)))

# 保存所有统计信息
save_all_stat('', 'chat_init', out)

# 手动触发垃圾回收
gc.collect()

# 释放 CUDA 缓存
torch.cuda.empty_cache()

# 服务器列表
srv_list = ['dummy_server']

# 遍历服务器列表
for s in srv_list:
    # 保存所有统计信息
    save_all_stat(s, 'chat', out)

# 定义回复消息的函数
def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')

# 处理消息的函数
def on_message(message):
    global model_tokens, model_state, user, bot, interface, init_prompt

    # 服务器名称
    srv = 'dummy_server'

    # 处理消息，替换特殊字符并去除首尾空格
    msg = message.replace('\\n','\n').strip()

    # 生成文本的温度和 top_p 参数
    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P

    # 解析消息中的温度参数
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")

    # 解析消息中的 top_p 参数
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")

    # 确保温度参数在合理范围内
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5

    # 确保 top_p 参数在合理范围内
    if x_top_p <= 0:
        x_top_p = 0

    # 去除消息首尾空格
    msg = msg.strip()
    
    # 如果消息为 '+reset'，则重置聊天状态
    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return
    
    # 如果消息以 '+prompt {path}' 开头，则加载新的提示信息
    elif msg[:8].lower() == '+prompt ':
        print("Loading prompt...")
        try:
            # 设置新的提示文件路径
            PROMPT_FILE = msg[8:].strip()
            user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)
            out = run_rnn(fix_tokens(pipeline.encode(init_prompt)))
            save_all_stat(srv, 'chat', out)
            print("Prompt set up.")
            gc.collect()
            torch.cuda.empty_cache()
        except:
            print("Path error.")
    # 如果消息以特定前缀开头，或者消息本身是特定字符串，则执行以下操作
    elif msg[:5].lower() == '+gen ' or msg[:3].lower() == '+i ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

        # 如果消息以'+gen '开头，则执行以下操作
        if msg[:5].lower() == '+gen ':
            # 提取出消息中的内容，并去除首尾空格
            new = '\n' + msg[5:].strip()
            # 保存模型状态和标记
            model_state = None
            model_tokens = []
            # 运行RNN模型，并传入编码后的消息
            out = run_rnn(pipeline.encode(new))
            # 保存生成的统计数据
            save_all_stat(srv, 'gen_0', out)

        # 如果消息以'+i '开头，则执行以下操作
        elif msg[:3].lower() == '+i ':
            # 去除消息中的换行符和多余的空行
            msg = msg[3:].strip().replace('\r\n','\n').replace('\n\n','\n')
            # 创建新的消息
            new = f'''
# 根据不同的语言设置不同的帮助信息
if CHAT_LANG == 'English':
    # 设置英文语言下的帮助信息
    HELP_MSG = '''Commands:
say something --> chat with bot. use \\n for new line.
+ --> alternate chat reply
+reset --> reset chat

+gen YOUR PROMPT --> free single-round generation with any prompt. use \\n for new line.
+i YOUR INSTRUCT --> free single-round generation with any instruct. use \\n for new line.
+++ --> continue last free generation (only for +gen / +i)
++ --> retry last free generation (only for +gen / +i)

Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B (especially https://huggingface.co/BlinkDL/rwkv-4-raven) for best results.
'''
elif CHAT_LANG == 'Chinese':
    # 设置中文语言下的帮助信息
    HELP_MSG = f'''指令:
直接输入内容 --> 和机器人聊天（建议问机器人问题），用\\n代表换行，必须用 Raven 模型
+ --> 让机器人换个回答
+reset --> 重置对话，请经常使用 +reset 重置机器人记忆

+i 某某指令 --> 问独立的问题（忽略聊天上下文），用\\n代表换行，必须用 Raven 模型
+gen 某某内容 --> 续写内容（忽略聊天上下文），用\\n代表换行，写小说用 testNovel 模型
+++ --> 继续 +gen / +i 的回答
++ --> 换个 +gen / +i 的回答

作者：彭博 请关注我的知乎: https://zhuanlan.zhihu.com/p/603840957
如果喜欢，请看我们的优质护眼灯: https://withablink.taobao.com

中文 Novel 模型，可以试这些续写例子（不适合 Raven 模型）：
+gen “区区
+gen 以下是不朽的科幻史诗长篇巨著，描写细腻，刻画了数百位个性鲜明的英雄和宏大的星际文明战争。\\n第一章
+gen 这是一个修真世界，详细世界设定如下：\\n1.
'''
elif CHAT_LANG == 'Japanese':
    # 设置日文语言下的帮助信息
    HELP_MSG = f'''コマンド:
直接入力 --> ボットとチャットする．改行には\\nを使用してください．
+ --> ボットに前回のチャットの内容を変更させる．
+reset --> 対話のリセット．メモリをリセットするために，+resetを定期的に実行してください．

+i インストラクトの入力 --> チャットの文脈を無視して独立した質問を行う．改行には\\nを使用してください．
+gen プロンプトの生成 --> チャットの文脈を無視して入力したプロンプトに続く文章を出力する．改行には\\nを使用してください．
+++ --> +gen / +i の出力の回答を続ける．
++ --> +gen / +i の出力の再生成を行う.

ボットとの会話を楽しんでください。また、定期的に+resetして、ボットのメモリをリセットすることを忘れないようにしてください。
'''

# 打印帮助信息
print(HELP_MSG)
# 打印语言、模型名称和策略
print(f'{CHAT_LANG} - {args.MODEL_NAME} - {args.strategy}')
# 打印经过 pipeline 解码的 model_tokens，并替换其中的换行符，然后以空字符结尾输出
print(f'{pipeline.decode(model_tokens)}'.replace(f'\n\n{bot}',f'\n{bot}'), end='')

########################################################################################################

# 无限循环，等待用户输入消息
while True:
    # 从用户处获取消息
    msg = prompt(f'{user}{interface} ')
    # 如果消息去除首尾空格后长度大于 0
    if len(msg.strip()) > 0:
        # 调用 on_message 函数处理消息
        on_message(msg)
    else:
        # 如果消息为空，打印错误提示
        print('Error: please say something')
```