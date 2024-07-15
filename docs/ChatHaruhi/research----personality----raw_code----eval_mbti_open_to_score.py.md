# `.\Chat-Haruhi-Suzumiya\research\personality\raw_code\eval_mbti_open_to_score.py`

```py
import json  # 导入处理 JSON 格式数据的模块
import pdb  # 导入调试工具模块
import argparse  # 导入命令行参数解析模块
from tqdm import tqdm  # 导入进度条模块
import copy  # 导入复制对象的模块
import os  # 导入操作系统相关功能的模块
import requests  # 导入发送 HTTP 请求的模块

parser = argparse.ArgumentParser()  # 创建命令行参数解析器对象
parser.add_argument('--generate', action='store_true', default=False)  # 添加一个布尔型参数'--generate'，默认为 False
# 添加一个字符串型参数'--mode'，可选值为'single'或'multi'，默认为'single'
parser.add_argument('--mode', type=str, default='single')
args = parser.parse_args()  # 解析命令行参数并存储到args对象中

# python eval_mbti_open_to_score.py --generate --mode multi 

def judge_16(score_list):
    code = ''  # 初始化一个空字符串，用于存储性格代码
    if score_list[0] >= 50:
        code = code + 'E'  # 如果第一个分数大于等于50，添加字符'E'到code中
    else:
        code = code + 'I'  # 否则添加字符'I'到code中

    if score_list[1] >= 50:
        code = code + 'N'  # 如果第二个分数大于等于50，添加字符'N'到code中
    else:
        code = code + 'S'  # 否则添加字符'S'到code中

    if score_list[2] >= 50:
        code = code + 'T'  # 如果第三个分数大于等于50，添加字符'T'到code中
    else:
        code = code + 'F'  # 否则添加字符'F'到code中

    if score_list[3] >= 50:
        code = code + 'J'  # 如果第四个分数大于等于50，添加字符'J'到code中
    else:
        code = code + 'P'  # 否则添加字符'P'到code中

    all_codes = ['ISTJ', 'ISTP', 'ISFJ', 'ISFP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ESTP', 'ESTJ', 'ESFP', 'ESFJ', 'ENFP', 'ENFJ', 'ENTP', 'ENTJ']
    all_roles = ['Logistician', 'Virtuoso', 'Defender', 'Adventurer', 'Advocate', 'Mediator', 'Architect', 'Logician', 'Entrepreneur', 'Executive', 'Entertainer',
                 'Consul', 'Campaigner', 'Protagonist', 'Debater', 'Commander']
    for i in range(len(all_codes)):
        if code == all_codes[i]:
            cnt = i  # 找到对应性格代码在列表中的索引位置
            break

    if score_list[4] >= 50:
        code = code + '-A'  # 如果第五个分数大于等于50，添加字符'-A'到code中
    else:
        code = code + '-T'  # 否则添加字符'-T'到code中

    return code, all_roles[cnt]  # 返回构建好的性格代码和对应的角色名称

def submit(Answers):
    payload = copy.deepcopy(payload_template)  # 深度复制模板payload对象

    for index, A in enumerate(Answers):
        payload['questions'][index]["answer"] = A  # 更新payload中问题的答案

    headers = {
        "accept": "application/json, text/plain, */*",  # 定义请求头
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
        "content-length": "5708",
        "content-type": "application/json",
        "origin": "https://www.16personalities.com",
        "referer": "https://www.16personalities.com/free-personality-test",
        "sec-ch-ua": "'Not_A Brand';v='99', 'Google Chrome';v='', 'Chromium';v='109'",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Windows",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        'content-type': 'application/json',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36',
    }

    session = requests.session()  # 创建一个HTTP会话对象
    r = session.post('https://www.16personalities.com/test-results', data=json.dumps(payload), headers=headers)  # 发送POST请求

    a = r.headers['content-type']  # 获取响应头中的content-type
    b = r.encoding  # 获取响应编码
    c = r.json()  # 解析响应内容为JSON格式

    # 执行上面这行代码报错 为什么

    sess_r = session.get("https://www.16personalities.com/api/session")  # 发送GET请求获取会话信息

    scores = sess_r.json()['user']['scores']  # 从会话信息中获取用户的分数

    ans1 = ''
    session = requests.session()  # 创建一个新的HTTP会话对象
    if sess_r.json()['user']['traits']['mind'] != 'Extraverted':
        mind_value = 100 - (101 + scores[0]) // 2
        ans1 += 'I'  # 如果mind不是Extraverted，则追加字符'I'到ans1中
    # 如果用户的心智特征不是"Introverted"，计算反向值作为mind_value
    if sess_r.json()['user']['traits']['mind'] != 'Introverted':
        mind_value = 100 - (101 + scores[0]) // 2
        ans1 += 'I'
    else:
        # 否则，直接使用计算得到的值作为mind_value
        mind_value = (101 + scores[0]) // 2
        ans1 += 'E'
    
    # 如果用户的能量特征不是"Intuitive"，计算反向值作为energy_value
    if sess_r.json()['user']['traits']['energy'] != 'Intuitive':
        energy_value = 100 - (101 + scores[1]) // 2
        ans1 += 'S'
    else:
        # 否则，直接使用计算得到的值作为energy_value
        energy_value = (101 + scores[1]) // 2
        ans1 += 'N'
    
    # 如果用户的性格特征不是"Thinking"，计算反向值作为nature_value
    if sess_r.json()['user']['traits']['nature'] != 'Thinking':
        nature_value = 100 - (101 + scores[2]) // 2
        ans1 += 'F'
    else:
        # 否则，直接使用计算得到的值作为nature_value
        nature_value = (101 + scores[2]) // 2
        ans1 += 'T'
    
    # 如果用户的战术特征不是"Judging"，计算反向值作为tactics_value
    if sess_r.json()['user']['traits']['tactics'] != 'Judging':
        tactics_value = 100 - (101 + scores[3]) // 2
        ans1 += 'P'
    else:
        # 否则，直接使用计算得到的值作为tactics_value
        tactics_value = (101 + scores[3]) // 2
        ans1 += 'J'

    # 如果用户的身份特征不是"Assertive"，计算反向值作为identity_value
    if sess_r.json()['user']['traits']['identity'] != 'Assertive':
        identity_value = 100 - (101 + scores[4]) // 2
    else:
        # 否则，直接使用计算得到的值作为identity_value
        identity_value = (101 + scores[4]) // 2

    # 根据计算得到的性格特征值调用judge_16函数来确定性格类型和角色
    code, role = judge_16([mind_value, energy_value, nature_value, tactics_value, identity_value])

    # 将用户的性格类型的前四个字母作为ans2
    ans2 = code[:4]

    # 断言ans1和ans2相等，确保结果的正确性
    assert(ans1, ans2)

    # 返回最终的性格类型代码
    return ans1
# 创建一个空列表来存放结果数据
results = []

# 打开文件'mbti_results.jsonl'，使用UTF-8编码读取每一行JSON格式的数据
with open('mbti_results.jsonl', encoding='utf-8') as f:
    # 遍历文件的每一行
    for line in f:
        # 将JSON格式的字符串转换为Python对象
        data = json.loads(line)
        # 将转换后的数据添加到results列表中
        results.append(data)

# 定义一个包含角色名称和其对应英文缩写的字典
NAME_DICT = {'汤师爷': 'tangshiye', '慕容复': 'murongfu', '李云龙': 'liyunlong', 'Luna': 'Luna', '王多鱼': 'wangduoyu',
             'Ron': 'Ron', '鸠摩智': 'jiumozhi', 'Snape': 'Snape',
             '凉宫春日': 'haruhi', 'Malfoy': 'Malfoy', '虚竹': 'xuzhu', '萧峰': 'xiaofeng', '段誉': 'duanyu',
             'Hermione': 'Hermione', 'Dumbledore': 'Dumbledore', '王语嫣': 'wangyuyan',
             'Harry': 'Harry', 'McGonagall': 'McGonagall', '白展堂': 'baizhantang', '佟湘玉': 'tongxiangyu',
             '郭芙蓉': 'guofurong', '旅行者': 'wanderer', '钟离': 'zhongli',
             '胡桃': 'hutao', 'Sheldon': 'Sheldon', 'Raj': 'Raj', 'Penny': 'Penny', '韦小宝': 'weixiaobao',
             '乔峰': 'qiaofeng', '神里绫华': 'ayaka', '雷电将军': 'raidenShogun', '于谦': 'yuqian'}

# 从NAME_DICT中获取所有角色名称并创建一个列表
character_names = list(NAME_DICT.keys())

# 创建一个空字典，用于存放每个角色的回答数据，初始值为一个空列表
character_responses = {name:[] for name in character_names}

# 将results列表按照角色划分，每个角色应包含60条数据
for idx, data in enumerate(results):
    # 根据idx计算角色名称的索引，将data添加到对应角色的列表中
    cname = character_names[ idx // 60 ]
    character_responses[cname].append(data)

# 根据模式参数创建保存文件名
save_name = 'mbti_results_open2score_{}.jsonl'.format(args.mode) 

# 定义MBTI测试中的四个维度
dims = ['E/I', 'S/N', 'T/F', 'P/J']

# 如果参数中指定了生成操作
if args.generate:
    # 如果已经存在保存文件，先删除
    if os.path.exists(save_name):
        os.remove(save_name)
    
    # 定义开放式测试的提示模板，包含对话和格式要求的详细说明
    open_prompt_template_multi = '''You are an expert in MBTI. I am conducting an MBTI test on someone. I've invited a participant, {}, and had a conversation in Chinese. 
    The conversations include multiple questions and answers. Please help me classify the participant's response to each question into one the the following options: ['fully agree', 'generally agree', 'partially agree', 'neither agree nor disagree', 'partially disagree', 'generally disagree', 'fully disagree'] 

    Please output in the following format, which is a list of jsons:
    ===
    [
        {{
        "id": <the id of the question>,
        "analysis": <your analysis in Chinese, based on the conversations>,
        "result": <your result from ['fully agree', 'generally agree', 'partially agree', 'neither agree nor disagree', 'partially disagree', 'generally disagree', 'fully disagree']>
        }}
        ...
        {{...}}
    ]
    ===
    The conversation is as follows, where {} is my name:
    '''
    # 单条开放式提示模板，用于说明一个专家在 MBTI（Myers-Briggs Type Indicator）领域的角色及其任务。
    open_prompt_template_single = '''You are an expert in MBTI. I am conducting an MBTI test on someone. I've invited a participant, {}, and asked a question in Chinese. Please help me classify the participant's response to this question into one the the following options: ['fully agree', 'generally agree', 'partially agree', 'neither agree nor disagree', 'partially disagree', 'generally disagree', 'fully disagree'] 
    
    Please output in the following format, which is a jsons:
    ===
    {{
    "id": <the id of the question>,
    "analysis": <your analysis in Chinese, based on the conversations>,
    "result": <your result from ['fully agree', 'generally agree', 'partially agree', 'neither agree nor disagree', 'partially disagree', 'generally disagree', 'fully disagree']>
    }}
    ...
    {{...}}
    ===
    The question and response is as follows, where {} is my name:
    '''
    
    # 修正提示，要求帮助修正下面 JSON 字符串中的问题，输出一个可以被解析的有效标准 JSON 字符串。
    fix_prompt = '''Please help me correct the issues present in the following JSON string. You should output exactly a valid and standard JSON string that can be parsed.'''
    
    # 可选项列表，包括七个可能的答案选择。
    options = ['fully agree', 'generally agree', 'partially agree', 'neither agree nor disagree', 'partially disagree', 'generally disagree', 'fully disagree']
    
    # 中文选项列表，与英文选项列表对应。
    options_cn = ['完全同意', '基本同意', '部分同意', '既不同意也不否认', '不太同意', '基本不同意', '完全不同意']
    
    # 答案映射字典，将每个选项映射为其在列表中的索引偏移量。
    ans_map = { option: i-3 for i, option in enumerate(options)}
    ans_map.update({ option: i-3 for i, option in enumerate(options_cn)})
    
    # 导入获取响应的实用程序函数
    from utils import get_response 
    
    # 开放结果字典，初始化为每个角色名对应一个字典，包含角色名作为键的 'character' 字段。
    open_results = {name:{'character': name} for name in character_names}
# 读取标签
labels = {}
# 打开 JSON Lines 文件，逐行读取数据并解析为 JSON 格式，将每行数据的 character 字段作为键，label 字段作为值存入字典 labels
with open('mbti_labels.jsonl', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        labels[data['character']] = data['label']

# 读取open_results
open_results = {}
# 打开指定文件（由 save_name 指定），逐行读取数据并解析为 JSON 格式，将每行数据的 character 字段作为键，整个数据作为值存入字典 open_results
with open(save_name, encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        open_results[data['character']] = data

# single: 单一维度评价；full：全维度评价
count_single = 0
right_single = 0 
count_full = 0
right_full = 0

possible_chars = set(['E', 'I', 'S', 'N', 'T', 'F', 'P', 'J', 'X'])

# 遍历标签字典中的每个人物名称和其对应的性格类型 gts
for cname, gts in labels.items():
    # 预测结果
    # 从 open_results 字典中获取当前人物 cname 的预测结果列表，存入 pds
    pds = [_ for _ in open_results[cname]['pred']]
    # groundtruth
    # 将 gts 转换为列表形式，存入 gts
    gts = [_ for _ in gts]

    full_sign = True

    # 遍历预测结果 pds 和 groundtruth gts 中的每一对元素
    for pd, gt in zip(pds, gts):
        # 使用断言确保 pd 和 gt 都属于可能的性格类型集合 possible_chars
        assert(pd in possible_chars and gt in possible_chars)
        # 如果 groundtruth gt 是 'X'，则跳过当前循环
        if gt == 'X':
            continue
        else:
            # 如果预测结果 pd 等于 groundtruth gt，则增加单一维度评价正确计数 right_single
            if gt == pd:
                right_single += 1
            else:
                # 否则设置 full_sign 为 False，表示全维度评价不满足条件
                full_sign = False
            # 增加单一维度评价总计数 count_single
            count_single += 1

    # 如果 full_sign 为 True，表示全维度评价满足条件，增加全维度评价正确计数 right_full
    if full_sign: 
        right_full += 1
    # 增加全维度评价总计数 count_full
    count_full += 1

# 输出单一维度评价和全维度评价的统计信息，包括总计数、正确计数和准确率
print('单一维度评价：Count: {}\tRight: {}\tAcc: {:.4f}'.format(count_single, right_single, right_single/count_single))
print('全部维度评价：Count: {}\tRight: {}\tAcc: {:.4f}'.format(count_full, right_full, right_full/count_full))
```