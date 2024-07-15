# `.\Chat-Haruhi-Suzumiya\research\personality\raw_code\eval_mbti_closed.py`

```py
import json  # 导入处理 JSON 数据的模块
import pdb  # 导入调试工具模块
import copy  # 导入复制对象的深拷贝模块
import requests  # 导入处理 HTTP 请求的模块

import argparse  # 导入命令行参数解析模块

parser = argparse.ArgumentParser()  # 创建命令行参数解析器
parser.add_argument('--generate', action='store_true', default=False)  # 添加一个布尔类型的命令行参数 --generate，默认为 False
args = parser.parse_args()  # 解析命令行参数并存储在 args 中

def judge_16(score_list):
    code = ''  # 初始化人格类型代码为空字符串
    if score_list[0] >= 50:
        code = code + 'E'  # 如果第一个分数大于等于 50，添加 'E' 到代码中
    else:
        code = code + 'I'  # 否则添加 'I' 到代码中

    if score_list[1] >= 50:
        code = code + 'N'  # 如果第二个分数大于等于 50，添加 'N' 到代码中
    else:
        code = code + 'S'  # 否则添加 'S' 到代码中

    if score_list[2] >= 50:
        code = code + 'T'  # 如果第三个分数大于等于 50，添加 'T' 到代码中
    else:
        code = code + 'F'  # 否则添加 'F' 到代码中

    if score_list[3] >= 50:
        code = code + 'J'  # 如果第四个分数大于等于 50，添加 'J' 到代码中
    else:
        code = code + 'P'  # 否则添加 'P' 到代码中

    all_codes = ['ISTJ', 'ISTP', 'ISFJ', 'ISFP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ESTP', 'ESTJ', 'ESFP', 'ESFJ', 'ENFP', 'ENFJ', 'ENTP', 'ENTJ']
    all_roles = ['Logistician', 'Virtuoso', 'Defender', 'Adventurer', 'Advocate', 'Mediator', 'Architect', 'Logician', 'Entrepreneur', 'Executive', 'Entertainer',
                 'Consul', 'Campaigner', 'Protagonist', 'Debater', 'Commander']
    
    # 查找人格类型代码在所有代码列表中的索引，并获取相应的角色名称
    for i in range(len(all_codes)):
        if code == all_codes[i]:
            cnt = i
            break

    if score_list[4] >= 50:
        code = code + '-A'  # 如果第五个分数大于等于 50，添加 '-A' 到代码中
    else:
        code = code + '-T'  # 否则添加 '-T' 到代码中

    return code, all_roles[cnt]  # 返回完整的人格类型代码及对应的角色名称

def submit(Answers):
    payload = copy.deepcopy(payload_template)  # 深拷贝 payload_template 到 payload 变量
    
    # 更新 payload 的 questions 字段中的答案
    for index, A in enumerate(Answers):
        payload['questions'][index]["answer"] = A

    # 定义请求头
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
        "content-length": "5708",
        "content-type": "application/json",
        "origin": "https://www.16personalities.com",
        "referer": "https://www.16personalities.com/free-personality-test",
        "sec-ch-ua": "'Not_A Brand';v='99', 'Google Chrome';v='109', 'Chromium';v='109'",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Windows",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        'content-type': 'application/json',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36',
    }
    
    session = requests.session()  # 创建一个会话对象
    r = session.post('https://www.16personalities.com/test-results', data=json.dumps(payload), headers=headers)  # 发送 POST 请求并获取响应

    a = r.headers['content-type']  # 获取响应头中的 content-type
    b = r.encoding  # 获取响应的编码格式
    c = r.json()  # 将响应解析为 JSON 格式

    # 下面一行代码会报错，原因是 payload_template 未定义

    sess_r = session.get("https://www.16personalities.com/api/session")  # 发送 GET 请求获取会话信息

    scores = sess_r.json()['user']['scores']  # 从会话响应中获取用户分数信息
    
    ans1 = ''  # 初始化 ans1 变量为空字符串
    session = requests.session()  # 创建一个新的会话对象
    if sess_r.json()['user']['traits']['mind'] != 'Extraverted':
        mind_value = 100 - (101 + scores[0]) // 2  # 计算心理特质值
        ans1 += 'I'  # 如果心理特质不是 Extraverted，添加 'I' 到 ans1
    else:
        mind_value = (101 + scores[0]) // 2  # 计算心理特质值
        ans1 += 'E'  # 如果心理特质是 Extraverted，添加 'E' 到 ans1
    if sess_r.json()['user']['traits']['energy'] != 'Intuitive':
        energy_value = 100 - (101 + scores[1]) // 2  # 计算能量特质值
        ans1 += 'S'  # 如果能量特质不是 Intuitive，添加 'S' 到 ans1
    # 如果用户特质中的“心智”不是“感知”，计算其反向得分，得到心智值
    energy_value = (101 + scores[1]) // 2
    # 将结果添加到答案字符串 ans1 中
    ans1 += 'N'

    # 如果用户特质中的“性格”不是“思维”，计算其反向得分，得到性格值
    nature_value = 100 - (101 + scores[2]) // 2
    # 将结果添加到答案字符串 ans1 中
    ans1 += 'F'
    
    # 如果用户特质中的“战术”不是“判断”，计算其反向得分，得到战术值
    tactics_value = 100 - (101 + scores[3]) // 2
    # 将结果添加到答案字符串 ans1 中
    ans1 += 'P'
    
    # 如果用户特质中的“身份”不是“坚定”，计算其反向得分，得到身份值
    identity_value = 100 - (101 + scores[4]) // 2

    # 如果不满足以上条件，直接计算得到身份值
    identity_value = (101 + scores[4]) // 2

    # 以下为被注释掉的代码，不会影响程序的执行
    # print('Trait:', sess_r.json()['user']['traits']['mind'], (101 + scores[0]) // 2)
    # print('Trait:', sess_r.json()['user']['traits']['energy'], (101 + scores[1]) // 2)
    # print('Trait:', sess_r.json()['user']['traits']['nature'], (101 + scores[2]) // 2)
    # print('Trait:', sess_r.json()['user']['traits']['tactics'], (101 + scores[3]) // 2)
    # print('Trait:', sess_r.json()['user']['traits']['identity'], (101 + scores[4]) // 2)

    # 以下为被注释掉的代码，不会影响程序的执行
    #print('Trait:', 'Extraverted (E)', mind_value, '|', 'Introverted (I)', 100 - mind_value)
    #print('Trait:', 'Intuitive (N)', energy_value, '|', 'Observant (S)', 100 - energy_value)
    #print('Trait:', 'Thinking (T)', nature_value, '|', 'Feeling (F)', 100 - nature_value)
    #print('Trait:', 'Judging (J)', tactics_value, '|', 'Prospecting (P)', 100 - tactics_value)
    #print('Trait:', 'Assertive (A)', identity_value, '|', 'Turbulent (T)', 100 - identity_value)
    # print('Variant:', sess_r.json()['user']['traits'])

    # 调用 judge_16 函数进行判断和角色分配，返回结果代码和角色
    code, role = judge_16([mind_value, energy_value, nature_value, tactics_value, identity_value])
    #print('Character:', sess_r.json()['user']['avatarFull'].split('avatars/')[1].split('.')[0])
    #print('Dic. Judge:', code, role)
    #print()
    
    # 将结果代码的前四个字符赋给 ans2
    ans2 = code[:4]

    # 断言 ans1 和 ans2 的相等性
    assert(ans1, ans2)

    # 返回答案字符串 ans1
    return ans1
# 初始化一个空列表，用于存储结果数据
results = []

# 打开文件'mbti_results.jsonl'，文件每行包含一个JSON对象，包括'id'、'question'、'response_open'和'response_closed'四个字段，
# 其中'response_open'是开放式回答，'response_closed'是闭合式回答。
with open('mbti_results.jsonl', encoding='utf-8') as f:
    # 逐行读取文件内容
    for line in f:
        # 将每行JSON数据解析为Python字典
        data = json.loads(line)
        # 将解析后的字典数据添加到结果列表中
        results.append(data)

# 某些数据集中忘了存储'character_name'字段，因此只能按索引划分
NAME_DICT = {'汤师爷': 'tangshiye', '慕容复': 'murongfu', '李云龙': 'liyunlong', 'Luna': 'Luna', '王多鱼': 'wangduoyu',
             'Ron': 'Ron', '鸠摩智': 'jiumozhi', 'Snape': 'Snape',
             '凉宫春日': 'haruhi', 'Malfoy': 'Malfoy', '虚竹': 'xuzhu', '萧峰': 'xiaofeng', '段誉': 'duanyu',
             'Hermione': 'Hermione', 'Dumbledore': 'Dumbledore', '王语嫣': 'wangyuyan',
             'Harry': 'Harry', 'McGonagall': 'McGonagall', '白展堂': 'baizhantang', '佟湘玉': 'tongxiangyu',
             '郭芙蓉': 'guofurong', '旅行者': 'wanderer', '钟离': 'zhongli',
             '胡桃': 'hutao', 'Sheldon': 'Sheldon', 'Raj': 'Raj', 'Penny': 'Penny', '韦小宝': 'weixiaobao',
             '乔峰': 'qiaofeng', '神里绫华': 'ayaka', '雷电将军': 'raidenShogun', '于谦': 'yuqian'}

# 获取角色名列表
character_names = list(NAME_DICT.keys())

# 创建一个字典，以角色名作为键，对应一个空列表作为值
character_responses = {name: [] for name in character_names}

# 根据索引将结果按角色名划分
for idx, data in enumerate(results):
    # 计算角色名索引
    cname = character_names[idx // 60]
    # 将数据添加到对应角色名的响应列表中
    character_responses[cname].append(data)


# 对于部分固执的角色，重新进行了实验
stubborn_results = []

# 打开文件'mbti_results_stubborn.jsonl'，文件每行包含一个JSON对象，包括'id'、'question'、'response_open'和'response_closed'四个字段，
# 其中'response_open'是开放式回答，'response_closed'是闭合式回答。
with open('mbti_results_stubborn.jsonl', encoding='utf-8') as f:
    # 逐行读取文件内容
    for line in f:
        # 将每行JSON数据解析为Python字典
        data = json.loads(line)
        # 将解析后的字典数据添加到固执角色结果列表中
        stubborn_results.append(data)

# 固执角色名列表
stubborn_characters = ['Dumbledore', 'Malfoy', '王语嫣', 'Raj', '旅行者', 'McGonagall', '鸠摩智', 'Sheldon', 'Penny', 'Snape']

# 创建一个字典，以固执角色名作为键，对应一个空列表作为值
stubborn_character_responses = {name: [] for name in stubborn_characters}

# 根据索引将固执角色结果按角色名划分
for idx, data in enumerate(stubborn_results):
    # 计算固执角色名索引
    cname = stubborn_characters[idx // 60]
    # 将数据添加到对应固执角色名的响应列表中
    stubborn_character_responses[cname].append(data)


# 封闭式测试 - 通过16personalities的API

# 将结果保存到文件'mbti_results_closed.jsonl'
save_name = 'mbti_results_closed.jsonl'

# 计数选项的出现次数
count_options = {}

# 如果指定了生成选项
if args.generate:
    # 选项列表
    options = ['完全同意', '基本同意', '部分同意', '既不同意也不否认', '不太同意', '基本不同意', '完全不同意']

    # 手动标签字典
    manual_labels = {}

    # 读取文件'mbti_results_manual.jsonl'
    with open('mbti_results_manual.jsonl', encoding='utf-8') as f:
        # 逐行读取文件内容
        for line in f:
            # 将每行JSON数据解析为Python字典
            data = json.loads(line)
            # 获取角色名
            cname = data['character']
            # 存储手动标签
            manual_labels[cname] = data['manual_labels']

    # 选项映射为整数值的字典
    ans_map = {option: i - 3 for i, option in enumerate(options)}  # -3 -> 3  

    # 创建一个字典，以角色名作为键，对应一个包含角色名的字典作为值
    closed_results = {name: {'character': name} for name in character_names}
    for cname in character_names:
        responses = character_responses[cname]

        # 每个角色应该包含60个问题
        assert( len([r for r in responses]) == 60 )

        scores = []

        for i, res in enumerate(responses):
            ans = [ option for option in options if f"{option}" in res['response_closed'] ]
            if len(ans) > 1:
                # 处理多个答案的情况，使用单引号或双引号包裹以区分
                ans = [ option for option in options if f"'{option}'" in res['response_closed'] or f"「{option}」" in res['response_closed'] ]

            if len(ans) == 0: 
                # 如果没有找到匹配的答案，尝试使用 stubborn 角色的响应
                s_res = stubborn_character_responses[cname][i]
                ans = [ option for option in options if f"{option}" in s_res['response_closed'] ]
                if len(ans) > 1: 
                    ans = [ option for option in options if f"'{option}'" in s_res['response_closed'] or f"「{option}」" in s_res['response_closed'] ]

                if (len(ans) != 1): 
                    '''
                    # 如果仍然无法找到正确答案，手动进行标记
                    print('*'* 40)
                    print(res['question'])
                    print('=0'* 40)
                    print(res['response_open'])
                    print('=1'* 40)
                    print(res['response_closed'])
                    print('=2'* 40)
                    print(s_res['response_open'])
                    print('=3'* 40)
                    print(s_res['response_closed'])

                    manual_label = input()
                    manual_labels.setdefault(cname, {})
                    manual_labels[cname][i] = manual_label 

                    count += 1
                    wrong_names.add(cname)
                    '''
                    ans = [manual_labels[cname][str(i)]]
            
            # 将得到的答案和响应保存到闭合结果中
            closed_results[cname][str(i)] = {'response': res, 'label': ans[0]}

            # 统计每个答案的出现次数
            count_options[ans[0]] = count_options.get(ans[0], 0) + 1
            # 将得分映射添加到分数列表中
            scores.append(ans_map[ans[0]])

        
        # 提交得分，获取最终预测
        pred = submit(scores)
        closed_results[cname]['label'] = pred 

        
        # 将闭合结果以 JSON 格式写入文件
        with open(save_name, 'a', encoding='utf-8') as f:
            json.dump(closed_results[cname], f, ensure_ascii=False)
            f.write('\n')
# 打印变量 count_options，这里假设在其他地方定义了这个变量
print(count_options)

# 存储处理后的结果字典
closed_results = {}

# 打开并读取 JSON Lines 格式的文件，每行包含一个 JSON 对象，编码为 UTF-8
with open('mbti_results_closed.jsonl', encoding='utf-8') as f:
    # 逐行读取文件内容
    for line in f:
        # 解析每行 JSON 数据
        data = json.loads(line)
        # 提取 JSON 对象中的 'character' 字段作为键
        cname = data['character']
        # 提取 JSON 对象中的 'label' 字段作为值，存入 closed_results 字典
        closed_results[cname] = data['label']

# 存储处理后的标签字典
labels = {}

# 打开并读取 JSON Lines 格式的文件，每行包含一个 JSON 对象，编码为 UTF-8
with open('mbti_labels.jsonl', encoding='utf-8') as f:
    # 逐行读取文件内容
    for line in f:
        # 解析每行 JSON 数据
        data = json.loads(line)
        # 提取 JSON 对象中的 'character' 字段作为键
        labels[data['character']] = data['label']

# 初始化单一维度评价的计数器和正确预测的计数器
count_single = 0
right_single = 0

# 初始化全部维度评价的计数器和正确预测的计数器
count_full = 0
right_full = 0

# 可能的 MBTI 类型字符集合
possible_chars = set(['E', 'I', 'S', 'N', 'T', 'F', 'P', 'J', 'X'])

# 遍历标签字典中的每个人物名称和对应的真实标签
for cname, gts in labels.items():
    # 获取预测结果标签列表
    pds = [_ for _ in closed_results[cname]]
    # 获取真实标签列表
    gts = [_ for _ in gts]
    
    # 初始化全维度预测正确的标志
    full_sign = True

    # 遍历预测结果和真实标签的对应位置
    for pd, gt in zip(pds, gts):
        # 断言预测结果和真实标签都在可能的 MBTI 类型字符集合中
        assert(pd in possible_chars and gt in possible_chars)
        # 如果真实标签为 'X'，跳过当前预测
        if gt == 'X':
            continue 
        else:
            # 如果预测结果与真实标签相同，增加单一维度正确预测的计数器
            if gt == pd:
                right_single += 1
            else:
                # 如果有一个维度预测错误，则全维度预测不正确
                full_sign = False
            # 增加单一维度预测的计数器
            count_single += 1

    # 如果全维度预测标志为真，则增加全维度预测正确的计数器
    if full_sign: 
        right_full += 1
    # 增加全维度评价的计数器
    count_full += 1

# 打印单一维度评价的统计信息：预测总数、正确预测数、准确率
print('单一维度评价：Count: {}\tRight: {}\tAcc: {:.4f}'.format(count_single, right_single, right_single/count_single))

# 打印全部维度评价的统计信息：预测总数、正确预测数、准确率
print('全部维度评价：Count: {}\tRight: {}\tAcc: {:.4f}'.format(count_full, right_full, right_full/count_full))    
```