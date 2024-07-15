# `.\Chat-Haruhi-Suzumiya\research\personality\code\api_16personality.py`

```py
import json  # 导入处理 JSON 数据的模块
import copy  # 导入复制对象的模块
import requests  # 导入发送 HTTP 请求的模块
import pdb  # 导入调试器模块

def judge_16(score_list):
    code = ''  # 初始化人格类型码为空字符串
    if score_list[0] >= 50:
        code = code + 'E'  # 如果第一个分数大于等于50，添加'E'到人格类型码中
    else:
        code = code + 'I'  # 否则添加'I'到人格类型码中

    if score_list[1] >= 50:
        code = code + 'N'  # 如果第二个分数大于等于50，添加'N'到人格类型码中
    else:
        code = code + 'S'  # 否则添加'S'到人格类型码中

    if score_list[2] >= 50:
        code = code + 'T'  # 如果第三个分数大于等于50，添加'T'到人格类型码中
    else:
        code = code + 'F'  # 否则添加'F'到人格类型码中

    if score_list[3] >= 50:
        code = code + 'J'  # 如果第四个分数大于等于50，添加'J'到人格类型码中
    else:
        code = code + 'P'  # 否则添加'P'到人格类型码中

    all_codes = ['ISTJ', 'ISTP', 'ISFJ', 'ISFP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ESTP', 'ESTJ', 'ESFP', 'ESFJ', 'ENFP', 'ENFJ', 'ENTP', 'ENTJ']
    all_roles = ['Logistician', 'Virtuoso', 'Defender', 'Adventurer', 'Advocate', 'Mediator', 'Architect', 'Logician', 'Entrepreneur', 'Executive', 'Entertainer',
                 'Consul', 'Campaigner', 'Protagonist', 'Debater', 'Commander']
    for i in range(len(all_codes)):
        if code == all_codes[i]:  # 查找匹配的人格类型码并获取其索引
            cnt = i
            break

    if score_list[4] >= 50:
        code = code + '-A'  # 如果第五个分数大于等于50，添加'-A'到人格类型码中
    else:
        code = code + '-T'  # 否则添加'-T'到人格类型码中

    return code, all_roles[cnt]  # 返回人格类型码和对应的角色名称

def submit_16personality_api(Answers):
    payload = copy.deepcopy(payload_template)  # 深拷贝初始负载模板

    for index, A in enumerate(Answers):
        payload['questions'][index]["answer"] = A  # 更新负载模板中的问题答案

    headers = {  # 定义 HTTP 请求头部信息
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

    session = requests.session()  # 创建持久化会话对象
    r = session.post('https://www.16personalities.com/test-results', data=json.dumps(payload), headers=headers)  # 发送 POST 请求

    a = r.headers['content-type']  # 获取响应头中的 content-type
    b = r.encoding  # 获取响应编码
    #c = r.json()  # 执行上面这行代码报错 为什么

    sess_r = session.get("https://www.16personalities.com/api/session")  # 发送 GET 请求获取会话信息

    scores = sess_r.json()['user']['scores']  # 解析响应中的用户分数信息

    ans1 = ''  # 初始化结果字符串
    session = requests.session()
    if sess_r.json()['user']['traits']['mind'] != 'Extraverted':  # 判断心理特质是否为内倾
        mind_value = 100 - (101 + scores[0]) // 2  # 计算内倾心理值
        ans1 += 'I'  # 添加'I'到结果字符串
    else:
        mind_value = (101 + scores[0]) // 2  # 计算外倾心理值
        ans1 += 'E'  # 添加'E'到结果字符串
    if sess_r.json()['user']['traits']['energy'] != 'Intuitive':  # 判断能量特质是否为直觉
        energy_value = 100 - (101 + scores[1]) // 2  # 计算感知能量值
        ans1 += 'S'  # 添加'S'到结果字符串
    else:
        energy_value = (101 + scores[1]) // 2  # 计算直觉能量值
        ans1 += 'N'  # 添加'N'到结果字符串
    # 检查用户特征中的性格属性是否为'Thinking'，如果不是则计算性格值和追加答案字母'F'，否则计算性格值和追加答案字母'T'
    if sess_r.json()['user']['traits']['nature'] != 'Thinking':
        nature_value = 100 - (101 + scores[2]) // 2
        ans1 += 'F'
    else:
        nature_value = (101 + scores[2]) // 2
        ans1 += 'T'

    # 检查用户特征中的战术属性是否为'Judging'，如果不是则计算战术值和追加答案字母'P'，否则计算战术值和追加答案字母'J'
    if sess_r.json()['user']['traits']['tactics'] != 'Judging':
        tactics_value = 100 - (101 + scores[3]) // 2
        ans1 += 'P'
    else:
        tactics_value = (101 + scores[3]) // 2
        ans1 += 'J'

    # 检查用户特征中的身份属性是否为'Assertive'，如果不是则计算身份值，否则计算身份值
    if sess_r.json()['user']['traits']['identity'] != 'Assertive':
        identity_value = 100 - (101 + scores[4]) // 2
    else:
        identity_value = (101 + scores[4]) // 2

    # 打印每个特征的结果及其分数（已注释掉的代码）

    # 调用judge_16函数，传入特征值数组，并获取返回的角色代码和角色
    code, role = judge_16([mind_value, energy_value, nature_value, tactics_value, identity_value])

    # 将生成的答案ans1的前四个字符与答案ans2进行断言比较
    ans2 = code[:4]
    assert(ans1 == ans2)

    # 返回一个包含每个性格特质结果和分数的字典
    return {
        "E/I": {"result": ans1[0], "score": {"E": mind_value, "I": 100 - mind_value}},
        "S/N": {"result": ans1[1], "score": {"S": 100 - energy_value, "N": energy_value}},
        "T/F": {"result": ans1[2], "score": {"T": nature_value, "F": 100 - nature_value}},
        "P/J": {"result": ans1[3], "score": {"P": 100 - tactics_value, "J": tactics_value}},
    }
```