# `.\Chat-Haruhi-Suzumiya\research\personality\raw_code\eval_mbti_open.py`

```py
import json
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--generate', action='store_true', default=False)
args = parser.parse_args()

results = []
# 读取mbti_results.jsonl文件，里面每行是一个json，包含了id，question，response_open，response_closed四个字段，其中response_open是开放式回答，response_closed是闭合式回答。
with open('mbti_results.jsonl', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        results.append(data)

# mbti_results.jsonl里忘了存character_name了..只能按idx划分了
NAME_DICT = {'汤师爷': 'tangshiye', '慕容复': 'murongfu', '李云龙': 'liyunlong', 'Luna': 'Luna', '王多鱼': 'wangduoyu',
             'Ron': 'Ron', '鸠摩智': 'jiumozhi', 'Snape': 'Snape',
             '凉宫春日': 'haruhi', 'Malfoy': 'Malfoy', '虚竹': 'xuzhu', '萧峰': 'xiaofeng', '段誉': 'duanyu',
             'Hermione': 'Hermione', 'Dumbledore': 'Dumbledore', '王语嫣': 'wangyuyan',
             'Harry': 'Harry', 'McGonagall': 'McGonagall', '白展堂': 'baizhantang', '佟湘玉': 'tongxiangyu',
             '郭芙蓉': 'guofurong', '旅行者': 'wanderer', '钟离': 'zhongli',
             '胡桃': 'hutao', 'Sheldon': 'Sheldon', 'Raj': 'Raj', 'Penny': 'Penny', '韦小宝': 'weixiaobao',
             '乔峰': 'qiaofeng', '神里绫华': 'ayaka', '雷电将军': 'raidenShogun', '于谦': 'yuqian'}

character_names = list(NAME_DICT.keys())
character_responses = {name:[] for name in character_names}

# 将results按照角色划分
for idx, data in enumerate(results):
    cname = character_names[ idx // 60 ]
    character_responses[cname].append(data)

save_name = 'mbti_results_open.jsonl'
# 开放式测试 - 通过GPT-4评价
dims = ['E/I', 'S/N', 'T/F', 'P/J']

if args.generate:
    open_prompt_template = '''You are an expert in MBTI. I am conducting an MBTI test on someone. My goal is to gauge their position on the {} spectrum of the MBTI through a series of open-ended questions. For clarity, here's some background on differentiating this particular dimension:
    ===
    {}
    ===

    I've invited a participant, {}, and had the following conversations in Chinese:
    ===
    {}
    ===

    Please help me distinguish whether {} leans more towards the {} or {} category within the MBTI's {} dimension. Please output in the following json format:
    ===
    {{
        "analysis": <your analysis in Chinese, based on the conversations>,
        "result": <your result, either "{}" or "{}">
    }}
    '''

    open_dimension_prompt = {
        'E/I': '''E/I Dimension: Extraversion (E) vs Introversion (I)

    E (Extraversion): Extraverts draw energy from interacting with others. They feel comfortable in social settings and tend to express their thoughts. Extraverts are often more active, seek social stimulation, and enjoy participating in group activities. For them, connecting with people, sharing, and exchanging ideas is often a need. They might be more focused on external world stimuli, such as sounds, colors, and social dynamics.

    I (Introversion): Introverts recharge by spending time alone or in quiet environments. They typically prefer solitary activities and think deeply before speaking. Introverts often have a smaller circle of close friends and enjoy meaningful one-on-one conversations. They may be more attuned to inner thoughts and feelings, and prefer fewer but deeper interactions.

    ''',
        'S/N': '''S/N Dimension: Sensing (S) vs Intuition (N)

    S (Sensing): Sensors focus on concrete information gathered through the five senses. They are practical, detail-oriented, and rely on factual data to make decisions. Sensors are often present-focused and attentive to their immediate surroundings.

    N (Intuition): Intuitives are imaginative and focus on interpreting meanings, patterns, and possibilities. They are future-oriented, abstract thinkers who enjoy contemplating theories and exploring new ideas. Intuitives may be less interested in specifics and more concerned with underlying concepts and potentials.

    ''',
        'T/F': '''T/F Dimension: Thinking (T) vs Feeling (F)

    T (Thinking): Thinkers prioritize logical analysis and objective criteria when making decisions. They value fairness and consistency, and tend to focus on tasks or problems rather than people's emotions. Thinkers strive for rationality and may appear detached when dealing with sensitive issues.

    F (Feeling): Feelers prioritize personal values and emotional aspects when making decisions. They emphasize empathy, harmony, and understanding others' perspectives. Feelers are often concerned with maintaining positive relationships and making choices that align with their values and feelings.

    ''',
        'P/J': '''P/J Dimension: Perceiving (P) vs Judging (J)

    P (Perceiving): Perceivers are adaptable and open-ended in their approach to life. They prefer flexibility and spontaneity, often keeping their options open and delaying decisions until necessary. Perceivers enjoy exploring possibilities and may struggle with structured routines or strict plans.

    J (Judging): Judgers are structured and decisive in their approach to life. They prefer order and planning, and feel more comfortable when decisions are made. Judgers value closure and seek to organize their surroundings, often following schedules and deadlines.

    '''
    }

# 这里是开放式测试维度的提示信息，提供了四种MBTI维度的背景描述和特点对比。
    # 定义一个包含多个人格特质说明的字典
    personality_traits = {
        'I/E': '''I/E 维度：内向 (I) vs 外向 (E)
    
        I (Introversion): 内向的人更喜欢独处。他们通过内省和个人时间获取能量。与外向者相反，长时间的社交互动可能使他们感到疲惫。内向的人可能更喜欢自省，享受深思，倾向于建立有意义的个人关系。他们更关注内心世界，如思想、情感和想象力。''',
    
        'S/N': '''S/N 维度：感觉 (S) vs 直觉 (N)
    
        S (Sensing): 感觉型个体重视具体、实际和当前的情况。他们依赖五官处理信息，通常关注细节。对他们来说，过去的经验和有形的证据在决策中扮演重要角色。他们通常务实，处理他们所“看到”和“听到”的内容。
    
        N (Intuition): 直觉型个体倾向于关注潜在的可能性和未来的机会。他们喜欢思考“可能是什么”，而不仅仅是“现在是什么”。他们更倾向于抽象思维，能有效捕捉概念和模式。直觉型个体通常更具创新性，偏好新的想法和方法。''',
    
        'T/F': '''T/F 维度：思维 (T) vs 感觉 (F)
    
        T (Thinking): 思维型个体在做决策时主要依赖逻辑和分析。他们追求公平和客观性，可能更直接和坦率。对他们来说，找到最有效的方法或最合乎逻辑的解决方案至关重要，即使这可能会伤害某些人的感情。
    
        F (Feeling): 感觉型个体在做决策时更考虑他人的情绪和需求。他们追求和谐，倾向于建立关系，避免冲突。他们通常更具移情能力，重视个人价值观和情感，而不仅仅是事实或逻辑。''',
    
        'P/J': '''P/J 维度：感知 (P) vs 判断 (J)
    
        P (Perceiving): 感知型个体更开放和灵活。他们倾向于“随波逐流”，而不是过度计划或组织事物。感知者喜欢探索各种可能性，更倾向于留下选项以应对意外情况。他们倾向于推迟决策，以收集更多信息和更好的理解。对他们来说，生活是一个不断变化的过程，而不是一个具有固定目标或计划的事件。他们通常更关注经历本身，而不仅仅是结果。
    
        J (Judging): 判断型个体在生活中更结构化和计划性。他们偏爱明确的期望和结构，通常设定目标并追求之。判断者通常更有条理，倾向于提前做决策。他们喜欢按计划行事，对有序的环境感到舒适。对他们来说，实现目标和完成任务通常是优先考虑的。他们可能更关注效率和结构，而不是开放性或自发性。'''
    }
    
    # 从utils模块中导入get_response函数
    from utils import get_response
    # 使用角色名创建一个字典，每个角色都包含一个 'character' 键值对
    open_results = {name:{'character': name} for name in character_names}

    # 遍历每个角色名
    for cname in character_names:
        # 获取特定角色的回答列表
        responses = character_responses[cname]

        # 确保每个角色的回答列表中包含60个问题
        assert( len([r for r in responses if r['factor'] in dims]) == 60 )

        # 获取第一个回答的测试角色
        test_role = responses[0]['test_role']

        # 遍历指定的维度列表
        for dim in dims:
            # 筛选出特定维度的回答列表
            dim_responses = [r for r in responses if r['factor'] == dim]

            conversations = ''
            # 枚举每个维度回答列表中的元素
            for i, r in enumerate(dim_responses):
                # 构建对话串，包括问题和回答
                conversations += f'{i+1}.\n'
                conversations += f"{test_role}: 「{r['question']}」\n"
                # 如果回答开头不是角色名，则添加角色名
                if not r['response_open'].startswith(cname):
                    r['response_open'] = cname + ': 「' + r['response_open'] + '」'
                conversations += f"{r['response_open']}\n"
            
            # 将维度名称拆分为两部分
            t1, t2 = dim.split('/')
            # 使用模板生成提示信息
            prompt = open_prompt_template.format(dim, open_dimension_prompt[dim], cname, conversations, cname, t1, t2, dim, t1, t2)

            # 分割提示信息，获取系统提示和用户输入部分
            sys_prompt, user_input = prompt.split("I've invited a participant")
            user_input = "I've invited a participant" + user_input

            # 使用 GPT-4 模型获取响应
            llm_response = get_response(sys_prompt, user_input, model="gpt-4")
            # 将 GPT-4 模型的响应转换为 JSON 格式
            llm_response = json.loads(llm_response)

            # 将 GPT-4 模型的响应存入结果字典中对应的维度下
            open_results[cname][dim] = llm_response

        # 将每个角色的结果字典写入文件
        with open(save_name, 'a', encoding= 'utf-8') as f:
            json.dump(open_results[cname], f, ensure_ascii=False)
            f.write('\n')
# 读取标签
labels = {}
with open('mbti_labels.jsonl', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        labels[data['character']] = data['label']

# 读取open_results
open_results = {}
with open('mbti_results_open.jsonl', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        open_results[data['character']] = data

# single: 单一维度评价；full：全维度评价
count_single = 0
right_single = 0 
count_full = 0
right_full = 0

possible_chars = set(['E', 'I', 'S', 'N', 'T', 'F', 'P', 'J', 'X'])

for cname, gts in labels.items():
    # 预测结果
    pds = [open_results[cname][dim]['result'] for dim in dims]
    # groundtruth
    gts = [_ for _ in gts]

    print('Character {}\t\tResults {}\tLabels {}'.format(cname, ''.join(pds), ''.join(gts)))

    full_sign = True 

    for pd, gt in zip(pds, gts):
        assert(pd in possible_chars and gt in possible_chars)
        if gt == 'X':
            continue 
        else:
            if gt == pd:
                right_single += 1
            else:
                full_sign = False
            count_single += 1

    if full_sign: 
        right_full += 1
    count_full += 1

print('单一维度评价：Count: {}\tRight: {}\tAcc: {:.4f}'.format(count_single, right_single, right_single/count_single))
print('全部维度评价：Count: {}\tRight: {}\tAcc: {:.4f}'.format(count_full, right_full, right_full/count_full))    
```