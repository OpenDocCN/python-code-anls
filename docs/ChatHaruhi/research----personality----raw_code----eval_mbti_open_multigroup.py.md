# `.\Chat-Haruhi-Suzumiya\research\personality\raw_code\eval_mbti_open_multigroup.py`

```py
import json
import pdb
import argparse
import math

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--generate', action='store_true', default=False)
parser.add_argument('--split', action='store_true', default=False)
args = parser.parse_args()

# 定义模型名称
model = "gpt-4"

results = []

# 打开并读取 mbti_results.jsonl 文件，每行为一个 JSON 对象，包含 id、question、response_open 和 response_closed 四个字段，其中 response_open 是开放式回答，response_closed 是闭合式回答。
with open('mbti_results.jsonl', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        results.append(data)

# 定义角色名与其简称的对应关系字典
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

# 将结果按角色名进行划分
for idx, data in enumerate(results):
    cname = character_names[idx // 60]
    character_responses[cname].append(data)

# 生成保存文件名，格式为 mbti_results_open_multigroup_split=<args.split>_<model>.jsonl
save_name = 'mbti_results_open_multigroup_split={}_{}.jsonl'.format(args.split, model)

# 定义 MBTI 测试维度
dims = ['E/I', 'S/N', 'T/F', 'P/J']

def split_list(input_list):
    # 尝试将输入列表每四个元素分割为一个子列表
    result = [input_list[i:i+4] for i in range(0, len(input_list), 4)]
    
    # 检查最后一个子列表的长度是否小于3
    if len(result[-1]) < 3:
        # 从倒数第二个子列表中各取一个元素来补足最后一个子列表
        result[-1].append(result[-2].pop())
        # 如果补足后仍然长度小于3，则再从倒数第三个子列表中取一个元素补足
        if len(result[-1]) < 3:
            result[-1].append(result[-3].pop())

    # 断言所有子列表的长度在3到4之间
    assert(all([len(_) >= 3 and len(_) <= 4 for _ in result]))

    return result

# 如果指定了 --generate 参数
if args.generate:
    # 清空 save_name 文件内容
    with open(save_name, 'w', encoding='utf-8') as f:
        pass

    open_prompt_template = '''You are an expert in MBTI. I am conducting an MBTI test on someone. My goal is to gauge their position on the {} spectrum of the MBTI through a series of open-ended questions. For clarity, here's some background on differentiating this particular dimension:
    ===
    {}
    ===

    I've invited a participant, {}, and had the following conversations in Chinese:
    ===
    {}
    ===
    # 请帮助我区分 {} 在 MBTI 的 {} 维度中更倾向于哪个类别：{} 还是 {}。您应该提供每个类别的百分比，总和为 100%，例如 30% A 和 70% B。
    # 请按照以下 JSON 格式输出：
    # ===
    # {{
    #     "analysis": <基于对话的中文分析>,
    #     "result": {{ "{}": <百分比 1>, "{}": <百分比 2> }} (百分比 1 和百分比 2 的总和应为 100%。输出时不带百分号。)
    # }}
    
    open_dimension_prompt = {
        'E/I': '''E/I Dimension: Extraversion (E) vs Introversion (I)
    
    E (Extraversion): Extraverts draw energy from interacting with others. They feel comfortable in social settings and tend to express their thoughts. Extraverts are often more active, seek social stimulation, and enjoy participating in group activities. For them, connecting with people, sharing, and exchanging ideas is often a need. They might be more focused on external world stimuli, such as sounds, colors, and social dynamics.
    
    I (Introversion): Introverts feel more comfortable when alone. They derive energy from inner reflection and personal time. Contrary to extraverts, prolonged social interaction might tire them. Introverts might be more introspective, enjoy deep thinking, and tend to have meaningful personal relationships. They are more concerned with the inner world, such as thoughts, emotions, and imaginations.''',
    
        'S/N': '''S/N Dimension: Sensing (S) vs Intuition (N)
    
    S (Sensing): Sensing individuals value the concrete, practical, and present situations. They rely on their five senses to process information and often focus on details. For them, past experiences and tangible evidence play a significant role in decision-making. They are typically pragmatic and tend to deal with what they "see" and "hear".
    
    N (Intuition): Intuitive individuals tend to focus on potential possibilities and future opportunities. They like to think about "what could be", rather than just "what is". They lean more towards abstract thinking and can capture concepts and patterns effectively. Intuitives are often more innovative, preferring new ideas and approaches.''',
    
        'T/F': '''T/F Dimension: Thinking (T) vs Feeling (F)
    
    T (Thinking): Thinking individuals rely primarily on logic and analysis when making decisions. They pursue fairness and objectivity and might be more direct and frank. For them, finding the most efficient method or the most logical solution is crucial, even if it might hurt some people's feelings.
    
    F (Feeling): Feeling individuals consider people's emotions and needs more when making decisions. They strive for harmony, tend to build relationships, and avoid conflicts. They are often more empathetic, valuing personal values and emotions, rather than just facts or logic.''',
    
        'P/J': '''P/J Dimension: Perceiving (P) vs Judging (J)
    
    P (Perceiving): Perceiving individuals prefer to keep their options open and tend to be more flexible and spontaneous. They enjoy adapting to new situations and exploring different possibilities before making decisions.
    
    J (Judging): Judging individuals prefer structure and organization. They tend to plan ahead, make decisions quickly, and seek closure. They value order and predictability in their lives.''',
    }
    # 导入自定义模块 utils 中的 get_response 函数
    from utils import get_response 

    # 使用列表 character_names 中的每个名称创建一个字典，每个字典的键为 'character'，对应的值为名称本身
    open_results = {name:{'character': name} for name in character_names}
    for cname in character_names:
        responses = character_responses[cname]

        # 每个角色应该包含60个问题
        assert( len([r for r in responses if r['factor'] in dims]) == 60 )

        test_role = responses[0]['test_role']

        for dim in dims:
            dim_responses = [r for r in responses if r['factor'] == dim]
            
            # 如果设置了分割标志，将dim_responses分成多个子列表，每个列表3-4个元素
            if args.split:
                dim_responses_list = split_list(dim_responses)
            else:
                dim_responses_list = [dim_responses]

            # 初始化角色的开放性结果列表
            open_results[cname][dim] = [] 

            # 遍历分组后的dim_responses列表
            for group_responses in dim_responses_list:
                conversations = ''
                for i, r in enumerate(group_responses):
                    # 构建对话内容，包括问题和回答
                    conversations += f'{i+1}.\n'
                    conversations += f"{test_role}: 「{r['question']}」\n"
                    # 如果回答的开头不是角色名，则添加角色名
                    if not r['response_open'].startswith(cname):
                        r['response_open'] = cname + ': 「' + r['response_open'] + '」'
                    conversations += f"{r['response_open']}\n"
                
                # 将维度dim按照'/'分隔成t1和t2
                t1, t2 = dim.split('/')
                # 使用模板生成提示文本
                prompt = open_prompt_template.format(dim, open_dimension_prompt[dim], cname, conversations, cname, t1, t2, dim, t1, t2)

                # 将提示文本分割为系统提示和用户输入
                sys_prompt, user_input = prompt.split("I've invited a participant")
                user_input = "I've invited a participant" + user_input

                # 调用模型获取语言生成模型的响应
                llm_response = get_response(sys_prompt, user_input, model=model)
                # 将语言生成模型的响应转为JSON格式
                llm_response = json.loads(llm_response)

                # 尝试将响应结果中的值转为整数，并确保总和为100
                try:
                    llm_response['result'] = {k: int(float(v)) for k, v in llm_response['result'].items()}
                    assert (sum(llm_response['result'].values()) == 100)
                except:
                    # 如果转换或断言失败，则输出错误信息并启动调试器
                    print('Wrong result : ', llm_response)
                    pdb.set_trace()

                # 将处理后的结果添加到角色的开放性结果列表中
                open_results[cname][dim].append({'group_responses': group_responses, 'results': llm_response})

        # 将每个角色的开放性结果写入JSON文件
        with open(save_name, 'a', encoding='utf-8') as f:
            json.dump(open_results[cname], f, ensure_ascii=False)
            f.write('\n')
# 读取标签文件 'mbti_labels.jsonl'，将每行的JSON数据加载为字典并存储在 labels 字典中
labels = {}
with open('mbti_labels.jsonl', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        labels[data['character']] = data['label']

# 从文件中读取保存的结果，将每行的JSON数据加载为字典并存储在 open_results 字典中
open_results = {}
with open(save_name, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        open_results[data['character']] = data

# 初始化计数器和正确率统计变量
count_single = 0    # 单一维度评价总数计数器
right_single = 0    # 单一维度评价正确预测数计数器
count_full = 0      # 全部维度评价总数计数器
right_full = 0       # 全部维度评价正确预测数计数器

# 可能的性格维度组合
possible_chars = set(['E', 'I', 'S', 'N', 'T', 'F', 'P', 'J', 'X'])

# 遍历标签字典，cname为角色名，gts为对应的性格标签列表
for cname, gts in labels.items():
    # 预测结果列表
    pds = []
    var_list = []
    
    # 遍历维度列表 dims
    for dim in dims:
        # 分割维度为两个类别
        dim_cls1, dim_cls2 = dim.split('/')
        
        # 初始化存储所有分数的字典
        all_score = {d: [] for d in [dim_cls1, dim_cls2]}
        
        # 获取当前角色的分组数
        count_group = len(open_results[cname][dim])
        
        # 遍历当前角色在当前维度下的结果
        for dim_res in open_results[cname][dim]:
            score = dim_res['results']['result']
            for d in [dim_cls1, dim_cls2]:
                all_score[d].append(score[d])
        
        # 计算两个类别的平均分数
        avg_score = {d: sum(all_score[d]) / count_group for d in [dim_cls1, dim_cls2]}
        
        # 如果分组数大于1，计算标准差
        if count_group > 1:
            var_score = {d: math.sqrt(sum([(s - avg_score[d])**2 for s in all_score[d]]) / (count_group - 1)) for d in [dim_cls1, dim_cls2]}
            var_list.append(var_score[dim_cls1])
        
        # 断言两个类别的平均分数之和为100
        assert sum(avg_score.values()) == 100
        
        # 预测角色在当前维度下的性格类别，选择平均分数较高的类别作为预测结果
        pred = max(avg_score, key=avg_score.get)
        pds.append(pred)
    
    # 打印角色名称、预测结果、和实际标签
    print('Character {}\t\tResults {}\tLabels {}'.format(cname, ''.join(pds), ''.join(gts)))
    
    # 初始化全维度正确标志为True
    full_sign = True
    
    # 遍历预测结果和实际标签，计算单一维度评价的正确预测数和总数
    for pd, gt in zip(pds, gts):
        assert pd in possible_chars and gt in possible_chars
        if gt == 'X':
            continue
        else:
            if gt == pd:
                right_single += 1
            else:
                full_sign = False
            count_single += 1
    
    # 如果全维度评价标志为True，则增加正确全维度评价的计数
    if full_sign:
        right_full += 1
    count_full += 1

# 打印单一维度评价和全部维度评价的统计结果
print('单一维度评价：Count: {}\tRight: {}\tAcc: {:.4f}'.format(count_single, right_single, right_single / count_single))
print('全部维度评价：Count: {}\tRight: {}\tAcc: {:.4f}'.format(count_full, right_full, right_full / count_full))

# 如果分组数大于1，打印平均方差的计算结果
if count_group > 1:
    print('平均方差Var\t {:.4f}'.format(sum(var_list) / len(var_list)))

# 执行脚本的命令提示
# python eval_mbti_open_multigroup.py --generate --split
```