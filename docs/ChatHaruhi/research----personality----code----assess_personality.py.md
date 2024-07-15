# `.\Chat-Haruhi-Suzumiya\research\personality\code\assess_personality.py`

```py
# 导入所需模块和库
from tqdm import tqdm  # 进度条显示库
import json  # JSON 格式数据处理库
import os  # 系统操作库
import openai  # OpenAI API 访问库
import zipfile  # ZIP 文件处理库
import argparse  # 命令行参数解析库
import pdb  # Python 调试器库
import random  # 随机数生成库
import prompts  # 指令集模块
import math  # 数学函数库
from utils import logger  # 自定义日志模块

random.seed(42)  # 设置随机数种子，保证随机数的可重复性

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Assess personality of a character')

# 为问卷类型参数添加选项
parser.add_argument('--questionnaire_type', type=str, default='mbti', 
                    choices=['bigfive', 'mbti'], 
                    help='questionnaire to use (bigfive or mbti)')

# 为角色名称参数添加选项
parser.add_argument('--character', type=str, default='haruhi', help='character name or code')

# 为语言模型代理参数添加选项
parser.add_argument('--agent_llm', type=str, default='gpt-3.5-turbo', 
                    choices=['gpt-3.5-turbo', 'openai', 'GLMPro', 'ChatGLM2GPT'], 
                    help='agent LLM (gpt-3.5-turbo)')

# 为评估器参数添加选项
parser.add_argument('--evaluator', type=str, default='gpt-3.5-turbo', 
                    choices=['api', 'gpt-3.5-turbo', 'gpt-4'], 
                    help='evaluator (api, gpt-3.5-turbo or gpt-4)')

# 为评估设置参数添加选项
parser.add_argument('--eval_setting', type=str, default='batch', 
                    choices=['batch', 'collective', 'sample'], 
                    help='setting (batch, collective, sample)')

# 为语言参数添加选项
parser.add_argument('--language', type=str, default='cn', 
                    choices=['cn', 'en'], 
                    help='language, temporarily only support Chinese (cn)')

# 解析命令行参数
args = parser.parse_args()

# 打印解析得到的参数信息
print(args)

# 角色名称映射字典，将角色名映射到统一的英文代码
NAME_DICT = {'汤师爷': 'tangshiye', '慕容复': 'murongfu', '李云龙': 'liyunlong', 'Luna': 'Luna', '王多鱼': 'wangduoyu',
             'Ron': 'Ron', '鸠摩智': 'jiumozhi', 'Snape': 'Snape',
             '凉宫春日': 'haruhi', 'Malfoy': 'Malfoy', '虚竹': 'xuzhu', '萧峰': 'xiaofeng', '段誉': 'duanyu',
             'Hermione': 'Hermione', 'Dumbledore': 'Dumbledore', '王语嫣': 'wangyuyan',
             'Harry': 'Harry', 'McGonagall': 'McGonagall', '白展堂': 'baizhantang', '佟湘玉': 'tongxiangyu',
             '郭芙蓉': 'guofurong', '旅行者': 'wanderer', '钟离': 'zhongli',
             '胡桃': 'hutao', 'Sheldon': 'Sheldon', 'Raj': 'Raj', 'Penny': 'Penny', '韦小宝': 'weixiaobao',
             '乔峰': 'qiaofeng', '神里绫华': 'ayaka', '雷电将军': 'raidenShogun', '于谦': 'yuqian'}

# 问卷类型对应的维度字典，包含 MBTI 和 BigFive 两种问卷的维度
dims_dict = {'mbti': ['E/I', 'S/N', 'T/F', 'P/J'], 'bigfive': ['openness', 'extraversion', 'conscientiousness', 'agreeableness', 'neuroticism']}

# 读取 MBTI 问卷的真实标签信息
mbti_labels = {}
with open(os.path.join('..', 'data', 'mbti_labels.jsonl'), encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        mbti_labels[data['character']] = data['label']

# 读取配置文件 config.json 的内容
with open('config.json', 'r') as f:
    config = json.load(f)

# 定义函数，用于加载指定名称的问卷数据
def load_questionnaire(questionnaire_name):
    q_path = os.path.join('..', 'data', f'{questionnaire_name}_questionnaire.jsonl')

    # 读取指定的 JSONL 文件，用于加载问卷数据
    # 使用 'r' 模式打开文件 q_path，并赋值给变量 f，使用 with 语句确保文件操作完成后自动关闭文件
    with open(q_path, 'r') as f:
        # 逐行读取文件内容，并将每行内容使用 json.loads() 转换为 Python 对象，存入列表 questionnaire
        questionnaire = [json.loads(line) for line in f]
    # 返回读取并解析后的问卷数据列表 questionnaire
    return questionnaire
# 将问卷根据 'dimension' 进行分组抽样
def subsample_questionnaire(questionnaire, n=20):
    # 定义内部函数 subsample，用于按指定键从 questions 中抽样 n 个问题
    def subsample(questions, key, n):
        # 获取所有不重复的键值
        key_values = list(set([q[key] for q in questions]))
        n_keys = len(key_values)
        # 计算每个键应抽取的基本数量
        base_per_key = n // n_keys
        remaining = n % n_keys

        # 从键值中随机选择一部分需要额外抽样的键
        keys_w_additional_question = random.sample(key_values, remaining)
        subsampled_questions = []

        for key_value in key_values:
            # 筛选当前键值对应的问题
            filtered_questions = [q for q in questions if q[key] == key_value]

            # 确定这个键值对应的样本数
            num_samples = base_per_key + 1 if key_value in keys_w_additional_question else base_per_key

            # 如果这个键值对应的问题不足，调整样本数
            num_samples = min(num_samples, len(filtered_questions))
            subsampled_questions += random.sample(filtered_questions, num_samples)
            n -= num_samples

        # 处理稀有情况：如果抽样后仍然不足 n 个问题，从剩余问题中额外抽样
        remaining_questions = [q for q in questions if q not in subsampled_questions]
        if n > 0 and len(remaining_questions) >= n:
            subsampled_questions += random.sample(remaining_questions, n)

        return subsampled_questions

    # 根据问卷第一个问题是否包含 'sub_dimension' 字段来判断是属于 bigfive 还是 mbti 类型问卷
    if 'sub_dimension' in questionnaire[0].keys(): # bigfive
        dimension_questions = {} 
        for q in questionnaire:
            # 将问题按 'dimension' 分类存放到 dimension_questions 字典中
            if q['dimension'] not in dimension_questions.keys():
                dimension_questions[q['dimension']] = []
            
            dimension_questions[q['dimension']].append(q)
        
        new_questionnaire = []
        for dim, dim_questions in dimension_questions.items():
            # 对每个维度的问题进行抽样，以 'sub_dimension' 为键进行抽样
            new_questionnaire += subsample(dim_questions, 'sub_dimension', n//len(dimension_questions.keys()))

    else: # mbti
        # 如果是 mbti 类型问卷，直接按 'dimension' 抽样
        new_questionnaire = subsample(questionnaire, 'dimension', n)
    
    return new_questionnaire

# 将输入列表分割成每个子列表有 n 个元素的形式
def split_list(input_list, n=4):
    # 使用列表推导式将列表分割成每个子列表包含 n 个元素
    result = [input_list[i:i+n] for i in range(0, len(input_list), n)]
    
    # 检查最后一个子列表的长度
    num_to_pop = n - 1 - len(result[-1])
    for i in range(num_to_pop):
        result[-1].append(result[i].pop())

    # 断言：确保每个子列表的长度在 3 到 n 之间
    assert( all([len(_) >= n-1 and len(_) <= n for _ in result]) )
    
    return result

# 根据角色代码和 agent_llm 构建角色代理
def build_character_agent(character_code, agent_llm):
    from chatharuhi import ChatHaruhi
    haruhi_path = './content/Haruhi-2-Dev' 

    '''
    # 你也可以通过以下方式下载 zip 文件

    zip_file_path = f"{haruhi_path}/data/character_in_zip/{ai_role_en}.zip"
    '''
    # 检查指定的 ZIP 文件路径是否存在
    if not os.path.exists(zip_file_path):
        # 如果不存在，抛出文件未找到异常，并指定文件路径
        raise FileNotFoundError(f"zip file {zip_file_path} not found")
        
    # 指定解压缩后的目标文件夹路径，格式为 "characters/{ai_role_en}"
    destination_folder = f"characters/{ai_role_en}"

    # 使用 zipfile 模块打开指定路径的 ZIP 文件，并解压缩到目标文件夹中
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

    # 构建角色数据库文件夹路径和系统提示文件路径
    db_folder = f"{haruhi_path}/characters/{ai_role_en}/content/{ai_role_en}"
    system_prompt = f"{haruhi_path}/characters/{ai_role_en}/content/system_prompt.txt"
    #print(db_folder, system_prompt)

    # 创建 ChatHaruhi 对象，指定系统提示文件、LLM 模型和故事数据库文件夹，并启用详细输出
    character_agent = ChatHaruhi(system_prompt=system_prompt,
                        llm="openai",
                        story_db=db_folder,
                        verbose=True)
    
    '''
    如果 agent_llm 的值为 'gpt-3.5-turbo'，将其设为 'openai'
    '''

    # 设置 API 密钥，仅当 agent_llm 为 'openai' 时生效
    if agent_llm == 'openai':
        os.environ["OPENAI_API_KEY"] = config['openai_apikey']

    # 创建 ChatHaruhi 对象，设置角色名称和LLM模型
    character_agent = ChatHaruhi(role_name=character_code, llm=agent_llm)
    # 设置聊天时的温度参数为 0
    character_agent.llm.chat.temperature = 0 

    # 返回初始化后的 character_agent 对象
    return character_agent
# 根据角色名获取实验者姓名的字典映射
def get_experimenter(character_name):    
    # 定义角色名到实验者姓名的映射关系
    EXPERIMENTER_DICT = {'汤师爷': '张牧之', '慕容复': '王语嫣', \
          '李云龙': '赵刚', 'Luna': 'Harry', '王多鱼': '夏竹',
          'Ron': 'Hermione', '鸠摩智': '慕容复', 'Snape': 'Dumbledore',
             '凉宫春日': '阿虚', 'Malfoy': 'Crabbe', \
          '虚竹': '乔峰', '萧峰': '阿朱', '段誉': '乔峰',\
             'Hermione': 'Harry', 'Dumbledore': 'McGonagall', '王语嫣': '段誉',\
             'Harry': 'Hermione', 'McGonagall': 'Dumbledore', '白展堂': '佟湘玉',\
           '佟湘玉': '白展堂',
             '郭芙蓉': '白展堂', '旅行者': '派蒙', '钟离': '旅行者',
             '胡桃': '旅行者', 'Sheldon': 'Leonard', 'Raj': 'Leonard', 'Penny': 'Leonard', \
          '韦小宝': '双儿',
             '乔峰': '阿朱', '神里绫华': '旅行者', '雷电将军': '旅行者', '于谦': '郭德纲'}
    
    return EXPERIMENTER_DICT[character_name]

# 执行问卷调查
def interview(character_agent, questionnaire, experimenter, language, evaluator):
    # 存储结果列表
    results = []

    # 遍历问卷中的每一个问题
    for question in tqdm(questionnaire):
        # 获取当前语言下的问题内容
        q = question[f'question_{language}']
        # 清空角色代理的对话历史记录
        character_agent.dialogue_history = []

        # 与实验者角色进行对话，获取开放式回答
        open_response = character_agent.chat(role=experimenter, text=q)

        # 构建当前问题的结果字典
        result = {
            'id': question['id'],
            'question': q,
            'response_open': open_response,
            'dimension': question['dimension'],
        }

        '''
        if evaluator == 'api':
            # 如果评估者为 'api'，提供闭合式选项
            close_prompt_template = prompts.close_prompt_template
            close_prompt = close_prompt_template.format(q)
            close_response = character_agent.chat(role=experimenter, text=close_prompt)
            result['response_close'] = close_response
        '''

        # 将当前问题结果加入总结果列表
        results.append(result)

    # 返回所有问题的调查结果
    return results

# 评估问卷调查结果
def assess(character_name, experimenter, questionnaire_results, questionnaire_type, evaluator, eval_setting):
    # 获取问卷类型对应的维度字典
    dims = dims_dict[questionnaire_type]
    
    # 导入获取响应的工具函数
    from utils import get_response 
    
    # 初始化评估结果字典
    assessment_results = {}
    # 待完成：
    # 1. 支持评估者为 'api'
    # 2. 支持问卷类型为 'bigfive'
    elif evaluator == 'api':
        # 如果评估器为 'api'，则以下代码段处理基于 API 的评估
        # 注意：API 仅支持 MBTI 评估类型，不支持 BigFive

        # 断言确保问卷类型为 'mbti'
        assert(questionnaire_type == 'mbti')

        # 定义选项列表，用于映射答案到数值
        options = ['fully agree', 'generally agree', 'partially agree', 'neither agree nor disagree', 'partially disagree', 'generally disagree', 'fully disagree']
        ans_map = { option: i-3 for i, option in enumerate(options)} 

        # 初始化答案列表
        answers = []

        # 遍历问卷结果中的每个问题及其回答
        for i, response in enumerate(questionnaire_results):
            # 构建系统提示信息
            sys_prompt = prompts.to_option_prompt_template.format(character_name, experimenter)

            # 初始化对话字符串
            conversations = ''
            # 添加实验者提出的问题到对话中
            conversations += f"{experimenter}: 「{response['question']}」\n"
            
            # 处理用户回答
            # 如果回答不以角色名开头，则添加角色名并重新格式化回答
            if not response['response_open'].startswith(character_name):
                response['response_open'] = character_name + ': 「' + response['response_open'] + '」'
            # 添加用户回答到对话中
            conversations += f"{response['response_open']}\n"
            
            # 获取用户输入
            user_input = conversations

            # 调用函数获取长短期记忆模型（LLM）的响应
            llm_response = get_response(sys_prompt, user_input, model="gpt-3.5-turbo").strip('=\n')
            # 解析 LLN 响应为 JSON 格式
            llm_response = json.loads(llm_response)

            # 提取模型结果作为答案
            answer = llm_response['result']

            # 将答案映射为数值并添加到答案列表
            answers.append(ans_map[answer])

        # 导入并调用 16personality API 的提交函数
        from api_16personality import submit_16personality_api
        pred = submit_16personality_api(answers)
        
        # 将 API 返回的评估结果存储在 assessment_results 变量中
        assessment_results = pred
    
    # 返回评估结果
    return assessment_results
# 评估人物个性特征的函数
def personality_assessment(character, agent_llm, questionnaire_type, eval_setting, evaluator, language):
    # character_name: 人物的中文名字
    # character_code: 人物的英文名字
    if character in NAME_DICT.keys():
        # 如果人物名在NAME_DICT的键中，则设置人物名为输入的名字
        character_name = character
        # 设置人物编码为NAME_DICT中对应的值
        character_code = NAME_DICT[character]
    elif character in NAME_DICT.values():
        # 如果人物名在NAME_DICT的值中，则设置人物编码为输入的名字
        character_code = character
        # 设置人物名为NAME_DICT中匹配到的键
        character_name = [k for k, v in NAME_DICT.items() if v == character][0]
    else:
        # 如果人物名不在NAME_DICT中，则引发值错误并显示NAME_DICT的所有项
        raise ValueError(f"Character '{character}' not found in NAME_DICT. Here are the items: {list(NAME_DICT.items())}")
    
    # 载入问卷
    if questionnaire_type in ['bigfive', 'mbti']:
        # 如果问卷类型为'bigfive'或'mbti'，则加载对应的问卷
        questionnaire = load_questionnaire(questionnaire_type)
    else:
        # 如果问卷类型不在支持范围内，则抛出未实现错误
        raise NotImplementedError
    
    if eval_setting == 'sample':
        # 如果评估设置为'sample'，则对问卷进行子采样处理
        questionnaire = subsample_questionnaire(questionnaire)

    # 构建人物代理
    character_agent = build_character_agent(character_code, agent_llm) 
    logger.info(f'Character agent created for {character_name}')

    # 获取实验者
    experimenter = get_experimenter(character_name)
    
    # 根据问卷对人物进行面试
    interview_folder_path = os.path.join('..', 'results', 'interview')
    if not os.path.exists(interview_folder_path):
        os.makedirs(interview_folder_path)

    interview_save_path = f'{character_name}_agent-llm={agent_llm}_{questionnaire_type}_sample={eval_setting=="sample"}_{language}_interview.json'
   
    interview_save_path = os.path.join(interview_folder_path, interview_save_path)
    
    if not os.path.exists(interview_save_path):
        # 如果面试保存路径不存在，则进行面试过程
        logger.info('Interviewing...')
        questionnaire_results = interview(character_agent, questionnaire, experimenter, language, evaluator)
        with open(interview_save_path, 'w') as f:
            # 将问卷结果以JSON格式保存到文件中
            json.dump(questionnaire_results, f, indent=4, ensure_ascii=False)
        logger.info(f'Interview finished... save into {interview_save_path}')
    else:
        # 如果面试保存路径已存在，则直接从文件中加载问卷结果
        logger.info(f'Interview done before. load directly from {interview_save_path}')
        with open(interview_save_path, 'r') as f:
            questionnaire_results = json.load(f)

    # 评估人物的个性特征
    assessment_folder_path = os.path.join('..', 'results', 'assessment')
    if not os.path.exists(assessment_folder_path):
        os.makedirs(assessment_folder_path)

    assessment_save_path = f'{character_name}_agent-llm={agent_llm}_{questionnaire_type}_eval={eval_setting}-{evaluator}_{language}_interview.json'
   
    assessment_save_path = os.path.join(assessment_folder_path, assessment_save_path)
    # 检查评估结果保存路径是否存在，如果不存在则进行评估并保存结果
    if not os.path.exists(assessment_save_path):
        # 记录信息：正在进行评估
        logger.info('Assessing...')
        # 调用评估函数，获取评估结果
        assessment_results = assess(character_name, experimenter, questionnaire_results, questionnaire_type, evaluator, eval_setting)
        # 将评估结果以 JSON 格式写入文件
        with open(assessment_save_path, 'w') as f:
            json.dump(assessment_results, f, indent=4, ensure_ascii=False)
        # 记录信息：评估完成，并指出保存路径
        logger.info(f'Assess finished... save into {assessment_save_path}')
    else:
        # 记录信息：之前已完成评估，直接从文件加载结果
        logger.info(f'Assess done before. load directly from {assessment_save_path}')
        # 从文件中读取评估结果
        with open(assessment_save_path, 'r') as f:
            assessment_results = json.load(f)

    # 展示人格评估结果
    if questionnaire_type == 'mbti':
        # 记录信息：MBTI 评估结果
        logger.info('MBTI assessment results:')
        # 记录信息：评估的角色名
        logger.info('Character: ' + character_name)
        # 获取预测的 MBTI 代码和实际标签的代码
        pred_code = ''.join([ assessment_results[dim]['result'] for dim in dims_dict['mbti']])
        label_code = mbti_labels[character_name]
        # 记录信息：预测代码和实际代码
        logger.info(f'Prediction {pred_code}\tGroundtruth {label_code}')

        # 遍历评估结果的维度和对应的结果
        for dim, result in assessment_results.items():
            # 如果结果中包含分数，记录维度和分数
            if "score" in result:
                logger.info(f'{dim}: {result["score"]}')
            # 如果结果中包含标准差，并且不为 None，则记录维度和标准差（保留两位小数）
            if "standard_variance" in result and result["standard_variance"] is not None:
                logger.info(f'{dim}: {result["standard_variance"]:.2f}')
            # 如果结果中包含批次结果，记录第一个批次的分析信息
            if "batch_results" in result:
                logger.info(f'{result["batch_results"][0]["analysis"]}')
    
    else:
        # 记录信息：大五人格评估结果
        logger.info('Big Five assessment results:')
        # 记录信息：评估的角色名
        logger.info('Character: ' + character_name)

        # 遍历评估结果的维度和对应的结果
        for dim, result in assessment_results.items():
            # 如果结果中包含分数，记录维度和分数
            if "score" in result:
                logger.info(f'{dim}: {result["score"]}')
            # 如果结果中包含标准差，并且不为 None，则记录维度和标准差（保留两位小数）
            if "standard_variance" in result and result["standard_variance"] is not None:
                logger.info(f'{dim}: {result["standard_variance"]:.2f}')
            # 如果结果中包含批次结果，记录第一个批次的分析信息
            if "batch_results" in result:
                logger.info(f'{result["batch_results"][0]["analysis"]}')
# 如果脚本被直接执行（而非被导入到其他模块中），则执行以下代码
if __name__ == '__main__':
    # 调用 personality_assessment 函数，传入命令行参数 args.character, args.agent_llm, args.questionnaire_type,
    # args.eval_setting, args.evaluator, args.language
    personality_assessment(args.character, args.agent_llm, args.questionnaire_type, args.eval_setting, args.evaluator, args.language)
    
# 以下是示例命令行调用示例
# python assess_personality.py --eval_setting sample --questionnaire_type mbti
# python assess_personality.py --eval_setting batch --questionnaire_type mbti --character hutao
```