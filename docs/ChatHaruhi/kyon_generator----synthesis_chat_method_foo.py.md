# `.\Chat-Haruhi-Suzumiya\kyon_generator\synthesis_chat_method_foo.py`

```py
# Chat凉宫春日 project (https://github.com/LC1332/Chat-Haruhi-Suzumiya)
# Chat凉宫春日项目模仿了凉宫春日等动漫角色，使用类似的语气和剧情进行聊天，开发者包括李鲁鲁、冷子昂等。
#
# 这个程序演示了一个简单的synthesis_chat方法。

from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import random
from configparser import ConfigParser
import os
import openai
from joblib import Parallel, delayed
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# 创建一个ChatOpenAI对象，用于聊天生成
chatModel = ChatOpenAI(temperature=0)

'''
[Synthesis]
OPENAI_API_KEY = sk
stop_words = 春日,阿虚,凉宫,凉宫春日
input1 = 阿虚:「我无意中听到一件事。」
    春日:「反正不会是什么重要的事。」
output1 = {"Entity": ["不重要的事","阿虚","春日"]}
input2 = 阿虚:「你为什么要剪头发啊？」
    春日:「没什么理由，就是想剪了而已。」
output2 = {"Entity": ["剪头发","没什么理由"]}
KEYWORD_PROMPT = 提取反引号文本中的关键字Entity，以list的形式输出在一个json中。
TANSFER_PROMPT = 根据keywords的内容补全text
    text为对于凉宫春日剧情的一些讨论问题，role不可以是春日或者凉宫春日
    role可以是阿虚、朝比奈、老师等凉宫春日中，非春日的其他角色
    role也可以是任意其他动漫中的角色
    用一致性的语言风格，根据每行中的json内容，根据keywords中的关键字，补全text的内容。
'''

# 存储提取的关键词和角色列表
keyword_list = []
role_list = []

def generate(input_file, output_file, additional_config=None):
    """
    核心函数，使用foo方法将input_file生成增广的jsonl文件保存到output_file
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        # 读取input_file的内容
        data = f.read()
    
    # 获取配置
    config = ConfigParser()
    config.read(additional_config)
    os.environ["OPENAI_API_KEY"] = config['Synthesis']['OPENAI_API_KEY']
    stop_words = config['Synthesis']['stop_words'].split(',')
    input1 = config['Synthesis']['input1']
    output1 = config['Synthesis']['output1']
    input2 = config['Synthesis']['input2']
    output2 = config['Synthesis']['output2']
    KEYWORD_PROMPT = config['Synthesis']['KEYWORD_PROMPT']
    TANSFER_PROMPT = config['Synthesis']['TANSFER_PROMPT']
 
    # 多线程提取全部关键词
    def extract_keywords(new_query):
        # 构建消息列表，包括系统消息、用户消息和AI回复消息
        messages = [
            SystemMessage(content=KEYWORD_PROMPT),
            HumanMessage(content=input1),
            AIMessage(content=output1),
            HumanMessage(content=input2),
            AIMessage(content=output2)
        ]
        # 添加新查询的用户消息内容
        messages.append(HumanMessage(content=new_query['text']))
        # 使用chatModel进行对话生成
        return_msg = chatModel(messages)
        response = return_msg.content
        # 将提取的关键词存储在new_query的keywords字段中
        new_query['keywords'] = response['Entity']

    # 并行或串行处理数据
    multiply_process = True  # 在测试的时候可以改成False
    if multiply_process:  # 并行运行
        Parallel(n_jobs=max(os.cpu_count() - 1, 1))(
            delayed(extract_keywords)(item)
            for item in tqdm(data)
        )
    else:  # 串行运行
        for item in tqdm(data):
            extract_keywords(item)
 
    # foo_sample, foo_input
    n = len(data)
    sel_all = random.sample(range(0, n), 20)
    sel_sample = sel_all[:10]
    sel_input = sel_all[10:]
    # 调用 foo_sample 函数，获取示例输入、输出和关键词
    sample_input, sample_output, sample_keywords = foo_sample(sel_sample)
    
    # 根据选择的输入和示例关键词，生成查询输入
    query_input = foo_input(sel_input, sample_keywords)

    # 调用 remove_stop_words 函数，去除数据和示例关键词中的停用词
    data = remove_stop_words(data, stop_words)
    sample_keywords = remove_stop_words(sample_keywords, stop_words)


    # 定义 generate_with_keywords 函数，以 sample_input、sample_output 和 query_input 为参数
    def generate_with_keywords(sample_input, sample_output, query_input):
        # 将 sample_input 和 sample_output 分成两部分，每部分的大小为总大小的四分之一
        div_k = 4
        input1 = list_to_string(sample_input[:div_k])
        output1 = list_to_string(sample_output[:div_k])
        input2 = list_to_string(sample_input[div_k:])
        output2 = list_to_string(sample_output[div_k:])

        # 将 query_input 转换为字符串
        query = list_to_string(query_input)

        # 创建包含系统信息、人类输入、AI输出的消息列表
        messages = [
            SystemMessage(content=TANSFER_PROMPT),  # 包含一个系统消息
            HumanMessage(content=input1),           # 包含人类输入的第一部分
            AIMessage(content=output1),             # 包含AI输出的第一部分
            HumanMessage(content=input2),           # 包含人类输入的第二部分
            AIMessage(content=output2),             # 包含AI输出的第二部分
            HumanMessage(content=query)             # 包含查询输入
        ]
        
        # 使用 chatModel 处理消息列表，获取返回消息对象
        return_msg = chatModel(messages)
        
        # 返回返回消息对象中的内容
        return return_msg.content   
    

    # 使用 DataLoader 类加载 sample_keywords 数据，每次加载大小为 10
    data_story = DataLoader(sample_keywords, 10)
    
    # 使用 DataLoader 类加载 data 数据，每次加载大小为 10，用于聊天故事
    data_chat_as_story = DataLoader(data, 10)
    
    # 使用 DataLoader 类加载 data 数据，每次加载大小为 10，用于聊天
    data_chat = DataLoader(data, 10)


    # 设置批处理大小为 10
    batch_size = 10
    # 对于700次迭代，显示进度条，描述为'autoGenerating'
    for iter_time in tqdm(range(700), desc='autoGenerating'):
        
        # 初始化一个空列表，用于存储聊天数据
        chat_data = []
        
        # 根据batch_size循环获取聊天数据，并添加到chat_data列表中
        for _ in range(batch_size):
            chat_data.append(data_chat.get_data())

        # 对聊天数据进行组织，获取样本输入、样本输出和关键词
        sample_input, sample_output, sample_keywords = organize_samples(chat_data)

        # 组织查询输入数据
        # 初始化空列表，用于存储查询输入
        query_input = []

        # 遍历样本输入
        for input in sample_input:
            # 确定目标关键词数量，至少为2个
            target_n = len(input['keywords'])
            target_n = max(2, target_n)

            count_time = 0
            max_len = -999
            max_len_plan = []

            # 循环直到找到合适的关键词集合或达到15次尝试
            while count_time < 15:
                # 每次循环增加计数器
                count_time = count_time + 1
                
                # 根据迭代次数选择故事关键词来源
                if iter_time % 2 == 0:
                    story_keyword = data_story.get_data()
                else:
                    story_keyword = data_chat_as_story.get_data()

                # 过滤掉已经在样本关键词中的关键词
                filtered_keyword = [w for w in story_keyword["keywords"] if w not in sample_keywords]

                # 如果过滤后的关键词数量足够，则随机选择一部分作为新的关键词集合
                if len(filtered_keyword) >= target_n:
                    story_keyword['keywords'] = random.sample(filtered_keyword, min(target_n, len(filtered_keyword)))
                    break
                else:
                    # 记录最大长度不够的情况下的关键词集合和长度
                    if len(filtered_keyword) > max_len:
                        max_len = len(filtered_keyword)
                        max_len_plan = filtered_keyword.copy()

            # 如果找不到足够数量的关键词，则使用最大长度计划的关键词集合
            if len(story_keyword['keywords']) < target_n:
                story_keyword['keywords'] = max_len_plan

            # 将处理后的关键词集合添加到查询输入列表中
            query_input.append({'keywords': story_keyword['keywords']})

            # 将这次处理的关键词加入样本关键词集合中
            for keyword in story_keyword['keywords']:
                sample_keywords.append(keyword)

        # 尝试生成使用关键词的响应
        try:
            response = generate_with_keywords(sample_input, sample_output, query_input)
        except Exception as e:
            # 如果生成过程中出现异常，打印错误信息并终止循环
            print(f"An error occurred while running the script: {e}")
            break

    # 打开文件准备写入增强数据
    with open(output_file, 'w') as f:
        # 将增强数据写入output_file
        f.write(data)
# 定义一个函数，从给定的索引列表中选择聊天数据
def foo_sample(sel_sample):
    # 根据选择的样本索引从全局变量 chat_datas 中获取对应的聊天数据列表
    sel_chat_data = [chat_datas[i] for i in sel_sample]

    # 调用 organize_samples 函数整理选择的聊天数据，得到输入样本、输出样本和关键词列表
    sample_input, sample_output, sample_keywords = organize_samples(sel_chat_data)

    # 返回整理好的样本输入、输出和关键词列表
    return sample_input, sample_output, sample_keywords
    

# 定义一个函数，从给定的输入索引列表中选择聊天数据的输入部分
def foo_input(sel_input, sample_keywords):
    # 根据选择的输入索引从全局变量 chat_datas 中获取对应的聊天数据列表
    sel_chat_data = [chat_datas[i] for i in sel_input]

    # 调用 organize_samples 函数整理选择的聊天数据，得到输入样本，忽略输出和关键词列表
    sample_input, _ , _ = organize_samples(sel_chat_data)

    # 返回整理好的样本输入
    return sample_input


# 定义一个函数，将列表转换为以换行符分隔的字符串
def list_to_string(lst):
    result = ''
    for item in lst:
        result += str(item) + '\n'
    return result


# 定义一个函数，移除数据中的停用词
def remove_stop_words(data, stop_words):
    # 将停用词列表转换为集合，以提高查找效率
    stop_words_set = set(stop_words)
    # 遍历数据中的每个元素，对其中的关键词列表进行停用词过滤操作
    for item in data:
        item["keywords"] = [w for w in item["keywords"] if w not in stop_words_set]
    # 返回处理后的数据
    return data


# 定义一个函数，整理选择的聊天数据，将关键词随机排序，并生成样本输入和输出
def organize_samples(sel_chat_datas: List[Dict[str, str]]) -> Tuple[List[Dict], List[Dict], List[str]]:
    # 初始化样本输入列表、样本输出列表和所有关键词集合
    sample_input = []
    sample_output = []
    all_keywords = set()

    # 遍历每个选择的聊天数据元素
    for element in sel_chat_datas:
        # 从当前聊天数据元素中获取关键词列表
        keywords = element['keywords']
        # 将关键词列表随机打乱顺序
        np.random.shuffle(keywords)
        # 将打乱顺序后的关键词列表作为字典存入样本输入列表中
        sample_input.append({'keywords': keywords})
        # 创建输出元素字典，包含关键词、角色和文本信息，并存入样本输出列表中
        output_element = {
            'keywords': keywords,
            'role': element['role'],
            'text': element['text'],
        }
        sample_output.append(output_element)
        # 将当前聊天数据元素中的所有关键词添加到所有关键词集合中
        for kw in keywords:
            all_keywords.add(kw)
    
    # 将所有关键词集合转换为列表并返回整理好的样本输入、输出和所有关键词列表
    return sample_input, sample_output, list(all_keywords)


# 定义一个数据加载器类
class DataLoader:
    # 初始化函数，接受数据和可选参数 k，默认为 10
    def __init__(self, data, k=10):
        # 将传入的数据存储在实例变量中
        self.data = data
        # 记录数据的数量
        self.n = len(data)
        # 设置数据加载器的批次大小
        self.k = k
        # 初始化当前数据加载位置的索引
        self.current_id = 0
        # 创建数据加载索引的随机排列列表
        self.shuffle_id = list(range(self.n))
        random.shuffle(self.shuffle_id)
        # 记录前一批次数据加载索引的尾部
        self.previous_tail = self.shuffle_id[-self.k+1:]
    # 定义一个方法用于打乱数据索引顺序
    def shuffle(self):
        # 如果数据长度小于等于两倍的分组大小，直接对索引列表进行随机打乱
        if self.n <= 2 * self.k:
            random.shuffle(self.shuffle_id)
        else:
            # 否则，先对整个索引列表进行一次随机打乱
            random.shuffle(self.shuffle_id)
            # 取出打乱后的列表的前 k-1 个元素作为头部
            head = self.shuffle_id[:self.k-1]
            # 设定一个标志位用于控制循环
            flag = True
            # 计数器，用于记录循环次数
            count = 0

            # 初始化最小重叠数量为一个较大的数值，最小重叠方案为空列表
            min_ovlp_num = 999
            min_ovlp_plan = []

            # 当计数小于10且标志位为真时执行循环
            while count < 10 and flag == True:
                count = count + 1
                inverse_flag = False
                ovlp_num = 0
                # 检查头部元素是否在之前的尾部元素列表中，计算重叠数量并设置反向标志
                for id in head:
                    if id in self.previous_tail:
                        ovlp_num = ovlp_num + 1
                        inverse_flag = True

                # 如果当前重叠数量小于记录的最小重叠数量，更新最小重叠数量及其对应的打乱方案
                if ovlp_num < min_ovlp_num:
                    min_ovlp_num = ovlp_num
                    min_ovlp_plan = self.shuffle_id.copy()

                # 如果没有反向重叠，则设置标志位为假并跳出循环
                if False == inverse_flag:
                    flag = False
                    break

                # 重新随机打乱索引列表，并取前 k-1 个元素作为新的头部
                random.shuffle(self.shuffle_id)
                head = self.shuffle_id[:self.k-1]

            # 如果记录的最小重叠数量大于0，将索引列表设置为最小重叠方案
            if min_ovlp_num > 0:
                self.shuffle_id = min_ovlp_plan

            # 取出索引列表中的第 k 个到最后一个元素作为头部，倒数第 k 个到第 k-1 个元素作为尾部
            head = self.shuffle_id[self.k-1:]
            tail = self.shuffle_id[-self.k+1:]

            # 将头部、前 k-1 个元素、尾部重新组合并随机打乱索引列表
            self.shuffle_id = head + self.shuffle_id[:self.k-1] + tail
            random.shuffle(self.shuffle_id)
            # 更新之前的尾部元素列表为当前的尾部
            self.previous_tail = tail

    # 定义一个方法用于获取打乱后的数据
    def get_data(self):
        # 如果当前索引超过数据长度，重新进行数据打乱
        if self.current_id >= self.n:
            self.shuffle()
            self.current_id = 0
        # 根据当前索引获取数据，并将索引加一
        data = self.data[self.shuffle_id[self.current_id]]
        self.current_id += 1
        # 返回获取到的数据
        return data
# 如果当前脚本被直接执行（而不是被导入到其他脚本中执行），则执行以下代码块
if __name__ == '__main__':
    # 定义输入文件路径，用于指定要处理的输入数据文件
    input_file = r"D:\Misc\Chat-Haruhi-Suzumiya\Haruhi_first_merge_res.jsonl"
    # 定义输出文件路径，指定生成处理结果的输出文件位置
    output_file = r"D:\Misc\Chat-Haruhi-Suzumiya\Haruhi_first_merge_res_out.jsonl"
    # 定义配置文件路径，指定程序运行所需的配置文件位置
    config_file = r"D:\Misc\Chat-Haruhi-Suzumiya\config.ini"
    # 调用 generate 函数，传入输入文件路径、输出文件路径和配置文件路径进行处理
    generate(input_file, output_file, config_file)
```