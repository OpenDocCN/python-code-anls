# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\ChatHaruhi_safe.py`

```py
from .ChromaDB import ChromaDB
import os

from .utils import luotuo_openai_embedding, tiktokenizer

from .utils import response_postprocess

from .utils import text_censor

class ChatHaruhi_safe:

    def check_system_prompt(self, system_prompt):
        # 检查系统提示是否以 .txt 结尾，如果是则以 utf-8 编码读取文件内容
        # 否则直接返回字符串本身
        if system_prompt.endswith('.txt'):
            with open(system_prompt, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return system_prompt
    

    def get_models(self, model_name):
        # 根据模型名称返回对应的语言模型、嵌入模型和分词器的组合

        if model_name == 'openai':
            from .LangChainGPT import LangChainGPT
            return (LangChainGPT(), tiktokenizer)
        elif model_name == 'debug':
            from .PrintLLM import PrintLLM
            return (PrintLLM(), tiktokenizer)
        elif model_name == 'spark':
            from .SparkGPT import SparkGPT
            return (SparkGPT(), tiktokenizer)
        elif model_name == 'GLMPro':
            from .GLMPro import GLMPro
            return (GLMPro(), tiktokenizer)
        elif model_name == 'ernie3.5':
            from .ErnieGPT import ErnieGPT
            return (ErnieGPT(), tiktokenizer)
        elif model_name == 'ernie4.0':
            from .ErnieGPT import ErnieGPT
            return (ErnieGPT(model="ernie-bot-4"), tiktokenizer)
        elif model_name == "ChatGLM2GPT":
            from .ChatGLM2GPT import ChatGLM2GPT, GLM_tokenizer
            return (ChatGLM2GPT(), GLM_tokenizer)
        elif model_name == "BaiChuan2GPT":
            from .BaiChuan2GPT import BaiChuan2GPT, BaiChuan_tokenizer
            return (BaiChuan2GPT(), BaiChuan_tokenizer)
        elif model_name == "BaiChuanAPIGPT":
            from .BaiChuanAPIGPT import BaiChuanAPIGPT
            return (BaiChuanAPIGPT(), tiktokenizer)
        else:
            # 如果模型名称未定义，输出警告并使用默认模型 openai
            print(f'warning! undefined model {model_name}, use openai instead.')
            from .LangChainGPT import LangChainGPT
            return (LangChainGPT(), tiktokenizer)
        
    def get_tokenlen_setting( self, model_name ):
        # 返回故事和历史 token 长度的设置
        if model_name == 'openai':
            return (1500, 1200)
        else:
            # 如果模型名称未定义，输出警告并使用默认模型 openai
            print(f'warning! undefined model {model_name}, use openai instead.')
            return (1500, 1200)
        
    def build_story_db_from_vec( self, texts, vecs ):
        # 创建 ChromaDB 实例并根据文本和向量初始化数据库
        self.db = ChromaDB()
        self.db.init_from_docs( vecs, texts)
    def build_story_db(self, text_folder):
        # 创建一个新的 ChromaDB 实例
        db = ChromaDB()

        # 用于存储读取的文本内容列表
        strs = []

        # 遍历指定文件夹下的所有文件
        for file in os.listdir(text_folder):
            # 如果文件以 .txt 结尾
            if file.endswith(".txt"):
                file_path = os.path.join(text_folder, file)
                # 使用 utf-8 编码打开文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    strs.append(f.read())

        # 如果设置了详细输出标志，打印正在提取嵌入向量的信息
        if self.verbose:
            print(f'starting extract embedding... for { len(strs) } files')

        # 用于存储计算得到的嵌入向量列表
        vecs = []

        ## TODO: 建立一个新的embedding batch test的单元测试
        ## 新的支持list batch test的embedding代码
        ## 用新的代码替换下面的for循环
        ## Luotuo-bert-en也发布了，所以可以避开使用openai
        
        # 对每个文本字符串进行嵌入向量的计算，并存储在 vecs 中
        for mystr in strs:
            vecs.append(self.embedding(mystr))

        # 初始化 ChromaDB 实例，使用提取的嵌入向量和原始文本内容
        db.init_from_docs(vecs, strs)

        # 返回构建好的数据库实例
        return db
    
    def save_story_db(self, db_path):
        # 将当前数据库保存到指定路径下
        self.db.save(db_path)
        
    def chat(self, text, role):
        # 初始化消息生成器
        self.llm.initialize_message()
        # 添加系统提示信息
        self.llm.system_message(self.system_prompt)
    

        # 添加故事
        query = self.get_query_string(text, role)
        self.add_story( query )

        # 添加历史记录
        self.add_history()

        # 添加用户查询
        self.llm.user_message(query)
        
        # 获取模型的响应
        response_raw = self.llm.get_response()

        # 对模型响应进行后处理，如去除特定标记
        response = response_postprocess(response_raw, self.dialogue_bra_token, self.dialogue_ket_token)

        # 记录对话历史
        self.dialogue_history.append((query, response))



        # 返回处理后的模型响应
        return response
    
    def get_query_string(self, text, role):
        # 根据角色生成查询字符串
        if role in self.narrator:
            return role + ":" + text
        else:
            return f"{role}:{self.dialogue_bra_token}{text}{self.dialogue_ket_token}"
        
    def add_story(self, query):
        # 如果数据库未初始化，则直接返回
        if self.db is None:
            return
        
        # 计算查询字符串的嵌入向量
        query_vec = self.embedding(query)

        # 在数据库中搜索与查询向量最相似的故事，返回匹配的故事列表
        stories = self.db.search(query_vec, self.k_search)
        
        # 初始化故事字符串
        story_string = self.story_prefix_prompt
        # 对初始故事字符串进行分词处理
        sum_story_token = self.tokenizer(story_string)
        
        # 遍历匹配的故事列表
        for story in stories:
            # 对每个故事进行分词处理，并添加对话分割标记的分词结果
            story_token = self.tokenizer(story) + self.tokenizer(self.dialogue_divide_token)
            # 如果累计的故事分词数量超过最大长度限制，则结束循环
            if sum_story_token + story_token > self.max_len_story:
                break
            else:
                # 将当前故事及其分词结果添加到累计的故事字符串中
                sum_story_token += story_token
                story_string += story + self.dialogue_divide_token
        
        # 对故事字符串进行审查，确保内容符合规定
        if text_censor(story_string):
            # 将审查通过的故事字符串作为用户消息发送
            self.llm.user_message(story_string)
    # 定义一个方法用来向对话历史中添加内容
    def add_history(self):
        # 如果对话历史为空，则直接返回
        if len(self.dialogue_history) == 0:
            return
        
        # 初始化变量，用于计算历史对话中的总 token 数量和标志符
        sum_history_token = 0
        flag = 0
        
        # 遍历对话历史，从最近的开始
        for query, response in reversed(self.dialogue_history):
            current_count = 0
            # 如果有查询内容，则计算其 token 数量并累加
            if query is not None:
                current_count += self.tokenizer(query) 
            # 如果有响应内容，则计算其 token 数量并累加
            if response is not None:
                current_count += self.tokenizer(response)
            # 将当前对话的 token 数量累加到总数中
            sum_history_token += current_count
            # 如果累计的 token 数量超过了设定的最大历史长度，则结束循环
            if sum_history_token > self.max_len_history:
                break
            else:
                # 否则增加标志符，表示可以添加这一对话
                flag += 1

        # 如果没有可以添加的历史对话，则输出警告信息
        if flag == 0:
            print('warning! no history added. the last dialogue is too long.')

        # 添加被选中的历史对话到模型中
        for (query, response) in self.dialogue_history[-flag:]:
            # 如果有查询内容，则将其作为用户消息传递给语言模型
            if query is not None:
                self.llm.user_message(query)
            # 如果有响应内容，则将其作为 AI 消息传递给语言模型
            if response is not None:
                self.llm.ai_message(response)
```