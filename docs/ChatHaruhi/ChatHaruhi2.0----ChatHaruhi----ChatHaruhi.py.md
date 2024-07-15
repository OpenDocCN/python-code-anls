# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\ChatHaruhi.py`

```py
from .ChromaDB import ChromaDB  # 导入ChromaDB模块，用于构建数据库
import os  # 导入os模块，用于操作系统相关功能

from .utils import luotuo_openai_embedding, tiktokenizer  # 从utils模块导入luotuo_openai_embedding和tiktokenizer函数

from .utils import response_postprocess  # 从utils模块导入response_postprocess函数

class ChatHaruhi:

    def extract_text_vec_from_datas( self, datas, embed_name ):
        # 从数据集中提取文本和向量，根据embed_name指定的字段名
        # 返回提取的文本列表texts和向量列表vecs
        from .utils import base64_to_float_array  # 从utils模块导入base64_to_float_array函数

        texts = []
        vecs = []
        for data in datas:
            if data[embed_name] == 'system_prompt':
                system_prompt = get_text_from_data( data )  # 获取系统提示文本
            elif data[embed_name] == 'config':
                pass  # 如果是配置信息则跳过
            else:
                vec = base64_to_float_array( data[embed_name] )  # 将base64编码的向量转换为浮点数数组
                text = get_text_from_data( data )  # 获取文本数据
                vecs.append( vec )
                texts.append( text )
        return texts, vecs, system_prompt  # 返回文本列表、向量列表和系统提示文本

        

    def check_system_prompt(self, system_prompt):
        # 检查系统提示文本，如果以.txt结尾则以utf-8编码读取文件内容，否则直接返回文本字符串
        if system_prompt.endswith('.txt'):
            with open(system_prompt, 'r', encoding='utf-8') as f:
                return f.read()  # 读取文件内容并返回
        else:
            return system_prompt  # 直接返回文本字符串
    

    def get_tokenlen_setting( self, model_name ):
        # 获取故事和历史记号长度设置
        if model_name == 'openai':
            return (1500, 1200)  # 如果模型名为'openai'，返回设定的长度
        else:
            print(f'warning! undefined model {model_name}, use openai instead.')
            return (1500, 1200)  # 否则输出警告信息并返回默认长度设置
        
    def build_story_db_from_vec( self, texts, vecs ):
        self.db = ChromaDB()  # 创建ChromaDB数据库实例

        self.db.init_from_docs( vecs, texts )  # 从文本和向量列表初始化数据库

    def build_story_db(self, text_folder):
        # 构建故事数据库，从文本文件夹中读取文本并抽取向量
        db = ChromaDB()  # 创建ChromaDB数据库实例

        strs = []

        # 扫描text_folder中所有的txt文件
        for file in os.listdir(text_folder):
            # 如果文件名以.txt结尾
            if file.endswith(".txt"):
                file_path = os.path.join(text_folder, file)  # 构建文件的完整路径
                with open(file_path, 'r', encoding='utf-8') as f:
                    strs.append(f.read())  # 读取文件内容并添加到strs列表中

        if self.verbose:
            print(f'starting extract embedding... for { len(strs) } files')  # 如果verbose为True，打印开始提取嵌入向量信息

        vecs = []

        ## TODO: 建立一个新的embedding batch test的单元测试
        ## 新的支持list batch test的embedding代码
        ## 用新的代码替换下面的for循环
        ## Luotuo-bert-en也发布了，所以可以避开使用openai
        for mystr in strs:
            vecs.append(self.embedding(mystr))  # 对每个文本应用嵌入函数并将结果添加到vecs列表中

        db.init_from_docs(vecs, strs)  # 从vecs和strs初始化数据库

        return db  # 返回构建好的数据库实例
    
    def save_story_db(self, db_path):
        self.db.save(db_path)  # 将数据库保存到指定路径
    def generate_prompt( self, text, role):
        from langchain.schema import (  # 导入所需模块
            AIMessage,  # 导入 AIMessage 类
            HumanMessage,  # 导入 HumanMessage 类
            SystemMessage  # 导入 SystemMessage 类
        )
        messages = self.generate_messages( text, role )  # 调用 generate_messages 方法生成消息列表
        prompt = ""  # 初始化提示字符串
        for msg in messages:  # 遍历消息列表
            if isinstance(msg, HumanMessage):  # 如果消息类型是 HumanMessage
                prompt += msg.content + "\n"  # 将消息内容加入提示字符串，并换行
            elif isinstance(msg, AIMessage):  # 如果消息类型是 AIMessage
                prompt += msg.content + "\n"  # 将消息内容加入提示字符串，并换行
            elif isinstance(msg, SystemMessage):  # 如果消息类型是 SystemMessage
                prompt += msg.content + "\n"  # 将消息内容加入提示字符串，并换行
        return prompt  # 返回生成的提示字符串


    def generate_messages( self, text, role):
        # 添加系统提示
        self.llm.initialize_message()  # 初始化消息处理器
        self.llm.system_message(self.system_prompt)  # 添加系统提示消息

        # 添加故事内容
        query = self.get_query_string(text, role)  # 获取查询字符串
        self.add_story( query )  # 添加故事到对话历史中
        self.last_query = query  # 记录最后一次查询

        # 添加用户查询
        self.llm.user_message(query)  # 将用户查询添加到消息处理器

        return self.llm.messages  # 返回消息处理器中的消息列表
    
    def append_response( self, response, last_query = None ):
        if last_query == None:  # 如果未指定最后查询
            last_query_record = ""  # 最后查询记录为空字符串
            if hasattr( self, "last_query" ):  # 如果对象具有 last_query 属性
                last_query_record = self.last_query  # 获取对象的最后查询记录
        else:
            last_query_record = last_query  # 否则，使用指定的最后查询记录

        # 记录对话历史
        self.dialogue_history.append((last_query_record, response))  # 将最后查询记录和响应添加到对话历史中
        
    def chat(self, text, role):
        # 添加系统提示
        self.llm.initialize_message()  # 初始化消息处理器
        self.llm.system_message(self.system_prompt)  # 添加系统提示消息

        # 添加故事内容
        query = self.get_query_string(text, role)  # 获取查询字符串
        self.add_story( query )  # 添加故事到对话历史中

        # 添加历史记录
        self.add_history()  # 添加历史记录

        # 添加用户查询
        self.llm.user_message(query)  # 将用户查询添加到消息处理器
        
        # 获取响应
        response_raw = self.llm.get_response()  # 获取处理器的响应消息

        # 对响应进行后处理
        response = response_postprocess(response_raw, self.dialogue_bra_token, self.dialogue_ket_token)  # 调用后处理函数处理响应

        # 记录对话历史
        self.dialogue_history.append((query, response))  # 将查询和响应添加到对话历史中

        return response  # 返回响应
    
    def get_query_string(self, text, role):
        if role in self.narrator:  # 如果角色在叙述者列表中
            return role + ":" + text  # 返回角色和文本连接的字符串
        else:
            return f"{role}:{self.dialogue_bra_token}{text}{self.dialogue_ket_token}"  # 返回角色和带有特定标记的文本字符串
        
    def add_story(self, query):
        if self.db is None:  # 如果数据库为空
            return  # 直接返回

        query_vec = self.embedding(query)  # 获取查询的嵌入向量

        stories = self.db.search(query_vec, self.k_search)  # 使用嵌入向量和搜索参数在数据库中搜索故事

        story_string = self.story_prefix_prompt  # 初始化故事字符串为故事前缀提示
        sum_story_token = self.tokenizer(story_string)  # 计算故事字符串的标记数

        for story in stories:  # 遍历搜索到的故事
            story_token = self.tokenizer(story) + self.tokenizer(self.dialogue_divide_token)  # 计算故事的标记数及对话分隔标记的标记数
            if sum_story_token + story_token > self.max_len_story:  # 如果总标记数超过最大故事长度
                break  # 跳出循环
            else:
                sum_story_token += story_token  # 更新总标记数
                story_string += story + self.dialogue_divide_token  # 添加故事和对话分隔标记到故事字符串

        self.llm.user_message(story_string)  # 将最终的故事字符串添加到消息处理器
    # 定义一个方法用于添加对话历史记录
    def add_history(self):

        # 如果对话历史记录为空，则直接返回
        if len(self.dialogue_history) == 0:
            return
        
        # 初始化总历史标记计数和标志位
        sum_history_token = 0
        flag = 0
        
        # 从最新到最旧遍历对话历史记录
        for query, response in reversed(self.dialogue_history):
            current_count = 0
            # 如果有查询内容，计算其token数量并累加
            if query is not None:
                current_count += self.tokenizer(query) 
            # 如果有回复内容，计算其token数量并累加
            if response is not None:
                current_count += self.tokenizer(response)
            # 将当前对话的token数量加入总历史标记计数中
            sum_history_token += current_count
            # 如果总历史标记计数超过了设定的最大历史长度，则结束遍历
            if sum_history_token > self.max_len_history:
                break
            else:
                flag += 1

        # 如果标志位为0，表示没有添加任何历史记录
        if flag == 0:
            print('warning! no history added. the last dialogue is too long.')

        # 遍历要添加的历史记录，并发送到相应的处理方法中
        for (query, response) in self.dialogue_history[-flag:]:
            if query is not None:
                self.llm.user_message(query)
            if response is not None:
                self.llm.ai_message(response)
```