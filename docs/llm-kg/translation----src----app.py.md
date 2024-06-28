# `.\translation\src\app.py`

```
# 导入Streamlit库，用于构建Web应用界面
import streamlit as st

# 从不同模块导入所需的类和函数
from classifier.language_classifier import LanguageDetector  # 导入语言检测器类
from encoder.encoder import Encoder  # 导入编码器类
from generator.generator import Generator  # 导入生成器类
from retriever.vector_db import VectorDatabase  # 导入向量数据库类
from translator.translator import Translator  # 导入翻译器类

# 初始化各个类的实例
generator = Generator()  # 实例化生成器
encoder = Encoder()  # 实例化编码器
vectordb = VectorDatabase(encoder.encoder)  # 实例化向量数据库，并使用编码器初始化
translator = Translator()  # 实例化翻译器
lang_classifier = LanguageDetector()  # 实例化语言检测器

# 创建应用的标题
st.title("Welcome")

# 创建消息历史的状态，如果状态中不存在 "messages" 键，则创建一个空列表
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染先前的消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 渲染聊天输入框，并获取用户输入的消息
prompt = st.chat_input("Enter your message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 渲染用户新消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 渲染助手的回复
    with st.chat_message("assistant"):
        # 获取产品ID和客户问题
        ID = prompt.split("|")[0]  # 从用户输入中获取产品ID
        QUERY = prompt.split("|")[1]  # 从用户输入中获取客户问题

        # 检测客户提问的语言，以便回复时使用相同语言
        user_detected_language = lang_classifier.detect_language(QUERY)

        # 获取与产品相关的上下文信息
        context = vectordb.retrieve_most_similar_document(QUERY, k=4, id=ID)

        # 将所有上下文信息转换为英语，以便用于语言模型
        english_context = []
        for doc in context:
            detected_language = lang_classifier.detect_language(doc)
            if detected_language != "en_XX":
                doc = translator.translate(doc, detected_language, "en_XX")
            english_context.append(doc)
        context = "\n".join(english_context)

        # 将客户问题翻译为英语
        if user_detected_language != "en_XX":
            QUERY = translator.translate(QUERY, user_detected_language, "en_XX")

        # 基于客户问题和上下文信息，使用语言模型生成回答
        answer = generator.get_answer(context, QUERY)

        # 如果客户的语言不是英语，则将回答翻译为客户语言
        if user_detected_language != "en_XX":
            answer = translator.translate(answer, "en_XX", user_detected_language)

        # 在应用界面上显示回答
        st.markdown(answer)

    # 将完整的回复添加到消息历史中
    st.session_state.messages.append({"role": "assistant", "content": answer})
```