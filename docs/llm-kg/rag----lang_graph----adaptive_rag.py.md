# `.\rag\lang_graph\adaptive_rag.py`

```
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import os
import pprint

# 设定本地的 LLAMA3 模型名称
local_llm = "llama3"

# 设置 Tavily API 密钥为空字符串
tavily_api_key = os.environ['TAVILY_API_KEY'] = ''

# 在 Streamlit 应用中设置标题
st.title("多 PDF 文件聊天机器人 - LLAMA3 & Adaptive RAG")

# 设置用于用户输入的文本框，并设置默认值
user_input = st.text_input("对 PDF 文件提问:", placeholder="请在输入框中输入您的提问", key='input', value="llm agent memory")

# 在 Streamlit 应用的侧边栏中设置文件上传控件
with st.sidebar:
    uploaded_files = st.file_uploader("上传 PDF 文件（可多个）", type=['pdf'], accept_multiple_files=True)
    process = st.button("导入并处理文件")

# 如果用户点击了处理按钮
if process:
    # 如果没有上传文件，显示警告信息并停止执行
    if not uploaded_files:
        st.warning("请上传至少一份 PDF 文件。")
        st.stop()

    # 确保临时目录存在，如果不存在则创建
    temp_dir = os.getcwd()
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    st.write("处理上传的文件.....")
    
    # 遍历每个上传的文件
    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # 将文件保存到磁盘
        with open(temp_file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())  # 使用 getbuffer() 保存 Streamlit 的 UploadedFile
        
        # 使用 PyPDFLoader 加载 PDF 文件内容
        try:
            loader = PyPDFLoader(temp_file_path)
            data = loader.load()  # 假设 loader.load() 是正确的调用方法
            st.write(f"加载成功，文件名 {uploaded_file.name}")
        except Exception as e:
            st.error(f"无法加载 {uploaded_file.name}: {str(e)}")

    st.write("对数据集进行分块.....")
    
    # 使用 RecursiveCharacterTextSplitter 分割文档内容为文本块
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    text_chunks = text_splitter.split_documents(data)

    # 将文本块转换为向量，并添加到 Chroma 向量数据库中
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        collection_name="rag-chroma",
        embedding=GPT4AllEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    # 初始化 ChatOllama 模型，用于后续的对话交互
    llm = ChatOllama(model=local_llm, format="json", temperature=0)
    prompt = PromptTemplate(
        template="""You are an expert at routing a user question to a vectorstore or web search. \n
        Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks. \n
        You do not need to be stringent with the keywords in the question related to these topics. \n
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
        Return the a JSON with a single key 'datasource' and no premable or explaination. \n
        Question to route: {question}""",
        input_variables=["question"],
    )
    # 定义一个提示模板，用于路由用户问题到 vectorstore 或 web 搜索

    question_router = prompt | llm | JsonOutputParser()
    # 创建问题路由器，通过提示模板、llm 和 JSON 输出解析器连接

    question = "llm agent memory"
    # 设置问题变量为 "llm agent memory"

    docs = retriever.invoke(question)
    # 使用检索器检索相关文档

    doc_txt = docs[1].page_content
    # 从检索结果中获取第二个文档的内容

    question_router.invoke({"question": question})
    # 调用问题路由器来处理给定的问题

    llm = ChatOllama(model=local_llm, format="json", temperature=0)
    # 设置llm为ChatOllama类的实例，使用本地模型 local_llm，输出格式为 JSON，温度为 0

    st.write("对数据集进行聊天机器人评分.....")
    # 在界面上显示消息："对数据集进行聊天机器人评分....."

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["question", "document"],
    )
    # 定义一个提示模板，用于评估检索到的文档与用户问题的相关性

    retrieval_grader = prompt | llm | JsonOutputParser()
    # 创建文档评分器，通过提示模板、llm 和 JSON 输出解析器连接

    question = "agent memory"
    # 设置问题变量为 "agent memory"

    docs = retriever.invoke(question)
    # 使用检索器检索相关文档

    doc_txt = docs[1].page_content
    # 从检索结果中获取第二个文档的内容

    st.write(retrieval_grader.invoke({"question": question, "document": doc_txt}))
    # 在界面上显示文档评分结果，调用文档评分器来处理给定的问题和文档内容

    # 使用 LangChain hub 来获取提示.....
    prompt = hub.pull("rlm/rag-prompt")
    # 从 LangChain hub 中获取指定提示 "rlm/rag-prompt"

    # LLM
    llm = ChatOllama(model=local_llm, temperature=0)
    # 设置llm为ChatOllama类的实例，使用本地模型 local_llm，温度为 0

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    # 定义一个函数，用于格式化文档列表的内容

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    # 创建链式处理器，通过 LangChain 提示、llm 和字符串输出解析器连接

    # Run
    question = "agent memory"
    # 设置问题变量为 "agent memory"

    generation = rag_chain.invoke({"context": docs, "question": question})
    # 调用链式处理器来生成基于给定上下文和问题的输出

    st.write('输出结果：')
    # 在界面上显示消息："输出结果："

    st.write(generation)
    # 在界面上显示生成的结果

    llm = ChatOllama(model=local_llm, format="json", temperature=0)
    # 设置llm为ChatOllama类的实例，使用本地模型 local_llm，输出格式为 JSON，温度为 0

    # 对提问和答案匹配度进行评分
    # Prompt
    # 继续对提问和答案匹配度进行评分的代码，未完整给出
    # 创建评估模板，用于评估答案是否基于一组事实进行支持
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "documents"],
    )

    # 创建 Hallucination Grader 实例，连接评估模板、语言模型和 JSON 输出解析器
    hallucination_grader = prompt | llm | JsonOutputParser()
    # 调用 Hallucination Grader，传递答案和相关文档
    hallucination_grader.invoke({"documents": docs, "generation": generation})

    # 初始化 ChatOllama 实例，用于处理语言模型的生成和格式化输出为 JSON
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # 评估给定答案在解决特定问题方面的实用性
    # Prompt 模板，用于评估答案是否有用来解决问题
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "question"],
    )

    # 创建 Answer Grader 实例，连接评估模板、语言模型和 JSON 输出解析器
    answer_grader = prompt | llm | JsonOutputParser()
    # 调用 Answer Grader，传递问题和答案
    answer_grader.invoke({"question": question,"generation": generation})

    # 重新初始化 ChatOllama 实例，用于处理语言模型的生成和格式化输出为 JSON
    llm = ChatOllama(model=local_llm, temperature=0)

    # 重写问题以提高其在向量存储中的检索适用性
    # Prompt 模板，用于优化问题以便于向量存储的检索
    re_write_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the initial and formulate an improved question. \n
        Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
        input_variables=["generation", "question"],
    )

    # 创建 Question Rewriter 实例，连接重写模板、语言模型和字符串输出解析器
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    # 调用 Question Rewriter，传递问题以优化
    question_rewriter.invoke({"question": question})

    # 使用 TavilySearchResults 工具进行网络搜索，以提取相关主题内容
    web_search_tool = TavilySearchResults(k=3,tavily_api_key=tavily_api_key)
        

    # 定义状态结构 GraphState，包括问题、生成的答案和文档列表
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents 
        """
        question : str
        generation : str
        documents : List[str]

    # 基于提供的问题获取相关文档
    # 检索文档
    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        st.write("---RETRIEVE---")
        # 从状态中获取问题
        question = state["question"]

        # 调用检索器来检索与问题相关的文档
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    # 生成答案
    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        st.write("---GENERATE---")
        # 从状态中获取问题和已检索到的文档
        question = state["question"]
        documents = state["documents"]
        
        # 使用 RAG 模型生成答案
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    # 检查文档的相关性
    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        st.write("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        # 从状态中获取问题和已检索到的文档
        question = state["question"]
        documents = state["documents"]
        
        # 评分每个文档，并根据评分筛选相关的文档
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score['score']
            if grade == "yes":
                st.write("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                st.write("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    # 改进原始问题以便更好地检索
    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        st.write("---TRANSFORM QUERY---")
        # 从状态中获取问题和已检索到的文档
        question = state["question"]
        documents = state["documents"]

        # 使用问题重写器重写问题
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    # 基于重述问题的网络搜索函数。它使用网络搜索工具检索网络结果，并将它们格式化为单个文档
    def web_search(state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        st.write("---WEB SEARCH---")  # 输出调试信息：显示正在进行网络搜索
        question = state["question"]  # 从状态字典中获取问题文本

        # Web search
        docs = web_search_tool.invoke({"query": question})  # 调用网络搜索工具进行搜索
        web_results = "\n".join([d["content"] for d in docs])  # 将搜索结果转换为文档内容的字符串
        web_results = Document(page_content=web_results)  # 创建文档对象

        return {"documents": web_results, "question": question}  # 返回更新后的状态字典，包括文档和原始问题

    ### Edges ###

    # 根据问题的来源决定是将问题引导到网络搜索还是 RAG
    def route_question(state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        st.write("---ROUTE QUESTION---")  # 输出调试信息：显示正在路由问题
        question = state["question"]  # 从状态字典中获取问题文本
        st.write(question)  # 输出问题文本
        source = question_router.invoke({"question": question})  # 使用问题路由器确定问题来源
        st.write(source)  # 输出问题来源信息
        st.write(source['datasource'])  # 输出问题的数据源信息

        if source['datasource'] == 'web_search':
            st.write("---ROUTE QUESTION TO WEB SEARCH---")  # 输出调试信息：显示问题被路由到网络搜索
            return "web_search"  # 返回下一个节点名称为 "web_search"
        elif source['datasource'] == 'vectorstore':
            st.write("---ROUTE QUESTION TO RAG---")  # 输出调试信息：显示问题被路由到 RAG
            return "vectorstore"  # 返回下一个节点名称为 "vectorstore"

    # 基于过滤文档的相关性决定是生成答案还是重新生成问题
    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        st.write("---ASSESS GRADED DOCUMENTS---")  # 输出调试信息：显示正在评估已分级的文档
        question = state["question"]  # 从状态字典中获取问题文本
        filtered_documents = state["documents"]  # 从状态字典中获取过滤后的文档列表

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            st.write("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"  # 如果没有相关的文档，则返回下一个节点名称为 "transform_query"
        else:
            # We have relevant documents, so generate answer
            st.write("---DECISION: GENERATE---")
            return "generate"  # 如果有相关的文档，则返回下一个节点名称为 "generate"

    # 这个函数通过检查生成是否基于提供的文档并回答了原始问题来评估生成答案的质量
    def grade_generation_v_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.
    
        Args:
            state (dict): The current graph state
    
        Returns:
            str: Decision for next node to call
        """
    
        # Write to the stream to indicate checking for hallucinations
        st.write("---CHECK HALLUCINATIONS---")
        
        # Extract question, documents, and generation from the state dictionary
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
    
        # Invoke hallucination grader to get a score
        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score['score']
    
        # Check if generation is grounded in documents
        if grade == "yes":
            # Write decision indicating generation is grounded in documents
            st.write("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            
            # Evaluate generation against the question
            st.write("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question,"generation": generation})
            grade = score['score']
            
            # Check if generation addresses the question
            if grade == "yes":
                # Write decision indicating generation addresses the question
                st.write("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"  # Return useful if generation is deemed useful
            else:
                # Write decision indicating generation does not address the question
                st.write("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"  # Return not useful if generation is not deemed useful
        else:
            # Write decision indicating generation is not grounded in documents
            st.write("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"  # Return not supported if generation is not supported
    
    
    workflow = StateGraph(GraphState)
    
    # Define the nodes in the workflow
    workflow.add_node("web_search", web_search)  # Node for web search
    workflow.add_node("retrieve", retrieve)  # Node for retrieval
    workflow.add_node("grade_documents", grade_documents)  # Node for grading documents
    workflow.add_node("generate", generate)  # Node for generation
    workflow.add_node("transform_query", transform_query)  # Node for transforming query
    
    # Build the graph structure with edges and conditional routing
    workflow.set_conditional_entry_point(
        route_question,
        {
            "web_search": "web_search",  # Route to web search node
            "vectorstore": "retrieve",  # Route to retrieve node
        },
    )
    workflow.add_edge("web_search", "generate")  # Edge from web search to generation
    workflow.add_edge("retrieve", "grade_documents")  # Edge from retrieval to document grading
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",  # Route to transform query if conditions met
            "generate": "generate",  # Route back to generation if conditions met
        },
    )
    workflow.add_edge("transform_query", "retrieve")  # Edge from query transformation back to retrieval
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",  # Route to generation if not supported
            "useful": END,  # End node if generation is useful
            "not useful": "transform_query",  # Route back to transform query if not useful
        },
    )
    
    # Compile the workflow
    app = workflow.compile()
    
    inputs = {"question": user_input}  # Define input dictionary with user question
    for output in app.stream(inputs):
        for key, value in output.items():
            # Write node name to indicate current processing node
            st.write(f"Node '{key}':")
            # Write value to display the current state at each node
            st.write(value)
            # Optionally, print full state using pprint
            # pprint.pprint(value["state"], indent=2, width=80, depth=None)
        print("\n---\n")
    
    # Final generation
    # 在页面上写入文本字符串 "---FINAL ANSWER---"
    st.write("---FINAL ANSWER---")
    # 在页面上写入来自变量 value 的 "generation" 键对应的值
    st.write(value["generation"])
```