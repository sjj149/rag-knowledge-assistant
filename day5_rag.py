import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ==========================================
# 1. 连接昨天建好的向量数据库
# ==========================================
# 直接读取本地 persist_directory 文件夹，不需要重新切分和存入了
embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    check_embedding_ctx_length=False 
)

# 确保你昨天生成了 ./chroma_db 文件夹
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 把向量库变成一个“检索器”（Retriever）
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 

# ==========================================
# 2. 准备大模型和【核心】提示词模板
# ==========================================
llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1 # 做知识库问答，温度调低，让它严谨
)

# 注意这个 Prompt 的写法，这是 RAG 的灵魂！
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的问答助手。请严格根据下面提供的上下文来回答用户的问题。\n如果你在上下文中找不到答案，请直接回答“我不知道”，绝对不要自己编造。\n\n【上下文】：\n{context}"),
    ("user", "{question}")
])

# ==========================================
# 3. 开始问答（RAG 全流程）
# ==========================================
question = "暨南大学的办学初心是什么？" # 换成你想问的问题

print(f"用户提问：{question}\n")

# 第一步：检索（Retrieval）
# 去向量库里找最相关的段落
docs = retriever.invoke(question)

# 第二步：把检索到的几个小段，拼成一段长文本
context_text = "\n\n".join([doc.page_content for doc in docs])
print(f"--- 检索到的参考资料 ---\n{context_text[:200]}...\n")

# 第三步：生成（Generation）
# 把 context 和 question 填进模板，发给大模型
print("--- 大模型正在思考并生成回答 ---")
chain = prompt | llm | StrOutputParser()

answer = chain.invoke({
    "context": context_text,
    "question": question
})

print(f"\n🤖 最终回答：\n{answer}")