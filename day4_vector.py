import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# 1. 加载文档（确保 news.txt 存在，或者换成你实际的 txt 文件名）
loader = TextLoader("./news.txt", encoding="utf-8")
docs = loader.load()
print(f"原始文档加载成功，共有 {len(docs)} 个文档对象。")

# 2. 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20
)
splits = text_splitter.split_documents(docs)
print(f"切分完成，现在变成了 {len(splits)} 个小块。")

# 3. 向量化并存入数据库
embeddings = OpenAIEmbeddings(
    model="embedding-3", 
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    check_embedding_ctx_length=False 
)

print("正在把数据存入向量数据库...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db" 
)
print("存入成功！")

# 4. 语义检索
query = "学校的办学初心是什么？" # 换成你想问的问题
print(f"\n正在检索：{query}")

results = vectorstore.similarity_search(query, k=2)

for i, res in enumerate(results):
    print(f"\n--- 匹配度第 {i+1} 的片段 ---")
    print(res.page_content)