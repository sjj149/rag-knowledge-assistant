import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="AI 私有知识库助手", page_icon="🤖", layout="wide")
st.title("🤖 企业级私有知识库 AI 问答助手 (RAG)")
st.caption("基于 LangChain + 智谱 GLM-4 + Chroma 构建")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

@st.cache_resource 
def get_embeddings():
    return OpenAIEmbeddings(
        model="embedding-3",
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        check_embedding_ctx_length=False
    )

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
    streaming=True 
)

with st.sidebar:
    st.header("📁 知识库文档")
    uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("处理并入库", type="primary"):
            with st.spinner("正在解析 PDF 并向量化，请稍候..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    splits = text_splitter.split_documents(docs)
                    
                    embeddings = get_embeddings()
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        persist_directory="./chroma_db_streamlit"
                    )
                    st.success(f"✅ 文档已入库！共切分为 {len(splits)} 个片段。")
                finally:
                    os.unlink(tmp_path)
    
    if st.session_state.vectorstore is not None:
        st.info("📚 知识库已就绪，可以在右侧开始提问了。")
    else:
        st.warning("⚠️ 请先上传 PDF 文档。")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入关于文档的问题..."):
    if st.session_state.vectorstore is None:
        st.warning("请先在左侧上传 PDF 文档。")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(prompt)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "你是一个专业的问答助手。请严格根据下面提供的上下文来回答用户的问题。\n"
                           "如果你在上下文中找不到答案，请直接回答“根据提供的文档，我无法回答该问题”，绝对不要自己编造。\n\n"
                           "【上下文】：\n{context}"),
                ("user", "{question}")
            ])
            
            chain = prompt_template | llm | StrOutputParser()
            response = st.write_stream(chain.stream({"context": context_text, "question": prompt}))
            
    st.session_state.messages.append({"role": "assistant", "content": response})