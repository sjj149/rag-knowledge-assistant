# 🤖 企业级私有知识库 AI 问答助手 (RAG)

基于 LangChain + 智谱 GLM-4 + Chroma 向量数据库构建的私有文档问答系统。

## 🌟 核心特性
- **完全私有化**：上传你的 PDF 文档，AI 只基于你的文档回答。
- **防幻觉机制**：通过严格设计 System Prompt，彻底杜绝大模型“胡编乱造”，若文档无答案，会明确告知用户。
- **流式输出**：前端采用 Streamlit 构建，支持类似 ChatGPT 的打字机交互体验。
- **数据向量化**：使用智谱 embedding-3 模型对文本切分块进行向量化，存入本地 ChromaDB。

## 🛠️ 技术栈
- **核心框架**: LangChain (LCEL 表达式)
- **大语言模型**: 智谱 GLM-4-Flash
- **向量数据库**: ChromaDB
- **前端 UI**: Streamlit
- **工具**: Python, dotenv

## 🚀 本地运行
1. 克隆代码 `git clone xxx`
2. 安装依赖 `pip install -r requirements.txt`
3. 根目录新建 `.env`，填入 `ZHIPU_API_KEY=你的智谱密钥`
4. 启动网页 `streamlit run app.py`


