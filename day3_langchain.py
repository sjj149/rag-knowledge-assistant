import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
load_dotenv()
print("测试 Key 读取:", os.getenv("ZHIPU_API_KEY"))

# 1. 初始化模型（把智谱的地址和 Key 塞进去）
llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPU_API_KEY") ,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 2. 直接调用
response = llm.invoke("你好，请用一句话介绍你自己")
print(response.content)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个极其专业的翻译官。请将用户提供的文本准确翻译成{target_language}，不要输出任何多余的废话。"),
    ("user", "{text}")
])

# 2. 把模板和模型连起来（这就是 LangChain 最强大的 Chain 概念）
# 解释一下：数据先经过 prompt_template 填空，再传给 llm 生成，最后经过 StrOutputParser 变成纯字符串
chain = prompt_template | llm | StrOutputParser()

# 3. 调用 Chain，传入变量（就像做填空题）
print("\n--- 翻译结果 ---")
result = chain.invoke({
    "target_language": "日语",
    "text": "人工智能正在改变世界。"
})
print(result)