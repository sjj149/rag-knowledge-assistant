from openai import OpenAI

# 1. 切换到智谱的 API Key 和地址
api_key = "71f32e2da1f7495ca00c24313c56b870.ERGzwTlyEPZzLhLL"
base_url = "https://open.bigmodel.cn/api/paas/v4/"

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

print("开始请求...")

try:
    response = client.chat.completions.create(
        # 2. 使用智谱的免费模型
        model="glm-4.7-flash", 
        messages=[
            {"role": "user", "content": "你好，请用一句话介绍你自己"}
        ]
    )
    
    print("请求成功，回复如下：")
    print(response.choices[0].message.content)

except Exception as e:
    print("--- 出错了 ---")
    print(e)