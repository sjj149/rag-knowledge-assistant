from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv() 
client = OpenAI(
    api_key="b27593f6d4e34671ac0f59a1feb3e3d2.0i18RIxncNH6N3RI",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

response = client.chat.completions.create(
    model="glm-4-flash",
    temperature=0.1,
    messages=[
        {"role": "system", "content": "你是一个诗人。"},
        {"role": "user", "content": "写一句关于风的诗"}
    ]
  
)

print(response.choices[0].message.content)