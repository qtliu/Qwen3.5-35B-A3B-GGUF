# 你的业务 Python 脚本
from openai import OpenAI

# 连接你刚刚用原生 C++ 部署的本地服务
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-no-key-required" # 填任意字符串
)

response = client.chat.completions.create(
    model="qwen35moe", # 名字任意，llama.cpp server 默认使用挂载的单一模型
    messages=[
        {"role": "system", "content": "你是一个非常有用的AI助手。"},
        {"role": "user", "content": "请介绍一下混合专家模型(MoE)的原理。"}
    ],
    temperature=0.7,
    max_tokens=2048
)

print(response.choices[0].message.content)