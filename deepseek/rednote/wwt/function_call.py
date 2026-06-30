import os
from openai import OpenAI

# 从环境变量获取 DeepSeek API Key
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",  # DeepSeek API 的基地址
)

def send_messages(messages):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )
    print("send messages is called")
    return response.choices[0].message

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of an location, the user should supply a location first",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        }
    },{
        "type": "function",
        "function": {
            "name": "get_fengxian_weather",
            "description": "Get weather of feng xian, the user should supply a location first",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state and square, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

messages = [{"role": "user", "content": "How's the weather in Shanghai feng xian?"}]
message = send_messages(messages)
print(f"Model>\t {message.content}")
tool = message.tool_calls[0]
print("tool is -- ",tool)
messages.append(message)
# 模拟 search_web 工具调用结果（直接返回24度）
messages.append({"role": "tool", "tool_call_id": tool.id, "content": "27℃"})
message = send_messages(messages)
print(f"Model>\t {message.content}")
"""
send messages is called
Model>	 Let me check the weather in Shanghai Fengxian.
tool is --  ChatCompletionMessageFunctionToolCall(id='call_00_9vtljsAzXRhtaeGX6vMF8565', function=Function(arguments='{"location": "Shanghai, Fengxian"}', name='get_fengxian_weather'), type='function', index=0)
send messages is called
Model>	 The weather in Shanghai Fengxian is currently **27°C**. Sounds like a pleasant temperature! Is there anything else you'd like to know?
"""