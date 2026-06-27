import os
from openai import OpenAI

# 从环境变量获取 DeepSeek API Key
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

print(api_key)
# 初始化 OpenAI 客户端（假设 DeepSeek 的 API 兼容 OpenAI 格式）
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",  # DeepSeek API 的基地址
)

# 定义提示词
# prompt = """请帮我用 HTML 生成一个五子棋游戏，所有代码都保存在一个 HTML 中。"""
prompt = """
请创建一个现代风格的五子棋游戏，保存在单个 HTML 文件中。
设计要求：
1. UI 风格：深色极简主题（Dark Mode），背景使用深蓝色或深灰色渐变。
2. 棋盘：使用 Canvas 绘制，具有逼真的木质质感或高对比度网格。
3. 交互：棋子要有 CSS 阴影和渐变使其看起来有立体感。
4. 功能：双人对战模式，必须包含“悔棋”和“重新开始”按钮，获胜时弹出美观的模态框（Modal）提示。
5. 代码格式：请尽量紧凑，确保在单次回答中能生成完整的代码。
注意：请只提供代码，不要添加 Markdown 标记。
"""
try:
    # 调用 DeepSeek Chat API
    response = client.chat.completions.create(
        model="deepseek-chat",  # 或 DeepSeek 提供的其他模型名称
        messages=[
            {"role": "system", "content": "你是一个专业的 Web 开发助手，擅长用 HTML/CSS/JavaScript 编写游戏。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        stream=False
    )

    # 提取生成的 HTML 内容
    if response.choices and len(response.choices) > 0:
        html_content = response.choices[0].message.content

        # 保存到文件
        with open("gomoku.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("五子棋游戏已保存为 gomoku.html")
    else:
        print("未收到有效响应")

except Exception as e:
    print(f"调用 API 出错: {e}")