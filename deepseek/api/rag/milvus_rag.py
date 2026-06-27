import os
from glob import glob
from openai import OpenAI
from pymilvus import model as milvus_model
# from pymilvus.model.dense import OpenAIEmbeddingFunction
from pymilvus import MilvusClient
from tqdm import tqdm
import json

# 从环境变量获取 DeepSeek API Key
api_key = os.getenv("DEEPSEEK_API_KEY")

text_lines = []

for file_path in glob("milvus_docs/faq/*.md"):
    with open(file_path, "r", encoding="utf-8") as file:
        file_text = file.read()
    text_lines += file_text.split("# ")

deepseek_client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",  # DeepSeek API 的基地址
)

# OpenAI国内代理 https://api.apiyi.com/token
embedding_model = milvus_model.DefaultEmbeddingFunction()
# embedding_model = OpenAIEmbeddingFunction(
#     model_name="text-embedding-3-large",
#     api_key='sk-74dfd0d430554d36be1bc6d1cdc38c70',
#     base_url="https://api.apiyi.com/v1",
#     dimensions=512
# )

test_embedding = embedding_model.encode_queries(["This is a test"])[0]
embedding_dim = len(test_embedding)
# print(embedding_dim)
# print(test_embedding[:10])

milvus_client = MilvusClient(uri="./milvus_demo.db")
collection_name = "my_rag_collection"

if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # 内积距离
    consistency_level="Strong",  # 支持的值为 (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`)。更多详情请参见 https://milvus.io/docs/consistency.md#Consistency-Level。
)

doc_embeddings = embedding_model.encode_documents(text_lines)

data = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": doc_embeddings[i], "text": line})

milvus_client.insert(collection_name=collection_name, data=data)

question = "How is data stored in milvus?"

# search_res = milvus_client.search(
#     collection_name=collection_name,
#     data=embedding_model.encode_queries(
#         [question]
#     ),  # 将问题转换为嵌入向量
#     limit=3,  # 返回前3个结果
#     output_fields=["text"],  # 返回 text 字段
# )
query_embedding = embedding_model.encode_queries([question])
search_res = milvus_client.search(
    collection_name=collection_name,
    data=query_embedding,
    limit=3,
    output_fields=["text"]        # 必须加
)

retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
print(json.dumps(retrieved_lines_with_distances, indent=4))

context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)

SYSTEM_PROMPT = """
Human: 你是一个 AI 助手。你能够从提供的上下文段落片段中找到问题的答案。
"""
USER_PROMPT = f"""
请使用以下用 <context> 标签括起来的信息片段来回答用 <question> 标签括起来的问题。最后追加原始回答的中文翻译，并用 <translated>和</translated> 标签标注。
<context>
{context}
</context>
<question>
{question}
</question>
<translated>
</translated>
"""

response = deepseek_client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print(response.choices[0].message.content)