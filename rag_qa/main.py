import os, sys
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)


import os
import sys
from base.config import config
from base.logger import logger
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem
from openai import OpenAI  # 使用 OpenAI 接口


# 1. 初始化 OpenAI 客户端
client = OpenAI(
    base_url=config.OPENAI_BASE_URL,
    api_key=config.OPENAI_API_KEY,
)

# 2. 定义 LLM 调用函数
def call_llm(prompt):
    try:
        # 创建聊天完成请求
        completion = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个有用的助手，能够根据用户输入的Prompt严格执行并返回可靠的结果"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1
        )
        # 返回完成结果
        return completion.choices[0].message.content if completion.choices else "错误: LLM返回无效响应"
    except Exception as e:
        logger.error(f"LLM API (call_llm) 调用失败: {e}")
        return f"错误: 调用LLM失败 - {e}"


def main():
    # 以下两步，只是为了填充类的参数，暂时不做调用
    vector_store = VectorStore()

    system = RAGSystem(vector_store, call_llm)

    while True:
        query = input("请输入您的问题（输入'exit'退出）：")
        if query.lower() == 'exit':
            break
        answer = system.generate_answer(query)
        print("AI:", answer)


if __name__ == "__main__":
    main()
