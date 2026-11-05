# 导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain_core.prompts import PromptTemplate

# 定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:

    # 定义 RAG 提示模板
    @staticmethod
    def rag_prompt():
        # 创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            你是一个智能助手，帮助用户回答问题。  
            如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。  
            如果答案来源于检索到的文档，请在回答中说明。

            上下文: {context}  
            历史对话：{history_context}
            问题: {question}

            如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”  
            回答:  
            """,
            #   定义输入变量
            input_variables=["context", "history_context", "question", "phone"],
        )

    # 定义假设问题生成的 Prompt 模板
    @staticmethod
    def hyde_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
            历史对话：{history_context}
            问题: {query}  
            假设答案:  
            """,
            #   定义输入变量
            input_variables=["query", "history_context"],
        )

    #   定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询分解为多个简单子查询，每行一个子查询：  
            历史对话：{history_context}
            查询: {query}  
            子查询:  
            """,
            #   定义输入变量
            input_variables=["query", "history_context"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询简化为一个更简单的问题：  
            历史对话：{history_context}
            查询: {query}  
            简化问题:  
            """,
            #   定义输入变量
            input_variables=["query", "history_context"],
        )


if __name__ == "__main__":
    prompt_template = RAGPrompts.rag_prompt()
    print(prompt_template.format(context="你好", question="你好", phone="123-456"))
