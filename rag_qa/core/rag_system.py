# 1. 导入必要的模块
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from rag_qa.core.prompts import RAGPrompts
#   导入 time 模块，用于计算时间
import time
from base.config import config
from base.logger import logger
from rag_qa.core.query_classifier import QueryClassifier  # 导入查询分类器
from rag_qa.core.strategy_selector import StrategySelector  # 导入策略选择器


conf = config

#  定义 RAGSystem 类，封装 RAG 系统的核心逻辑
class RAGSystem:
    #  2.初始化方法，设置 RAG 系统的基本参数
    def __init__(self, vector_store, llm, llm_stream=None):
        #   设置向量数据库对象
        self.vector_store = vector_store
        #  设置大语言模型调用函数
        self.llm = llm
        #   设置流式大语言模型调用函数（可选）
        self.llm_stream = llm_stream
        #   获取 RAG 提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        #   初始化查询分类器
        self.query_classifier = QueryClassifier()
        #   初始化策略选择器
        self.strategy_selector = StrategySelector()

    # 3.定义私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(self, query, source_filter, history_context=""):
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        #   获取假设问题生成的 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt() # 使用 template 后缀区分
        #   调用大语言模型生成假设答案
        try:
            prompt = hyde_prompt_template.format(query=query, history_context=history_context)
            hypo_answer = self.llm(prompt).strip()
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            #   使用假设答案进行检索，并返回检索结果
            #   注意：HyDE 通常只用于生成检索向量，不一定需要 rerank 这一步，但这里复用了
            query_text = f"问题：{query} 假设答案：{hypo_answer}"
            return self.vector_store.hybrid_search_with_rerank(
                query_text, k=conf.CANDIDATE_M, source_filter=source_filter  # 使用 M
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []

    #  定义私有方法，使用回溯问题进行检索
    def _retrieve_with_backtracking(self, query, source_filter, history_context=""):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        #   获取回溯问题生成的 Prompt 模板
        backtrack_prompt_template = RAGPrompts.backtracking_prompt() # 使用 template 后缀区分
        try:
            #   调用大语言模型生成回溯问题
            prompt = backtrack_prompt_template.format(query=query, history_context=history_context)
            simplified_query = self.llm(prompt).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            #  使用回溯问题进行检索，并返回检索结果
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=conf.CANDIDATE_M, source_filter=source_filter  # 使用 M
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    # 定义私有方法，使用子查询进行检索
    def _retrieve_with_subqueries(self, query, source_filter, history_context=""):
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        #   获取子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt() # 使用 template 后缀区分
        try:
            # 调用大语言模型生成子查询列表
            subqueries_text = self.llm(subquery_prompt_template.format(
                query=query, 
                history_context=history_context
            )).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                 logger.warning("未能生成有效的子查询")
                 return []
            
            #   初始化空列表，用于存储所有子查询的检索结果
            all_docs = []
            #   遍历每个子查询
            for sub_q in subqueries:
                #   使用子查询进行检索，并将结果添加到列表中
                #   这里对每个子查询都执行了 hybrid search + rerank，开销可能较大
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=conf.CANDIDATE_M, source_filter=source_filter
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")
            
            #   更可靠的去重方式是基于文档内容或 ID
            unique_docs_dict = {doc.metadata['parent_id']: doc for doc in all_docs} # 基于内容去重
            unique_docs = list(unique_docs_dict.values())
            # print(unique_docs[0], len(unique_docs))

            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            return unique_docs

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []

    #  定义方法，检索并合并相关文档
    def retrieve_and_merge(self, query, source_filter=None, history_context=""): 
        #  选择查询策略
        strategy = self.strategy_selector.select_strategy(query)
        logger.info(f"查询问题 '{query}'， 选择的策略: {strategy}")

        # 策略判断，并且调用对应的召回方法
        docs = []
        if strategy == "假设问题检索":
            docs = self._retrieve_with_hyde(query, source_filter, history_context)
        elif strategy == "回溯问题检索":
            docs = self._retrieve_with_backtracking(query, source_filter, history_context)
        elif strategy == "子查询检索":
            docs = self._retrieve_with_subqueries(query, source_filter, history_context)
        else:
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            docs = self.vector_store.hybrid_search_with_rerank(
                query, k=conf.CANDIDATE_M, source_filter=source_filter
            )
        logger.info(f"召回到 {len(docs)} 个文档")
        return docs

    # 定义方法，生成答案
    def generate_answer(self, query, source_filter=None, history_context=""):
        
        # 判断查询类型
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果：{query_category} (查询: '{query}')")
        #   如果查询属于“通用知识”类别，则直接使用 LLM 回答
        if query_category == "通用知识":
            prompt_input = self.rag_prompt.format(
                context="", history_context=history_context, question=query, phone=config.CUSTOMER_SERVICE_PHONE
            )
            answer = self.llm(prompt_input)
            return answer

        logger.info("查询为专业咨询，执行 RAG 流程")
        # 选择检索策略
        docs = self.retrieve_and_merge(query, source_filter, history_context)
        # 拼接上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        # 生成 RAG 提示
        prompt_input = self.rag_prompt.format(
            context=context, history_context=history_context, question=query, phone=config.CUSTOMER_SERVICE_PHONE
        )
        answer = self.llm(prompt_input)
        return answer

    # 定义流式答案生成方法，检索逻辑与 generate_answer 一致，最终答案逐段产出
    def generate_answer_stream(self, query, source_filter=None, history_context=""):
        # 判断查询类型
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果：{query_category} (查询: '{query}')")

        # 如果查询属于“通用知识”类别，则直接使用 LLM 回答
        if query_category == "通用知识":
            prompt_input = self.rag_prompt.format(
                context="", history_context=history_context, question=query, phone=config.CUSTOMER_SERVICE_PHONE
            )
        else:
            logger.info("查询为专业咨询，执行 RAG 流程")
            # 选择检索策略
            docs = self.retrieve_and_merge(query, source_filter, history_context)
            # 拼接上下文
            context = "\n\n".join([doc.page_content for doc in docs])
            # 生成 RAG 提示
            prompt_input = self.rag_prompt.format(
                context=context, history_context=history_context, question=query, phone=config.CUSTOMER_SERVICE_PHONE
            )

        # 优先使用流式调用；未注入时退化为整段生成后切片
        if self.llm_stream:
            yield from self.llm_stream(prompt_input)
        else:
            answer = self.llm(prompt_input)
            for i in range(0, len(answer), 20):
                yield answer[i:i + 20]


if __name__ == "__main__":
    # 以下两步，只是为了填充类的参数，暂时不做调用
    from rag_qa.core.vector_store import VectorStore
    vector_store = VectorStore()

    from rag_qa.core.strategy_selector import call_llm as llm

    system = RAGSystem(vector_store, llm)

    # # 测试过程，自己调用方法
    # docs = system.retrieve_and_merge("人工智能专业毕业后的薪资待遇如何？今天的天气怎么样？")
    # print(len(docs))

    # 直接生成答案
    answer = system.generate_answer("大模型的就业方向有哪些？")
    print(answer)



