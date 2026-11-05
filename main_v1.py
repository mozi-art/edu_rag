# 导入 MySQL 系统组件，用于数据库操作和搜索
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MYSQLClient
from mysql_qa.retrieval.bm25_search import BM25Search
# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem
# 导入配置和日志工具，用于系统配置和日志记录
from base.config import config
from base.logger import logger
# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI
# 导入时间库，用于记录处理时间
import time



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


class QASystem:
    def __init__(self):
         # 初始化日志
        self.logger = logger
        # 初始化 MySQL 客户端
        self.mysql_client = MYSQLClient()
        # 初始化 Redis 客户端
        self.redis_client = RedisClient()
        # 初始化 BM25 搜索
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        # 初始化向量存储
        self.vector_store = VectorStore()
        # 初始化 RAG 系统
        self.rag_system = RAGSystem(self.vector_store, call_llm)

    def query(self, query, source_filter=None):
        # 定义查询方法，处理用户输入的查询
        start_time = time.time()  # 记录查询开始时间
        # 记录查询信息到日志
        self.logger.info(f"处理查询: '{query}'")
        # 执行 BM25 搜索，获取答案和是否需要 RAG 的标志
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)
        if answer:
            # 如果找到可靠答案，记录答案到日志
            self.logger.info(f"MySQL 答案: {answer}")
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 返回 MySQL 答案
            return answer
        elif need_rag:
            # 如果需要 RAG，记录回退信息到日志
            self.logger.info("无可靠 MySQL 答案，回退到 RAG")
            # 调用 RAG 系统生成答案，支持学科过滤
            answer = self.rag_system.generate_answer(query, source_filter=source_filter)
            # 记录 RAG 答案到日志
            self.logger.info(f"RAG 答案: {answer}")
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 返回 RAG 答案
            return answer
        else:
            # 如果无答案，记录信息到日志
            self.logger.info("未找到答案")
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 返回默认答案
            return "未找到答案"


if __name__ == "__main__":
    system = QASystem()
    while True:
        query = input("请输入您的问题（输入'exit'退出）：")
        if query.lower() == 'exit':
            break
        answer = system.query(query)
        print("AI:", answer)

