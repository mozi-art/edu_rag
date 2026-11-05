# 导入 MySQL 系统组件，用于数据库操作和搜索
import pymysql

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
import uuid


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


# 3. 定义流式 LLM 调用函数
def call_llm_stream(prompt):
    """以流式方式调用 LLM，逐段返回生成内容；异常时返回错误提示，避免中断 WebSocket"""
    try:
        # 创建流式聊天完成请求
        stream = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个有用的助手，能够根据用户输入的Prompt严格执行并返回可靠的结果"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            stream=True
        )
        # 逐个返回增量内容
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
    except Exception as e:
        logger.error(f"LLM API (call_llm_stream) 调用失败: {e}")
        yield f"错误: 调用LLM失败 - {e}"


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
        # 初始化 RAG 系统（同时注入同步与流式两种 LLM 调用方式）
        self.rag_system = RAGSystem(self.vector_store, call_llm, call_llm_stream)
        # 初始化对话历史表，用于存储会话记录
        self.init_conversation_table()

    def init_conversation_table(self):
        """初始化MySQL中的conversations表，用于存储对话历史"""
        try:
            # 创建 conversations 表，包含会话 ID、问题、答案和时间戳
            self.mysql_client.cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    INDEX idx_session_id (session_id)
                )
            """)
            # 提交数据库事务
            self.mysql_client.connection.commit()
            # 记录表初始化成功的日志
            self.logger.info("对话历史表初始化成功")
        except pymysql.MySQLError as e:
            # 记录表初始化失败的错误日志
            self.logger.error(f"初始化对话历史表失败: {e}")
            # 抛出异常，终止初始化
            raise

    def get_session_history(self, session_id: str, limit: int = 5) -> list:
        """获取最近5轮对话历史"""
        try:
            # 执行 SQL 查询，获取最近 5 轮对话
            self.mysql_client.cursor.execute("""
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (session_id, limit))
            # 将查询结果转换为字典列表
            history = [{"question": row[0], "answer": row[1]} for row in self.mysql_client.cursor.fetchall()]
            # 反转结果，按时间正序返回
            return history[::-1]
        except pymysql.MySQLError as e:
            # 记录查询失败的错误日志
            self.logger.error(f"获取对话历史失败: {e}")
            # 返回空列表
            return []

    def add_session_history(self, session_id: str, question: str, answer: str) -> list:
        try:
            # 插入新的对话记录
            self.mysql_client.cursor.execute("""
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
            """, (session_id, question, answer))
            # 提交数据库事务
            self.mysql_client.connection.commit()
            # 记录插入成功的日志
            self.logger.info(f"成功添加对话记录: {session_id}, {question}, {answer}")
        except pymysql.MySQLError as e:
            # 记录插入失败的错误日志
            self.logger.error(f"添加对话记录失败: {e}")
            # 抛出异常，终止操作
            raise

    def delete_session_history(self, session_id: str) -> bool:
        """删除指定会话历史"""
        try:
            # 删除指定 session_id 的所有对话记录
            self.mysql_client.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s
            """, (session_id,))
            # 提交事务
            self.mysql_client.connection.commit()
            # 记录清除成功的日志
            self.logger.info(f"会话 {session_id} 历史已清除")
            # 返回 True 表示成功
            return True
        except pymysql.MySQLError as e:
            # 记录清除失败的错误日志
            self.logger.error(f"清除会话历史失败: {e}")
            # 回滚事务
            self.mysql_client.connection.rollback()
            # 返回 False 表示失败
            return False
    

    def query(self, query, source_filter=None, session_id: str = None):

        # 检查会话 ID 是否为空
        if not session_id:
            # 生成新的会话 ID
            session_id = str(uuid.uuid4())

        # 定义查询方法，处理用户输入的查询
        start_time = time.time()  # 记录查询开始时间
        # 记录查询信息到日志
        self.logger.info(f"处理查询: '{query}'")
        # 执行 BM25 搜索，获取答案和是否需要 RAG 的标志
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)
        if answer:
            # 记录对话历史
            self.add_session_history(session_id, query, answer)
            # 如果找到可靠答案，记录答案到日志
            self.logger.info(f"MySQL 答案: {answer}")
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 返回 MySQL 答案
            return answer

        elif need_rag:
            # 前面的faq环节，没有历史对话消息，只做单次的对话处理
            histories = self.get_session_history(session_id, 5)
            history_context = "\n\n".join([f"问题：{item['question']}\n回答：{item['answer']}" for item in histories])

            # 如果需要 RAG，记录回退信息到日志
            self.logger.info("无可靠 MySQL 答案，回退到 RAG")
            # 调用 RAG 系统生成答案，支持学科过滤
            answer = self.rag_system.generate_answer(query, source_filter=source_filter, history_context=history_context)
            # 记录对话历史
            self.add_session_history(session_id, query, answer)
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

    def query_stream(self, query, source_filter=None, session_id: str = None):
        """流式查询：按 (token, is_complete) 逐段返回，供 WebSocket 使用"""
        # 检查会话 ID 是否为空
        if not session_id:
            # 生成新的会话 ID
            session_id = str(uuid.uuid4())

        # 记录查询开始时间
        start_time = time.time()
        # 记录查询信息到日志
        self.logger.info(f"处理流式查询: '{query}'")

        # 执行 BM25 搜索，获取答案和是否需要 RAG 的标志
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)

        if answer:
            # 记录对话历史
            self.add_session_history(session_id, query, answer)
            # 记录答案到日志
            self.logger.info(f"MySQL 答案: {answer}")
            # 按固定长度切片，模拟流式输出
            for i in range(0, len(answer), 20):
                yield answer[i:i + 20], False
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {time.time() - start_time:.2f}秒")
            # 发送结束标志
            yield "", True
            return

        if need_rag:
            # 获取最近对话历史
            histories = self.get_session_history(session_id, 5)
            history_context = "\n\n".join([f"问题：{item['question']}\n回答：{item['answer']}" for item in histories])

            # 记录回退信息到日志
            self.logger.info("无可靠 MySQL 答案，回退到 RAG（流式）")
            # 调用 RAG 系统流式生成答案，并累积完整答案用于记录历史
            collected_answer = ""
            for token in self.rag_system.generate_answer_stream(
                query, source_filter=source_filter, history_context=history_context
            ):
                collected_answer += token
                yield token, False

            # 记录对话历史
            self.add_session_history(session_id, query, collected_answer)
            # 记录 RAG 答案到日志
            self.logger.info(f"RAG 答案: {collected_answer}")
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {time.time() - start_time:.2f}秒")
            # 发送结束标志
            yield "", True
            return

        # 如果无答案，记录信息到日志
        self.logger.info("未找到答案")
        # 记录处理时间到日志
        self.logger.info(f"查询处理耗时 {time.time() - start_time:.2f}秒")
        # 返回默认答案，并标记完成
        yield "未找到答案", True


qa_system = QASystem()

if __name__ == "__main__":
    system = QASystem()

    # 插入数据
    # system.add_session_history("123", "你好", "你好，我是123")
    # system.add_session_history("123", "我给你起个名字叫贾维斯", "好的好的")

    # 获取数据
    # history = system.get_session_history("123", 5)
    # print(history)

    # 删除数据
    # system.delete_session_history("123")

    session_id = "66ea1960-cf7a-4ddc-8488-69921f2c5144"

    while True:
        query = input("请输入您的问题（输入'exit'退出）：")
        if query.lower() == 'exit':
            break
        answer = system.query(query, session_id=session_id)
        print("AI:", answer)

