import os,sys
project_dir=os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,project_dir)

# 导入Mysql客户端
from mysql_qa.db.mysql_client import MYSQLClient
# 导入Redis客户端
from mysql_qa.cache.redis_client import RedisClient
# 导入BM25搜索器
from mysql_qa.retrieval.bm25_search import BM25Search
# 导入日志记录器
from base import logger
import time

class MySQLQASystem:
    def __init__(self):
        # 初始化日志记录器
        self.logger=logger
        # 初始化Mysql客户端
        self.mysql_client=MYSQLClient()
        # 初始化Redis客户端
        self.redis_client=RedisClient()
        # 初始化BM25搜索器
        self.bm25_search=BM25Search(self.redis_client,self.mysql_client)
    
    def query(self, query):
        # 查询 MySQL 系统
        start_time = time.time()
        # 记录查询信息
        self.logger.info(f"处理查询: '{query}'")
        # 执行 BM25 搜索
        answer, _ = self.bm25_search.search(query, threshold=0.85)
        if answer:
            # 记录 MySQL 答案
            self.logger.info(f"MySQL 答案: {answer}")
        else:
            # 记录无答案
            self.logger.info("SQL中未找到答案, 需要调用RAG系统")
            # 设置默认答案
            answer = "SQL未找到答案"
        # 计算处理时间
        processing_time = time.time() - start_time
        # 记录处理时间
        self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
        # 返回答案
        return answer

def main():
    system=None
    try:
        # 初始化系统
        system=MySQLQASystem()
        # 测试查询
        query="不同模块下的类无法导入!"
        answer=system.query(query)
        print(answer)
    except Exception as e:
        logger.error(f"查询过程中发生错误: {e}")
        return "查询过程中发生错误"
    finally:
        # 只有系统初始化成功过,才需要关闭连接
        if system:
            system.mysql_client.close()

if __name__ == "__main__":
    main()