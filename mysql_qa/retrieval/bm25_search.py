import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)


from rank_bm25 import BM25Okapi
import numpy as np
from mysql_qa.utils.preprocess import preprocess_text
from mysql_qa.db.mysql_client import MYSQLClient
from mysql_qa.cache.redis_client import RedisClient
from base import logger

class BM25Search:
    def __init__(self,redis_client,mysql_client):
        # 客户端从外面进来，本类不负责建链
        self.logger=logger
        self.mysql_client = mysql_client
        self.redis_client = redis_client
        self.bm25 = None
        self.questions = None            # 分词后的问题
        self.original_questions = None   # 原始问题
        self._load_data()
    
    def _load_data(self):
        # 设置redis缓存名
        original_key = "original_questions"
        tokenized_key = "tokenized_questions"
        
        # 先读缓存
        self.original_questions = self.redis_client.get_data(original_key)
        tokenized_questions = self.redis_client.get_data(tokenized_key)

        # 如果没有缓存，就从mysql数据库读
        if not self.original_questions or not tokenized_questions:
            # fetch_questions 返回的是行元组，问题在 q[0]
            self.original_questions = [q[0] for q in self.mysql_client.fetch_questions()]
            if not self.original_questions:
                self.logger.error("未加载到问题")
                return
        
            tokenized_questions = [preprocess_text(q) for q in self.original_questions]
            # 缓存到redis(此时已是字符串列表)
            self.redis_client.set_data(tokenized_key, tokenized_questions)
            self.redis_client.set_data(original_key, self.original_questions)
            self.logger.info("从mysql数据库加载问题到redis")
        
        self.questions = tokenized_questions
        self.bm25 = BM25Okapi(self.questions)
        self.logger.info("BM25 模型初始化完成")
    
    def _softmax(self,scores):
        # 先减最大值，避免exp溢出
        exp_scores=np.exp(scores-np.max(scores))
        return exp_scores/exp_scores.sum()

    def search(self,query,threshold=0.85):
        # 返回，(答案，是否需要走RAG)
        if not query or not isinstance(query,str):
            self.logger.error("查询问题不能为空或不是字符串类型")
            return None,False
        
        # 这句问过,直接返回缓存的答案
        cached_answer=self.redis_client.get_answer(query)
        if cached_answer:
            return cached_answer, False
        
        try:
            # 分词 → BM25 → Softmax，取最高占比
            query_tokens=preprocess_text(query)
            # 计算bm25得分
            scores=self.bm25.get_scores(query_tokens)
            # 转换为softmax
            softmax_scores=self._softmax(scores)
            # 取最高占比
            best_idx=softmax_scores.argmax()
            best_score=softmax_scores[best_idx]

            print(f"最高占比问题索引: {best_idx}, 占比: {best_score:.3f}")

            # 判断最大分数是否超过阈值
            if best_score>=threshold:
                # 获取原始问题和答案
                original_question = self.original_questions[best_idx]

                # 原始问题的答案：先Redis,没有再Mysql，注意和上面的区别
                redis_result=self.redis_client.get_answer(original_question)

                if redis_result:
                    answer=redis_result
                    self.logger.info(f"从redis获取到了答案: {answer}")
                else:
                    answer=self.mysql_client.fetch_answer(original_question)
                    # 从mysql拿到答案后才缓存,避免把None写进redis
                    if answer:
                        self.redis_client.set_answer(original_question,answer)
                        self.logger.info(f"从mysql获取到答案并缓存到redis: {answer}")
                
                if answer:
                    # 缓存答案
                    self.redis_client.set_answer(query,answer)
                    self.logger.info(f"搜索成功，Softmax 相似度: {best_score:.3f}")
                    return answer, False
            
            self.logger.info(f"未找到可靠答案，最高 Softmax 相似度: {best_score:.3f}")
            return None, True
        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            return None, True


if __name__ == "__main__":
    # 实例化数据库客户端
    mysql_client = MYSQLClient()
    # 实例化 Redis 客户端
    redis_client = RedisClient()
    bm25_search = BM25Search(redis_client, mysql_client)
    print("实例化成功", bm25_search)

    # query_tokens = preprocess_text("测试的问题")

    # scores = bm25_search.bm25.get_scores(query_tokens)
    # print("BM25:", scores, len(scores))

    # # 对BM25得分进行softmax
    # scores = bm25_search._softmax(scores)
    # print("BM25:", scores, len(scores))

    # 测试搜索
    query = "不同模块下的类无法导入?"
    result = bm25_search.search(query)
    print("搜索结果:", result)


