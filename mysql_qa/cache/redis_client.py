import os,sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_dir)

import redis
import json
from base import config

class RedisClient:
    def __init__(self):
        self.client = redis.StrictRedis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD,
            db=config.REDIS_DB,
            decode_responses=True
        )
    
        print("Redis连接成功")
    
    def set_data(self, key, value):
        # 存储数据格式为Json字符串
        self.client.set(key, json.dumps(value))
        print(f"set_data存储数据到Redis:{key}")
    
    def get_data(self,key):
        # 从Redis获取数据并解析为Json对象
        data = self.client.get(key)
        
        return json.loads(data) if data else None
    
    def delete_data(self,key):
        # 删除Redis中的数据
        self.client.delete(key)
        print(f"delete_data删除Redis数据:{key}")

    def delete_all(self):
        # 删除Redis中的所有数据
        self.client.flushdb()
        print("delete_all删除Redis所有数据")


        

    def set_answer(self, question, answer):
        # 存储数据格式为Json字符串
        self.set_data(f"answer:{question}", answer)
        print(f"set_answer存储数据到Redis:{question}")
    
    def get_answer(self, question):
        # 从Redis获取数据并解析为Json对象
        answer = self.get_data(f"answer:{question}")
        if answer:
            print(f"get_answer从Redis获取数据:{question}")
            return answer
    
    def delete_answer(self, question):
        # 删除Redis中的数据
        self.delete_data(f"answer:{question}")
        print(f"delete_answer删除Redis数据:{question}")
