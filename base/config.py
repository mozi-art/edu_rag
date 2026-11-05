import configparser
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()



# 项目根目录
project_dir = Path(__file__).parent.parent

class Config:
    def __init__(self,config_file="config.ini"):
        # 将配置文件补充为绝对路径
        config_file=project_dir / config_file
        # 创建配置解析器
        self.config = configparser.ConfigParser()
        # 读取配置文件
        self.config.read(config_file)

        # 项目根目录
        self.PROJECT_DIR = project_dir
        self.DATA_DIR = os.path.join(self.PROJECT_DIR, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_DIR, 'rag_qa/models')
        self.EDU_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_DIR, 'rag_qa/edu_document_loaders')
       
        # MySQL 配置
        # MySQL 主机地址, fallback 默认值
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost')) 
        # MySQL 端口
        self.MYSQL_PORT = int(os.getenv('MYSQL_PORT', self.config.getint('mysql', 'port', fallback=3306)))  
        # MySQL 用户名
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        # MySQL 密码
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='123456') 
        # MySQL 数据库名
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='subjects_kg')  


        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost')) 
        # Redis 端口
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.getint('redis', 'port', fallback=6379)))  
        # Redis 密码
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='1234') 
        # Redis 数据库编号
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)  

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        # Milvus 端口
        self.MILVUS_PORT = int(os.getenv('MILVUS_PORT', self.config.getint('milvus', 'port', fallback=19530)))
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name', fallback='itcast')
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='edurag_collection')

        # LLM 配置
        # LLM 模型名
        self.MODEL_NAME = os.getenv('MODEL_NAME', '')
        # OPENAI API 密钥
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        # OPENAI API 地址
        self.OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', '')


        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)


        # 日志文件路径
        log_file = self.config.get('logger', 'log_file', fallback='logs/app.log')
        # 日志路径加工
        self.LOG_FILE = project_dir / log_file
        # 日志目录创建
        if not self.LOG_FILE.parent.exists():
            self.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
         # 应用配置
        # 有效来源列表
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback='["ai", "java", "test", "ops", "bigdata"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='1234-5678')


config = Config()

if __name__ == "__main__":
    print(config.LOG_FILE)
