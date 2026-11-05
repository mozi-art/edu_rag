import os,sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,project_dir)

# 导入数据库连接库
import pymysql
# 导入配置类
from base import config,logger
import pandas as pd



class MYSQLClient:
    def __init__(self):
        # 初始化日志
        self.logger = logger
        try:
            # 连接 MySQL 数据库
            self.connection = pymysql.connect(
                host=config.MYSQL_HOST,
                port=config.MYSQL_PORT,
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DATABASE
            )
            # 创建游标
            self.cursor = self.connection.cursor()
            # 记录连接成功
            self.logger.info("MySQL 连接成功")
        except pymysql.MySQLError as e:
            # 记录连接失败
            self.logger.error(f"MySQL 连接失败: {e}")
            raise
    
    # 创建表
    def create_table(self):
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS jpkb (
            id INT AUTO_INCREMENT PRIMARY KEY,
            subject_name VARCHAR(20),
            question VARCHAR(1000),
            answer VARCHAR(1000))
        '''
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            self.logger.info("表创建成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"表创建失败: {e}")
            raise
    
    # 插入数据
    def insert_data(self, csv_path):
        try:
            data = pd.read_csv(csv_path, encoding="utf-8")
            for _, row in data.iterrows():
                if self.fetch_answer(row['问题']):
                    continue
                insert_query = "INSERT INTO jpkb (subject_name, question, answer) VALUES (%s, %s, %s)"
                self.cursor.execute(insert_query, (row['学科名称'], row['问题'], row['答案']))
            self.connection.commit()
            self.logger.info("数据插入成功")
        except Exception as e:
            self.logger.error(f"数据插入失败: {e}")
            self.connection.rollback()
            raise
    
    # 查询所有问题列表
    def fetch_questions(self):
        # 获取所有问题
        try:
            # 执行查询
            self.cursor.execute("SELECT question FROM jpkb")
            # 获取结果
            results = self.cursor.fetchall()
            # 记录获取成功
            self.logger.info("成功获取问题")
            # 返回结果
            return results
        except pymysql.MySQLError as e:
            # 记录查询失败
            self.logger.error(f"查询失败: {e}")
            # 返回空列表
            return []
    
    # 根据问题查询答案,精准查询
    def fetch_answer(self, question):
        # 获取指定问题的答案
        try:
            # 执行查询
            # GROUP BY  HAVING
            # JOIN   LEFT JOIN   RIGHT JOIN
            # DISTINCT
            # ORDER BY id DESC  按id字段倒序排列
            self.cursor.execute("SELECT answer FROM jpkb WHERE question=%s ORDER BY id DESC", (question,))
            # 获取结果
            result = self.cursor.fetchone()   # first()
            # 返回答案或 None
            return result[0] if result else None
        except pymysql.MySQLError as e:
            # 记录答案获取失败
            self.logger.error(f"答案获取失败: {e}")
            # 返回 None
            return None

    # 关闭数据库连接
    def close(self):
        # 关闭数据库连接
        try:
            # 关闭游标
            self.cursor.close()
            # 关闭连接
            self.connection.close()
            # 记录关闭成功
            self.logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            # 记录关闭失败
            self.logger.error(f"关闭连接失败: {e}")



if __name__ == '__main__':
    # 实例化数据库客户端
    mysql_client = MYSQLClient()
    # 创建表
    mysql_client.create_table()
    # 插入数据
    csv_path = os.path.join(project_dir, 'mysql_qa', 'data', 'JP学科知识问答.csv')
    mysql_client.insert_data(csv_path)

    # 查询所有问题列表
    # results = mysql_client.fetch_questions()
    # questions = [item[0] for item in results]
    # print(questions)

    # 根据问题查询答案
    # answer = mysql_client.fetch_answer('Tomcat在docker里下载什么版本呢？')
    # print(answer)


    
    