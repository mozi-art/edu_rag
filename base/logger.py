import os,sys
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,project_dir)


# 导入日志库
import logging
# 导入路径操作库
import os
# 导入配置类
from base.config import config


def setup_logging(log_file=config.LOG_FILE):
    # 获取日志器
    logger = logging.getLogger("EduRAG")
    # 设置日志级别
    logger.setLevel(logging.DEBUG)
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        # 设置文件处理器级别,分闸
        file_handler.setLevel(logging.INFO)
        # 创建控制台处理器,分闸
        console_handler = logging.StreamHandler()
        # 设置控制台处理器级别
        console_handler.setLevel(logging.DEBUG)
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # 为文件处理器设置格式
        file_handler.setFormatter(formatter)
        # 为控制台处理器设置格式
        console_handler.setFormatter(formatter)
        # 添加文件处理器
        logger.addHandler(file_handler)
        # 添加控制台处理器
        logger.addHandler(console_handler)
    # 返回日志器
    return logger

# 初始化日志器
logger = setup_logging()

if __name__ == '__main__':
    logger.info('这是一条 info日志')
    logger.debug('这是一条 debug日志')