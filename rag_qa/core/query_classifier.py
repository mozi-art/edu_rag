# 导入标准库
import os, sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_dir)

# 导入 PyTorch
import torch
# 导入日志
from base.logger import logger
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification


class QueryClassifier:
    # 1、初始化分类器
    def __init__(self, model_path='models/bert_query_classifier'):
        # 模型路径
        self.model_path = os.path.join(project_dir, 'rag_qa', model_path)
        # 加载 BERT 分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        # 初始化模型
        self.model = None
        # 确定设备（GPU 或 CPU）
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu" )
        # 记录设备信息
        logger.info(f"使用设备: {self.device}")
        # 加载模型
        self.load_model()

    # 2、加载模型
    def load_model(self):
        # 加载预训练模型
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        # 将模型移到指定设备
        self.model.to(self.device)
        # 记录加载成功的日志
        logger.info(f"加载模型: {self.model_path}")

    # 3、预测查询类别
    def predict_category(self, query):
        # 对查询进行编码
        encoding = self.tokenizer(query, truncation=True, padding=True, max_length=128, return_tensors="pt")
        # 将编码移到指定设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        # 不计算梯度，进行预测
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(**encoding)
            # 获取预测结果
            prediction = torch.argmax(outputs.logits, dim=1).item()
        # 根据预测结果返回类别
        return "专业咨询" if prediction == 1 else "通用知识"


if __name__ == "__main__":
    classifier = QueryClassifier()
    print(classifier.predict_category("袁文杰是gay"))

