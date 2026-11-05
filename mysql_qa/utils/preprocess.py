import jieba
from pathlib import Path

def _get_stopwords():
    file_path = Path(__file__).parent.parent / "data/cn_stopwords.txt"
    with open(file_path,encoding="utf-8") as f:
        stopwords = set(f.read().splitlines())
    return stopwords

stopwords = _get_stopwords()

def preprocess_text(text):
    text = text.lower()
    # 对文本进行分词
    tokens = jieba.lcut(text)
    # 移除停用词
    filtered_tokens = [token for token in tokens if token not in stopwords and token.strip()]
    return filtered_tokens


if __name__ == "__main__":
    # 读取停用词
    print(preprocess_text("你好，我是小明，很高兴认识你"))
   