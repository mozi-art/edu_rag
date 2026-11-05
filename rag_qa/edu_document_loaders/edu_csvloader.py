"""
CSV文件加载器介绍:
    用于加载csv格式的数据文件, csv文件第一行通常是表头(列名),
    后面每一行是一条完整的数据记录.
    本加载器的要求: 返回 '表头: 内容' 格式的数据, 例如:
        科目: Java学科
        问题: Java中如何实现匿名函数
        答案: xxxxx
    即把每一行数据与其对应的表头拼接成可读的文本.
"""

import os, sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_dir)


import csv
from typing import Iterator
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class EduCSVLoader(BaseLoader):
    """一个按 '表头: 内容' 格式解析csv文件的文档加载器."""

    def __init__(self, file_path: str, encoding: str = "utf-8") -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the csv file to load.
            encoding: The encoding of the csv file.
        """
        self.file_path = file_path
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        # <-- Does not take any arguments
        """A lazy loader that reads the csv file row by row.

        每一行数据生成一个Document, page_content为 '表头: 内容' 格式的文本.
        """
        with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
            # DictReader会把第一行作为表头, 之后每行数据自动映射成 {表头: 内容} 的字典
            reader = csv.DictReader(csvfile)
            for row in reader:
                # 按表头顺序拼接: '表头: 内容', 多个字段之间用换行分隔
                lines = [f"{field}: {row.get(field, '')}" for field in reader.fieldnames]
                page_content = "\n".join(lines)
                # source不变, 把该行label_id对应的数字一并放入metadata
                metadata = {"source": self.file_path}
                if row.get("label_id"):
                    metadata["label_id"] = int(row["label_id"])
                yield Document(page_content=page_content, metadata=metadata)


if __name__ == '__main__':
    # 使用原始字符串r''表示windows路径, 避免\被当作转义字符
    csv_loader = CSVLoader(file_path=r"F:\RAG_data\5-素材\train_sample.csv")
    docs = csv_loader.load()

    print(type(docs))
    for doc in docs:
        print(doc)
