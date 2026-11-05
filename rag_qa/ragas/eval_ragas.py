import json
import math
import random
import sys
import types
from datetime import datetime
from pathlib import Path

RAGAS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAGAS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ragas / langchain 可能间接导入 vertexai，缺包时用空壳占位避免 ImportError
_VERTEXAI = "langchain_community.chat_models.vertexai"
if _VERTEXAI not in sys.modules:
    _stub = types.ModuleType(_VERTEXAI)
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[_VERTEXAI] = _stub

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from base.config import config
from base.logger import logger
from rag_qa.core.rag_system import RAGSystem
from rag_qa.core.strategy_selector import call_llm
from rag_qa.core.vector_store import VectorStore

DATA_PATH = RAGAS_DIR / "data" / "ragas_eval.json"
OUT_DIR = RAGAS_DIR / "output"   # 评估结果输出目录，会自动创建


class _BGEEmbeddings:
    """用项目已有的 BGE-M3 提供 RAGAS 需要的 embed_query / embed_documents。"""

    def __init__(self, embedding_function):
        self.embedding_function = embedding_function

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        all_vectors = []
        step = 16
        for i in range(0, len(texts), step):
            batch = texts[i : i + step]
            dense = self.embedding_function(batch)["dense"]
            all_vectors.extend([list(vec) for vec in dense])
        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _judge_llm():
    # ragas 0.2.x 的 llm_factory 不支持 client=；直接包一层 Langchain ChatOpenAI
    chat = ChatOpenAI(
        model="deepseek-v4.1-flash",
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        temperature=0.1,
        extra_body={"enable_thinking": False},
    )
    return LangchainLLMWrapper(chat)


# 因为给的数据只有两列，但是要评估的数据，是四列，所以需要填充。
# 填充规则：question -> contexts -> answer -> ground_truth
def _fill_row(rag: RAGSystem, item: dict) -> dict:
    question = item["question"]
    docs = rag.retrieve_and_merge(question)          # 只检索这一次
    contexts = [d.page_content for d in docs] if docs else []

    if contexts:
        # 用同一批 contexts 拼 prompt 生成 answer，绕过 generate_answer 的二次检索
        context = "\n\n".join(contexts)
        prompt_input = rag.rag_prompt.format(
            context=context,
            history_context="",  # 评估为单轮提问，无历史对话
            question=question,
            phone=config.CUSTOMER_SERVICE_PHONE,
        )
        answer = rag.llm(prompt_input)
    else:
        answer = "未检索到相关内容"
    ...

    return {
        "question": question,
        "ground_truth": item["ground_truth"],
        "contexts": contexts,
        "answer": answer,
    }


def _scores_dict(result) -> dict:
    """RAGAS EvaluationResult 不支持 dict(result)，需读聚合分数字典。"""
    raw = getattr(result, "_repr_dict", None) or {}
    return {
        k: (None if (isinstance(v, float) and math.isnan(v)) else float(v))
        for k, v in raw.items()
    }


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"评测数据不存在: {DATA_PATH}")

    items = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    SAMPLE_N = 20  # 抽样条数，改为 None 则跑全量
    if SAMPLE_N and len(items) > SAMPLE_N:
        random.seed(42)                  # 固定随机种子，保证两次评估抽样一致
        items = random.sample(items, SAMPLE_N)
    logger.info(f"开始 RAGAS 评估，样本数={len(items)}")

    vector_store = VectorStore()
    rag = RAGSystem(vector_store, call_llm)
    rows = [_fill_row(rag, it) for it in items]

    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "contexts": [r["contexts"] for r in rows],
            "ground_truth": [r["ground_truth"] for r in rows],
        }
    )

    answer_relevancy.strictness = 1
    result = evaluate(
        dataset=dataset,
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
        llm=_judge_llm(),
        embeddings=_BGEEmbeddings(vector_store.embedding_function),
    )
    scores = _scores_dict(result)
    logger.info(f"RAGAS 结果: {scores}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"ragas_result_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scores": scores,
        "n_samples": len(rows),
        "rows": [
            {
                "question": r["question"],
                "n_contexts": len(r["contexts"]),
                "answer": r["answer"],
                "ground_truth": r["ground_truth"],
            }
            for r in rows
        ],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"已写出评估结果: {out_path}")
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print(f"result -> {out_path}")


if __name__ == "__main__":
    main()
