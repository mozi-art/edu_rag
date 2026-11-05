FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# cv2 需要 libgl1/libglib2.0-0，torch 需要 libgomp1，unstructured 需要 libmagic1
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libgomp1 libmagic1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# CPU 版 torch：pytorch 索引只作为附加源，其余依赖仍走主索引（清华源）
# 注意不能用 --index-url，否则 torch 的依赖也会去 pytorch 索引找，导致解析失败
RUN pip install --no-cache-dir --retries 8 --timeout 60 \
      torch==2.10.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --retries 8 --timeout 60 -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]