FROM python:3.10-slim

WORKDIR /app

# 先装依赖，便于利用镜像层缓存
COPY requirements.txt .
RUN pip install -U pip \
 && pip install -r requirements.txt

# 再拷贝业务代码
COPY . .

EXPOSE 8000

# 启动 Web 服务（模块路径按项目修改，如 app:app）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]