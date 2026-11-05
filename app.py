import os
project_dir = os.path.dirname(__file__)

from fastapi import FastAPI
import uvicorn
from router import router

app = FastAPI()
# 挂载路由（页面、健康检查、会话、HTTP 查询与 WebSocket 流式接口均在 router 中定义）
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000, reload=True)
