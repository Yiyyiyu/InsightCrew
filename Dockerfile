FROM python:3.11-slim

# 时区与基础工具：curl 用于健康检查，tzdata 让日志时间可读
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖单独一层：改代码不会让 pip 重装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 产品代码（.dockerignore 已排除 Markdown/tools/tests/scripts/*.db 等）
COPY . .

# SQLite 落盘目录：必须与 docker-compose 的 db-data 卷一致（/app/data）
RUN mkdir -p /app/data

# 非 root 运行（4C4G 上跑公网服务，没必要给容器 root）
RUN useradd --create-home --uid 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"

# 默认启动 API；worker 容器由 compose 覆盖 command 为 `python -m src.core.worker`
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
