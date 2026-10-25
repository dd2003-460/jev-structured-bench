# jev-structured-bench — benchmark runner container
#
# 完整开箱即用（两端都是容器）：
#   docker compose up -d                                    # 1. vLLM 服务器
#   cp .env.example .env && $EDITOR .env                    # 2. 配置（VLLM_BASE 等）
#   docker build -t jev-bench .                             # 3. 构建 benchmark 容器
#   docker run --rm --env-file .env jev-bench                # 4. 一键跑全量
#   docker run --rm --env-file .env jev-bench --limit 30     #    或 5 分钟冒烟
#
# 也可以直接宿主机跑（python3 >= 3.9 + curl，零 pip 依赖）：
#   ./scripts/reproduce.sh
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /bench
COPY harness/ ./harness/
COPY scripts/ ./scripts/
COPY data/ ./data/

# 容器里 VLLM_BASE 指向宿主机的 vLLM：用 --env-file .env 传
# （docker compose 同机部署时写 http://host.docker.internal:8000 或主机 IP）
ENTRYPOINT ["./scripts/reproduce.sh"]
