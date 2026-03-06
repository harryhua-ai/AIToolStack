#!/bin/bash
# NE301 更新和服务重启脚本

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 1. 检查是否在 AIToolStack 目录
AITOOLSTACK_DIR="/home/harry/Desktop/github/AIToolStack"
cd "$AITOOLSTACK_DIR" || {
    log_error "无法进入 AIToolStack 目录: $AITOOLSTACK_DIR"
    exit 1
}

# 2. 更新 NE301 仓库
log_info "正在更新 NE301 仓库..."
cd ne301

# 检查是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    log_warn "检测到 NE301 仓库有未提交的更改"
    git status --short
    echo
    read -p "是否要暂存这些更改？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git stash push -m "自动暂存：$(date '+%Y%m%d_%H%M%S')"
        log_info "更改已暂存"
    fi
fi

# 拉取最新代码
log_info "从远程仓库拉取最新代码..."
git fetch origin
git checkout main
git pull origin main

log_info "NE301 仓库已更新"
git log -1 --oneline

cd "$AITOOLSTACK_DIR"

# 3. 停止所有容器
log_info "正在停止容器..."
docker compose down

# 4. 拉取最新的 Docker 镜像
log_info "正在拉取最新的 Docker 镜像..."
docker pull camthink/aitoolstack:latest
docker pull camthink/ne301-dev:latest

# 5. 启动服务
log_info "正在启动服务..."
docker compose up -d

# 6. 等待服务启动
log_info "等待服务启动..."
sleep 10

# 7. 验证服务状态
log_info "验证服务状态..."
docker compose ps

# 检查健康状态
if docker ps | grep -q "camthink-aitoolstack.*healthy"; then
    log_info "✓ 服务已成功启动并健康运行"
else
    log_warn "服务启动中，请稍后检查健康状态"
    log_info "可以使用以下命令查看状态："
    echo "  docker ps"
    echo "  docker logs camthink-aitoolstack"
fi

log_info "更新完成！"
echo ""
echo "访问地址: http://localhost:8000"
