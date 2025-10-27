#!/bin/bash
# 设置带时间戳的版本号
# shellcheck disable=SC2155
export TAG=$(date +%Y%m%d%H%M%S)
docker-compose down
# 构建并启动容器
docker-compose up -d --build

# 清理旧镜像（保留最近的2个版本）
docker rmi $(docker images 'ocr-server' --format '{{.ID}}' | tail -n +3)
