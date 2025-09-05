#!/bin/bash
# 设置带时间戳的版本号
# shellcheck disable=SC2155
export TAG=$(date +%Y%m%d%H%M%S)

# 构建并启动容器
docker-compose up -d --build

# 清理旧镜像（保留最近的2个版本）
docker images --filter=reference='ocr-server:*' --format "{{.Tag}}" | \
grep -E '^[0-9]{14}$' | \
sort -r | \
awk -v keep=2 'NR > keep' | \
xargs -I {} docker rmi flet-server:{} 2>/dev/null