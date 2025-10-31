#!/bin/bash

# 自动选择 docker compose 命令
if docker compose version &>/dev/null; then
    DOCKER_COMPOSE="docker compose"
elif docker-compose --version &>/dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "Error: Neither 'docker compose' nor 'docker-compose' is available." >&2
    exit 1
fi

export TAG=$(date +%Y%m%d%H%M%S)

$DOCKER_COMPOSE down
$DOCKER_COMPOSE up -d --build

# === 清理旧 TAG（保留最新2个）===
echo "Cleaning up old 'ocr-server' tags (keeping latest 2)..."

tags=$(docker images 'ocr-server' --format '{{.Tag}}' --filter "dangling=false" 2>/dev/null | grep -v '^<none>$' | head -n -0)

if [ -z "$tags" ]; then
    echo "No tags found."
else
    total=$(echo "$tags" | wc -l)
    if [ "$total" -gt 2 ]; then
        old_tags=$(echo "$tags" | tail -n +3)
        echo "$old_tags" | while IFS= read -r tag; do
            if [ -n "$tag" ]; then
                echo "Removing tag: ocr-server:$tag"
                docker rmi "ocr-server:$tag" 2>/dev/null || true
            fi
        done
    else
        echo "Only $total tag(s) exist. Nothing to remove."
    fi
fi
