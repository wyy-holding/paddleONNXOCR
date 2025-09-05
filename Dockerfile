FROM python:3.12-slim-bookworm

# 设置时区
ENV TZ=Asia/Shanghai

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "-c", "conf.py", "main:app"]