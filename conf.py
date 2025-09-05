import psutil

bind = "0.0.0.0:80"
workers = psutil.cpu_count()
worker_class = "uvicorn.workers.UvicornWorker"
