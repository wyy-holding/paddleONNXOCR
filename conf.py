import os

bind = "0.0.0.0:80"
workers = os.cpu_count()
worker_class = "uvicorn.workers.UvicornWorker"
preload_app = True
