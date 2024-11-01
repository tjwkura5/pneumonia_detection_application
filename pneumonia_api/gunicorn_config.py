import multiprocessing

# Gunicorn configuration
bind = "0.0.0.0:5000"
workers = multiprocessing.cpu_count() * 2 + 1
timeout = 120
keepalive = 5
worker_class = "sync"
worker_connections = 1000
accesslog = "access.log"
errorlog = "error.log"
loglevel = "info"