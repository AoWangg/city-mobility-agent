## 环境要求
- Python 3.11

### 安装依赖
pip install -r requirements.txt

### 运行
开发环境：uvicorn app:app --reload
生产环境：
gunicorn -w 8 -k uvicorn.workers.UvicornWorker app:app \
--bind 0.0.0.0:8000 \
--timeout 20000 \
--access-logfile /var/log/gunicorn/access.log \
--error-logfile /var/log/gunicorn/error.log \
--log-level info \
--daemon



### swagger
http://127.0.0.1:8000/docs#/