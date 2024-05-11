release: python nltk_download.py
web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app