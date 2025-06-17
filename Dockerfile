# 使用官方 Python 映像檔作為基礎
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 以安裝相依套件
COPY requirements.txt ./

# 安裝相依套件
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案檔案到容器中
COPY . .

# 設定 Streamlit 預設執行 port
EXPOSE 8501

# 執行 Streamlit 應用程式
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
