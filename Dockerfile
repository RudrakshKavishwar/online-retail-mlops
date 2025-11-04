FROM python:3.10-slim

WORKDIR /app

# system deps for some python packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

# default: run streamlit app
CMD ["streamlit", "run", "app_streamlit.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
