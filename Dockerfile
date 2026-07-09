# ds4 API Server Docker Image for Railway/Render
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ python3 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only needed source files
COPY ds4pro/ds4pro.hpp ds4pro/
COPY ds4pro/ds4pro_train.hpp ds4pro/
COPY ds4pro/gguf_loader.hpp ds4pro/
COPY ds4pro/train_main.cpp ds4pro/
COPY ds4pro/ds4_final.gguf ds4pro/
COPY web/server.py web/

# Compile inference engine
RUN cd ds4pro && g++ -std=c++17 -O3 -pthread -o train_ds4 train_main.cpp

EXPOSE 8080

CMD ["python3", "web/server.py", "--port", "8080"]
