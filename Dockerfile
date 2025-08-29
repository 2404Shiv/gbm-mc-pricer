FROM ubuntu:22.04
RUN apt-get update && apt-get install -y build-essential cmake clang && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
CMD ["./build/pricer"]
