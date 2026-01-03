# 使用 Python 基础镜像
FROM python:3.11-slim

# 安装系统依赖和 iverilog
RUN apt-get update && apt-get install -y \
    iverilog \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建 output 目录
RUN mkdir -p output/dut output/bdd output/testbench output/results output/quality_reports

# 暴露端口
EXPOSE 10000

# 启动命令
CMD ["python", "main.py"]