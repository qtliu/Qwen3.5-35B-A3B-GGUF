#!/bin/bash
# ==========================================================
# Qwen3.5-35B-A3B MoE 模型离线多卡部署脚本 (基于 llama.cpp)
# ==========================================================

# 1. 基础路径配置 (请根据实际情况修改路径)
LLAMA_SERVER_PATH="/path/to/llama.cpp/build/bin/llama-server"
MODEL_PATH="/path/to/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf"
LOG_FILE="qwen_moe_server.log"

# 2. 显卡配置：指定使用所有的 4 张显卡
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 3. 运行参数设置
# -m:           模型文件路径
# -c:           上下文窗口大小 (例如 8192 或 32768)
# -ngl 999:     将所有层卸载到 GPU (999 意为尽最大可能 offload)
# --host & -port: 绑定的 IP 和端口
# -t 8:         CPU 处理线程数（预处理时使用，根据 CPU 物理核心数设置即可）
# -fa:          开启 Flash Attention (极大优化显存占用和长文本速度)
# -sm row:      对多显卡进行按行切分矩阵 (Split Matrix)，优化多卡并行效率 (可选参数)

echo "正在启动 Qwen3.5-35B-A3B MoE API 服务..."
echo "使用显卡: GPU $CUDA_VISIBLE_DEVICES"
echo "日志将输出至: $LOG_FILE"

nohup $LLAMA_SERVER_PATH \
    -m "$MODEL_PATH" \
    -c 8192 \
    -ngl 999 \
    --host 0.0.0.0 \
    --port 8000 \
    -t 8 \
    -fa auto \
    > "$LOG_FILE" 2>&1 &

# 4. 提取进程 ID 并提示
PID=$!
echo "服务已放入后台运行，进程 PID: $PID"
echo "可通过命令查看实时日志: tail -f $LOG_FILE"