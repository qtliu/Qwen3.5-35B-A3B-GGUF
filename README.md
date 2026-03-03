# Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf 模型部署

Qwen3.5-35B-A3B 是阿里刚发布不久的最新款 MoE（混合专家）架构模型。为了支持这个全新的结构，llama.cpp 官方在 **2026年2月底** 才刚刚向底层 C++ 核心代码中合并了名为 qwen35moe 的架构支持。如果直接拉取 abetlen 仓库提供的 Python 预编译 Wheel 包（llama-cpp-python）。**会由于该模型太新，第三方 Python 包的更新速度滞后**，导致下载到的预编译包里，底层的 C++ 核心还没有包含 qwen35moe 的代码，因此根本不认识这个模型。

最佳解决方案是弃用滞后的 Python 包，直连官方原生 C++ 服务。**业内在部署这种极新模型时，最标准的做法是：直接使用 llama.cpp 官方每日更新（Nightly）的原生 C++ 二进制服务程序（llama-server）。该服务能提供完全兼容 OpenAI 格式的 API 接口。**

------

### 第一阶段：准备工作

1. **下载最新的 llama.cpp 源码**
   由于第三方包更新滞后，需要直接获取原作者最新的主分支源码。打开终端：

   ```
   git clone https://github.com/ggerganov/llama.cpp.git
   # 如果没有 git，直接去 github 页面点击 "Code -> Download ZIP" 并解压
   ```

2. **打包源码**：将 llama.cpp 文件夹打包（如 tar -czvf llama.cpp.tar.gz llama.cpp/）。

3. **拷贝至内网**：通过 U盘 / 堡垒机 将源码压缩包和 Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf 模型（[bartowski/Qwen_Qwen3.5-35B-A3B-GGUF at main](https://hf-mirror.com/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF/tree/main)）文件传入离线服务器。

------



### 第二阶段：编译支持新 MoE 的核心（在离线服务器上进行）

假设离线服务器已安装好了 NVIDIA 驱动和 CUDA Toolkit（nvcc --version 正常工作），并且有基本的编译工具（gcc, make, cmake）。

1. 解压并进入源码目录：

   ```
   tar -xzvf llama.cpp.tar.gz
   cd llama.cpp
   ```

2. 使用 CMake 开启 CUDA 支持进行编译（注：最新版 llama.cpp 已使用 GGML_CUDA 替代旧版的 LLAMA_CUBLAS）：

   ```
   mkdir build && cd build
   cmake .. -DGGML_CUDA=ON 
   cmake --build . --config Release -j 8
   ```

   *编译完成后，在 build/bin/ 目录下会生成 llama-server（旧版本名为 server）可执行文件。这就是最新支持该模型的 C++ 核心引擎。*

------



### 第三阶段：多卡部署脚本

在存放模型或者源码的同级目录下，新建一个部署脚本 start_qwen_moe.sh。

```
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
    -fa \
    > "$LOG_FILE" 2>&1 &

# 4. 提取进程 ID 并提示
PID=$!
echo "服务已放入后台运行，进程 PID: $PID"
echo "可通过命令查看实时日志: tail -f $LOG_FILE"
```

赋予执行权限并启动：

```
chmod +x start_qwen_moe.sh
./start_qwen_moe.sh
```

------



### 第四阶段：在 Python 中调用（绕过第三方包不支持的问题）

由于底层的 llama-server 暴露的是 **完全兼容 OpenAI 标准的 HTTP 接口**，在写业务代码或进行模型推理时，**根本不需要安装滞后的 llama-cpp-python**。只需使用标准的 requests 库或官方的 openai 库调用本地的接口即可：

```
# 业务 Python 脚本
from openai import OpenAI

# 连接刚刚用原生 C++ 部署的本地服务
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-no-key-required" # 填任意字符串
)

response = client.chat.completions.create(
    model="qwen35moe", # 名字任意，llama.cpp server 默认使用挂载的单一模型
    messages=[
        {"role": "system", "content": "你是一个非常有用的AI助手。"},
        {"role": "user", "content": "请介绍一下混合专家模型(MoE)的原理。"}
    ],
    temperature=0.7,
    max_tokens=2048
)

print(response.choices[0].message.content)
```

### 这种方案的核心优势：

1. **解决模型不兼容**：只要 GitHub 上的主分支代码一并入了 Qwen 的最新 PR，拉下来编译立马就能用。
2. **免外网环境**：规避了离线服务器使用 pip 安装时解决各种 C++ 构建环境依赖链的痛苦。
3. **显存自动分配**：llama.cpp 检测到4张卡后，会自动使用张量切分（Tensor Splitting）将 35B 的模型权重均匀拆分到 4 张卡上，对于 MoE 模型尤其高效。