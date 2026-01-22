# CosyVoice TTS API 服务文档

## 📖 概述

CosyVoice TTS API 是一个高性能的文本转语音服务，提供OpenAI兼容的REST API和WebSocket流式API。服务基于阿里云CosyVoice规范构建，支持中文、英文、多语言合成，具备流式处理、实时响应等特性。

### 🎯 主要特性

- **OpenAI API兼容**：完全兼容OpenAI `/v1/audio/speech` 接口
- **WebSocket流式**：支持实时流式文本输入和音频输出
- **多语言支持**：中文、英文、俄语、日语等多语言合成
- **高质量音频**：支持8k/16k/24k采样率，PCM/WAV/MP3格式
- **参数控制**：语速、语调、音量精确调节
- **阿里云规范**：严格按照阿里云CosyVoice接口规范实现

## 🚀 快速开始

### 启动服务

```bash
# 启动TTS服务（默认端口50000）
python openai_server.py --port 50000 --host 0.0.0.0

# 自定义模型路径
python openai_server.py --port 50000 --model_dir /path/to/your/model
```

### 检查服务状态

```bash
# 获取服务信息
curl http://localhost:50000/v1/tts/info

# 获取可用模型
curl http://localhost:50000/v1/models
```

## 🔌 API接口

### 1. REST API - OpenAI兼容

#### 🎵 音频合成 - `/v1/audio/speech`

**请求方式**: `POST`

**请求参数**:

```json
{
    "model": "cosyvoice-tts",
    "input": "要合成的文本内容",
    "voice": "中文女",
    "response_format": "mp3",
    "speed": 1.0
}
```

**参数说明**:

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | string | 否 | "cosyvoice-tts" | 模型名称 |
| `input` | string | 是 | - | 要合成的文本（最大10000字符） |
| `voice` | string | 否 | "中文女" | 音色名称 |
| `response_format` | string | 否 | "mp3" | 音频格式：pcm/wav/mp3 |
| `speed` | float | 否 | 1.0 | 语速（0.5-2.0） |

**curl示例**:

```bash
# 基本合成
curl -X POST "http://localhost:50000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好，欢迎使用CosyVoice文本转语音服务！",
    "voice": "中文女",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output output.wav

# 英文合成
curl -X POST "http://localhost:50000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, welcome to CosyVoice text-to-speech service!",
    "voice": "english",
    "response_format": "mp3",
    "speed": 1.2
  }' \
  --output output.mp3
```

**Python示例**:

```python
import requests

def synthesize_speech(text, voice="中文女", format="wav", speed=1.0):
    url = "http://localhost:50000/v1/audio/speech"
    data = {
        "input": text,
        "voice": voice,
        "response_format": format,
        "speed": speed
    }

    response = requests.post(url, json=data, stream=True)

    if response.status_code == 200:
        with open(f"output.{format}", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Audio saved to output.{format}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# 使用示例
synthesize_speech("今天天气真不错！", voice="中文女", format="wav")
```

### 2. WebSocket API - 流式合成

#### 🌊 WebSocket端点 - `/ws/v1/tts`

WebSocket API支持流式文本输入和实时音频输出，适合实时对话、直播等场景。

**连接URL**: `ws://localhost:50000/ws/v1/tts`

#### 消息格式

所有消息都使用以下JSON格式：

```json
{
    "header": {
        "name": "消息类型",
        "namespace": "FlowingSpeechSynthesizer",
        "task_id": "任务ID",
        "message_id": "消息ID"
    },
    "payload": {
        // 具体参数
    }
}
```

#### 通信流程

1. **StartSynthesis** - 开始合成会话
2. **RunSynthesis** - 发送文本块（可多次调用）
3. **StopSynthesis** - 结束合成会话

#### 🎯 StartSynthesis - 开始合成

**发送消息**:
```json
{
    "header": {
        "name": "StartSynthesis",
        "namespace": "FlowingSpeechSynthesizer",
        "task_id": "task_12345"
    },
    "payload": {
        "voice": "中文女",
        "format": "PCM",
        "sample_rate": 16000,
        "speech_rate": 0,
        "pitch_rate": 0,
        "volume": 50
    }
}
```

**服务端响应**:
```json
{
    "header": {
        "name": "SynthesisStarted",
        "status": 20000000,
        "status_message": "SUCCESS"
    },
    "payload": {
        "voice": "中文女",
        "format": "PCM",
        "sample_rate": 16000
    }
}
```

#### 📝 RunSynthesis - 发送文本

**发送消息**:
```json
{
    "header": {
        "name": "RunSynthesis",
        "namespace": "FlowingSpeechSynthesizer",
        "task_id": "task_12345"
    },
    "payload": {
        "text": "要合成的文本片段"
    }
}
```

**服务端响应**:
```json
// 开始处理
{
    "header": {
        "name": "SentenceBegin",
        "status": 20000000
    },
    "payload": {
        "text": "要合成的文本片段",
        "char_count": 16
    }
}

// 音频数据（二进制）
// ... binary audio data ...

// 处理完成
{
    "header": {
        "name": "SentenceEnd",
        "status": 20000000
    },
    "payload": {
        "text": "要合成的文本片段"
    }
}
```

#### 🏁 StopSynthesis - 结束会话

**发送消息**:
```json
{
    "header": {
        "name": "StopSynthesis",
        "namespace": "FlowingSpeechSynthesizer",
        "task_id": "task_12345"
    }
}
```

**服务端响应**:
```json
{
    "header": {
        "name": "SynthesisCompleted",
        "status": 20000000,
        "status_message": "SUCCESS"
    }
}
```

### 3. 服务信息API

#### 📋 获取服务信息 - `/v1/tts/info`

**请求方式**: `GET`

**响应示例**:
```json
{
    "service": "CosyVoice TTS",
    "version": "1.0",
    "supported_formats": ["pcm", "wav", "mp3"],
    "supported_sample_rates": [8000, 16000, 24000],
    "limits": {
        "max_single_request_chars": 10000,
        "max_total_chars": 200000,
        "char_calculation": "1个汉字=2字符，1个英文/标点/空格=1字符"
    },
    "available_voices": {
        "sft_voices": ["中文女", "中文男"],
        "zero_shot_voices": ["english", "russian", "voice_1", "voice_2"]
    },
    "parameters": {
        "speech_rate": {"range": [-100, 100], "default": 0},
        "pitch_rate": {"range": [-100, 100], "default": 0},
        "volume": {"range": [0, 100], "default": 50}
    }
}
```

## 🎛️ 参数详解

### 音色 (Voice)

| 音色名称 | 语言 | 特点 |
|----------|------|------|
| `中文女` | 中文 | 标准女声，清晰自然 |
| `english` | 英文 | 英语女声 |
| `russian` | 俄语 | 俄语女声 |

### 音频格式

| 格式 | 描述 | 适用场景 |
|------|------|----------|
| `pcm` | 原始PCM数据，无压缩 | 实时处理，低延迟 |
| `wav` | WAV格式，无损 | 高质量存储 |
| `mp3` | MP3格式，有损压缩 | 网络传输，存储优化 |

### 采样率

| 采样率 | 质量 | 文件大小 | 适用场景 |
|--------|------|----------|----------|
| 8000Hz | 电话质量 | 小 | 语音通话 |
| 16000Hz | 标准质量 | 中等 | 一般应用 |
| 24000Hz | 高质量 | 大 | 高保真场景 |

### 语音参数

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| `speed` (REST) | 0.5-2.0 | 1.0 | 语速倍数 |
| `speech_rate` (WS) | -100~100 | 0 | 语速调节，0为正常 |
| `pitch_rate` | -100~100 | 0 | 语调调节，0为正常 |
| `volume` | 0~100 | 50 | 音量大小 |

## 🛠️ 客户端工具

### WebSocket客户端示例

```bash
# 基本使用
python websocket_tts_demo.py --text "你好，这是一个测试。"

# 完整参数
python websocket_tts_demo.py \
    --host localhost \
    --port 50000 \
    --text "春天来了，花儿开了。小鸟在枝头歌唱。" \
    --voice "中文女" \
    --speed 1.2 \
    --pitch-rate 10 \
    --volume 80 \
    --sample-rate 16000 \
    --format WAV \
    --chunk-mode sentence \
    --output my_audio.wav

# 调试模式
python websocket_tts_demo.py \
    --text "测试调试输出" \
    --debug

# 参数验证
python websocket_tts_demo.py \
    --text "测试参数" \
    --validate-only
```

### 分块模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `sentence` | 按句子分块 | 自然语音，推荐使用 |
| `word` | 按词语分块 | 精细控制 |
| `char` | 按字符分块 | 逐字输出，演示效果 |

## 📊 字符计算规则

根据阿里云CosyVoice规范：
- **1个汉字** = 2个字符
- **1个英文字母** = 1个字符
- **1个标点符号** = 1个字符
- **1个空格** = 1个字符

**示例**：
- "你好世界！" = 2+2+2+2+1 = 9个字符
- "Hello World!" = 5+1+5+1 = 12个字符

## 🚨 错误代码

### HTTP状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |

### WebSocket状态码

| 状态码 | 含义 | 说明 |
|--------|------|------|
| 20000000 | SUCCESS | 操作成功 |
| 40000001 | INVALID_REQUEST | 请求格式错误 |
| 40000002 | INVALID_PARAMETER | 参数无效 |
| 40000003 | TEXT_TOO_LONG | 文本过长 |
| 40000004 | UNSUPPORTED_FORMAT | 格式不支持 |
| 40000005 | SYNTHESIS_NOT_STARTED | 合成未开始 |
| 40000006 | SYNTHESIS_ALREADY_STARTED | 合成已开始 |
| 50000000 | SERVER_ERROR | 服务器错误 |

## 💡 最佳实践

### 1. 文本优化

```python
# ✅ 推荐：句子完整，标点正确
text = "今天天气很好。我们去公园散步吧！"

# ❌ 避免：文本过长，无标点
text = "今天天气很好我们去公园散步吧然后可以去喝茶聊天看看风景拍拍照片..."
```

### 2. 音频质量选择

```python
# 高质量场景
params = {
    "response_format": "wav",
    "sample_rate": 24000,  # WebSocket
    "voice": "中文女"
}

# 网络传输场景
params = {
    "response_format": "mp3",
    "sample_rate": 16000,
    "voice": "中文女"
}

# 实时对话场景
params = {
    "response_format": "pcm",
    "sample_rate": 8000,
    "voice": "中文女"
}
```

### 3. WebSocket流式处理

```python
import asyncio
import websockets
import json

async def streaming_tts():
    uri = "ws://localhost:50000/ws/v1/tts"

    async with websockets.connect(uri) as websocket:
        # 1. 开始会话
        await websocket.send(json.dumps({
            "header": {"name": "StartSynthesis", "task_id": "task1"},
            "payload": {"voice": "中文女", "format": "PCM", "sample_rate": 16000}
        }))

        # 2. 流式发送文本
        sentences = ["第一句话。", "第二句话。", "第三句话。"]
        for sentence in sentences:
            await websocket.send(json.dumps({
                "header": {"name": "RunSynthesis", "task_id": "task1"},
                "payload": {"text": sentence}
            }))
            await asyncio.sleep(0.1)  # 适当延迟

        # 3. 结束会话
        await websocket.send(json.dumps({
            "header": {"name": "StopSynthesis", "task_id": "task1"}
        }))

        # 4. 接收音频数据
        with open("output.wav", "wb") as f:
            async for message in websocket:
                if isinstance(message, bytes):
                    f.write(message)
                else:
                    response = json.loads(message)
                    if response["header"]["name"] == "SynthesisCompleted":
                        break

asyncio.run(streaming_tts())
```

### 4. 错误处理

```python
import requests

def safe_tts_request(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:50000/v1/audio/speech",
                json={"input": text, "voice": "中文女"},
                timeout=30
            )

            if response.status_code == 200:
                return response.content
            elif response.status_code == 400:
                print(f"参数错误: {response.text}")
                break  # 不重试参数错误
            else:
                print(f"服务器错误 (尝试 {attempt+1}/{max_retries}): {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"请求超时 (尝试 {attempt+1}/{max_retries})")
        except requests.exceptions.ConnectionError:
            print(f"连接失败 (尝试 {attempt+1}/{max_retries})")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # 指数退避

    return None
```

## 🔧 故障排除

### 常见问题

#### 1. 连接失败

**问题**: `Connection refused` 或 `Connection timeout`

**解决**:
```bash
# 检查服务是否启动
curl http://localhost:50000/v1/tts/info

# 检查端口是否正确
netstat -an | grep 50000

# 重启服务
python openai_server.py --port 50000
```

#### 2. 音频质量差

**问题**: 音频有杂音或不清晰

**解决**:
```python
# 提高采样率
{"sample_rate": 24000}  # WebSocket
{"response_format": "wav"}  # REST

# 选择合适的音色
{"voice": "中文女"}  # 对于中文文本
```

#### 3. 文本过长错误

**问题**: `TEXT_TOO_LONG` 错误

**解决**:
```python
def split_text(text, max_chars=5000):
    """按句子分割长文本"""
    import re
    sentences = re.split(r'[。！？.!?]+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + "。"
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence + "。"

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
```

#### 4. WebSocket连接断开

**问题**: WebSocket意外断开

**解决**:
```python
async def robust_websocket_tts():
    max_retries = 3

    for attempt in range(max_retries):
        try:
            async with websockets.connect(
                "ws://localhost:50003/ws/v1/tts",
                ping_interval=30,  # 心跳检测
                ping_timeout=10
            ) as websocket:
                # 正常处理逻辑
                pass

        except websockets.exceptions.ConnectionClosed:
            print(f"连接断开，重试中... ({attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
```

## 📈 性能优化

### 1. 批量处理

```python
# ✅ 推荐：批量处理多个文本
texts = ["文本1", "文本2", "文本3"]

async with websockets.connect(uri) as websocket:
    # 一次连接处理多个文本
    await start_synthesis(websocket)

    for text in texts:
        await send_text(websocket, text)

    await stop_synthesis(websocket)
```

### 2. 音频格式选择

```python
# 实时场景：使用PCM减少编码开销
{"response_format": "pcm"}

# 存储场景：使用MP3减少空间占用
{"response_format": "mp3"}
```

### 3. 并发控制

```python
import asyncio
from asyncio import Semaphore

# 限制并发连接数
semaphore = Semaphore(5)

async def process_text_with_limit(text):
    async with semaphore:
        return await synthesize_speech(text)
```
