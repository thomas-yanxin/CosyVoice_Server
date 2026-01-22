import argparse
import asyncio
import json
import sys
import uuid
import wave

import aiohttp
import numpy as np
import websockets

# 阿里云CosyVoice规范常量
SUPPORTED_SAMPLE_RATES = {8000, 16000, 24000}
SUPPORTED_FORMATS = {"pcm", "wav", "mp3"}
MAX_SINGLE_REQUEST_CHARS = 10000
MAX_TOTAL_CHARS = 200000

# 状态码定义
class StatusCode:
    SUCCESS = 20000000
    CLIENT_ERROR = 40000000
    SERVER_ERROR = 50000000


def _new_message_id():
    """Generate a new message ID"""
    return uuid.uuid4().hex[:32]


def _build_ws_message(name, task_id=None, message_id=None, payload=None):
    """Build a WebSocket message in the format expected by the server"""
    header = {
        "name": name,
        "namespace": "FlowingSpeechSynthesizer",
        "task_id": task_id,
        "message_id": message_id or _new_message_id(),
    }
    if task_id:
        header["task_id"] = task_id
    return {"header": header, "payload": payload or {}}


def _count_text_characters(text: str) -> int:
    """
    根据阿里云规范计算字符数：
    1个汉字算作2个字符，1个英文字母、1个标点或1个句子中间空格均算作1个字符
    """
    char_count = 0
    for char in text:
        # 判断是否为中文字符（汉字）
        if '\u4e00' <= char <= '\u9fff':
            char_count += 2  # 汉字算2个字符
        else:
            char_count += 1  # 其他字符（英文、标点、空格等）算1个字符
    return char_count


def _validate_parameters(sample_rate, format_str, speed, pitch_rate, volume, text):
    """验证所有参数是否符合阿里云规范"""
    errors = []

    # 验证采样率
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        errors.append(f"Unsupported sample_rate: {sample_rate}. Supported: {list(SUPPORTED_SAMPLE_RATES)}")

    # 验证格式
    format_lower = format_str.lower()
    if format_lower not in SUPPORTED_FORMATS:
        errors.append(f"Unsupported format: {format_str}. Supported: {list(SUPPORTED_FORMATS)}")

    # 验证语速
    speed_rate = int((speed - 1.0) * 100)
    if speed_rate < -100 or speed_rate > 100:
        errors.append(f"speed results in speech_rate {speed_rate}, must be between -100 and 100")

    # 验证语调
    if not isinstance(pitch_rate, (int, float)) or pitch_rate < -100 or pitch_rate > 100:
        errors.append(f"pitch_rate must be between -100 and 100")

    # 验证音量
    if not isinstance(volume, (int, float)) or volume < 0 or volume > 100:
        errors.append(f"volume must be between 0 and 100")

    # 验证文本长度
    char_count = _count_text_characters(text)
    if char_count > MAX_SINGLE_REQUEST_CHARS:
        errors.append(f"Text too long: {char_count} characters (max {MAX_SINGLE_REQUEST_CHARS})")

    # 验证UTF-8编码
    try:
        text.encode('utf-8')
    except UnicodeEncodeError:
        errors.append("Text must be UTF-8 encoded")

    return errors


async def get_server_info(host, port):
    """获取服务器信息"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{host}:{port}/v1/tts/info") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"⚠️ Failed to get server info: {response.status}")
    except Exception as e:
        print(f"⚠️ Could not fetch server info: {e}")
    return None


async def stream_tts(text_chunks, host="localhost", port=50000, voice="中文女", speed=1.0,
                     output_file="output.wav", sample_rate=16000, format_str="PCM",
                     pitch_rate=0, volume=50, show_progress=True, chunk_mode="sentence", debug=False):
    """流式TTS合成"""
    uri = f"ws://{host}:{port}/ws/v1/tts"

    wf = None
    audio_data_received = 0
    task_id = _new_message_id()

    # 合并所有文本块进行验证
    full_text = "".join(text_chunks)
    char_count = _count_text_characters(full_text)

    # 参数验证
    validation_errors = _validate_parameters(sample_rate, format_str, speed, pitch_rate, volume, full_text)
    if validation_errors:
        print("❌ Parameter validation failed:")
        for error in validation_errors:
            print(f"   • {error}")
        return False

    print(f"📊 Text analysis:")
    print(f"   • Total characters (Aliyun counting): {char_count}")
    print(f"   • Text chunks: {len(text_chunks)}")
    print(f"   • Speech rate: {int((speed - 1.0) * 100)}")

    # 获取并显示服务器信息
    if show_progress:
        server_info = await get_server_info(host, port)
        if server_info:
            print(f"🔧 Server info: {server_info.get('service', 'Unknown')} {server_info.get('version', '')}")

    try:
        async with websockets.connect(uri, ping_interval=None, ping_timeout=None) as websocket:
            print(f"✅ Connected to WebSocket TTS server at {uri}")

            # 打开音频文件
            wf = wave.open(output_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(sample_rate)

            # 状态跟踪
            synthesis_completed = False
            task_failed = False
            sentences_processed = 0
            sentences_completed = 0  # 新增：跟踪完成的句子数
            total_chunks = len(text_chunks)
            chunks_sent = 0  # 新增：跟踪发送的块数

            async def receive_messages():
                nonlocal audio_data_received, synthesis_completed, task_failed, sentences_processed, sentences_completed
                while True:
                    try:
                        message = await websocket.recv()
                        if isinstance(message, bytes):
                            # 接收到音频数据
                            if wf and not task_failed:
                                wf.writeframes(message)
                                audio_data_received += len(message)
                                if show_progress:
                                    print(f"🎵 Audio chunk: {len(message)} bytes (Total: {audio_data_received:,})")
                        else:
                            # 接收到JSON消息
                            try:
                                response = json.loads(message)
                                header = response.get("header", {})
                                payload = response.get("payload", {})
                                name = header.get("name")
                                status = header.get("status", 0)
                                status_message = header.get("status_message", "")
                                status_text = header.get("status_text", "")

                                # 根据状态码处理
                                if status == StatusCode.SUCCESS:
                                    status_icon = "✅"
                                elif status >= StatusCode.SERVER_ERROR:
                                    status_icon = "🔥"
                                elif status >= StatusCode.CLIENT_ERROR:
                                    status_icon = "❌"
                                else:
                                    status_icon = "📩"

                                if show_progress:
                                    print(f"{status_icon} {name}: {status_message}")

                                if name == "SynthesisStarted":
                                    if show_progress:
                                        print(f"   📋 Configuration: {payload}")
                                elif name == "SentenceBegin":
                                    sentences_processed += 1
                                    sentence_char_count = payload.get("char_count", "?")
                                    if show_progress:
                                        print(f"   🎯 Processing chunk {sentences_processed}/{total_chunks} ({sentence_char_count} chars): '{payload.get('text', '')}'")
                                elif name == "SentenceEnd":
                                    sentences_completed += 1
                                    if show_progress:
                                        print(f"   ✅ Completed chunk {sentences_completed}/{total_chunks}")

                                    # 检查是否所有块都已完成，如果是则发送StopSynthesis
                                    if sentences_completed == chunks_sent and chunks_sent > 0:
                                        if debug:
                                            print(f"🔄 All {chunks_sent} chunks completed, sending StopSynthesis...")
                                        stop_message = _build_ws_message(
                                            name="StopSynthesis",
                                            task_id=task_id
                                        )
                                        await websocket.send(json.dumps(stop_message))
                                        if show_progress:
                                            print("📤 Sent StopSynthesis - waiting for completion...")
                                    elif debug:
                                        print(f"🔄 Progress: {sentences_completed}/{chunks_sent} chunks completed, waiting for more...")
                                elif name == "SynthesisCompleted":
                                    synthesis_completed = True
                                    print(f"🏁 Synthesis completed! Processed {sentences_completed}/{chunks_sent} chunks")
                                    break
                                elif name == "TaskFailed":
                                    task_failed = True
                                    print(f"❌ Task Failed [{status}]: {status_text}")
                                    break
                                elif status_text and status != StatusCode.SUCCESS:
                                    if show_progress:
                                        print(f"   ⚠️ Details: {status_text}")

                            except json.JSONDecodeError:
                                print("⚠️ Received non-JSON message:", message[:100])
                    except websockets.exceptions.ConnectionClosed:
                        print("🔗 WebSocket connection closed")
                        break
                    except Exception as e:
                        print(f"❌ Message handling error: {e}")
                        break

            receive_task = asyncio.create_task(receive_messages())

            # 1. 发送StartSynthesis消息
            speech_rate = int((speed - 1.0) * 100)

            start_message = _build_ws_message(
                name="StartSynthesis",
                task_id=task_id,
                payload={
                    "voice": voice,
                    "format": format_str.upper(),
                    "sample_rate": sample_rate,
                    "speech_rate": speech_rate,
                    "pitch_rate": pitch_rate,
                    "volume": volume,
                }
            )
            await websocket.send(json.dumps(start_message))
            print(f"📤 Sent StartSynthesis with {len(text_chunks)} text chunks to process")

            # 等待StartSynthesis响应
            await asyncio.sleep(0.2)

            # 2. 流式发送文本 - RunSynthesis
            for i, chunk in enumerate(text_chunks, 1):
                # 只在任务失败时停止发送
                if task_failed:
                    print(f"⏹️ Stopping due to task failure")
                    break

                chunk_char_count = _count_text_characters(chunk)
                run_message = _build_ws_message(
                    name="RunSynthesis",
                    task_id=task_id,
                    payload={"text": chunk}
                )
                await websocket.send(json.dumps(run_message))
                chunks_sent += 1  # 跟踪发送的块数

                if show_progress or debug:
                    print(f"📤 Sent chunk {i}/{total_chunks}: '{chunk}' ({chunk_char_count} chars)")

                # 适应性延迟：确保服务端有足够时间处理
                if chunk_mode == "char":
                    delay = 0.3  # 单字符需要更长延迟
                elif chunk_mode == "word":
                    delay = 0.2  # 单词模式中等延迟
                else:  # sentence mode
                    delay = max(0.1, min(0.5, chunk_char_count / 50))

                if debug:
                    print(f"   ⏱️ Waiting {delay}s before next chunk...")
                await asyncio.sleep(delay)

            if debug:
                print(f"🚀 Sent all {chunks_sent} chunks, waiting for completion...")

            # 不再手动发送StopSynthesis - 会在所有SentenceEnd收到后自动发送
            # 但是添加一个fallback机制，以防万一
            async def fallback_stop():
                # 等待一个合理的时间让所有块完成处理
                timeout = max(10, chunks_sent * 2)  # 每个块最多2秒，最少10秒
                await asyncio.sleep(timeout)

                # 如果还没有完成且没有失败，发送StopSynthesis
                if not synthesis_completed and not task_failed:
                    print(f"⏰ Timeout after {timeout}s, sending fallback StopSynthesis...")
                    try:
                        stop_message = _build_ws_message(
                            name="StopSynthesis",
                            task_id=task_id
                        )
                        await websocket.send(json.dumps(stop_message))
                    except Exception as e:
                        print(f"❌ Failed to send fallback StopSynthesis: {e}")

            # 启动fallback任务
            fallback_task = asyncio.create_task(fallback_stop())

            # 等待接收完成（现在等待SynthesisCompleted或TaskFailed）
            await receive_task

            # 取消fallback任务
            if not fallback_task.done():
                fallback_task.cancel()

            return not task_failed

    except websockets.exceptions.ConnectionClosed:
        print("❌ WebSocket connection lost")
        return False
    except websockets.exceptions.InvalidURI:
        print(f"❌ Invalid WebSocket URI: {uri}")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if wf:
            wf.close()
        success_icon = "✅" if audio_data_received > 0 else "❌"
        print(f"{success_icon} Audio saved to {output_file} ({audio_data_received:,} bytes)")


def split_text_by_char(text, chunk_size=1):
    """按字符分块（模拟逐字输出）"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def split_text_by_word(text):
    """按词分块（简单空格/标点分割，中文可按字）"""
    import re
    return [token for token in re.findall(r'[\w\W]', text) if token.strip()]


def split_text_by_sentence(text):
    """按句子分块"""
    import re
    sentences = re.split(r'[。！？.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


async def main():
    parser = argparse.ArgumentParser(description="WebSocket Streaming TTS Client for CosyVoice")
    parser.add_argument("--host", default="172.21.8.46", help="Server host")
    parser.add_argument("--port", type=int, default=50003, help="Server port")
    parser.add_argument("--voice", default="voice_1", help="Voice name")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (0.5-2.0)")
    parser.add_argument("--pitch-rate", type=int, default=0, help="Pitch rate (-100 to 100)")
    parser.add_argument("--volume", type=int, default=50, help="Volume (0 to 100)")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    parser.add_argument("--chunk-size", type=int, default=3, help="Characters per chunk (when using char mode)")
    parser.add_argument("--chunk-mode", choices=["char", "word", "sentence"], default="sentence",
                       help="Text chunking mode")
    parser.add_argument("--sample-rate", type=int, default=16000, choices=[8000, 16000, 24000],
                       help="Audio sample rate (8000/16000/24000)")
    parser.add_argument("--format", default="PCM", choices=["PCM", "WAV", "MP3"],
                       help="Audio format")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--validate-only", action="store_true", help="Only validate parameters, don't synthesize")

    args = parser.parse_args()

    # 验证参数
    validation_errors = _validate_parameters(
        args.sample_rate, args.format, args.speed,
        args.pitch_rate, args.volume, args.text
    )

    if validation_errors:
        print("❌ Parameter validation failed:")
        for error in validation_errors:
            print(f"   • {error}")
        sys.exit(1)

    # 文本分块
    if args.chunk_mode == "char":
        text_chunks = split_text_by_char(args.text, args.chunk_size)
    elif args.chunk_mode == "word":
        text_chunks = split_text_by_word(args.text)
    elif args.chunk_mode == "sentence":
        text_chunks = split_text_by_sentence(args.text)
    else:
        text_chunks = [args.text]  # 整体发送

    # 显示配置信息
    char_count = _count_text_characters(args.text)
    print("🚀 CosyVoice WebSocket TTS Client")
    print("=" * 50)
    print(f"🎤 Text: '{args.text}'")
    print(f"📊 Characters (Aliyun standard): {char_count}")
    print(f"🗣️ Voice: {args.voice}")
    print(f"⚡ Speed: {args.speed} (speech_rate: {int((args.speed - 1.0) * 100)})")
    print(f"🎵 Format: {args.format}, Sample Rate: {args.sample_rate}Hz")
    print(f"🔊 Pitch: {args.pitch_rate}, Volume: {args.volume}")
    print(f"📦 Chunks: {len(text_chunks)} ({args.chunk_mode} mode)")
    print(f"💾 Output: {args.output}")
    print("=" * 50)

    if args.validate_only:
        print("✅ All parameters are valid!")
        return

    # 执行TTS合成
    success = await stream_tts(
        text_chunks=text_chunks,
        host=args.host,
        port=args.port,
        voice=args.voice,
        speed=args.speed,
        output_file=args.output,
        sample_rate=args.sample_rate,
        format_str=args.format,
        pitch_rate=args.pitch_rate,
        volume=args.volume,
        show_progress=not args.quiet,
        chunk_mode=args.chunk_mode,
        debug=args.debug
    )

    if success:
        print("🎉 TTS synthesis completed successfully!")
    else:
        print("💥 TTS synthesis failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
    
