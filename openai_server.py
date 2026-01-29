import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import uuid
from typing import Generator, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from vllm import ModelRegistry

# Adjust path to include CosyVoice modules
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "third_party", "Matcha-TTS"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openai_server")

try:
    from cosyvoice.cli.cosyvoice import AutoModel
except ImportError as e:
    print(f"Error importing CosyVoice: {e}")
    sys.exit(1)

# 导入音频速度处理器
try:
    from audio_speed_processor import AudioSpeedProcessor

    SPEED_PROCESSOR_AVAILABLE = True
    logger.info("Audio speed processor loaded successfully")
except ImportError:
    SPEED_PROCESSOR_AVAILABLE = False
    logger.warning("Audio speed processor not available - speed parameter may not work")


app = FastAPI(title="CosyVoice OpenAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration
cosyvoice = None
MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
DEFAULT_PROMPT_WAV = "asset/zero_shot_prompt.wav"
DEFAULT_PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

# CosyVoice API Limits (based on Aliyun specifications)
MAX_SINGLE_REQUEST_CHARS = 10000  # 单次合成不超过1万字符
MAX_TOTAL_CHARS = 200000  # 总计不超过20万字符
SUPPORTED_SAMPLE_RATES = {8000, 16000, 24000}  # 支持的采样率 8k/16k/24k
SUPPORTED_FORMATS = {"pcm", "wav", "mp3"}  # 支持的音频格式


# 状态码定义 (按照阿里云规范)
class StatusCode:
    SUCCESS = 20000000
    CLIENT_ERROR = 40000000  # 客户端错误
    SERVER_ERROR = 50000000  # 服务端错误

    # 具体错误码
    INVALID_REQUEST = 40000001
    INVALID_PARAMETER = 40000002
    TEXT_TOO_LONG = 40000003
    UNSUPPORTED_FORMAT = 40000004
    SYNTHESIS_NOT_STARTED = 40000005
    SYNTHESIS_ALREADY_STARTED = 40000006


# CUSTOM VOICE MAP
VOICE_MAP = {
    "中文女": {
        "text": DEFAULT_PROMPT_TEXT,
        "wav": DEFAULT_PROMPT_WAV,
    },
    "russian": {
        "text": "You are a helpful assistant.<|endofprompt|>Всем привет, дорогие друзья! Сейчас 6.20 и мы с вами успели. Сегодня мы с вами встречаем восход солнца.",
        "wav": "asset/russian_prompt.wav",
    },
    "english": {
        "text": "You are a helpful assistant.<|endofprompt|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that's coming into the family is a reason why sometimes we don't buy the whole thing.",
        "wav": "asset/cross_lingual_prompt.wav",
    },
}


class SpeechRequest(BaseModel):
    model: Optional[str] = "tts-1"
    input: str
    voice: Optional[str] = "中文女"
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = (
        "mp3"
    )
    speed: Optional[float] = 1.0


def get_ffmpeg_cmd(output_format: str, input_sample_rate: int, output_sample_rate: int):
    cmd = [
        "ffmpeg",
        "-f",
        "s16le",
        "-ar",
        str(input_sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
    ]
    if output_sample_rate and output_sample_rate != input_sample_rate:
        cmd.extend(["-ar", str(output_sample_rate)])
    if output_format == "pcm":
        cmd.extend(["-f", "s16le", "pipe:1"])
    elif output_format == "mp3":
        cmd.extend(["-f", "mp3", "pipe:1"])
    elif output_format == "opus":
        cmd.extend(["-f", "opus", "-c:a", "libopus", "pipe:1"])
    elif output_format == "aac":
        cmd.extend(["-f", "adts", "pipe:1"])
    elif output_format == "flac":
        cmd.extend(["-f", "flac", "pipe:1"])
    elif output_format == "wav":
        cmd.extend(["-f", "wav", "pipe:1"])
    else:
        raise ValueError(f"Unsupported format via ffmpeg: {output_format}")
    return cmd


def _new_message_id():
    return uuid.uuid4().hex[:32]


def _build_ws_response(
    name: str,
    task_id: Optional[str],
    message_id: Optional[str],
    status: int,
    status_message: str,
    status_text: str,
    payload: Optional[dict],
):
    header = {
        "message_id": message_id or _new_message_id(),
        "task_id": task_id,
        "namespace": "FlowingSpeechSynthesizer",
        "name": name,
        "status": status,
        "status_message": status_message,
        "status_text": status_text,
    }
    return {"header": header, "payload": payload or {}}


def _resolve_voice(spk_id: str):
    prompt_text = DEFAULT_PROMPT_TEXT
    prompt_wav = DEFAULT_PROMPT_WAV
    for key, val in VOICE_MAP.items():
        if key.lower() in spk_id.lower():
            prompt_text = val["text"]
            prompt_wav = val["wav"]
            break
    available_spks = cosyvoice.list_available_spks()
    matched_sft = next((s for s in available_spks if s.lower() == spk_id.lower()), None)
    if matched_sft:
        return matched_sft, False, prompt_text, prompt_wav
    if os.path.exists(spk_id):
        return spk_id, True, prompt_text, spk_id
    return spk_id, True, prompt_text, prompt_wav


def _count_text_characters(text: str) -> int:
    """
    根据阿里云规范计算字符数：
    1个汉字算作2个字符，1个英文字母、1个标点或1个句子中间空格均算作1个字符
    """
    char_count = 0
    for char in text:
        # 判断是否为中文字符（汉字）
        if "\u4e00" <= char <= "\u9fff":
            char_count += 2  # 汉字算2个字符
        else:
            char_count += 1  # 其他字符（英文、标点、空格等）算1个字符
    return char_count


def _validate_text_length(text: str) -> tuple[bool, str]:
    """验证文本长度是否符合规范"""
    char_count = _count_text_characters(text)
    if char_count > MAX_SINGLE_REQUEST_CHARS:
        return (
            False,
            f"Text too long: {char_count} characters (max {MAX_SINGLE_REQUEST_CHARS})",
        )
    return True, ""


def _normalize_output_format(value: Optional[str]) -> Optional[str]:
    """标准化音频输出格式，仅支持阿里云规范的格式"""
    if not value:
        return "pcm"
    fmt = value.lower()
    # 严格按照阿里云支持的格式：PCM、WAV、MP3
    if fmt in SUPPORTED_FORMATS:
        return fmt
    return None


def _validate_sample_rate(sample_rate: int) -> bool:
    """验证采样率是否符合阿里云规范：8k/16k/24k"""
    return sample_rate in SUPPORTED_SAMPLE_RATES


def _resolve_speed(speech_rate: Optional[float]):
    if speech_rate is None:
        return 1.0
    try:
        speed = 1.0 + float(speech_rate) / 100.0
    except Exception:
        return 1.0
    return max(0.5, min(2.0, speed))


async def _stream_tts_audio(
    text: str,
    spk_id: str,
    prompt_text: str,
    prompt_wav: str,
    use_zero_shot: bool,
    speed: float,
    output_format: str,
    output_sample_rate: int,
):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    sentinel = object()
    ffmpeg_proc = None
    use_ffmpeg = output_format != "pcm" or output_sample_rate != cosyvoice.sample_rate

    if use_ffmpeg:
        try:
            cmd = get_ffmpeg_cmd(
                output_format, cosyvoice.sample_rate, output_sample_rate
            )
            ffmpeg_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.error(f"ffmpeg error: {e}")
            return

    def producer_thread():
        try:
            # 检查是否需要速度调整
            need_speed_adjustment = (
                SPEED_PROCESSOR_AVAILABLE
                and abs(speed - 1.0) > 0.01  # 只有当速度明显不是1.0时才调整
            )

            if need_speed_adjustment:
                logger.info(f"Applying audio speed adjustment: {speed}x")

            # 调用CosyVoice进行推理 (不传递speed参数，由后处理器处理)
            if use_zero_shot:
                generator = cosyvoice.inference_zero_shot(
                    text, prompt_text, prompt_wav, stream=True
                )
            else:
                generator = cosyvoice.inference_sft(
                    text, spk_id, stream=True
                )

            # 如果需要速度调整，使用我们的处理器包装原始生成器
            if need_speed_adjustment:

                def audio_generator():
                    for i in generator:
                        raw_data = (
                            (i["tts_speech"].numpy() * (2**15))
                            .astype(np.int16)
                            .tobytes()
                        )
                        yield raw_data

                # 使用速度处理器包装生成器
                speed_adjusted_generator = (
                    AudioSpeedProcessor.create_speed_aware_generator(
                        audio_generator(), speed, cosyvoice.sample_rate
                    )
                )

                for processed_chunk in speed_adjusted_generator:
                    if ffmpeg_proc:
                        try:
                            ffmpeg_proc.stdin.write(processed_chunk)
                            ffmpeg_proc.stdin.flush()
                        except (BrokenPipeError, OSError):
                            break
                    else:
                        loop.call_soon_threadsafe(queue.put_nowait, processed_chunk)
            else:
                # 原始处理逻辑（无速度调整）
                for i in generator:
                    raw_data = (
                        (i["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
                    )
                    if ffmpeg_proc:
                        try:
                            ffmpeg_proc.stdin.write(raw_data)
                            ffmpeg_proc.stdin.flush()
                        except (BrokenPipeError, OSError):
                            break
                    else:
                        loop.call_soon_threadsafe(queue.put_nowait, raw_data)

            if ffmpeg_proc:
                ffmpeg_proc.stdin.close()
            else:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)
        except Exception as e:
            logger.error(f"Producer error: {e}")
            if ffmpeg_proc:
                ffmpeg_proc.terminate()
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    def reader_thread():
        if not ffmpeg_proc:
            return
        try:
            while True:
                chunk = ffmpeg_proc.stdout.read(4096)
                if not chunk:
                    break
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
            ffmpeg_proc.wait()
        except Exception as e:
            logger.error(f"Reader error: {e}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    threading.Thread(target=producer_thread, daemon=True).start()
    if ffmpeg_proc:
        threading.Thread(target=reader_thread, daemon=True).start()

    while True:
        chunk = await queue.get()
        if chunk is sentinel:
            break
        yield chunk


@app.on_event("startup")
async def startup_event():
    global cosyvoice
    logger.info(f"Loading model from {MODEL_DIR}...")
    try:
        if not os.path.exists(MODEL_DIR):
            logger.warning(
                f"Model directory {MODEL_DIR} not found. Please run download_models.py first."
            )
        else:
            cosyvoice = AutoModel(
                model_dir=MODEL_DIR, load_trt=True, load_vllm=True, fp16=False
            )
            logger.info(f"Model loaded. Sample rate: {cosyvoice.sample_rate}")
            available_spks = cosyvoice.list_available_spks()
            logger.info(f"Available speakers: {available_spks}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/v1/models")
async def list_models():
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": "cosyvoice-tts",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "cosyvoice",
                }
            ],
        }
    )


@app.get("/v1/tts/info")
async def get_tts_info():
    """获取TTS服务信息和支持的参数"""
    available_voices = []
    if cosyvoice:
        try:
            available_voices = cosyvoice.list_available_spks()
        except Exception:
            available_voices = []

    return JSONResponse(
        content={
            "service": "CosyVoice TTS",
            "version": "1.0",
            "supported_formats": list(SUPPORTED_FORMATS),
            "supported_sample_rates": list(SUPPORTED_SAMPLE_RATES),
            "limits": {
                "max_single_request_chars": MAX_SINGLE_REQUEST_CHARS,
                "max_total_chars": MAX_TOTAL_CHARS,
                "char_calculation": "1个汉字=2字符，1个英文/标点/空格=1字符",
            },
            "available_voices": {
                "sft_voices": available_voices,
                "zero_shot_voices": list(VOICE_MAP.keys()),
            },
            "parameters": {
                "speech_rate": {"range": [-100, 100], "default": 0},
                "pitch_rate": {"range": [-100, 100], "default": 0},
                "volume": {"range": [0, 100], "default": 50},
            },
        }
    )


@app.post("/v1/audio/speech")
async def text_to_speech(req: SpeechRequest):
    if not cosyvoice:
        raise HTTPException(
            status_code=500, detail="Model not loaded or invalid model directory."
        )

    text = req.input
    spk_id = req.voice
    speed = req.speed if req.speed else 1.0
    format = _normalize_output_format(req.response_format)

    # 验证格式支持
    if not format:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format. Supported formats: {list(SUPPORTED_FORMATS)}",
        )

    # 验证文本长度
    is_valid, error_msg = _validate_text_length(text)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    spk_id, use_zero_shot, prompt_text, prompt_wav = _resolve_voice(spk_id)

    char_count = _count_text_characters(text)
    logger.info(
        f"REST TTS Request: chars={char_count}, voice='{spk_id}', format={format}, speed={speed}, zero_shot={use_zero_shot}"
    )

    if use_zero_shot and prompt_wav == DEFAULT_PROMPT_WAV:
        logger.warning(
            f"Voice '{spk_id}' not found in SFT list or VOICE_MAP. Using default zero-shot prompt."
        )

    async def audio_generator():
        async for chunk in _stream_tts_audio(
            text,
            spk_id,
            prompt_text,
            prompt_wav,
            use_zero_shot,
            speed,
            format,
            cosyvoice.sample_rate,
        ):
            yield chunk

    media_type = f"audio/{format}"
    if format == "pcm":
        media_type = "application/octet-stream"
    elif format == "mp3":
        media_type = "audio/mpeg"
    elif format == "wav":
        media_type = "audio/wav"

    return StreamingResponse(audio_generator(), media_type=media_type)


@app.websocket("/ws/v1/tts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")

    if not cosyvoice:
        await websocket.close(code=1011, reason="Model not loaded")
        return

    task_id = None
    appkey = None
    voice = "中文女"
    output_format = "pcm"
    output_sample_rate = cosyvoice.sample_rate
    speech_rate = 0
    pitch_rate = 0
    volume = 50
    started = False
    stop_requested = False
    completed = False
    run_lock = asyncio.Lock()
    running_tasks = set()

    async def send_response(
        name: str,
        status: int = StatusCode.SUCCESS,
        status_message: str = "SUCCESS",
        status_text: str = "OK",
        payload: Optional[dict] = None,
        message_id: Optional[str] = None,
    ):
        try:
            # 检查WebSocket连接状态
            if websocket.client_state.value == 3:  # CLOSED
                logger.warning("WebSocket connection is closed, skipping response")
                return

            response_data = _build_ws_response(
                name,
                task_id,
                message_id,
                status,
                status_message,
                status_text,
                payload,
            )
            await websocket.send_text(json.dumps(response_data))
        except Exception as e:
            logger.error(f"Failed to send WebSocket response: {e}")
            # 不要重新抛出异常，避免级联错误

    async def send_completed(message_id: Optional[str] = None):
        nonlocal completed
        if completed:
            return
        # 检查连接状态
        if websocket.client_state.value == 3:  # CLOSED
            logger.warning(
                "WebSocket connection closed, cannot send completion message"
            )
            return
        completed = True
        await send_response("SynthesisCompleted", message_id=message_id)

    async def handle_run(text: str, message_id: Optional[str]):
        nonlocal voice, speech_rate, output_format, output_sample_rate
        async with run_lock:
            try:
                # 检查连接状态
                if websocket.client_state.value == 3:  # CLOSED
                    logger.warning("WebSocket connection closed during handle_run")
                    return

                spk_id, use_zero_shot, prompt_text, prompt_wav = _resolve_voice(voice)
                speed = _resolve_speed(speech_rate)

                # 计算字符数用于日志
                char_count = _count_text_characters(text)
                logger.info(f"Processing text: {char_count} chars, voice: {voice}")

                await send_response(
                    "SentenceBegin",
                    payload={
                        "text": text,
                        "char_count": char_count,
                        "timestamp": asyncio.get_event_loop().time(),
                    },
                    message_id=message_id,
                )

                # 流式发送音频数据
                try:
                    async for chunk in _stream_tts_audio(
                        text,
                        spk_id,
                        prompt_text,
                        prompt_wav,
                        use_zero_shot,
                        speed,
                        output_format,
                        output_sample_rate,
                    ):
                        # 在发送每个音频块前检查连接状态
                        if websocket.client_state.value == 3:  # CLOSED
                            logger.warning(
                                "WebSocket connection closed during audio streaming"
                            )
                            break

                        await websocket.send_bytes(chunk)

                    # 只有在连接仍然有效时才发送结束消息
                    if websocket.client_state.value != 3:
                        await send_response(
                            "SentenceEnd",
                            payload={
                                "text": text,
                                "char_count": char_count,
                                "timestamp": asyncio.get_event_loop().time(),
                            },
                            message_id=message_id,
                        )
                except Exception as audio_error:
                    logger.error(f"Audio streaming error: {audio_error}")
                    # 只有在连接仍然有效时才发送错误响应
                    if websocket.client_state.value != 3:
                        await send_response(
                            "TaskFailed",
                            status=StatusCode.SERVER_ERROR,
                            status_message="FAILED",
                            status_text=f"Audio streaming failed: {str(audio_error)}",
                            message_id=message_id,
                        )

            except Exception as e:
                logger.error(f"TTS synthesis error: {e}")
                # 只有在连接仍然有效且不是连接关闭错误时才发送错误响应
                if (
                    websocket.client_state.value != 3
                    and "close message has been sent" not in str(e)
                ):
                    try:
                        await send_response(
                            "TaskFailed",
                            status=StatusCode.SERVER_ERROR,
                            status_message="FAILED",
                            status_text=f"TTS synthesis failed: {str(e)}",
                            message_id=message_id,
                        )
                    except Exception as response_error:
                        logger.error(f"Failed to send error response: {response_error}")

            # 不要在这里发送completed消息 - 让StopSynthesis处理逻辑来决定何时完成
            # if stop_requested and websocket.client_state.value != 3:
            #     await send_completed(message_id=message_id)

    def on_task_done(task: asyncio.Task):
        running_tasks.discard(task)
        if (
            stop_requested
            and not running_tasks
            and not completed
            and hasattr(websocket, "client_state")
            and websocket.client_state.value != 3
        ):
            asyncio.create_task(send_completed())

    try:
        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.disconnect":
                break
            if "text" not in raw:
                continue
            try:
                message = json.loads(raw["text"])
            except Exception:
                await send_response(
                    "TaskFailed",
                    status=StatusCode.INVALID_REQUEST,
                    status_message="FAILED",
                    status_text="Invalid JSON format",
                )
                continue

            header = message.get("header", {})
            payload = message.get("payload", {}) or {}
            name = header.get("name")
            namespace = header.get("namespace")
            message_id = header.get("message_id")

            if namespace and namespace != "FlowingSpeechSynthesizer":
                await send_response(
                    "TaskFailed",
                    status=StatusCode.INVALID_PARAMETER,
                    status_message="FAILED",
                    status_text="Invalid namespace. Must be 'FlowingSpeechSynthesizer'",
                    message_id=message_id,
                )
                continue

            if name == "StartSynthesis":
                if started:
                    await send_response(
                        "TaskFailed",
                        status=StatusCode.SYNTHESIS_ALREADY_STARTED,
                        status_message="FAILED",
                        status_text="Synthesis already started",
                        message_id=message_id,
                    )
                    continue

                task_id = header.get("task_id") or _new_message_id()
                appkey = header.get("appkey")
                voice = payload.get("voice") or voice

                # 验证音频格式
                output_format_value = payload.get("format") or "PCM"
                output_format = _normalize_output_format(output_format_value)
                if output_format not in SUPPORTED_FORMATS:
                    await send_response(
                        "TaskFailed",
                        status=StatusCode.UNSUPPORTED_FORMAT,
                        status_message="FAILED",
                        status_text=f"Unsupported format: {output_format_value}. Supported: {list(SUPPORTED_FORMATS)}",
                        message_id=message_id,
                    )
                    continue

                # 验证采样率
                output_sample_rate = payload.get("sample_rate") or cosyvoice.sample_rate
                if not _validate_sample_rate(output_sample_rate):
                    await send_response(
                        "TaskFailed",
                        status=StatusCode.INVALID_PARAMETER,
                        status_message="FAILED",
                        status_text=f"Unsupported sample_rate: {output_sample_rate}. Supported: {list(SUPPORTED_SAMPLE_RATES)}",
                        message_id=message_id,
                    )
                    continue

                # 验证语速、语调、音量参数范围
                speech_rate = payload.get("speech_rate", 0)
                if (
                    not isinstance(speech_rate, (int, float))
                    or speech_rate < -100
                    or speech_rate > 100
                ):
                    await send_response(
                        "TaskFailed",
                        status=StatusCode.INVALID_PARAMETER,
                        status_message="FAILED",
                        status_text="speech_rate must be between -100 and 100",
                        message_id=message_id,
                    )
                    continue

                pitch_rate = payload.get("pitch_rate", 0)
                if (
                    not isinstance(pitch_rate, (int, float))
                    or pitch_rate < -100
                    or pitch_rate > 100
                ):
                    await send_response(
                        "TaskFailed",
                        status=StatusCode.INVALID_PARAMETER,
                        status_message="FAILED",
                        status_text="pitch_rate must be between -100 and 100",
                        message_id=message_id,
                    )
                    continue

                volume = payload.get("volume", 50)
                if not isinstance(volume, (int, float)) or volume < 0 or volume > 100:
                    await send_response(
                        "TaskFailed",
                        status=StatusCode.INVALID_PARAMETER,
                        status_message="FAILED",
                        status_text="volume must be between 0 and 100",
                        message_id=message_id,
                    )
                    continue

                started = True
                await send_response(
                    "SynthesisStarted",
                    status=StatusCode.SUCCESS,
                    payload={
                        "voice": voice,
                        "format": output_format_value,
                        "sample_rate": output_sample_rate,
                        "speech_rate": speech_rate,
                        "pitch_rate": pitch_rate,
                        "volume": volume,
                        "appkey": appkey,
                    },
                    message_id=message_id,
                )
                continue

            if name == "RunSynthesis":
                if not started:
                    await send_response(
                        "TaskFailed",
                        status=StatusCode.SYNTHESIS_NOT_STARTED,
                        status_message="FAILED",
                        status_text="Synthesis not started. Please send StartSynthesis first",
                        message_id=message_id,
                    )
                    continue

                text = payload.get("text", "")
                if not text or not str(text).strip():
                    await send_response(
                        "TaskFailed",
                        status=StatusCode.INVALID_PARAMETER,
                        status_message="FAILED",
                        status_text="Empty text provided",
                        message_id=message_id,
                    )
                    continue

                # 验证文本长度（按照阿里云规范）
                text_str = str(text).strip()
                is_valid, error_msg = _validate_text_length(text_str)
                if not is_valid:
                    await send_response(
                        "TaskFailed",
                        status=StatusCode.TEXT_TOO_LONG,
                        status_message="FAILED",
                        status_text=error_msg,
                        message_id=message_id,
                    )
                    continue

                # 检查UTF-8编码
                try:
                    text_str.encode("utf-8")
                except UnicodeEncodeError:
                    await send_response(
                        "TaskFailed",
                        status=StatusCode.INVALID_PARAMETER,
                        status_message="FAILED",
                        status_text="Text must be UTF-8 encoded",
                        message_id=message_id,
                    )
                    continue

                task = asyncio.create_task(handle_run(text_str, message_id))
                running_tasks.add(task)
                task.add_done_callback(on_task_done)
                continue

            if name == "StopSynthesis":
                stop_requested = True
                if not running_tasks and not completed:
                    await send_completed(message_id=message_id)
                continue

            await send_response(
                "TaskFailed",
                status=StatusCode.INVALID_PARAMETER,
                status_message="FAILED",
                status_text=f"Unsupported command: {name}",
                message_id=message_id,
            )
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"Global WS Error: {e}")
        # 只有在连接仍然有效且不是连接关闭错误时才尝试发送错误响应
        if (
            "close message has been sent" not in str(e)
            and hasattr(websocket, "client_state")
            and websocket.client_state.value != 3
        ):
            try:
                await send_response(
                    "TaskFailed",
                    status=StatusCode.SERVER_ERROR,
                    status_message="FAILED",
                    status_text=f"Server internal error: {str(e)}",
                )
            except Exception as send_error:
                logger.error(f"Failed to send global error response: {send_error}")
    finally:
        # 取消所有正在运行的任务
        for task in running_tasks.copy():
            if not task.done():
                task.cancel()

        try:
            if hasattr(websocket, "client_state") and websocket.client_state.value != 3:
                await websocket.close()
        except Exception as close_error:
            logger.error(f"Error closing WebSocket: {close_error}")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50003)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--model_dir",
        type=str,
    )
    args = parser.parse_args()

    MODEL_DIR = args.model_dir
    uvicorn.run(app, host=args.host, port=args.port)
