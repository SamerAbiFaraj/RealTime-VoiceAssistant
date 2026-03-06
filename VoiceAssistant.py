import ollama
import pyaudio
import wave
import tempfile
import subprocess
import os
import threading
import queue
import numpy as np
import webrtcvad
import pyttsx3
from datetime import datetime

WHISPER_PATH = "C:\\whisper.cpp\\bin\\Release\\whisper-cli.exe" 
WHISPER_MODEL = "C:\\whisper.cpp\\models\\ggml-large-v3.bin"

# Audio settings - much watch what whisper expects
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30 # WebRTC VAD requires 10,20 or 30 ms
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
SILENCE_THRESHOLD_CHUNKS = 30 # ~900ms of silence = end of speech


# --- TEXT TO SPEECH ---

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 175) # Words per minute

def speak(text:str) -> None:
    """Speak text aloud. Runs in main thread"""
    # Clean up text - remove markdown
    clean = text.replace("**","").replace("*","").replace("#","")
    tts_engine.say(clean)
    tts_engine.runAndWait()
    

# --- VOICE ACTIVITY DETECTION ---

vad = webrtcvad.Vad(2)  # 0-3, higher = more aggressive filtering

def is_speech(audio_chunk: bytes) -> bool:
    """Returns True if this chunk contains speech"""
    try:
        return vad.is_speech(audio_chunk, SAMPLE_RATE)
    except Exception:
        return False


# --- AUDIO CAPTURE ---

def record_until_silence()-> bytes | None:
    """Record audio until the user stops speaking. Returns raw PCM bytes"""
    p = pyaudio.PyAudio()
    stream = p.open(
        format = pyaudio.paInt16,
        channels = 1,
        rate = SAMPLE_RATE,
        input = True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    print("Listening ...", end="", flush=True)
    
    frames = []
    silence_count = 0
    started_speaking = False
    max_silence_before_start = 50 # Dont record until speech detected
    pre_speech_buffer = []
    
    try:
        while True:
            chunk = stream.read(CHUNK_SIZE,exception_on_overflow=False)
            
            if is_speech(chunk):
                if not started_speaking:
                    print(" [Speaking]", end="",flush=True)
                    started_speaking = True
                    frames.extend(pre_speech_buffer)  # Include pre-speech buffer
                
                frames.append(chunk)
                silence_count=0
            else:
                if started_speaking:
                    frames.append(chunk)
                    silence_count += 1
                    
                    if silence_count >= SILENCE_THRESHOLD_CHUNKS:
                        print( "[Done]")
                        break
                
                else:
                    # Keep a rolling buffer before speech starts
                    pre_speech_buffer.append(chunk)
                    if len(pre_speech_buffer)>10:
                        pre_speech_buffer.pop(0)
                    
                    max_silence_before_start -= 1
                    if max_silence_before_start <= 0:
                        # No speech detected within timeout
                        return None
        
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    if not frames or len(frames) < 5:
        return None
    
    return b"".join(frames)
            

def save_audio_to_temp(pcm_bytes: bytes) ->str:
    """Save PCM bytes as a WAV file for Whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as tmp:
        tmp_path = tmp.name
    
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    
    return tmp_path

# --- TRANSCRIPTION ---

def transcribe_audio(wav_path: str)-> str:
    """Run Whisper on the WAV file, return transcript"""
    result = subprocess.run(
        [WHISPER_PATH, "-m", WHISPER_MODEL, "-f", wav_path, "--no-timestamps"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    transcript = result.stdout.strip()
    
    # Whisper sometimes adds filler markers - clean them
    for marker in ["[BLANK_AUDIO]","[MUSIC]","(music)","(sound)"]:
        transcript = transcript.replace(marker,"").strip()
        
    return transcript


# --- LLM RESPONSE ---

conversation_history = []
SYSTEM_PROMPT = """You are a helpful voice assistant. Keep responses conversational and concise -
2-4 sentences max unless asked for more detail. You are being read aloud, so avoid using markdown,
 bullet points, lists, or special characters. Speak naturally."""


def get_response(user_message: str) ->str:
    conversation_history.append({
        "role":"user",
        "content": user_message})
    
    response = ollama.chat(
        model="qwen2.5-coder:7b",
        messages=[{
            "role":"system",
            "content": SYSTEM_PROMPT
        }, *conversation_history[-8:] # keep last 4 turns of context
        ],
        options = {"temperature": 0.7}
    )
        
    reply = response["message"]["content"].strip()
    conversation_history.append({
        "role": "assistant",
        "content": reply
    })
    
    return reply

# --- MAIN LOOP ---

def run_voice_assistant():
    print("Voice Assistant Started")
    print("Speak naturally. There will be a short delay while processing")
    print("Press Ctrl+C to quit. \n")
    
    speak("Voice assistant ready. How can I help?")
    
    while True:
        try:
            # Step 1: Record
            pcm = record_until_silence()
            if pcm is None:
                continue
            
            # Step 2: Transcribe
            wav_path = save_audio_to_temp(pcm)
            try:
                transcript = transcribe_audio(wav_path)
            finally:
                os.unlink(wav_path)
            
            
            if not transcript or len(transcript.strip())< 3:
                continue
            
            print(f"\n You: {transcript}")
            
            
            # Check for exit commands
            if any(cmd in transcript.lower() for cmd in ["goodbye","quit","stop assistant", "shut down"]):
                speak("Goodbye!")
                break
            
            # Step 3: Get response
            print("Thinking...", end="",flush=True)
            reply = get_response(transcript)
            print(f"\nAssistant: {reply}")
            
            # Step 4: Speak
            speak(reply)
        
        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

if __name__ == "__main__":
    run_voice_assistant()
