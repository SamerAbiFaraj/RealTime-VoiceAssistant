# Real-Time Voice Assistant (Whisper + Ollama)

A simple **real-time voice assistant** that listens to your microphone, transcribes speech with **whisper.cpp**, generates responses using **Ollama** + a local LLM (e.g., `qwen2.5-coder:7b`), and speaks back using **pyttsx3**.

It uses **WebRTC Voice Activity Detection (VAD)** to detect when you stop speaking, then runs a transcription and sends the text to the LLM.

---

## 🔎 What this code does

1. **Records audio** from your default microphone until you stop speaking (using WebRTC VAD).
2. **Saves the recording** to a temporary WAV file (16 kHz, mono, 16-bit PCM).
3. **Transcribes the audio** using `whisper.cpp` (`whisper-cli.exe`) and a downloaded Whisper model.
4. **Sends the transcript** to an LLM via the **Ollama** Python client.
5. **Speaks the reply** back to you using `pyttsx3`.

This creates a simple conversational loop that listens, transcribes, thinks, and speaks.

---

## ✅ Prerequisites

### 1) Python
- Python 3.9+ (3.11 recommended)

### 2) Install system dependencies
- Make sure you have a working microphone.
- On Windows you may need to install the Visual C++ build tools for `pyaudio`.

### 3) `whisper.cpp`
This script relies on the **whisper.cpp** command-line tool (`whisper-cli.exe`).

1. Clone and build whisper.cpp (or use a prebuilt binary):
   - https://github.com/ggerganov/whisper.cpp

2. Download a model (e.g. `ggml-large-v3.bin`):
   - https://huggingface.co/ggerganov/whisper.cpp/tree/main/models

3. Update the paths at the top of `VoiceAssistant.py` (or set your own):

```python
WHISPER_PATH = "C:\\whisper.cpp\\bin\\Release\\whisper-cli.exe"
WHISPER_MODEL = "C:\\whisper.cpp\\models\\ggml-large-v3.bin"
```

> 💡 Tip: Use a smaller model (e.g., `ggml-small.bin`) if you need faster transcription at the cost of accuracy.

### 4) Ollama (local LLM runner)

This script uses `ollama` (https://ollama.ai/) to run a local LLM.

1. Install Ollama and confirm it works:
   ```powershell
   ollama --help
   ```

2. Install a model (example):
   ```powershell
   ollama pull qwen2.5-coder:7b
   ```

> ✅ The script uses `qwen2.5-coder:7b` by default. You can change it in `get_response()`.

---

## 🧩 Python Dependencies

Install the required Python packages:

```powershell
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install directly:

```powershell
pip install ollama pyaudio numpy webrtcvad pyttsx3
```

> ⚠️ `pyaudio` can be tricky to install on Windows. If `pip install pyaudio` fails, try installing a wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio.

---

## ▶️ Running the Voice Assistant

Run the script with:

```powershell
python VoiceAssistant.py
```

### What to expect
- The assistant will say: **"Voice assistant ready. How can I help?"**
- Speak naturally. When you stop speaking for ~1 second, it will:
  1. Transcribe your speech.
  2. Send it to the LLM.
  3. Speak back the LLM response.

### Exit
Say any of these phrases to stop the assistant:
- "goodbye"
- "quit"
- "stop assistant"
- "shut down"

Or press **Ctrl+C**.

---

## 🛠️ Customization

### Change the LLM model
Edit `get_response()` in `VoiceAssistant.py`:

```python
response = ollama.chat(
    model="qwen2.5-coder:7b",
    ...
)
```

### Adjust speech detection sensitivity
- `SILENCE_THRESHOLD_CHUNKS`: how many chunks (30ms each) of silence before the assistant decides you’re done speaking.
- `vad = webrtcvad.Vad(2)`: value 0–3 (higher = more aggressive silence detection).

---

## ✅ Notes / Troubleshooting

- If transcription takes too long, try a smaller Whisper model (e.g., `ggml-small.bin`).
- If audio is choppy or you get `OSError` from PyAudio, make sure the correct microphone is selected and no other app locks it.
- If you see errors from Ollama, ensure the Ollama daemon is running and the model is installed.

---

## 📄 License
This repository does not include an explicit license file. Use or modify it under your own terms.
