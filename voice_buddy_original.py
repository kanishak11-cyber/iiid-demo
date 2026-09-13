"""
Voice Study Buddy — a hands-free, tool-using voice agent on Sarvam AI.

  You speak (Hinglish / Hindi / English)
    -> Sarvam Saaras v3   : speech -> text
    -> Agent loop         : LLM decides, calls tools (wiki / calculator / date)
    -> Sarvam Bulbul v3   : text -> Hindi speech
    -> Your speaker

Setup
  pip install requests sounddevice soundfile numpy
  export SARVAM_API_KEY=...            # dashboard.sarvam.ai (free credits, no card)

  # Optional: use an NVIDIA NIM model as the "brain" instead of Sarvam-105B
  export BRAIN=nvidia
  export NVIDIA_API_KEY=nvapi-...       # build.nvidia.com (free, no card)

Run
  python voice_buddy.py
  -> it listens automatically after each answer. Say "bye" / "alvida" to stop.
"""
import os, io, sys, json, math, time, base64, datetime
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf

SARVAM_KEY = os.environ.get("SARVAM_API_KEY", "YOUR_SARVAM_API_KEY")
SARVAM = "https://api.sarvam.ai"
H = {"api-subscription-key": SARVAM_KEY}

# ---------------------------------------------------------------- brain
BRAIN = os.environ.get("BRAIN", "sarvam")
if BRAIN == "nvidia":
    CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    CHAT_HEADERS = {"Authorization": "Bearer " + os.environ.get("NVIDIA_API_KEY", "")}
    MODEL = os.environ.get("NVIDIA_MODEL", "mistralai/mistral-nemotron")  # strong tool calling
else:
    CHAT_URL = SARVAM + "/v1/chat/completions"
    CHAT_HEADERS = H
    MODEL = "sarvam-105b"


def ask(messages, tools=None):
    body = {"model": MODEL, "messages": messages}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    r = requests.post(CHAT_URL, json=body, headers=CHAT_HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


# ---------------------------------------------------------------- tools
def calculator(expression):
    safe = {"__builtins__": {}}
    return str(eval(expression, safe, vars(math)))


def wiki(query):
    slug = query.replace(" ", "_")
    r = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/" + slug,
                     headers={"User-Agent": "voice-buddy/0.1"}, timeout=15)
    return r.json().get("extract", "Not found") if r.ok else "Not found"


def today(_=""):
    return datetime.datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


RUN = {"calculator": calculator, "wiki": wiki, "today": today}


def tool(name, desc, arg, arg_desc):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object",
                       "properties": {arg: {"type": "string", "description": arg_desc}},
                       "required": [arg]}}}


TOOLS = [
    tool("calculator", "Do a math calculation. Use Python syntax, e.g. sqrt(144) or 2**10.",
         "expression", "the math expression"),
    tool("wiki", "Look up a person, place, or topic on Wikipedia.",
         "query", "topic name in English"),
    tool("today", "Get today's date and current time.",
         "ignore", "always pass an empty string"),
]

SYSTEM = ("You are a friendly study buddy for Indian college students. "
          "The user speaks to you by voice, so keep answers SHORT — 2 to 4 sentences, "
          "no lists, no markdown, no emojis. Use tools when a fact or calculation is needed. "
          "Always reply in simple spoken Hindi (Devanagari script).")

MAX_STEPS = 6
messages = [{"role": "system", "content": SYSTEM}]   # memory persists across turns


def agent(user_text):
    messages.append({"role": "user", "content": user_text})
    for _ in range(MAX_STEPS):                          # step cap
        msg = ask(messages, TOOLS)                      # THINK
        messages.append(msg)
        if not msg.get("tool_calls"):
            return msg["content"]                       # DONE
        for call in msg["tool_calls"]:                  # ACT
            fn = call["function"]
            try:
                args = json.loads(fn["arguments"] or "{}")
                out = RUN[fn["name"]](**args) if args else RUN[fn["name"]]()
            except Exception as e:
                out = f"tool error: {e}"
            print(f"   -> {fn['name']}({args}) = {str(out)[:70]}")
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": str(out)})       # OBSERVE
    return "माफ़ कीजिए, मैं यह पूरा नहीं कर पाया।"


# ---------------------------------------------------------------- ears
SR = 16000            # Sarvam STT works best at 16 kHz
SILENCE_RMS = 0.012   # tweak if it cuts you off / never stops
SILENCE_SEC = 1.3
MAX_SEC = 25          # REST STT limit is 30 s


def record_until_silence():
    """Start recording when you speak; stop after SILENCE_SEC of quiet."""
    print("\n🎤  Listening... (speak now)")
    chunks, quiet, started, t0 = [], 0.0, False, time.time()
    block = int(SR * 0.1)
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=block) as s:
        while time.time() - t0 < MAX_SEC:
            data, _ = s.read(block)
            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms > SILENCE_RMS:
                started, quiet = True, 0.0
            elif started:
                quiet += 0.1
            if started:
                chunks.append(data.copy())
            if started and quiet >= SILENCE_SEC:
                break
    if not chunks:
        return None
    audio = np.concatenate(chunks)
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def speech_to_text(wav_bytes):
    r = requests.post(SARVAM + "/speech-to-text", headers=H,
                      files={"file": ("speech.wav", wav_bytes, "audio/wav")},
                      data={"model": "saaras:v3", "mode": "codemix"},   # Hinglish-friendly
                      timeout=60)
    r.raise_for_status()
    return r.json().get("transcript", "").strip()


# ---------------------------------------------------------------- mouth
def speak(text, speaker="shubh"):
    r = requests.post(SARVAM + "/text-to-speech", headers=H, json={
        "text": text[:2500],
        "target_language_code": "hi-IN",
        "model": "bulbul:v3",
        "speaker": speaker,
        "pace": 1.05,
    }, timeout=60)
    r.raise_for_status()
    wav = base64.b64decode(r.json()["audios"][0])
    audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
    sd.play(audio, sr)
    sd.wait()


# ---------------------------------------------------------------- loop
EXIT_WORDS = ("bye", "alvida", "अलविदा", "bas karo", "stop", "band karo")

if __name__ == "__main__":
    print(f"Voice Study Buddy  |  brain: {MODEL}  |  ears: saaras:v3  |  mouth: bulbul:v3")
    speak("नमस्ते! मैं आपका स्टडी बडी हूँ। बोलिए, क्या पढ़ना है?")
    while True:
        wav = record_until_silence()
        if wav is None:
            continue
        try:
            heard = speech_to_text(wav)
        except requests.HTTPError as e:
            print("STT error:", e.response.text[:200]); continue
        if not heard:
            continue
        print(f"👂  You: {heard}")
        if any(w in heard.lower() for w in EXIT_WORDS):
            speak("ठीक है, अलविदा! पढ़ाई अच्छी रहे।"); break
        try:
            answer = agent(heard)
        except requests.HTTPError as e:
            print("LLM error:", e.response.text[:200]); continue
        print(f"🤖  Buddy: {answer}")
        try:
            speak(answer)
        except requests.HTTPError as e:
            print("TTS error:", e.response.text[:200])
