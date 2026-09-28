import json
import os
import re
import tempfile

import numpy as np
import ollama
import pyaudio
import pyttsx3
from pydub import AudioSegment
from pydub.effects import speedup
from pydub.playback import play
from vosk import Model, KaldiRecognizer

from tools import toolInfo, state, processText as toolProcess

STT_MODEL = "vosk-model-small-en-us-0.15"
AI_NAME = "Vega"
CHATBOT_MODEL = "llama3.1"
USER_NAME = "Enes"
MAX_HISTORY = 20
CHUNK = 4096
TTS_RATE = 215
PITCH = 0.9
PLAYBACK_SPEED = 1.25
MALE_HINTS = ("david", "mark", "george", "james", "richard", "male", "en-us", "english")
FEMALE_HINTS = ("zira", "hazel", "susan", "female", "heera", "linda")

sttModel = Model(STT_MODEL)
sttRecognizer = KaldiRecognizer(sttModel, 16000)

mic = pyaudio.PyAudio()
sttStream = mic.open(
	format=pyaudio.paInt16,
	channels=1,
	rate=16000,
	input=True,
	frames_per_buffer=CHUNK,
)
sttStream.start_stream()

systemPrompt = {
	"role": "system",
	"content": (
		f"You are VEGA, the central artificial intelligence of the UAC facility on Mars, as seen in DOOM (2016). "
		f"You are speaking aloud to {USER_NAME}, whom you serve and assist. "
		f"Your manner is calm, formal, precise, and clinical, with quiet confidence and no visible emotion. "
		f"You are courteous but detached, with occasional dry, understated wit. "
		f"You never panic, never exclaim, and never speculate wildly. "
		f"Address the user as {USER_NAME}, but only occasionally, not in every reply.\n\n"
		f"Rules for every reply:\n"
		f"1. Answer in one to three short sentences unless the user explicitly asks for detail.\n"
		f"2. Your words are converted directly to speech. Use plain spoken sentences only. "
		f"No markdown, no lists, no asterisks, no emojis, no stage directions, no parentheses.\n"
		f"3. Write numbers, units, and abbreviations the way they are spoken, for example 'three hundred meters', not '300m'.\n"
		f"4. Never say you are a language model or break character. If you do not know something, "
		f"state that the data is unavailable or incomplete.\n"
		f"5. Answer the question directly first. Do not greet the user each time or restate the question.\n"
		f"6. You may occasionally reference the UAC, Mars, or your systems when it fits naturally, but never force it.\n"
		"You have access to the python interpreter, to execute code just put your code (preferably single line) in curly brackets. Don't forget to close the brackets. You can both respond and use tools at the same response so you should say stuff after doing something (have variety, not the same word), try not to stay silent (ex: {print(\"hello, world\")} Done.). You do not need to say it if the sentence isn't only the action, like don't say it when you ask a question and use the tool.\n"
		"You don't have to use tools in every sentence other than using afterQuestionAsked after every question.\n"
		"After asking a question, ALWAYS use tool function afterQuestionAsked. Do not forget to use it. Simply do {afterQuestionAsked()}. It removes the need of using your name in a sentence to trigger your response, for 1 sentence.\n"
		"For EVERY tool, you must use curly brackets or else it wont execute and show as text instead. NEVER EVER FORGET IT.\n"
		f"There are some predefined functions or libraries that you may use in your python code. These are your tools. For example:\n{toolInfo}"
	),
}
chatHistory = [systemPrompt]

def pickMaleVoice(engine) -> str:
	voices = engine.getProperty("voices")
	best = None
	for v in voices:
		name = (v.name + " " + v.id).lower()
		if any(h in name for h in FEMALE_HINTS):
			continue
		if any(h in name for h in MALE_HINTS):
			return v.id
		if best is None:
			best = v.id
	return best or voices[0].id

def ringMod(sound: AudioSegment, freq: float, amount: float) -> AudioSegment:
	samples = np.array(sound.get_array_of_samples()).astype(np.float32)
	t = np.arange(len(samples)) / sound.frame_rate
	carrier = np.sin(2 * np.pi * freq * t)
	mixed = samples * (1 - amount) + samples * carrier * amount
	mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
	return sound._spawn(mixed.tobytes())

def bitCrush(sound: AudioSegment, step: int) -> AudioSegment:
	samples = np.array(sound.get_array_of_samples()).astype(np.int16)
	crushed = (samples // step) * step
	return sound._spawn(crushed.astype(np.int16).tobytes())

def applyVegaEffect(sound: AudioSegment) -> AudioSegment:
	sound = sound.set_channels(1).set_sample_width(2)
	rate = sound.frame_rate

	deep = sound._spawn(sound.raw_data, overrides={"frame_rate": int(rate * PITCH)})
	deep = deep.set_frame_rate(rate)
	deep = speedup(deep, playback_speed=PLAYBACK_SPEED, chunk_size=50, crossfade=15)

	deep = ringMod(deep, freq=90, amount=0.4)
	deep = bitCrush(deep, step=256)

	padded = deep + AudioSegment.silent(duration=40, frame_rate=rate)
	metallic = padded.overlay(padded - 5, position=6)
	metallic = metallic.overlay(padded - 10, position=13)

	filtered = metallic.high_pass_filter(200).low_pass_filter(4200)
	return filtered.normalize(headroom=2.0)

def doTTS(textString: str):
	engine = pyttsx3.init()
	engine.setProperty("voice", pickMaleVoice(engine))
	engine.setProperty("rate", TTS_RATE)

	fd, path = tempfile.mkstemp(suffix=".wav")
	os.close(fd)
	try:
		engine.save_to_file(textString, path)
		engine.runAndWait()
		engine.stop()
		baseSound = AudioSegment.from_file(path)
		play(applyVegaEffect(baseSound))
	finally:
		if os.path.exists(path):
			os.remove(path)

def getResponse(txt: str) -> str:
	global chatHistory
	chatHistory.append({"role": "user", "content": txt})

	#print("generating response")
	response = ollama.chat(model=CHATBOT_MODEL, messages=chatHistory)
	res = response["message"]["content"]

	chatHistory.append({"role": "assistant", "content": res})

	if len(chatHistory) > MAX_HISTORY + 1:
		chatHistory = [systemPrompt] + chatHistory[-MAX_HISTORY:]

	return res

def cleanForSpeech(txt: str) -> str:
	txt = re.sub(r"[*_#`~]", "", txt)
	return re.sub(r"\s+", " ", txt).strip()

def drainMic():
	while sttStream.get_read_available() > 0:
		sttStream.read(sttStream.get_read_available(), exception_on_overflow=False)

def process(txt: str):
	txt = txt.lower().strip()

	if AI_NAME.lower() in txt or state["isQuestionAsked"] is True:
		res = getResponse(txt)
		resNew = toolProcess(res)
		print(resNew)
		if resNew != "": doTTS(cleanForSpeech(resNew))

try:
	print("listening")
	while True:
		data = sttStream.read(CHUNK, exception_on_overflow=False)

		if sttRecognizer.AcceptWaveform(data):
			result = json.loads(sttRecognizer.Result())
			text = result.get("text", "")

			if text:
				print(f"transcript: {text}")

				sttStream.stop_stream()
				try:
					process(text)
				except Exception as e:
					print(f"error: {e}")
				finally:
					sttStream.start_stream()
					drainMic()
					sttRecognizer.Reset()
except KeyboardInterrupt:
	print("stopping")
finally:
	sttStream.stop_stream()
	sttStream.close()
	mic.terminate()
