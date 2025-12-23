import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import queue
import json
import time
import pickle  # For loading .pkl tokenizers
import ssl
import numpy as np
import os # <-- ADDED IMPORT

# --- STT (Vosk) Imports ---
import vosk
import sounddevice as sd

# --- TTS (Silero) Imports ---
import torch

# --- Chatbot (Keras) Imports ---
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Import all the custom layers from your 'transformer_chatbot.py' file
from transformer_chatbot import PositionalEncoding, DecoderBlock, ExpandMask, build_transformer

from tensorflow.keras.utils import register_keras_serializable


# -----------------------------------------------
# Keras Custom Layer Fix (MUST be here for loading)
# -----------------------------------------------
@register_keras_serializable()
class PositionalEncoding(PositionalEncoding):
    pass

@register_keras_serializable()
class DecoderBlock(DecoderBlock):
    pass

@register_keras_serializable()
class ExpandMask(ExpandMask):
    pass

# -----------------------------------------------
# Chatbot Tokenizer Class (from your notebook)
# -----------------------------------------------
class Vocabulary:
    def __init__(self):
        self._word_to_index = {"<unk>": 0}
        self._index_to_word = {0: "<unk>"}
        self._count = 1

    def add_words(self, tokens):
        for token in tokens:
            if token not in self._word_to_index:
                self._word_to_index[token] = self._count
                self._index_to_word[self._count] = token
                self._count += 1

    def stoi(self, word):
        return self._word_to_index.get(word, 0)

    def itos(self, index):
        return self._index_to_word.get(index, "<unk>")

    def __len__(self):
        return self._count

# -----------------------------------------------
# Main Application Class
# -----------------------------------------------

class ChatApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Based Chatbot")
        self.root.geometry("600x450")

        self.chat_history = scrolledtext.ScrolledText(root, state='disabled', width=70, height=20, wrap=tk.WORD, font=("Arial", 10))
        self.chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(root, text="Loading models... Please wait.", font=("Arial", 9), fg="blue")
        self.status_label.pack(pady=5)

        self.listen_button = tk.Button(root, text="Start Listening", command=self.toggle_listening, state='disabled', font=("Arial", 11, "bold"), bg="#4CAF50", fg="white")
        self.listen_button.pack(pady=10, ipadx=10, ipady=5)

        self.stt_queue = queue.Queue()
        self.is_listening = False
        self.is_speaking = False

        # --- Model Objects ---
        self.vosk_model = None
        self.tts_model = None
        self.chatbot_model = None
        self.vocab_q = None
        self.vocab_a = None

        # --- ABSOLUTE PATH FIX (Line 1) ---
        # Get the absolute path to the directory this script is in
        self.SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

        # --- Model Config ---
        # --- ABSOLUTE PATH FIX (Lines 2-5) ---
        # These paths are now ABSOLUTE
        self.VOSK_MODEL_PATH = os.path.join(self.SCRIPT_DIR, "vosk-model-small-en-us-0.15")
        self.CHATBOT_MODEL_PATH = os.path.join(self.SCRIPT_DIR, "chatbot_model_skyrim.keras")
        self.VOCAB_Q_PATH = os.path.join(self.SCRIPT_DIR, "skyrim_vocab_q.pkl")
        self.VOCAB_A_PATH = os.path.join(self.SCRIPT_DIR, "skyirm_vocab_a.pkl") # Make sure spelling is correct!
        # --- END FIX ---

        # Values from your notebook
        self.MAX_ENCODER_LEN = 7
        self.MAX_DECODER_LEN = 16
        self.START_ID = -1
        self.END_ID = -1

        # --- SSL Fix for Silero ---
        try:
            _default_context = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception:
            pass

        # Start loading all models in a separate thread
        self.add_message("System", "Loading models. This may take a moment...")
        threading.Thread(target=self.load_all_models, daemon=True).start()

        self.root.after(100, self.process_stt_queue)

    def update_status(self, text, color="blue"):
        self.root.after(0, lambda: self.status_label.config(text=text, fg=color))

    def add_message(self, sender, message):
        self.root.after(0, self._add_message_gui, sender, message)

    def _add_message_gui(self, sender, message):
        self.chat_history.config(state='normal')
        self.chat_history.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_history.config(state='disabled')
        self.chat_history.see(tk.END)

    def load_all_models(self):
        try:
            # 1. Load Vosk (STT)
            self.update_status("Loading STT (Vosk)...")
            # Check if VOSK path exists before loading
            if not os.path.exists(self.VOSK_MODEL_PATH):
                raise FileNotFoundError(f"Vosk model folder not found at: {self.VOSK_MODEL_PATH}")
            self.vosk_model = vosk.Model(self.VOSK_MODEL_PATH)
            self.vosk_recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000)
            self.add_message("System", "STT Model Loaded.")

            # 2. Load Silero (TTS)
            self.update_status("Loading TTS (Silero)... (This is slow the first time)")
            self.tts_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="en",
                speaker="v3_en",
                trust_repo=True
            )
            self.tts_sample_rate = 48000
            self.add_message("System", "TTS Model Loaded.")

            # 3. Load Keras Chatbot Tokenizers
            self.update_status("Loading Chatbot Tokenizers...")
             # Check if pickle paths exist before loading
            if not os.path.exists(self.VOCAB_Q_PATH):
                raise FileNotFoundError(f"Question vocab file not found at: {self.VOCAB_Q_PATH}")
            if not os.path.exists(self.VOCAB_A_PATH):
                raise FileNotFoundError(f"Answer vocab file not found at: {self.VOCAB_A_PATH}")

            with open(self.VOCAB_Q_PATH, "rb") as f:
                self.vocab_q = pickle.load(f)
            with open(self.VOCAB_A_PATH, "rb") as f:
                self.vocab_a = pickle.load(f)

            self.START_ID = self.vocab_a.stoi("startseq")
            self.END_ID = self.vocab_a.stoi("endseq")
            self.add_message("System", "Chatbot Tokenizers Loaded.")

            # 4. Load Keras Chatbot Model
            self.update_status("Loading Chatbot Model (Keras)...")
            # Check if Keras path exists before loading
            if not os.path.exists(self.CHATBOT_MODEL_PATH):
                raise FileNotFoundError(f"Keras model file not found at: {self.CHATBOT_MODEL_PATH}")
            self.chatbot_model = load_model(
                self.CHATBOT_MODEL_PATH,
                compile=False,
                custom_objects={
                    'PositionalEncoding': PositionalEncoding,
                    'DecoderBlock': DecoderBlock,
                    'ExpandMask': ExpandMask
                }
            )
            _ = self.get_chatbot_response("hello") # Initialize
            self.add_message("System", "Chatbot Model Initialized.")

            self.update_status("All models loaded. Ready!", "green")
            self.root.after(0, lambda: self.listen_button.config(state='normal'))

        except Exception as e:
            self.update_status(f"Error loading models: {e}", "red")
            messagebox.showerror("Model Load Error", fr"Could not load models: {e}\n\Check file paths and console.")
            self.add_message("System", f"FATAL ERROR: {e}")
            # Optional: Disable button on critical error
            # self.root.after(0, lambda: self.listen_button.config(state='disabled'))


    def toggle_listening(self):
        if self.is_listening:
            self.is_listening = False
            self.listen_button.config(text="Start Listening", bg="#4CAF50")
            self.add_message("System", "Speech recognition paused.")
        else:
            self.is_listening = True
            self.listen_button.config(text="Stop Listening", bg="#f44336")
            self.add_message("System", "Listening...")

            if not hasattr(self, 'stt_thread') or not self.stt_thread.is_alive():
                self.stt_thread = threading.Thread(target=self.run_stt_listener, daemon=True)
                self.stt_thread.start()

    def run_stt_listener(self):
        audio_q = queue.Queue()
        def audio_callback(indata, frames, time, status):
            if status: print(status, flush=True)
            audio_q.put(bytes(indata))

        try:
            with sd.RawInputStream(samplerate=16000, blocksize=8000,
                                   dtype='int16', channels=1, callback=audio_callback):
                while True:
                    data = audio_q.get()
                    if self.is_listening and not self.is_speaking:
                        if self.vosk_recognizer.AcceptWaveform(data):
                            result_json = self.vosk_recognizer.Result()
                            result = json.loads(result_json)
                            if result.get("text"):
                                self.stt_queue.put(result["text"])
                    else:
                        time.sleep(0.1)
        except Exception as e:
            print(f"STT Thread Error: {e}")
            self.stt_queue.put(f"STT_ERROR:{e}")

    def process_stt_queue(self):
        try:
            text = self.stt_queue.get_nowait()
            if "STT_ERROR" in text:
                self.add_message("System", f"STT Error: {text}")
            else:
                self.add_message("You", text)
                if self.is_listening:
                    self.toggle_listening()

                threading.Thread(target=self.run_chatbot_and_tts, args=(text,), daemon=True).start()

        except queue.Empty:
            pass

        self.root.after(100, self.process_stt_queue)

    def run_chatbot_and_tts(self, user_text):
        try:
            self.update_status("Bot is thinking...", "orange")
            bot_response = self.get_chatbot_response(user_text)
            self.add_message("Chatbot", bot_response)

            self.update_status("Bot is speaking...", "orange")
            self.is_speaking = True
            self.play_tts_response(bot_response)
            self.is_speaking = False
            self.update_status("Ready!", "green")

            if not self.is_listening:
                self.root.after(0, self.toggle_listening)

        except Exception as e:
            self.add_message("System", f"Chatbot/TTS Error: {e}")
            self.update_status(f"Error: {e}", "red")
            self.is_speaking = False

    def get_chatbot_response(self, user_input):
        if self.chatbot_model is None or self.vocab_q is None:
            return "Chatbot model not loaded."

        try:
            user_input = user_input.lower().strip()
            user_input = re.sub(r'[\\/:;\\-_+@&!?$()<>.,@#%^&*\"]', '', user_input)
            input_tokens = [self.vocab_q.stoi(tok) for tok in user_input.split()]
            enc_input = pad_sequences([input_tokens], maxlen=self.MAX_ENCODER_LEN, padding='post')
            enc_input_tf = tf.constant(enc_input, dtype=tf.int32)

            generated = [self.START_ID]

            for _ in range(self.MAX_DECODER_LEN - 1):
                dec_input = pad_sequences([generated], maxlen=self.MAX_DECODER_LEN, padding='post')
                dec_input_tf = tf.constant(dec_input, dtype=tf.int32)

                logits = self.chatbot_model([enc_input_tf, dec_input_tf], training=False)

                next_logits = logits[:, len(generated) - 1, :]
                next_id = tf.argmax(next_logits, axis=-1, output_type=tf.int32).numpy()[0]

                if next_id == self.END_ID:
                    break
                generated.append(next_id)

            response = " ".join([self.vocab_a.itos(tok) for tok in generated[1:]
                                 if tok not in (0, self.START_ID, self.END_ID)])
            return response if response else "..."

        except Exception as e:
            print(f"Keras Chatbot Error: {e}")
            return f"Error in chatbot: {e}"

    def play_tts_response(self, text_to_speak):
        if self.tts_model is None:
            return

        try:
            audio = self.tts_model.apply_tts(
                text=text_to_speak,
                speaker="en_1",
                sample_rate=self.tts_sample_rate
            )
            audio_numpy = np.array(audio, dtype=np.float32)
            sd.play(audio_numpy, samplerate=self.tts_sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Silero-TTS Error: {e}")
            self.add_message("System", f"TTS Error: {e}")

# --- Main entry point ---
if __name__ == "__main__":
    import re
    root = tk.Tk()
    app = ChatApplication(root)
    root.mainloop()