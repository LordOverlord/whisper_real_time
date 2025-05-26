#! python3.12

import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta, timezone
from queue import Queue
from time import sleep
from sys import platform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use (e.g., tiny, base, small, medium, large, large-v3, large-v3-turbo)")
    parser.add_argument("--input_device", default=None, help="Index or partial name of input device (use 'list' to show all)", type=str)
    parser.add_argument("--energy_threshold", default=1000, type=int)
    parser.add_argument("--record_timeout", default=4, type=float)
    parser.add_argument("--phrase_timeout", default=6, type=float)
    parser.add_argument("--save_name", default="transcripcion", help="Base filename without extension")
    parser.add_argument("--save_format", choices=["txt"], default="txt", help="Only txt is supported in this implementation")

    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse', help="Linux mic name")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"{args.save_name}_{timestamp}.{args.save_format}"

    phrase_time = None
    data_queue = Queue()
    phrase_bytes = bytes()

    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    input_device = args.input_device
    device_index = None

    if input_device == "list":
        print("Available audio input devices:")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"[{index}] {name}")
        return
    elif input_device is not None:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if input_device.isdigit():
                if int(input_device) == index:
                    device_index = index
                    break
            elif input_device.lower() in name.lower():
                device_index = index
                break

    source = sr.Microphone(sample_rate=16000, device_index=device_index)
    audio_model = whisper.load_model(args.model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")

    while True:
        try:
            now = datetime.now(timezone.utc)
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_bytes = bytes()
                    phrase_complete = True
                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                phrase_bytes += audio_data
                audio_np = np.frombuffer(phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                text = ""
                try:
                    start = datetime.now()
                    result = audio_model.transcribe(
                        audio_np,
                        fp16=torch.cuda.is_available(),
                        #language='es',
                        temperature=0.0
                    )
                    end = datetime.now()
                    #print(f"⏱️ Tiempo de transcripción: {(end - start).total_seconds():.2f}s")
                    text = result['text'].strip()
                except Exception as e:
                    print(f"\u26a0\ufe0f Error en transcripci\u00f3n: {e}")

                if text:
                    if phrase_complete:
                        transcription.append(text)
                    else:
                        transcription[-1] = text

                if args.save_format == "txt":
                    try:
                        with open(output_filename, "w", encoding="utf-8") as f:
                            for line in transcription:
                                f.write(f"{line.strip()}\n")
                    except Exception as e:
                        print(f"\u26a0\ufe0f No se pudo guardar el archivo TXT: {e}")

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                print('', end='', flush=True)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)

if __name__ == "__main__":
    main()
