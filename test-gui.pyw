import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from threading import Thread, Event
from queue import Queue, Empty
from time import sleep

# Cola para mensajes de fondo → GUI
text_queue = Queue()

# Evento para detener el hilo cuando se cierre la ventana
stop_event = Event()

def background_task():
    count = 1
    while not stop_event.is_set():
        text_queue.put(f"Línea de prueba #{count}")
        count += 1
        sleep(1)  # Simula retardo de transcripción

def start_gui():
    root = tk.Tk()
    root.title("Mini GUI de Prueba")

    text_area = ScrolledText(root, wrap=tk.WORD, height=20, width=60)
    text_area.pack(padx=10, pady=10)

    def update_gui():
        try:
            while True:
                line = text_queue.get_nowait()
                text_area.insert(tk.END, line + "\n")
                text_area.see(tk.END)
        except Empty:
            pass
        root.after(100, update_gui)

    def on_close():
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    update_gui()
    Thread(target=background_task, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    start_gui()
