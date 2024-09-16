import base64
from threading import Lock, Thread

import cv2
import numpy as np
import openai
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

# Для захвата экрана на macOS
import mss  # Используйте mss для захвата экрана
# Если mss не работает на macOS, используйте Quartz
# from PIL import Image
# import Quartz.CoreGraphics as CG

load_dotenv()

class CaptureDevice:
    def start(self):
        pass

    def read(self, encode=False):
        pass

    def stop(self):
        pass

class DesktopCapture(CaptureDevice):
    def __init__(self):
        self.running = False
        self.lock = Lock()
        self.frame = None

        # Для mss
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Измените индекс для нескольких мониторов

        # Для Quartz (раскомментируйте, если используете Quartz)
        # self.display_id = CG.CGMainDisplayID()

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update)
        self.thread.start()
        return self

    def update(self):
        while self.running:
            # Используем mss
            img = self.sct.grab(self.monitor)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Используем Quartz (раскомментируйте, если используете Quartz)
            # img = self.capture_screen()

            with self.lock:
                self.frame = img

    # Для Quartz (раскомментируйте, если используете Quartz)
    # def capture_screen(self):
    #     image = CG.CGWindowListCreateImage(
    #         CG.CGRectInfinite,
    #         CG.kCGWindowListOptionOnScreenOnly,
    #         CG.kCGNullWindowID,
    #         CG.kCGWindowImageDefault)
    #     width = CG.CGImageGetWidth(image)
    #     height = CG.CGImageGetHeight(image)
    #     data_provider = CG.CGImageGetDataProvider(image)
    #     data = CG.CGDataProviderCopyData(data_provider)
    #     img = Image.frombytes("RGBA", (width, height), data, "raw", "RGBA")
    #     img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
    #     return img

    def read(self, encode=False):
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()

        if encode:
            _, buffer = cv2.imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

class WebcamCapture(CaptureDevice):
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        self.running = False
        self.lock = Lock()
        self.frame = None

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update)
        self.thread.start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.stream.read()
            if not ret:
                continue

            with self.lock:
                self.frame = frame

    def read(self, encode=False):
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()

        if encode:
            _, buffer = cv2.imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words in your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

def main():
    # Выбор между 'desktop' и 'webcam'
    capture_choice = input("Введите 'desktop' для захвата экрана или 'webcam' для использования камеры: ").strip().lower()

    if capture_choice == 'desktop':
        capture_device = DesktopCapture()
    elif capture_choice == 'webcam':
        capture_device = WebcamCapture()
    else:
        print("Неверный выбор. По умолчанию выбран захват экрана.")
        capture_device = DesktopCapture()

    capture_device.start()

    # Используем модель GPT-4o
    model = ChatOpenAI(model="gpt-4o")
    # model = ChatOpenAI(model="gpt-4o-mini")

    assistant = Assistant(model)

    def audio_callback(recognizer, audio):
        try:
            prompt = recognizer.recognize_whisper(audio, model="base", language="english")
            image_data = capture_device.read(encode=True)
            if image_data is not None:
                assistant.answer(prompt, image_data)
            else:
                print("Нет доступных данных изображения.")
        except UnknownValueError:
            print("Произошла ошибка при обработке аудио.")

    recognizer = Recognizer()
    microphone = Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    stop_listening = recognizer.listen_in_background(microphone, audio_callback)

    while True:
        frame = capture_device.read()
        if frame is not None:
            window_title = "Desktop Capture" if isinstance(capture_device, DesktopCapture) else "Webcam Capture"
            cv2.imshow(window_title, frame)
        if cv2.waitKey(1) in [27, ord("q")]:
            break

    capture_device.stop()
    cv2.destroyAllWindows()
    stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    main()