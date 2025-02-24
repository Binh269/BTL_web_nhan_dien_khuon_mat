import paho.mqtt.client as mqtt
import base64
import cv2
import numpy as np
import threading
import time
from django.http import StreamingHttpResponse

BROKER = "192.168.195.22"
PORT = 1883
TOPIC = "Camera_vip/#"

latest_frame = None
frame_lock = threading.Lock()
data_chunks = {}

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(TOPIC)
    else:
        print(f"❌ Kết nối thất bại. Mã lỗi: {rc}")

def on_message(client, userdata, msg):
    global latest_frame

    topic_name = msg.topic.split("/")[-1]
    payload = msg.payload.decode("utf-8")

    if topic_name.startswith("chunk_"):
        data_chunks[topic_name] = payload

    elif topic_name == "end" and payload.strip() == "done":

        sorted_chunks = sorted(data_chunks.items(), key=lambda x: int(x[0].split("_")[-1]))
        full_data = "".join(chunk_data for _, chunk_data in sorted_chunks)

        try:
            image_data = base64.b64decode(full_data)
            image_np = np.frombuffer(image_data, dtype=np.uint8)
            frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            if frame is not None and frame.size > 0:
                with frame_lock:
                    latest_frame = frame
            else:
                pass

        except Exception as e:
            print(f"⚠️ Lỗi giải mã Base64: {e}")

        data_chunks.clear()  

def start_mqtt():
    """ Khởi chạy MQTT trong luồng riêng để không chặn Django """
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    
    mqtt_thread = threading.Thread(target=client.loop_forever, daemon=True)
    mqtt_thread.start()