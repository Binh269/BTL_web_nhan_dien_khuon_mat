import paho.mqtt.client as mqtt
import json

# Cấu hình MQTT
MQTT_BROKER = "192.168.0.102"  # Thay bằng broker của bạn
MQTT_PORT = 1883
MQTT_TOPIC = "MQTT_DC_DongCo"

def send_mqtt_message(payload):
    """ Gửi dữ liệu lên MQTT """
    try:
        client = mqtt.Client()
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.publish(MQTT_TOPIC, json.dumps(payload))
        client.disconnect()
        print("Gửi dữ liệu MQTT thành công:", payload)
    except Exception as e:
        print(f"Lỗi khi gửi MQTT: {e}")
