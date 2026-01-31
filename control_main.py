import paho.mqtt.client as mqtt
import tkinter as tk
import json
from uicontrol import RobotUI  

# --- Cấu hình MQTT ---
MQTT_BROKER = "127.0.0.1" # Dùng loopback IP cho máy cục bộ
TOPIC_CONTROL = "mqtt/control"
TOPIC_ASTAR_TARGET = "astar/target"

class MQTTController:
    def __init__(self):
        self.root = tk.Tk()
        
        # 1. Khởi tạo MQTT Client ĐÚNG CÁCH (Gán vào self.client)
        # Sửa lỗi: Bạn dùng 'client =' (biến cục bộ) thay vì 'self.client' (biến của class)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="Python_Control")
        
        # Gán callback trước khi kết nối
        self.client.on_connect = self.on_connect
        
        # 2. Khởi tạo Giao diện (Giả định RobotUI của bạn đã nhận 3 tham số này)
        self.ui = RobotUI(self.root, self.send_mqtt_command, self.send_astar_target)

        # 3. Kết nối Broker
        print(f"Đang kết nối đến Broker {MQTT_BROKER}...")
        try:
            # Chỉ cần gọi connect 1 lần duy nhất ở đây
            self.client.connect(MQTT_BROKER, 1883, 60)
            self.client.loop_start() 
        except Exception as e:
            print(f"Lỗi kết nối: {e}")
            # self.ui.update_status("Lỗi kết nối Broker!", "red")

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(">>> Connected to Broker thành công!")
            # self.ui.update_status("Đã kết nối Broker", "green")
            self.root.after(0, lambda: self.ui.update_status("Đã kết nối Broker", "green"))
        else:
            print(f">>> Lỗi kết nối: Mã lỗi {rc}")

    def send_mqtt_command(self, cmd):
        if self.client.is_connected():
            self.client.publish(TOPIC_CONTROL, cmd)
        else:
            print("Mất kết nối MQTT!")

    def send_astar_target(self, target_str):
        """Gửi tọa độ đích dưới dạng chuỗi 'x,y' tới Broker"""
        if self.client.is_connected():
            # Gửi chuỗi thuần 'x,y'
            self.client.publish(TOPIC_ASTAR_TARGET, target_str)
            print(f"[SENT TO A*]: {target_str}")
        else:
            print("Mất kết nối MQTT! Không thể gửi tọa độ.")
            self.ui.update_status("Mất kết nối MQTT!", "red")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MQTTController()
    app.run()