import tkinter as tk

class RobotUI:
    def __init__(self, root, send_command_callback, send_target_callback):
        self.root = root
        self.root.title("Robot Control Panel")
        self.root.geometry("500x550") # Tăng chiều cao để chứa thêm các nút mới
        self.send_command = send_command_callback
        self.send_target = send_target_callback 

          # --- PHẦN 1: ĐIỀU KHIỂN THỦ CÔNG (GIỮ NGUYÊN) ---

        manual_frame = tk.LabelFrame(root, text="Manual Control", padx=10, pady=10)

        manual_frame.pack(pady=10)



        btn_forward = tk.Button(manual_frame, text="FORWARD", width=10, height=2,

                                bg="#4CAF50", fg="white", font=('Arial', 10, 'bold'),

                                command=lambda: self.send_command("forward"))

        btn_forward.grid(row=0, column=1, pady=5)

       

        btn_backward = tk.Button(manual_frame, text="BACKWARD", width=10, height=2,

                                 bg="#4CAF50", fg="white", font=('Arial', 10, 'bold'),

                                 command=lambda: self.send_command("backward"))

        btn_backward.grid(row=2, column=1, pady=5)



        btn_left = tk.Button(manual_frame, text="LEFT", width=10, height=2,

                             bg="#2196F3", fg="white", font=('Arial', 10, 'bold'),

                             command=lambda: self.send_command("left"))

        btn_left.grid(row=1, column=0, padx=5)



        btn_stop = tk.Button(manual_frame, text="STOP", width=10, height=2,

                             bg="#F44336", fg="white", font=('Arial', 10, 'bold'),

                             command=lambda: self.send_command("stop"))

        btn_stop.grid(row=1, column=1, padx=5)



        btn_right = tk.Button(manual_frame, text="RIGHT", width=10, height=2,

                              bg="#2196F3", fg="white", font=('Arial', 10, 'bold'),

                              command=lambda: self.send_command("right"))

        btn_right.grid(row=1, column=2, padx=5)

        # (Các nút Left, Right, Backward tương tự...)

        # --- PHẦN 2: ĐIỀU KHIỂN CHÍNH XÁC (MOVE & ROTATE) - THÊM MỚI ---
        precision_frame = tk.LabelFrame(root, text="Precision Control (Value)", padx=10, pady=10)
        precision_frame.pack(pady=10, fill="x", padx=20)

        # Dòng nhập cho MOVE
        tk.Label(precision_frame, text="Distance (0-1m):").grid(row=0, column=0, sticky="w")
        self.entry_move = tk.Entry(precision_frame, width=10)
        self.entry_move.grid(row=0, column=1, padx=5, pady=5)
        self.entry_move.insert(0, "0.5") # Mặc định 0.5 mét

        btn_move = tk.Button(precision_frame, text="MOVE", width=10, bg="#FF9800", fg="white",
                             font=('Arial', 9, 'bold'), command=self.handle_move)
        btn_move.grid(row=0, column=2, padx=5)

        # Dòng nhập cho ROTATE
        tk.Label(precision_frame, text="Angle (deg):").grid(row=1, column=0, sticky="w")
        self.entry_rotate = tk.Entry(precision_frame, width=10)
        self.entry_rotate.grid(row=1, column=1, padx=5, pady=5)
        self.entry_rotate.insert(0, "90")

        btn_rotate = tk.Button(precision_frame, text="ROTATE", width=10, bg="#00BCD4", fg="white",
                               font=('Arial', 9, 'bold'), command=self.handle_rotate)
        btn_rotate.grid(row=1, column=2, padx=5)

        # --- PHẦN 3: TỌA ĐỘ A* (TARGET) ---
        astar_frame = tk.LabelFrame(root, text="A* Target Navigation", padx=10, pady=10)
        astar_frame.pack(pady=10, fill="x", padx=20)

        tk.Label(astar_frame, text="X:").grid(row=0, column=0)
        self.entry_x = tk.Entry(astar_frame, width=8)
        self.entry_x.grid(row=0, column=1)
        
        tk.Label(astar_frame, text="Y:").grid(row=0, column=2)
        self.entry_y = tk.Entry(astar_frame, width=8)
        self.entry_y.grid(row=0, column=3)

        btn_send_astar = tk.Button(astar_frame, text="SET TARGET", bg="#9C27B0", fg="white",
                                   command=self.handle_send_target)
        btn_send_astar.grid(row=0, column=4, padx=5)

        # Nhãn trạng thái
        self.status_label = tk.Label(root, text="Trạng thái: Sẵn sàng", fg="blue")
        self.status_label.pack(side=tk.BOTTOM, pady=10)

    # Xử lý gửi lệnh MOVE <giá trị>
    def handle_move(self):
        try:
            val = float(self.entry_move.get())
            
            # Kiểm tra giới hạn từ 0 đến 1 mét
            if 0 <= val <= 1.0:
                command = f"MOVE {val}"
                self.send_command(command)
                self.update_status(f"Gửi lệnh: {command}m", "green")
            else:
                self.update_status("Lỗi: Chỉ cho phép từ 0 đến 1m!", "red")
                
        except ValueError:
            self.update_status("Lỗi: Vui lòng nhập số thập phân!", "red")

    # --- Logic xử lý ROTATE (giữ nguyên độ) ---
    def handle_rotate(self):
        try:
            val = float(self.entry_rotate.get())
            
            # Kiểm tra giới hạn góc quay từ -180 đến 180 độ
            if -180 <= val <= 180:
                command = f"ROTATE {val}"
                self.send_command(command)
                self.update_status(f"Gửi lệnh: {command}°", "green")
            else:
                self.update_status("Lỗi: Góc phải từ -180 đến 180°!", "red")
                
        except ValueError:
            self.update_status("Lỗi: Nhập số cho góc quay!", "red")

    def handle_send_target(self):
        try:
            tx, ty = float(self.entry_x.get()), float(self.entry_y.get())
            target_str = f"{tx},{ty}"
            self.send_target(target_str)
            self.update_status(f"Đã gửi đích: {target_str}", "green")
        except ValueError:
            self.update_status("Lỗi: Tọa độ phải là số!", "red")

    def update_status(self, text, color="black"):
        self.status_label.config(text=f"Trạng thái: {text}", fg=color)