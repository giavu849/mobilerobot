import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider  # Bắt buộc
import numpy as np
from matplotlib.transforms import Affine2D
import queue
class MapVisualizer:
    def __init__(self, robot_data, grid_map):

        self.robot_data = robot_data

        self.grid_map = grid_map

        self.user_view_angle = 0.0

        # 1. Tạo Figure với kích thước đủ lớn
        self.fig = plt.figure(figsize=(8, 9))
        # 2. Tạo vùng vẽ bản đồ (chiếm phần trên)
        self.ax = self.fig.add_axes([0.1, 0.25, 0.8, 0.65])
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        with self.grid_map.grid_lock:
            self.img = self.ax.imshow(self.grid_map.grid.T, origin='lower',
                                  cmap='gray_r', vmin=0, vmax=255,
                                  extent=[-3, 3, -3, 3])

        # Trong __init__ của MapVisualizer

        self.robot_marker, = self.ax.plot([], [], 'ro', markersize=10, zorder=5, label="Robot")

        self.robot_dir_line, = self.ax.plot([], [], 'r-', linewidth=1, zorder=5)

        self.ax.legend(loc='upper right')

        self.ax.set_title("Real-time SLAM")
       # 3. TẠO THANH TRƯỢT (Slider)

        # Vị trí: [cách lề trái, cách lề dưới, độ rộng, độ cao]

        ax_slider = self.fig.add_axes([0.2, 0.1, 0.6, 0.03])
     # Gán vào self.slider để không bị xóa khỏi bộ nhớ

        self.slider = Slider(ax_slider, 'Rotate View', 0.0, 360.0, valinit=0.0)
      # Kết nối sự kiện
        self.slider.on_changed(self.update_view_angle)
        
        # Tạo một đối tượng Transform trống
        self.view_transform = Affine2D()    
        # --- TẠO LƯỚI TỌA ĐỘ CỐ ĐỊNH (KHÔNG XOAY) ---
        self.grid_lines = []
        ticks = np.arange(-5, 6, 1) # Vẽ rộng ra một chút từ -5m đến 5m
        
        # Màu xám 50/255
        for t in ticks:
            # Đường ngang cố định theo trục Y của hệ tọa độ gốc
            line_h, = self.ax.plot([-5, 5], [t, t], 
                                  color="#5A625A", 
                                  linestyle='-', 
                                  linewidth=0.7, 
                                  alpha=0.5, 
                                  zorder=1) # zorder thấp để nằm dưới robot
            # KHÔNG gọi set_transform(self.view_transform...) ở đây
            self.grid_lines.append(line_h)
            
            # Đường dọc cố định theo trục X của hệ tọa độ gốc
            line_v, = self.ax.plot([t, t], [-5, 5], 
                                  color="#5A625A", 
                                  linestyle='-', 
                                  linewidth=0.7, 
                                  alpha=0.5, 
                                  zorder=1)
            self.grid_lines.append(line_v)
        # Áp dụng transform cho tất cả các đối tượng vẽ
        # Lưu ý: Ta cộng với self.ax.transData để giữ các điểm nằm đúng hệ tọa độ của trục vẽ
        self.img.set_transform(self.view_transform + self.ax.transData)
        self.robot_marker.set_transform(self.view_transform + self.ax.transData)
        self.robot_dir_line.set_transform(self.view_transform + self.ax.transData)
    def update_view_angle(self, val):

        self.user_view_angle = val
    
        # 1. Xóa bỏ phép biến đổi cũ và tạo mới
        # Xoay quanh tâm (0,0) - nơi robot bắt đầu
        self.view_transform.clear()
        self.view_transform.rotate_deg(val)
        
        # 2. Cập nhật tiêu đề để người dùng biết góc xoay hiện tại
        self.ax.set_title(f"SLAM Map - View Rotation: {val:.1f}°")
        
        # 3. Yêu cầu vẽ lại khung nhìn ngay lập tức
        self.fig.canvas.draw_idle()
    def update(self, frame):
        # 1. LUÔN LUÔN lấy Pose hiện tại trước tiên để tránh lỗi UnboundLocalError
        # Việc lấy pose nằm ngoài khối lệnh 'if queue' để robot marker luôn được cập nhật
        with self.robot_data.pose_lock:
            rx = self.robot_data.pose_x
            ry = self.robot_data.pose_y
            rt = self.robot_data.pose_theta

        nx, ny = [], []
        # 2. Chỉ xử lý điểm Lidar nếu có tín hiệu từ hàng đợi
        if not self.robot_data.plot_queue.empty():
            try:
                self.robot_data.plot_queue.get_nowait()
                with self.robot_data.data_lock:
                    nx = self.robot_data.data['x_coords']
                    ny = self.robot_data.data['y_coords']
                    self.robot_data.data['x_coords'] = []
                    self.robot_data.data['y_coords'] = []
            except queue.Empty:
                pass

        # 3. Cập nhật bản đồ Grid nếu có điểm Lidar mới
        if nx:
            nx1, ny1 = self.robot_data.remove_outliers(nx, ny)
            # nx2, ny2 = self.robot_data.voxel_downsample(nx1, ny1, self.grid_map.resolution)
            with self.grid_map.grid_lock:
                self.grid_map.update_map(nx1, ny1, rx, ry)
                self.img.set_data(self.grid_map.grid.T)

        # 4. VẼ ROBOT (Marker) - Bây giờ rx, ry luôn tồn tại
        self.robot_marker.set_data([rx], [ry])

        # 5. VẼ HƯỚNG ROBOT
        line_len = 0.3      
        lx = [rx, rx + line_len * np.cos(rt)]
        ly = [ry, ry + line_len * np.sin(rt)]
        self.robot_dir_line.set_data(lx, ly)

        # 6. Cập nhật tiêu đề
        degree = np.degrees(rt)
        self.ax.set_title(f"SLAM - Robot: x={rx:.2f}, y={ry:.2f}, θ={degree:.1f}°")
         # VẼ ĐƯỜNG WAYPOINT
        with self.robot_data.data_lock:
            if len(self.robot_data.waypoints_m) > 0:
                waypoints = np.array(self.robot_data.waypoints_m)
                
                # 1. Kẻ đường thẳng màu xanh lá đè lên (Vector line, không phải pixel)
                # zorder=10 đảm bảo đường này nằm trên cùng của Grid
                self.ax.plot(waypoints[:, 0], waypoints[:, 1], 
                            color='#00FF00',      # Màu xanh lá sáng
                            linestyle='-',        # Nối các điểm bằng đường thẳng
                            linewidth=2,          # Độ dày đường kẻ
                            alpha=0.8,            # Độ trong suốt nhẹ
                            zorder=10)            # Lớp hiển thị cao hơn bản đồ
                
                # 2. Vẽ các điểm tròn xanh lá tại mỗi Waypoint
                self.ax.scatter(waypoints[:, 0], waypoints[:, 1], 
                                color='#00FF00',    # Màu xanh lá
                                s=50,               # Kích thước điểm tròn
                                marker='o',         # Hình dạng tròn
                                edgecolors='white', # Viền trắng cho rõ
                                linewidths=1,
                                zorder=11)          # Nằm trên cả đường thẳng
        
            
        return self.img, self.robot_marker, self.robot_dir_line
        
            
    def show(self):

        # Đặt interval cao một chút (200ms) để ổn định

        self.ani = FuncAnimation(self.fig, self.update, interval=500, cache_frame_data=False)

        plt.show()