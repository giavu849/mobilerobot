import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import numpy as np
from matplotlib.transforms import Affine2D
import queue

class MapVisualizer:
    def __init__(self, robot_data, grid_map):
        self.robot_data = robot_data
        self.grid_map = grid_map
        self.user_view_angle = 0.0

        # 1. Khởi tạo Figure
        self.fig = plt.figure(figsize=(8, 9))
        self.ax = self.fig.add_axes([0.1, 0.25, 0.8, 0.65])
        self.ax.set_xlim([-4, 4])
        self.ax.set_ylim([-4, 4])
        
        # Tạo đối tượng Transform để xoay
        self.view_transform = Affine2D()

        # 2. Khởi tạo các đối tượng vẽ CỐ ĐỊNH (Reusable Objects)
        with self.grid_map.grid_lock:
            self.img = self.ax.imshow(self.grid_map.grid.T, origin='lower',
                                  cmap='gray_r', vmin=0, vmax=255,
                                  extent=[-4, 4, -4, 4])

        self.robot_marker, = self.ax.plot([], [], 'ro', markersize=10, zorder=5, label="Robot")
        self.robot_dir_line, = self.ax.plot([], [], 'r-', linewidth=1, zorder=5)
        
        # ---  Khởi tạo trước các đối tượng Waypoint ---
        self.path_line, = self.ax.plot([], [], color='#00FF00', linewidth=2, zorder=10)
        self.path_points = self.ax.scatter([], [], color='#00FF00', s=50, edgecolors='white', zorder=11)

        # Áp dụng Transform
        transform = self.view_transform + self.ax.transData
        self.img.set_transform(transform)
        self.robot_marker.set_transform(transform)
        self.robot_dir_line.set_transform(transform)
        self.path_line.set_transform(transform)
        # Lưu ý: Scatter transform phức tạp hơn nên cập nhật offset thủ công trong update sẽ tốt hơn
        # Hoặc gán trực tiếp: self.path_points.set_transform(transform)

        # Slider điều khiển xoay
        ax_slider = self.fig.add_axes([0.2, 0.1, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Rotate View', 0.0, 360.0, valinit=0.0)
        self.slider.on_changed(self.update_view_angle)
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
    def update_view_angle(self, val):
        self.user_view_angle = val
        self.view_transform.clear()
        self.view_transform.rotate_deg(val)
        self.fig.canvas.draw_idle()

    def update(self, frame):
        # 1. Cập nhật Pose
        with self.robot_data.pose_lock:
            rx, ry, rt = self.robot_data.pose_x, self.robot_data.pose_y, self.robot_data.pose_theta

        # 2. Xử lý Map reset
        # Lưu ý: Check đúng tên biến reset_map hay load_map từ Lidardata
        

        # 3. Cập nhật Grid Map khi có dữ liệu mới
        if not self.robot_data.plot_queue.empty():
            try:
                self.robot_data.plot_queue.get_nowait()
                if hasattr(self.robot_data, 'reset_map') and self.robot_data.reset_map.is_set():
                    self.grid_map.clear_map()
                    self.robot_data.reset_map.clear()
                with self.robot_data.data_lock:
                    nx = self.robot_data.data['x_coords']
                    ny = self.robot_data.data['y_coords']
                    # Không xóa ngay dữ liệu gốc để class Lidardata có thể reuse
                
                if nx:
                    nx_f, ny_f = self.robot_data.remove_outliers(nx, ny)
                    with self.grid_map.grid_lock:
                        self.grid_map.update_map(nx_f, ny_f, rx, ry)
                        self.img.set_data(self.grid_map.grid.T)
            except:
                pass

        # 4. Vẽ Robot
        self.robot_marker.set_data([rx], [ry])
        line_len = 0.3      
        self.robot_dir_line.set_data([rx, rx + line_len * np.cos(rt)], [ry, ry + line_len * np.sin(rt)])

        # 5. Vẽ Waypoint 
        with self.robot_data.data_lock:
            if self.robot_data.waypoints_m:
                wp = np.array(self.robot_data.waypoints_m)
                self.path_line.set_data(wp[:, 0], wp[:, 1])
                self.path_points.set_offsets(wp)
            else:
                self.path_line.set_data([], [])
                self.path_points.set_offsets(np.empty((0, 2)))

        self.ax.set_title(f"SLAM - Pose: {rx:.2f}, {ry:.2f}, {np.degrees(rt):.1f}°")
        return self.img, self.robot_marker, self.robot_dir_line, self.path_line, self.path_points

    def show(self):
        # Để interval=100 hoặc 200 tùy tốc độ xử lý CPU
        self.ani = FuncAnimation(self.fig, self.update, interval=400, blit=False )
        plt.show()
