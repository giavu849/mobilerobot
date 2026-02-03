import paho.mqtt.client as mqtt
import queue
import heapq
import os 
from scipy.spatial import KDTree, cKDTree
import threading
import numpy as np
from math import pi, cos, sin , atan2
from scipy.optimize import least_squares
from visualizer2 import MapVisualizer
# from scipy.optimize import least_squares
# --- Cấu hình MQTT ---
MQTT_BROKER = "127.0.0.1"
TOPIC_DATA = "mqtt/data" # nhận data từ esp32
TOPIC_ASTAR_TARGET = "astar/target"   # nhận tọa độ (x, y) 
TOPIC_ASTAR_STATUS = "astar/status"   # nhận tin hieu esp32
TOPIC_MAP = "mqtt/map"
class EKFSLAM:
    def __init__(self,initial_pose):
        # state =[x,y,theta]
        self.state = np.array(initial_pose,dtype=float)
        # ma trận hiệp phương sai
        self.cov = np.eye(3) * 0.1
    def predict(self,delta_s,delta_theta,motion_cov):
        theta = self.state[2]
        dx = delta_s * cos(theta + delta_theta / 2)
        dy = delta_s * sin(theta + delta_theta / 2)
        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += delta_theta
        self.state[2] = atan2(sin(self.state[2]), cos(self.state[2]))
        F = np.array([
            [1, 0, -delta_s * sin(theta + delta_theta / 2)],
            [0, 1,  delta_s * cos(theta + delta_theta / 2)],
            [0, 0, 1]
        ])
        self.cov = F @ self.cov @ F.T + motion_cov    
    def update(self, observation, obs_cov):
        """
        observation: np.array([z_x, z_y, z_theta]) - Kết quả từ ICP
        obs_cov: np.array(3x3) - Ma trận R (độ nhiễu của ICP)
        """
        # 1. Lấy trạng thái hiện tại (sau bước predict)
        x_est = self.state 
        # 2. Ma trận quan sát H 
        # Vì ICP trả về trực tiếp x, y, theta nên H là ma trận đơn vị 3x3
        H = np.eye(3)

        # 3. Tính Innovation (Sai số giữa thực tế đo được và dự đoán)
        # z là kết quả ICP, H @ x_est là vị trí Robot nghĩ mình đang đứng
        z = observation
        y = z - (H @ x_est)

        # Chuẩn hóa góc quay (rất quan trọng!)
        # Đảm bảo y[2] luôn nằm trong khoảng [-pi, pi] tránh robot xoay vòng lỗi
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi

        # 4. Tính Innovation Covariance (S)
        # S = H*P*H^T + R
        S = H @ self.cov @ H.T + obs_cov

        # 5. Tính Kalman Gain (K)
        # K = P*H^T * inv(S)
        K = self.cov @ H.T @ np.linalg.inv(S)

        # 6. Cập nhật Trạng thái (State Update)
        # x_new = x_old + K * y
        self.state = x_est + (K @ y)

        # 7. Cập nhật Hiệp phương sai (Covariance Update)
        # P_new = (I - K*H) * P_old
        I = np.eye(len(self.state))
        self.cov = (I - K @ H) @ self.cov
        return self.state, self.cov
class OccupancyGridMap:
    def __init__(self, width_m=6, height_m=6, resolution=0.02):
        self.resolution = resolution
        self.width = int(width_m / resolution)
        self.height = int(height_m / resolution)
        self.grid = np.full((self.width, self.height), 0, dtype=np.uint8) 
        self.origin_x = self.width // 2
        self.origin_y = self.height // 2
        self.grid_lock = threading.Lock()
        self.map_origin = -(self.width // 2) * self.resolution
        
        self.save_dir = "D:\map"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def world_to_grid(self, x_m, y_m):
        """Chuyển từ mét sang chỉ số mảng (index)"""
        ix = int(round(x_m / self.resolution)) + self.origin_x
        iy = int(round(y_m / self.resolution)) + self.origin_y
        return ix, iy

    def update_map(self, x_coords_m, y_coords_m, robot_x_m, robot_y_m):
        """Tất cả đầu vào là Mét"""
        r_ix, r_iy = self.world_to_grid(robot_x_m, robot_y_m)

        for x_m, y_m in zip(x_coords_m, y_coords_m):
            ix, iy = self.world_to_grid(x_m, y_m)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                current_val = int(self.grid[ix, iy])
                # 1. Vẽ vùng trống 
                self.ray_trace(r_ix, r_iy, ix, iy)
                # 2. Vật cản (Màu 255)
                self.grid[ix, iy] = min(255, current_val + 255)
    def ray_trace(self, x0, y0, x1, y1, step=1):
    # step=1: tô màu mọi pixel (mặc định)
    # step=2: tô màu cách 1 ô   
    # step=5: tô màu cách 4 ô
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        count = 0 # Biến đếm bước đi
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                # Chỉ tô màu dựa trên tần suất 'step'
                if count % step == 0:
                    if 0 <= x < self.width and 0 <= y < self.height:
                        if self.grid[x, y] != 255: 
                            current_val = int(self.grid[x, y])
                            self.grid[x, y] = max(50, current_val - 25)
                
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
                count += 1
        else:
            err = dy / 2.0
            while y != y1:
                if count % step == 0:
                    if 0 <= x < self.width and 0 <= y < self.height:
                        if self.grid[x, y] != 255: 
                            current_val = int(self.grid[x, y])
                            self.grid[x, y] = max(50, current_val - 25)
                
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                count += 1
    def save_map_to_laptop(self, filename="my_robot_map.npz"):
        """Lưu bản đồ lưới xuống ổ cứng laptop"""
        filepath = os.path.join(self.save_dir, filename)
        with self.grid_lock:
            # Lưu mảng grid và các thông số đi kèm
            np.savez_compressed(filepath, 
                                grid=self.grid, 
                                res=self.resolution, 
                                ox=self.origin_x, 
                                oy=self.origin_y)
        print(f"✅ Đã lưu bản đồ thành công tại: {os.path.abspath(filepath)}")

    def load_map_from_laptop(self, filename="my_robot_map.npz"):
        """Tải bản đồ từ ổ cứng laptop lên lại chương trình"""
        filepath = os.path.join(self.save_dir, filename)
        if os.path.exists(filepath):
            data = np.load(filepath)
            with self.grid_lock:
                self.grid = data['grid']
                self.resolution = float(data['res'])
                self.origin_x = int(data['ox'])
                self.origin_y = int(data['oy'])
            print(f"📂 Đã tải bản đồ thành công từ: {os.path.abspath(filepath)}")
            return True
        else:
            print(f"❌ Không tìm thấy file bản đồ tại: {filepath}")
            return False
my_grid = OccupancyGridMap(width_m=10, height_m=10, resolution=0.02)
class Lidardata:
    
    MAX_DISTANCE = 3  # m
    MIN_DISTANCE = 0.04   # m
    MIN_NEIGHBORS = 4
    def __init__(self,mqtt_client):
        # Đây là nơi lưu trữ dữ liệu để bạn sử dụng
        self.data = {
            # 'angles': [], 
            # 'distances': [], 
            'x_coords': [], 
            'y_coords': [],
            'x_buffers': [], 
            'y_buffers': [],
        }
        self.client = mqtt_client
        # bien khoi dau
        self.robot_distance = 0.0  # m
        self.robot_theta = 0.0 # rad
        self.initial_encoder_left = 0.0
        self.initial_encoder_right = 0.0
        self.packet_counter = 0 # Đếm số gói đã nhận
        self.waypoints_m = [] # Lưu danh sách (x, y) đơn vị Mét để vẽ màu xanh
        #thiết lập biến quản lí
        self.data_lock = threading.Lock()
        self.received_lock=threading.Lock()
        self.pose_lock =threading.Lock()
        self.plot_queue = queue.Queue()
        self.raw_data_queue = queue.Queue(maxsize=100)
        self.counter = 0
        self.slam_counter = 0
        self.load_map = threading.Event()
        # 1. 
        self.slam_queue = queue.Queue(maxsize=20) 
        self.is_running = True
        self.slam_thread = threading.Thread(target=self._slam_core_worker, daemon=True)
        self.slam_thread.start()
         # ... luồng cho pose graph ...
        self.graph_queue = queue.Queue(maxsize=50)
        self.posegraph_is_running = True
        self.graph_thread = threading.Thread(target=self._posegraph_worker, daemon=True)
        self.graph_thread.start()
         # luồng A *
        self.target_goal = None
        self.new_goal_event = threading.Event()
        self.planning_thread = threading.Thread(target=self._planning_loop, daemon=True)
        self.planning_thread.start()
        self.command_done_event = threading.Event()
        # Thông số vật lý
        self.wheel_diameter = 0.07  # m
        self.ppr = 250             # pulses per revolution
        self.wheel_circumference = self.wheel_diameter * pi
        self.wheel_base = 0.159    # m (khoảng cách 2 bánh)
        self.last_encoder_left = None
        self.encoder_right = None
        self.encoder_left = None
        self.last_encoder_right = None
          # Biến định vị (Pose)
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0
        self.ekf_slam = EKFSLAM([self.pose_x, self.pose_y, self.pose_theta])
        #   keyframe for posegraph
        self.keyframe_trans_thresh = 0.5 # m
        self.keyframe_rot_thresh = 10 # deg
        self.current_keyframe_trans = 0
        self.current_keyframe_rots = 0
        self.last_keyframe_pose = None
        self.keyframes = []
        # Danh sách constraint: (i, j, measurement [dx, dy, dtheta], information_matrix 3x3)
        self.edges = []
        # Thêm keyframe đầu tiên
        self._add_keyframe(self.pose_x, self.pose_y, self.pose_theta, np.zeros((0, 2))) # lấy điểm đàu tiên làm keyframe đầu tiê
    def update_to_main(self, base_angle, n, h, distances, gyro_Z, encoder_count, encoder_count2, current_time_raw):
        # 1. Khởi tạo lần đầu
        if self.last_encoder_left is None:
            self.last_encoder_left = encoder_count2
            self.last_encoder_right = encoder_count
            self.last_time = current_time_raw
            return
        # 2. Tính toán sai lệch thời gian
        current_time = current_time_raw
        delta_t = current_time - self.last_time
        self.last_time = current_time
        # 3. Tính delta_s, delta_theta
        
        delta_left = (encoder_count2 - self.last_encoder_left) * self.wheel_circumference / self.ppr
        delta_right = (encoder_count - self.last_encoder_right) * self.wheel_circumference / self.ppr
        delta_s = (delta_left + delta_right) / 2.0
        # gyro_Z_rad = gyro_Z * (pi / 180.0)
        # if gyro_Z_rad < 0.02: gyro_Z_rad = 0
        # delta_theta = gyro_Z_rad * delta_t 
        # omega = gyro_Z_rad
        delta_theta = ( delta_right - delta_left) / self.wheel_base
        omega = delta_theta / delta_t
        if delta_t != 0:
            v = delta_s / delta_t
        else: v = delta_s / 0.01
        # cap nhat encoder
        self.last_encoder_left = encoder_count2
        self.last_encoder_right =encoder_count
        #predict
        motion_cov = np.diag([1e-3, 1e-3, 1e-4])
        self.ekf_slam.predict(delta_s, delta_theta, motion_cov)
        with self.pose_lock:
            self.pose_x = self.ekf_slam.state[0]
            self.pose_y = self.ekf_slam.state[1]
            self.pose_theta = self.ekf_slam.state[2]
            x0 = self.pose_x 
            y0 = self.pose_y 
            theta0 =  self.pose_theta
        # bu tru chuyen dong
        dt =  delta_t / n
        v = delta_s / delta_t if delta_t > 0 else 0 # (m/s)
        t_i = np.flip(np.arange(n-h,n)) * dt # thoi gian tuong doi moi diem 
        # tính local_x, local_y 
        angles_np = (base_angle - np.arange(h) * (22.5 / n)) * (np.pi / 180)
        distances_np = np.array(distances) 
        valid_mask = (distances_np >= self.MIN_DISTANCE) & (distances_np <= self.MAX_DISTANCE)
        valid_angles = angles_np[valid_mask]
        valid_distances = distances_np[valid_mask]  
        valid_t_i = t_i[valid_mask]         
        # diem lidar theo pose x0 
        local_x =0 - v * valid_t_i * np.cos( omega * valid_t_i / 2) +  valid_distances * np.cos(valid_angles - omega * valid_t_i)
        local_y =0 - v * valid_t_i * np.sin( omega * valid_t_i / 2) +  valid_distances * np.sin(valid_angles - omega * valid_t_i)
        # diem lidar theo pose x_i
        local_x_i = valid_distances * np.cos(valid_angles)
        local_y_i = valid_distances * np.sin(valid_angles)
        # Ước lượng pose tại mỗi thời điểm t_i theo global
        theta_i = theta0 - omega * valid_t_i
        x_i = x0 - v * valid_t_i * np.cos(theta0 - omega * valid_t_i / 2)
        y_i = y0 - v * valid_t_i * np.sin(theta0 - omega * valid_t_i / 2)
        # diem lidar theo global 
        global_x = x_i + local_x_i * np.cos(theta_i) - local_y_i * np.sin(theta_i)
        global_y = y_i + local_x_i * np.sin(theta_i) + local_y_i * np.cos(theta_i)
        # ---- LỌC NaN / Inf SAU KHI TÍNH GLOBAL ----
        finite_mask = np.isfinite(global_x) & np.isfinite(global_y)

        global_x = global_x[finite_mask]
        global_y = global_y[finite_mask] 
        if len(global_x) == 0:
            return
        if self.counter == 0:
            self.data['x_coords'].extend(global_x.tolist())
            self.data['y_coords'].extend(global_y.tolist())
            
            # Khi đủ 500 điểm thì vẽ "phát súng đầu tiên"
            if len(self.data['x_coords']) >= 1000:
                self.counter = 1 # Đánh dấu đã vẽ xong lần đầu
                self.plot_queue.put(True)
                    
        self.robot_distance += abs(delta_s)
        self.robot_theta += abs(delta_theta)

        # Chỉ gửi dữ liệu khi robot thực sự di chuyển để giảm tải cho ICP
        if self.counter == 1:
            # 1. Nếu chưa vẽ lần đầu, tích lũy điểm
            self.data['x_buffers'].extend(global_x.tolist())
            self.data['y_buffers'].extend(global_y.tolist())
            if len(self.data['x_buffers']) > 100:
                package = {
                    'dx0': list(self.data['x_buffers']),
                    'dy0': list(self.data['y_buffers']),
                    'lx0': local_x.tolist(),
                    'ly0': local_y.tolist(),
                }
                self.data['x_buffers'] = []
                self.data['y_buffers'] = []
                try:
                    self.slam_queue.put(package, block=False)
    
                
                except queue.Full:
                    # Nếu SLAM quá tải, bỏ qua gói này để lấy gói mới nhất sau
                    pass
        else:
            pass

    def _slam_core_worker(self):
        while self.is_running:
            try:            
                # 1. Lấy dữ liệu từ hàng đợi 
                package = self.slam_queue.get()
                dx0 = package['dx0']
                dy0 = package['dy0']
                lx0 = package['lx0']
                ly0 = package['ly0']
                # GIAI ĐOẠN 1 & 2: Lọc thô và Downsample (đã làm lúc gom gói)
                
                gx0, gy0 = self.remove_outliers(dx0, dy0)
                # vx0, vy0 = self.voxel_downsample( gx0, gy0, my_grid.resolution)
                
                points_filtered = np.column_stack((gx0, gy0))
                # points = self.upsample_simple(local_points_filtered, max_gap=0.005)
                refined_pose, obs_cov, correction  = self.map_to_map(points_filtered, my_grid.grid)    
                self.ekf_slam.update( refined_pose, obs_cov)
                # 4. Cập nhật Pose chính thức và vẽ bản đồ
                rx, ry, rt = correction
                with self.pose_lock:
                    self.pose_x, self.pose_y, self.pose_theta = self.ekf_slam.state.flatten()
        

                # 4. QUAN TRỌNG: VẼ TIẾP VÀO MAP DỰA TRÊN POSE CHUẨN
                # Nếu không có dòng này, Robot sẽ đi vào vùng tối và ICP bị sai
                c, s = np.cos(rt), np.sin(rt)
                gx = rx + (points_filtered[:, 0] * c - points_filtered[:, 1] * s)
                gy = ry + (points_filtered[:, 0] * s + points_filtered[:, 1] * c)
                
                # Cập nhật Visualizer
                with self.data_lock:
                    self.data['x_coords'] = gx.tolist()
                    self.data['y_coords'] = gy.tolist()
                self.plot_queue.put(True)
            

                # pose queue:
                if self.robot_distance >= 0.01 or self.robot_theta >= (2 * pi / 180) :
                    self.robot_distance = 0
                    self.robot_theta = 0
                    fx0, fy0 = self.remove_outliers(lx0, ly0)
                    local_points_filtered = np.column_stack((fx0, fy0))
                    pose_keyframes = {
                        'refined_pose': refined_pose,
                        'local_points_filtered' :local_points_filtered
                    }
                    try:
                        self.graph_queue.put(pose_keyframes, block=False)
                    except queue.Full:
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Lỗi trong luồng SLAM worker: {e}")

    def upsample_simple(self, points, max_gap=0.04):
        """
        Input: points (np.array) shape (N, 2) - Dạng column_stack [vx0, vy0]
        Output: upsampled_points (np.array) shape (M, 2) - M >= N
        """
        if len(points) < 2:
            return points

        # 1. Tính toán hiệu tọa độ và khoảng cách giữa các điểm liên tiếp
        diffs = np.diff(points, axis=0)  # Lấy P_{i+1} - P_i
        dists = np.linalg.norm(diffs, axis=1) # Khoảng cách Euclid giữa các điểm

        upsampled_list = []

        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            dist = dists[i]

            # Thêm điểm gốc p1
            upsampled_list.append(p1)

            # 2. Kiểm tra điều kiện khoảng trống (Gap)
            # Chỉ chèn điểm nếu gap lớn hơn ngưỡng và nhỏ hơn một mức 'hợp lý'
            # (Để tránh nối nhầm giữa 2 bức tường khác nhau hoặc qua các khe cửa)
            if max_gap < dist < (max_gap * 10): 
                num_to_insert = int(dist // max_gap)
                
                # Tạo các trọng số nội suy từ 0 đến 1 (Vectorized)
                # t = [0.2, 0.4, 0.6, 0.8] chẳng hạn
                t = np.linspace(0, 1, num=num_to_insert + 2)[1:-1]
                
                # Nội suy tất cả điểm mới cùng lúc: P = P1 + t * (P2 - P1)
                # Reshape t để nhân được với ma trận diffs[i]
                interp_points = p1 + t[:, np.newaxis] * diffs[i]
                upsampled_list.append(interp_points)

        # Thêm điểm cuối cùng của mảng
        upsampled_list.append(points[-1:])
        
        # 3. Kết hợp lại thành một mảng NumPy duy nhất (Column Stack)
        return np.vstack(upsampled_list)
    def map_to_map(self, recent_global_points, grid_map, max_iterations=60, tolerance=1e-4):
        """
        Trả về: 
        1. refined_pose (Tọa độ mới chuẩn sau ICP)
        2. obs_cov (Ma trận hiệp phương sai)
        3. correction (dx, dy, dtheta  - Độ lệch mà ICP đã tìm ra để khớp vào map)
        """
        old_pose = self.ekf_slam.state
        
        # 1. Lấy tọa độ vật cản từ Grid làm tham chiếu (Target)
        occupied_indices = np.argwhere(grid_map > 200)
        if len(occupied_indices) < 10:
            return old_pose, np.diag([0.2, 0.2, 0.1]), (0, 0, 0)

        map_points = occupied_indices * my_grid.resolution + my_grid.map_origin
        tree = cKDTree(map_points)
        
        # Khởi tạo ma trận biến đổi đồng nhất (Identity)
        T_total = np.eye(3)
        current_pts = np.copy(recent_global_points)

        # Biến lưu khoảng cách để tính RMSE sau này
        distances = np.zeros(len(recent_global_points))
        valid = np.zeros(len(recent_global_points), dtype=bool)

        for i in range(max_iterations):
            distances, indices = tree.query(current_pts)
            
            # Ngưỡng khớp điểm (10cm)
            valid = distances < 0.1
            if np.sum(valid) < 5: break
            
            src = current_pts[valid]
            dst = map_points[indices[valid]]
 
            # SVD để tìm phép biến đổi giữa src và dst
            mu_s = np.mean(src, axis=0)
            mu_d = np.mean(dst, axis=0)
            S = (src - mu_s).T @ (dst - mu_d)
            U, _, Vt = np.linalg.svd(S)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[1,:] *= -1
                R = Vt.T @ U.T
            t = mu_d - R @ mu_s

            # Cập nhật các điểm đang xét để lặp tiếp
            current_pts = (R @ current_pts.T).T + t
            
            # Cập nhật ma trận biến đổi tích lũy T_step * T_total
            T_step = np.eye(3)
            T_step[:2, :2] = R
            T_step[:2, 2] = t
            T_total = T_step @ T_total

            if np.linalg.norm(t) < tolerance: 
                break

        # --- TRÍCH XUẤT PHÉP DỊCH CHUYỂN (CORRECTION) ---
        # dx, dy là các phần tử tịnh tiến trong ma trận 3x3
        dx_corr = T_total[0, 2]
        dy_corr = T_total[1, 2]
        # dtheta trích xuất từ phần ma trận xoay 2x2
        dtheta_corr = np.arctan2(T_total[1, 0], T_total[0, 0])

        # --- TÍNH POSE MỚI ---
        # Cách 1: Áp dụng T_total vào old_pose (Nếu điểm đầu vào là global chuẩn)
        new_x = old_pose[0] + dx_corr
        new_y = old_pose[1] + dy_corr
        new_yaw = old_pose[2] + dtheta_corr
        new_yaw = np.arctan2(np.sin(new_yaw), np.cos(new_yaw)) # Chuẩn hóa góc
        
        refined_pose = np.array([new_x, new_y, new_yaw])
        
        # --- TÍNH ĐỘ TIN CẬY (COVARIANCE) ---
        final_rmse = np.mean(distances[valid]) if np.any(valid) else 0.1
        obs_cov = np.diag([final_rmse*0.5, final_rmse*0.5, final_rmse*1.0])

        return refined_pose, obs_cov, (dx_corr, dy_corr, dtheta_corr)
    def scan_to_map(self, local_points, grid_map, max_iterations=60, tolerance=1e-4):
        """
        Sửa đổi: Thêm điều kiện kiểm tra lỗi để hủy scan nếu không khớp chính xác.
        """
        # Lưu lại pose cũ để trả về nếu khớp thất bại
        previous_pose = self.ekf_slam.state.copy()
        
        # Mặc định Covariance cao khi không khớp được (giảm tin cậy)
        default_cov = np.diag([0.5, 0.5, 0.2]) 

        # 1. Chuyển đổi Grid Map (Giữ nguyên logic của bạn)
        occupied_indices = np.argwhere(grid_map > 200)
        if len(occupied_indices) == 0:
            return previous_pose, default_cov

        map_points = occupied_indices * my_grid.resolution + my_grid.map_origin
        tree = KDTree(map_points)
        current_pose = previous_pose.copy()
        
        # Khai báo biến bên ngoài vòng lặp để dùng sau
        distances = np.array([])
        valid_mask = np.array([])

        for i in range(max_iterations):
            tx, ty, theta = current_pose
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            
            transformed_points = (local_points @ rot_matrix.T) + [tx, ty]
            distances, indices = tree.query(transformed_points)
            
            # Ngưỡng lọc điểm quá xa trong quá trình lặp (ví dụ 0.2m)
            valid_mask = distances < 0.2 
            num_valid = np.sum(valid_mask)
            
            if num_valid < 10: # Tăng lên 10 điểm tối thiểu để ổn định hơn
                break
            
            matched_map_points = map_points[indices[valid_mask]]
            source_points = transformed_points[valid_mask]

            mu_s = np.mean(source_points, axis=0)
            mu_m = np.mean(matched_map_points, axis=0)
            
            S = (source_points - mu_s).T @ (matched_map_points - mu_m)
            U, _, Vt = np.linalg.svd(S)
            R_optimal = Vt.T @ U.T
            
            if np.linalg.det(R_optimal) < 0:
                Vt[1, :] *= -1
                R_optimal = Vt.T @ U.T

            delta_theta = np.arctan2(R_optimal[1, 0], R_optimal[0, 0])
            t_optimal = mu_m - R_optimal @ mu_s
            
            current_pose[0] += t_optimal[0]
            current_pose[1] += t_optimal[1]
            current_pose[2] += delta_theta

            if np.linalg.norm(t_optimal) < tolerance:
                break

        # --- PHẦN KIỂM TRA ĐIỀU KIỆN ĐỂ HỦY SCAN ---
        
        # Tính toán sai số trung bình cuối cùng
        final_error = np.mean(distances[valid_mask]) if np.any(valid_mask) else float('inf')
        num_matched = np.sum(valid_mask)
        
        # Ngưỡng chấp nhận (Bạn có thể điều chỉnh 2 con số này)
        ERROR_THRESHOLD = 0.01  # Nếu sai số trung bình > 8cm -> Hủy
        MIN_MATCH_RATIO = 0.20   # Phải khớp được ít nhất 30% tổng số điểm local

        match_ratio = num_matched / len(local_points)

        if final_error > ERROR_THRESHOLD or match_ratio < MIN_MATCH_RATIO:
            # Nếu không thỏa mãn, trả về Pose cũ và đặt Covariance cực lớn
            print(f"⚠️ Scan rejected: Error {final_error:.3f} > {ERROR_THRESHOLD} or Ratio {match_ratio:.2f}")
            return previous_pose, np.diag([1.0, 1.0, 0.5])

        # 2. Ước lượng Observation Covariance (Chỉ khi khớp thành công)
        # Thêm yếu tố num_matched vào covariance để tăng độ tin cậy khi khớp được nhiều điểm
        obs_cov = np.diag([final_error * 0.3, final_error * 0.3, final_error * 0.6])

        return current_pose, obs_cov
        # lọc nhiễu
    def remove_outliers(self, x_coords, y_coords):
        if not x_coords or not y_coords or len(x_coords) < self.MIN_NEIGHBORS:
            return [], []
        # Trả về ngay nếu không đủ điểm để tính toán
        if len(x_coords) < self.MIN_NEIGHBORS + 1:
            return x_coords, y_coords

        # Chuyển đổi nhanh sang numpy array
        points = np.column_stack((x_coords, y_coords))
        tree = cKDTree(points)

        # 1. Tính toán bán kính động dựa trên k-lân cận
        # Lấy k+1 vì điểm gần nhất luôn là chính nó (distance = 0)
        distances, _ = tree.query(points, k=self.MIN_NEIGHBORS)
        
        # Lấy khoảng cách tới hàng xóm xa nhất trong nhóm k
        dist_k = distances[:, -1]
        mean_d = np.mean(dist_k)
        std_d = np.std(dist_k)
        dynamic_radius = mean_d + 2 * std_d

        # 2. Lọc theo mật độ điểm (Radius Outlier Removal)
        neighbor_counts = tree.query_ball_point(points, r=dynamic_radius, return_length=True)
        mask = neighbor_counts >= self.MIN_NEIGHBORS
        filtered_points = points[mask]

        # Kiểm tra nếu sau khi lọc không còn điểm nào
        if len(filtered_points) == 0:
            return [], []

        # 3. Lọc thống kê dựa trên khoảng cách tới gốc tọa độ (Lidar)
        # Bước này giúp loại bỏ các tia laser "bắn" quá xa hoặc nhiễu không khí
        dist_origin = np.linalg.norm(filtered_points, axis=1)
        mean_o = np.mean(dist_origin)
        std_o = np.std(dist_origin)
        
        # Ngưỡng 3 * std thường an toàn hơn để tránh mất vật cản thật ở xa
        stat_mask = dist_origin <= (mean_o + 3 * std_o)
        final_points = filtered_points[stat_mask]

        return final_points[:, 0].tolist(), final_points[:, 1].tolist()
    # giảm mật độ điểm
    def voxel_downsample(self, x_coords, y_coords, voxel_size):
        # Cách kiểm tra an toàn cho cả List và Numpy Array
        if x_coords is None or len(x_coords) == 0:
            return [], []
        
        # 1. Chuyển sang mảng Numpy (đảm bảo là mảng 2D)
        pts = np.column_stack((x_coords, y_coords))
        
        # 2. Tính chỉ số Voxel cho từng điểm
        # Dùng np.floor để gộp các điểm vào cùng một "ô lưới" (grid cell)
        voxel_indices = np.floor(pts / voxel_size).astype(int)
        
        # 3. Lấy các voxel duy nhất
        # return_index=True trả về vị trí của điểm đầu tiên xuất hiện trong mỗi voxel
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        
        downsampled_pts = pts[unique_indices]
        
        # Trả về list
        return downsampled_pts[:, 0].tolist(), downsampled_pts[:, 1].tolist()
    #####################
     # KHối lệnh cho pose graph
     #################
    def _add_keyframe(self, x, y, theta, local_scan):
        # 1. Tạo node mới
        new_node = {
            'pose': np.array([x, y, theta]),
            'scan': local_scan.copy() if local_scan is not None else None
        }
        self.keyframes.append(new_node)
        new_node_idx = len(self.keyframes) - 1
        
        # Nếu đây là Keyframe đầu tiên (idx = 0)
        if new_node_idx == 0:
            self.last_keyframe_pose = new_node
            # Kết thúc hàm vì không có node trước đó để tạo Edge
            return 

        # 2. Lấy pose của node trước đó (i)
        prev_node_idx = new_node_idx - 1
        prev_node = self.last_keyframe_pose # Đây là dictionary {'pose': ..., 'scan': ...}
        
        # SỬA LỖI TRUY CẬP TẠI ĐÂY:
        prev_pose_values = prev_node['pose'] # Lấy array [x, y, theta]
        
        # Tính sai lệch trong hệ tọa độ Global (New - Prev)
        dx_global = x - prev_pose_values[0]
        dy_global = y - prev_pose_values[1]
        dtheta = theta - prev_pose_values[2]
        
        # Chuẩn hóa dtheta về khoảng [-pi, pi]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        
        # Xoay về hệ tọa độ của prev_pose để lấy dx, dy local
        # Sử dụng góc theta của node phía trước
        cos_p = np.cos(prev_pose_values[2])
        sin_p = np.sin(prev_pose_values[2])
        
        dx_local = dx_global * cos_p + dy_global * sin_p
        dy_local = -dx_global * sin_p + dy_global * cos_p

        measurement = [dx_local, dy_local, dtheta]

        # 3. Information Matrix (giữ nguyên)
        info_matrix = np.array([
            [20.0, 0.0,  0.0],
            [0.0,  20.0, 0.0],
            [0.0,  0.0,  20.0]
        ])

        # 4. Lưu cạnh
        new_edge = (prev_node_idx, new_node_idx, measurement, info_matrix)
        self.edges.append(new_edge)

        # 5. Cập nhật last_keyframe_pose cho lần kế tiếp
        self.last_keyframe_pose = new_node
    def optimize_posegraph(self):
        """
        Tối ưu hóa Pose Graph để khử sai số tích lũy (Drift).
        Sử dụng Scipy Least Squares (Levenberg-Marquardt).
        """
        if len(self.keyframes) < 2 or not self.edges:
            return

        # 1. Chuẩn bị x0 (Giá trị khởi tạo): Phẳng hóa danh sách poses thành mảng 1D
        # Ta giữ Keyframe đầu tiên (0,0,0) cố định để làm mỏ neo (Anchor)
        all_poses = np.array([kf['pose'] for kf in self.keyframes])
        x0 = all_poses[1:].flatten()

        def residual_function(x_flat):
            # Đưa x_flat về dạng [node1, node2, ...] với mỗi node là [x, y, theta]
            # Thêm node 0 (cố định) vào đầu
            current_poses = np.zeros((len(self.keyframes), 3))
            current_poses[1:] = x_flat.reshape(-1, 3)
            current_poses[0] = all_poses[0] # Node đầu tiên luôn là [0,0,0] hoặc pose ban đầu

            residuals = []

            for (i, j, measurement, info_matrix) in self.edges:
                # measurement: [dx_local, dy_local, dtheta] từ ICP/Odometry
                p_i = current_poses[i]
                p_j = current_poses[j]

                # Tính sai lệch dự đoán (Prediction) giữa node i và j theo hệ tọa độ i
                cos_i = np.cos(p_i[2])
                sin_i = np.sin(p_i[2])
                
                dx_global = p_j[0] - p_i[0]
                dy_global = p_j[1] - p_i[1]
                
                # Chuyển sai lệch global sang hệ tọa độ local của node i
                pred_dx = dx_global * cos_i + dy_global * sin_i
                pred_dy = -dx_global * sin_i + dy_global * cos_i
                pred_dtheta = p_j[2] - p_i[2]
                
                # Chuẩn hóa góc quay
                pred_dtheta = (pred_dtheta + np.pi) % (2 * np.pi) - np.pi
                
                # Tính sai số (Error)
                err_x = pred_dx - measurement[0]
                err_y = pred_dy - measurement[1]
                err_theta = (pred_dtheta - measurement[2] + np.pi) % (2 * np.pi) - np.pi

                # Nhân với trọng số (sqrt của Information Matrix) để có độ ưu tiên
                # Ở đây ta dùng căn bậc hai của các phần tử đường chéo info_matrix
                weight = np.sqrt(np.diag(info_matrix))
                residuals.extend([err_x * weight[0], err_y * weight[1], err_theta * weight[2]])

            return np.array(residuals)

        # 2. Chạy tối ưu hóa
        # 'lm' là Levenberg-Marquardt, cực kỳ ổn định cho SLAM
        result = least_squares(residual_function, x0, method='lm', ftol=1e-4, xtol=1e-4)

        # 3. Cập nhật lại kết quả vào hệ thống
        optimized_poses_flat = result.x.reshape(-1, 3)
        
        with self.pose_lock:
            # Cập nhật danh sách keyframes
            for idx in range(1, len(self.keyframes)):
                self.keyframes[idx]['pose'] = optimized_poses_flat[idx-1]
            
            # Cập nhật state hiện tại của EKF về vị trí mới nhất đã tối ưu
            latest_optimized_pose = self.keyframes[-1]['pose']
            self.ekf_slam.state = latest_optimized_pose.copy()
            self.pose_x, self.pose_y, self.pose_theta = latest_optimized_pose
        # --- PHẦN QUAN TRỌNG: VẼ LẠI TOÀN BỘ GRID MAP ---
        with self.data_lock:
            # Xóa dữ liệu cũ để vẽ lại từ đầu dựa trên pose đã tối ưu
            self.data['x_coords'] = []
            self.data['y_coords'] = []
            
            for kf in self.keyframes:
                if kf['scan'] is not None:
                    pose = kf['pose']
                    scan = kf['scan']
                    
                    # Chuyển local scan sang global dựa trên pose đã tối ưu
                    c, s = np.cos(pose[2]), np.sin(pose[2])
                    gx = pose[0] + (scan[:, 0] * c - scan[:, 1] * s)
                    gy = pose[1] + (scan[:, 0] * s + scan[:, 1] * c)
                    
                    self.data['x_coords'].extend(gx.tolist())
                    self.data['y_coords'].extend(gy.tolist())
            
            # Kích hoạt vẽ lại
            self.plot_queue.put(True)
    def _posegraph_worker(self):
        """Luồng chuyên biệt để tối ưu hóa Pose Graph"""
        while self.posegraph_is_running:
            try:
                # Đợi cho đến khi có tín hiệu set() từ luồng chính
                keyget = self.graph_queue.get()
                refined_pose = keyget['refined_pose']
                local_points_filtered = keyget['local_points_filtered']
                icp_x, icp_y, icp_theta = refined_pose.flatten()
                self._add_keyframe(icp_x, icp_y, icp_theta, local_scan=local_points_filtered)
                match_idx = self.check_loop_closure(refined_pose)
                if match_idx is not None:
                    # Lấy dữ liệu điểm (đã qua lọc nhiễu và downsample)
                    current_scan = local_points_filtered
                    past_scan = self.keyframes[match_idx]['scan']
                    # Chạy ICP để tìm sai số thực tế giữa 2 vị trí
                    dx, dy, dyaw = self.run_icp(current_scan, past_scan)
                    info_loop = np.eye(3) * 50 
                    self.edges.append((len(self.keyframes)-1, match_idx, [dx, dy, dyaw], info_loop))
                elif len(self.keyframes) % 5 == 0:
                    # Hoặc cứ mỗi 10 keyframes thì tối ưu 1 lần
                    self.optimize_posegraph()   
            except queue.Empty:
                continue # Nếu không có dữ liệu thì lặp lại vòng while
            except Exception as e:
                print(f"Lỗi PoseGraph Worker: {e}")
    ###################################
### LOOP_closure
###############################
# icp for loopclosure
    def run_icp(self, source_points, target_points, max_iterations=20, tolerance=0.001):
        """
        Tìm phép biến đổi để khớp source_points vào target_points.
        source_points: np.array([[x1, y1], [x2, y2], ...]) - Scan hiện tại
        target_points: np.array([[x1, y1], [x2, y2], ...]) - Scan cũ trong quá khứ
        """
        if len(source_points) == 0 or len(target_points) == 0:
            return 0.0, 0.0, 0.0
        src = np.copy(source_points)
        dst = np.copy(target_points)
        
        # Xây dựng cây để tìm lân cận nhanh hơn
        tree = cKDTree(dst)
        
        prev_error = 0
        T = np.eye(3) # Ma trận biến đổi tích lũy

        for i in range(max_iterations):
            # 1. Tìm các điểm gần nhất trong target cho mỗi điểm trong source
            distances, indices = tree.query(src)
            matched_dst = dst[indices]
            
            # Tính sai số trung bình
            mean_error = np.mean(distances)
            if abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error
            
            # 2. Tính toán trọng tâm (Centroid)
            mu_src = np.mean(src, axis=0)
            mu_dst = np.mean(matched_dst, axis=0)
            
            # 3. Tính ma trận hiệp phương sai
            S = (src - mu_src).T @ (matched_dst - mu_dst)
            
            # 4. Sử dụng SVD để tìm ma trận xoay R
            U, _, Vt = np.linalg.svd(S)
            R = Vt.T @ U.T
            
            # Đảm bảo là phép xoay thuần túy (tránh hiện tượng phản xạ)
            if np.linalg.det(R) < 0:
                Vt[1, :] *= -1
                R = Vt.T @ U.T
                
            # 5. Tính toán vector tịnh tiến t
            t = mu_dst - R @ mu_src
            
            # 6. Cập nhật các điểm source cho lần lặp tiếp theo
            src = (R @ src.T).T + t
            
            # 7. Cập nhật ma trận biến đổi tổng hợp T_step
            T_step = np.eye(3)
            T_step[:2, :2] = R
            T_step[:2, 2] = t
            T = T_step @ T # Tích lũy biến đổi
            
        # Trả về delta_x, delta_y, delta_yaw
        dx = T[0, 2]
        dy = T[1, 2]
        dyaw = np.arctan2(T[1, 0], T[0, 0])
        
        return dx, dy, dyaw
    def check_loop_closure(self, current_pose, dist_threshold=1.0, skip_recent=30):
        """
        Tìm xem pose hiện tại có gần pose nào trong quá khứ không.
        skip_recent: Bỏ qua n keyframe gần nhất để tránh nhận nhầm chính mình.
        """
        if len(self.keyframes) < skip_recent + 1:
            return None

        # Lấy tọa độ x, y của các keyframe cũ (loại bỏ các frame gần nhất)
        past_poses = np.array([kf['pose'][:2] for kf in self.keyframes[:-skip_recent]])
        curr_xy = current_pose[:2]

        # Tính khoảng cách Euclidean
        distances = np.linalg.norm(past_poses - curr_xy, axis=1)
        min_idx = np.argmin(distances)

        if distances[min_idx] < dist_threshold:
            return min_idx  # Trả về ID của keyframe cũ khớp với hiện tại
        return None
    #######################
    ## tìm đường đi A*
    ####################
    def _heuristic(self, a, b):
        """Tính khoảng cách Euclid giữa 2 điểm (x, y)"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    def _reconstruct_path(self, came_from, current):
        """Duyệt ngược từ đích về đầu để lấy danh sách tọa độ"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Đảo ngược lại để có từ Start -> Goal   
    def run_a_star(self, grid_map, start, goal):
        """
        grid_map: numpy array (0: trống, 255: vật cản, 100: vùng đã khám phá)
        start, goal: tuple (ix, iy)
        """
        rows, cols = grid_map.shape
        
        # open_set lưu: (f_score, x, y)
        open_set = []
        heapq.heappush(open_set, (0, start[0], start[1]))
        
        came_from = {}
        g_score = {start: 0}
        
        # Các hướng di chuyển: 8 hướng (ngang, dọc, chéo)
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while open_set:
            # Lấy ô có f_score thấp nhất (nhờ heapq)
            _, curr_x, curr_y = heapq.heappop(open_set)
            current = (curr_x, curr_y)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)

                # Kiểm tra ranh giới bản đồ
                if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                    continue
                
                # Kiểm tra vật cản (Chỉ đi vào vùng có giá trị 100 - vùng trống)
                # Nếu giá trị là 255 (vật cản) hoặc 0 (chưa biết), coi như không đi được
                val = grid_map[neighbor[0], neighbor[1]]
                # if val > 100 or val == 0: # Không đi vào vật cản hoặc vùng chưa biết
                if val > 150:
                    continue

                # Chi phí: đường chéo là 1.41, đường thẳng là 1.0   
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor[0], neighbor[1]))

        return None # Không tìm thấy đường
    def is_clear_path(self, grid, start, end):
        """Kiểm tra xem từ start đến end có vướng vật cản không"""
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # Nếu chạm vật cản (>) hoặc vùng chưa biết (0)
            # if grid[x0, y0] > 100 or grid[x0, y0] == 0:
            if grid[x0, y0] > 150: 
                return False
            if x0 == x1 and y0 == y1:   
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return True
    def get_all_waypoints(self, grid, path):
        """
        Nén đường đi A* thành danh sách các điểm nút (Waypoints).
        Hành động: Từ điểm hiện tại, tìm điểm xa nhất trong path có thể đi thẳng tới,
        sau đó nhảy đến điểm đó và lặp lại cho đến Goal.
        """
        if not path:
            return []
        if len(path) < 2:
            return path

        waypoints = [path[0]]  # Điểm bắt đầu luôn là waypoint đầu tiên
        current_idx = 0
        goal_idx = len(path) - 1

        while current_idx < goal_idx:
            # Duyệt ngược từ cuối đường đi về vị trí hiện tại
            found_next = False
            for i in range(goal_idx, current_idx, -1):  
                if self.is_clear_path(grid, path[current_idx], path[i]):
                    waypoints.append(path[i])
                    current_idx = i  # Nhảy đến điểm vừa tìm được
                    found_next = True
                    break
            
            # Phòng trường hợp lỗi logic không tìm thấy điểm kế tiếp (hiếm gặp)
            if not found_next:
                current_idx += 1
                waypoints.append(path[current_idx])
        return waypoints
    def generate_commands(self, waypoints):
        """
        Tạo danh sách lệnh di chuyển.
        Góc quay tương đối ngắn nhất: Sang Phải (+), Sang Trái (-), không quá 180 độ.
        """
        if not waypoints or len(waypoints) < 2:
            return []

        commands = []
        
        # 1. Khởi tạo vị trí và hướng hiện tại
        with self.pose_lock:
            current_pos = np.array([self.pose_x, self.pose_y])
            last_yaw = self.pose_theta  

        # 2. Duyệt từ waypoint thứ 2 (chỉ số 1)
        for i in range(1, len(waypoints)):
            tx_m = (waypoints[i][0] - my_grid.origin_x) * my_grid.resolution
            ty_m = (waypoints[i][1] - my_grid.origin_y) * my_grid.resolution
            target_pos = np.array([tx_m, ty_m])

            # Tính toán vector khoảng cách
            diff = target_pos - current_pos
            distance = np.linalg.norm(diff)
            dist_meters = distance   # Giả sử 5cm/ô

            # 3. Tính góc tuyệt đối của mục tiêu (Target Heading)
            target_angle_rad = np.arctan2(diff[1], diff[0])

            # 4. Tính góc quay tương đối ban đầu
            relative_angle = np.degrees(target_angle_rad - last_yaw)
            
            # 5. CHUẨN HÓA VỀ KHOẢNG [-180, 180] (Góc quay ngắn nhất)
            # Bước a: Đưa về [0, 360)
            relative_angle = relative_angle % 360.0
            # Bước b: Nếu > 180 thì trừ 360 để sang góc âm (quay ngược lại cho gần)
            if relative_angle > 180:
                relative_angle -= 360
            # 7. Thêm lệnh vào danh sách
            if abs(relative_angle) > 1.0: # Ngưỡng 1 độ để tránh rung
                commands.append(f"ROTATE {relative_angle:.2f}")
            if dist_meters > 0.005: 
                commands.append(f"MOVE {dist_meters:.2f}")
            # 8. Cập nhật trạng thái giả định
            current_pos = target_pos
            last_yaw = target_angle_rad
        return commands
    def _planning_loop(self):
        while True:
            # Đợi cho đến khi có đích mới
            self.new_goal_event.wait() 
            
            if self.target_goal is not None:
                # Tính toán A* và nén đường đi
                with self.pose_lock:
                    # Chuyển từ mét sang chỉ số grid
                    ix, iy = my_grid.world_to_grid(self.pose_x, self.pose_y)
                    start_pos = (ix, iy)
                current_target = self.target_goal    
                with my_grid.grid_lock:
                    path = self.run_a_star(my_grid.grid, start_pos, current_target  )
                
                if path:
                    with my_grid.grid_lock:
                        waypoints = self.get_all_waypoints(my_grid.grid, path)
                     # --- CHỈ LƯU TỌA ĐỘ WAYPOINT (MÉT) ---
                    temp_waypoints = []
                    for (wx, wy) in waypoints:
                        mx = (wx - my_grid.origin_x) * my_grid.resolution
                        my = (wy - my_grid.origin_y) * my_grid.resolution
                        temp_waypoints.append([mx, my])
                    
                    with self.data_lock:
                        self.waypoints_m = temp_waypoints # Lưu để visualizer lấy vẽ
                    
                    # Thông báo visualizer cập nhật
                    self.plot_queue.put(True)
                    
                    commands = self.generate_commands(waypoints)
                    
                    # Trước khi bắt đầu chuỗi lệnh, xóa event đích mới để sẵn sàng nhận đích tiếp theo
                    self.new_goal_event.clear()

                    for cmd in commands:
                        # KIỂM TRA: Nếu trong lúc đang chạy mà có Goal mới được gửi đến
                        if self.new_goal_event.is_set():
                            print("Có mục tiêu mới! Hủy chuỗi lệnh cũ.")
                            break

                        # BƯỚC 1: Xóa trạng thái 'done' cũ
                        self.command_done_event.clear()

                        # BƯỚC 2: Gửi lệnh xuống ESP32
                        print(f"Đang gửi: {cmd}")
                        self.client.publish("mqtt/control", cmd)

                        # BƯỚC 3: Đợi ESP32 gửi 'done' lên topic astar/status
                        # Nó sẽ dừng tại đây cho đến khi on_message gọi .set()
                        success = self.command_done_event.wait(timeout=5.0) 

                        if not success:
                            print(f"Lỗi: Quá thời gian chờ phản hồi cho lệnh {cmd}")
                            break
                    self.target_goal = None
                    print("Đã thực hiện xong chuỗi lệnh hoặc bị ngắt.")

# --- Các hàm Callback MQTT ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        client.subscribe([(TOPIC_DATA, 0), (TOPIC_ASTAR_TARGET, 0), ("astar/status", 0)])
        print("Đã đăng ký nhận dữ liệu từ tất cả các Topic.")

def on_message(client, userdata, message):
    try:
        if message.topic == TOPIC_DATA:
            raw_data = message.payload.decode('utf-8').strip()
            rows = raw_data.split('\n')
            
            for row in rows:
                if not row.strip(): continue
                cols = row.split(',')
                
                if len(cols) >= 32: # Đảm bảo đủ số cột tối thiểu
                    # Trích xuất dữ liệu
                    raw_base_angle = - float(cols[0]) / 10 + 88
                    raw_n_points =  int(cols[1])   # số điểm có trong  1 gói
                    h_points    = int(cols[2])  # số điểm lấy trong 1 gói 
                    raw_distances  = np.array(cols[3:h_points+3], dtype=float) / 4000
                    raw_gyro_Z     = float(cols[28]) / 1000.0
                    raw_encoder_count = float(cols[29])
                    raw_encoder_count2 = float(cols[30])
                    current_time =  float(cols[31]) / 1000
                    package = (raw_base_angle, raw_n_points, h_points, raw_distances, raw_gyro_Z, raw_encoder_count, raw_encoder_count2,    current_time)
                    try:
                        robot_data.raw_data_queue.put(package, block=False)
                    except queue.Full:
                        pass # Xử lý nếu queue đầy
        # Xử lý tọa độ đích cho A*
        elif message.topic == TOPIC_ASTAR_TARGET:
            # Giả định tin nhắn gửi lên có định dạng chuỗi: "1.5,2.0" (x,y tính bằng mét)
            raw_target = message.payload.decode('utf-8').strip()
            target_parts = raw_target.split(',')
            
            if len(target_parts) == 2:  
                target_x = float(target_parts[0])
                target_y = float(target_parts[1])
                tx, ty = my_grid.world_to_grid(target_x, target_y)
                # Lưu điểm đích vào robot_data để luồng worker hoặc visualizer xử lý
                robot_data.target_goal = (tx, ty)
                    # Cờ báo hiệu cần tính toán lại đường đi
                   # SỬA TẠI ĐÂY: Sử dụng phương thức .set() của đối tượng Event
                robot_data.new_goal_event.set() 
                print(f"Đã nhận mục tiêu mới: {target_x}, {target_y}")
        elif message.topic == TOPIC_ASTAR_STATUS:
            payload = message.payload.decode().lower()
            if payload == "done":
                # print("ESP32 đã hoàn thành lệnh, cho phép gửi lệnh tiếp theo.")
                robot_data.command_done_event.set() # Mở khóa cho luồng planning
        elif message.topic == TOPIC_MAP:
            payload = message.payload.decode('utf-8').strip().lower()
            
            if payload == "save":
                # Gọi hàm lưu xuống laptop
                my_grid.save_map_to_laptop()
                
            elif payload == "load":
                # Gọi hàm tải từ laptop lên
                if my_grid.load_map_from_laptop():
                    # Sau khi load xong, cần báo cho Visualizer vẽ lại màn hình
                    robot_data.load_map.set()
            return
    except Exception as e:
        print(f"Lỗi xử lý tin nhắn tại topic {message.topic}: {e}")

# --- Thiết lập Client MQTT ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="Python_Astar")
client.max_inflight_messages_set(100)
client.max_queued_messages_set(100)
client.on_connect = on_connect
client.on_message = on_message
robot_data = Lidardata(client)
# hàm vẽ map
visualizer = MapVisualizer(robot_data, my_grid)

# --- Chạy chương trình ---
if __name__ == "__main__":
    try:    
        client.connect(MQTT_BROKER, 1883, 60)
        client.loop_start() 

        def data_received_loop():
            while True:
                try:
                    # Lấy data từ queue thay vì check biến has_new_data
                    package = robot_data.raw_data_queue.get(timeout=0.1)
                    robot_data.update_to_main(*package)
                except queue.Empty:
                    continue
                
        # Chạy luồng xử lý dữ liệu
        data_thread = threading.Thread(target=data_received_loop, daemon=True)
        data_thread.start()
        # Luồng chính chạy hiển thị (Luôn để cuối cùng vì nó là hàm chặn)
        print(">>> Đang hiển thị bản đồ...")
        visualizer.show()

    except KeyboardInterrupt:
        print("\nĐang dừng...")
        client.loop_stop()
        client.disconnect()
