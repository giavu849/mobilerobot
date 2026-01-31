  #include <HardwareSerial.h>
  #include <Wire.h>
  #include <WiFi.h>
  #include <PubSubClient.h>
  #include <Arduino.h>

  // Robot parameters 
  const float WHEEL_RADIUS = 0.035;       // m
  const float WHEEL_BASE   = 0.159;       // m (khoảng cách tâm hai bánh)
  const int   ENCODER_PPR  = 250;        // xung/rev
  const float WHEEL_CIRCUMFERENCE = 2 * PI * WHEEL_RADIUS;
  // === MOTION COMMAND STATE MACHINE ===
  enum State { IDLE, RUN_DIST, RUN_ANGLE } state = IDLE;
  long targetTicks = 0;
  long startEnc1 = 0, startEnc2 = 0;
  float targetRad = 0;
  float Ksync = 1.5;

  // --- Cấu hình WiFi & MQTT ---
  const char* ssid = "KhueT4";
  const char* password = "tinoichonhe";
  const char* mqtt_server = "192.168.88.107";

  WiFiClient espClient;
  PubSubClient client(espClient);

  // --- CẤU HÌNH CHÂN 2 motor---
    const byte motor1_ena = 15;
    const byte motor1_in1 = 18;
    const byte motor1_in2 = 19;
    const byte encoderPinM1 = 27;
      
    const byte motor2_enb = 26;
    const byte motor2_in3 = 23;
    const byte motor2_in4 = 25;
    const byte encoderPinM2 = 4;

    // --- THÔNG SỐ PI ---
    float Kp = 30.0;         
    float Ki = 6;         
    int setpoint = 0.0; // (+) là Tiến, (-) là Lùi, 0 là Dừng
    int setpoint2 = 0.0; // (+) là Tiến, (-) là Lùi, 0 là Dừng
    float input = 0, output = 0, error = 0, integral = 0;
    float input2 = 0, output2 = 0, error2 = 0, integral2 = 0;
    float minPWM = 100.0;    

    // --- BIẾN ENCODER ---
    volatile long pulseCount = 0;
    long lastPulseCount = 0;
    int direction = 0;
    volatile long pulseCount2 = 0;
    long lastPulseCount2 = 0;
    int direction2 = 0;
    const int encoderMin = -32768;
    const int encoderMax = 32767;

    void IRAM_ATTR handleEncoder() {
      if (direction == 1)
      pulseCount ++; 
      else if ( direction == -1 )
      pulseCount --;
    }
    void IRAM_ATTR handleEncoder2() {
      if (direction2 == 1)
      pulseCount2 ++; 
      else if (direction2 == -1)
      pulseCount2 --;
    }
  // --- MPU6050 ---
  #define MPU6050_CONFIG 0x1A 
  const int MPU = 0x68; 
  float gyroZBias = 0.0; 
  float gyZ_deg = 0.0;

  // --- LiDAR ---
  HardwareSerial lidarSerial(1);  
  const uint8_t HEADER_BYTE = 0xAA;
  const int MAX_PACKET_SIZE = 200;
  const int MAX_DATA_SIZE = 28;
  const int BUFFER_SIZE = 20;
  const int motorPin1 = 32; 
  const int motorPin2 = 33; 
  #define RX_PIN 16
  #define BAUDRATE_SENSOR 115200

  uint8_t packet[MAX_PACKET_SIZE];
  int16_t dataArr[MAX_DATA_SIZE];  
  int32_t lidarBuffer[32 * BUFFER_SIZE];
  bool waitPacket = true;
  unsigned int packetIndex = 0;
  int packet_size = 0;
  int bufferIndex = 0;

  // --- Prototypes ---
  void reconnect();
  void callback(char* topic, byte* payload, unsigned int length);
  void handleLidarData();
  void sendDataToClient();
  uint16_t calcChecksum(uint8_t data[], int size);
  void decodePacket(uint8_t packet[], int packet_size);
  void resetPacket();

  void setup() {
    // 1. ƯU TIÊN SỐ 1: Cấu hình Motor và ép dừng ngay lập tức
    // Việc này chỉ mất vài miligiây, triệt tiêu nhiễu ngay khi chip vừa chạy code
    pinMode(motor1_ena, OUTPUT);
    pinMode(motor1_in1, OUTPUT);
    pinMode(motor1_in2, OUTPUT);
    pinMode(motor2_enb, OUTPUT);
    pinMode(motor2_in3, OUTPUT);
    pinMode(motor2_in4, OUTPUT);

    digitalWrite(motor1_ena, LOW);
    digitalWrite(motor1_in1, LOW);
    digitalWrite(motor1_in2, LOW);
    digitalWrite(motor2_enb, LOW);
    digitalWrite(motor2_in3, LOW);
    digitalWrite(motor2_in4, LOW);
    
    // Gọi hàm stop để chắc chắn mức logic trong biến phần mềm đồng bộ
    stopmotor(); 

    // 2. Khởi tạo Serial
    Serial.begin(115200);

    // 3. Cấu hình Encoder (Tránh khai báo encoder trước motor để không nhận xung ảo)
    pinMode(encoderPinM1, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(encoderPinM1), handleEncoder, RISING);
    pinMode(encoderPinM2, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(encoderPinM2), handleEncoder2, RISING);

    // 4. Khởi tạo I2C & MPU6050
    Wire.begin(21, 22);
    Wire.beginTransmission(MPU);
    Wire.write(0x6B); Wire.write(0); Wire.endTransmission(true);
    Wire.beginTransmission(MPU);
    Wire.write(MPU6050_CONFIG); Wire.write(0x04); Wire.endTransmission(true);
    Wire.beginTransmission(MPU);
    Wire.write(0x1B); Wire.write(0x00); Wire.endTransmission(true);

    // 5. Hiệu chỉnh Gyro (Mất khoảng 2-3 giây)
    Serial.println("Calibrating Gyro...");
    float sumGyroZ = 0.0;
    for (int i = 0; i < 1000; i++) {
        Wire.beginTransmission(MPU);
        Wire.write(0x47);
        Wire.endTransmission(false);
        Wire.requestFrom(MPU, 2, true);
        if (Wire.available() >= 2) {
            int16_t rawGyZ = Wire.read() << 8 | Wire.read();
            sumGyroZ += (rawGyZ / 131.0);
        }
        delay(2);
    }
    gyroZBias = sumGyroZ / 1000.0;
    Serial.println("Done Calib.");

    // 6. Kết nối WiFi (Mất nhiều thời gian nhất, có thể gây sụt áp làm nhiễu motor)
    Serial.print("Connecting WiFi");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi Connected");

    // 7. Cấu hình các thành phần còn lại
    pinMode(motorPin1, OUTPUT);
    pinMode(motorPin2, OUTPUT);
    analogWrite(motorPin1, 160); // Bật motor Lidar cuối cùng
    analogWrite(motorPin2, 0);

    lidarSerial.begin(BAUDRATE_SENSOR, SERIAL_8N1, RX_PIN, -1);

    client.setServer(mqtt_server, 1883);
    client.setCallback(callback);
    client.setBufferSize(1100);
}

  // --- HÀM KẾT NỐI LẠI MQTT (Thay thế Task) ---
  void reconnect() {
    static unsigned long lastReconnectAttempt = 0;
    if (millis() - lastReconnectAttempt > 5000) { // Thử lại sau mỗi 5s
      lastReconnectAttempt = millis();
      Serial.println("Đang thử kết nối lại MQTT..."); // Thêm dòng này để debug
      if (client.connect("ESP32_Device")) {
        client.subscribe("mqtt/control");
        Serial.println("MQTT Reconnected");
      }
      else {
        Serial.print("Kết nối thất bại, rc=");
        Serial.println(client.state());
      }
    }
  }
  void callback(char* topic, byte* payload, unsigned int length) {
    // 1. Chuyển payload (byte array) sang String
  String message = "";
    for (int i = 0; i < length; i++) {
      message += (char)payload[i];
    }
    message.trim(); // Loại bỏ khoảng trắng thừa hoặc ký tự xuống dòng
   // Lệnh STOP luôn được ưu tiên
  if (message == "stop") {
    stopmotor();
    state = IDLE;
    // Serial.println("Dừng khẩn cấp");
    return;
  }
    stopmotor();
  // CHỈ nhận lệnh mới nếu đang RẢNH (IDLE)
  if (state == IDLE) {
    if (message.startsWith("MOVE ")) {
       stopmotor();
      state = RUN_DIST;
      float D = message.substring(5).toFloat();
      targetTicks = (D / WHEEL_CIRCUMFERENCE) * ENCODER_PPR;
      startEnc1 = pulseCount ;
      startEnc2 = pulseCount2 ;
      // Serial.println("Nhận lệnh MOVE");
    } 
    else if (message.startsWith("ROTATE ")) {
       stopmotor();
      state = RUN_ANGLE;
      float D = message.substring(7).toFloat();
      targetRad = D * PI /180 ;
      startEnc1 = pulseCount ;
      startEnc2 = pulseCount2 ;
      // Serial.println("Nhận lệnh ROTATE");
    }
    // Các lệnh di chuyển tự do (không theo khoảng cách)
    else if (message == "forward") move_forward();
    else if (message == "backward") move_backward();
    else if (message == "left") turn_left();
    else if (message == "right") turn_right();
  } 
  }
  void loop() {
    // 1. Duy trì kết nối MQTT
    if (!client.connected()) {
      reconnect();
    }
    client.loop(); 
    // dieu khien dong co
    unsigned long currentTime = millis();
 static unsigned long lastPIDTime = 0;
if (currentTime - lastPIDTime >= 50) {
 
    updatemotorPID();
  
  lastPIDTime = currentTime;
}


    // 2. Đọc MPU6050
    Wire.beginTransmission(MPU);
    Wire.write(0x47);
    Wire.endTransmission(false);
    if (Wire.requestFrom(MPU, 2, true) == 2) {
      int16_t rawGyZ = Wire.read() << 8 | Wire.read();
      gyZ_deg = (rawGyZ / 131.0) - gyroZBias;
      // float dtYaw = (currentTime - lastYawTime) / 1000.0;  // s
      // yaw += gyZ_deg * dtYaw;                             // tích phân ra °
      // lastYawTime = currentTime;
    }
    // --- QUẢN LÝ TRẠNG THÁI DI CHUYỂN ---
  if (state == RUN_DIST) {
    long d1 = abs(pulseCount - startEnc1);
    long d2 = abs(pulseCount2 - startEnc2);
    long dAvg = (d1 + d2) / 2;
    
    if (dAvg >= abs(targetTicks)) {
      stopmotor();
       analogWrite(motor1_ena, 0);
        analogWrite(motor2_enb, 0);
      state = IDLE;
      client.publish("astar/status", "Done"); // Quan trọng: Báo cho laptop
      return;
    } else {
      if (targetTicks > 0) move_forward(); else move_backward();
    }
  } 
  else if (state == RUN_ANGLE) {
    long d1 = abs(pulseCount - startEnc1);
    long d2 = abs(pulseCount2 - startEnc2);
    // Tính toán góc đã quay thực tế (Radian)
    float turnedRad = (float)(d1 + d2) * WHEEL_CIRCUMFERENCE / (ENCODER_PPR * WHEEL_BASE);
    
    if (turnedRad >= abs(targetRad)) {
      stopmotor();
      state = IDLE;
      client.publish("astar/status", "Done"); // Quan trọng: Báo cho laptop
    } else {
      if (targetRad > 0) turn_left(); else turn_right();
    }
  }
      // 3. Xử lý dữ liệu Lidar
    handleLidarData();
    // 4. Gửi dữ liệu định kỳ (10Hz)
    static unsigned long lastMsg = 0;
    if (currentTime - lastMsg > 40) {
      lastMsg = currentTime;
      if (client.connected() && bufferIndex > 0) {
        sendDataToClient();
        bufferIndex = 0; // Reset buffer sau khi gửi
      }
    }

  }
  // ================== LIDAR HANDLER ==================
  void handleLidarData() {
    // Đọc hết dữ liệu đang có trong Serial buffer
    while (lidarSerial.available() > 0 && bufferIndex < (BUFFER_SIZE * 32 - 32)) {
      uint8_t receivedByte = lidarSerial.read();

      if (waitPacket) {
        if (receivedByte == HEADER_BYTE) {
          waitPacket = false;
          packetIndex = 0;
          packet_size = 0;
          packet[packetIndex++] = receivedByte;
        }
        continue;
      }

      if (packetIndex >= MAX_PACKET_SIZE) {
        resetPacket();
        continue;
      }

      packet[packetIndex++] = receivedByte;

      if (packet_size == 0 && packetIndex >= 3) {
        uint16_t lenField = ((uint16_t)packet[1] << 8) | packet[2];
        if (lenField < 10 || lenField > MAX_PACKET_SIZE) {
          // Serial.println("ERROR SPEED");
          resetPacket();
          continue;
        }
        packet_size = (int)lenField + 2;
      }

      if (packet_size > 0 && packetIndex >= packet_size) {
        uint16_t checksum_calc = calcChecksum(packet, packet_size - 2);
        uint16_t checksum_recv = ((uint16_t)packet[packet_size - 2] << 8) | packet[packet_size - 1];

        if (checksum_calc == checksum_recv) {
          decodePacket(packet, packet_size);
          for (int i = 0; i < MAX_DATA_SIZE; i++) {
            lidarBuffer[bufferIndex + i] = dataArr[i];
          }
          lidarBuffer[bufferIndex + 28] = (int32_t)(gyZ_deg * 1000);
          lidarBuffer[bufferIndex + 29] = (int32_t)pulseCount;
          lidarBuffer[bufferIndex + 30] = (int32_t)pulseCount2;
          lidarBuffer[bufferIndex + 31] = (int32_t)millis();
          bufferIndex += 32;
        }
        resetPacket();
      }
    }
  }

  void resetPacket() {
    waitPacket = true;
    packetIndex = 0;
    packet_size = 0;
  }

  void decodePacket(uint8_t packet[], int packet_size) {
    for (int i = 0; i < MAX_DATA_SIZE; i++) dataArr[i] = 0;
    if (packet_size <= 13)
    // Serial.println("Packet too small");
    return;
    int n_points = (packet_size - 13) / 3;
    dataArr[0] = (((uint16_t)packet[11] << 8) | packet[12]) / 10; // angle
    dataArr[1] = n_points;

    int h = (n_points <= 25) ? n_points : 25; // lấy h điểm
    dataArr[2] = h;
    for (int i = 0; i < h; i++) {
      int hi = 14 + 3 * i ;
      int lo = hi + 1;
      dataArr[i+3] = ((uint16_t)packet[hi] << 8) | packet[lo];
    }
  }

  uint16_t calcChecksum(uint8_t data[], int size) {
    uint32_t sum = 0;
    for (int i = 0; i < size; i++) sum += data[i];
    return (uint16_t)(sum & 0xFFFF);
  }

  void sendDataToClient() {
    if (bufferIndex == 0) return;

    // Duyệt qua từng dòng (mỗi dòng 32 số)
    for (int row = 0; row < (bufferIndex / 32); row++) {
      char msgBuffer[230]; // Tạo "khay" chứa cố định 250 byte cho 1 dòng
      int offset = 0;      // Biến theo dõi vị trí hiện tại trong msgBuffer
      int startIdx = row * 32;

      for (int i = 0; i < 32; i++) {
        // In số vào msgBuffer, trả về số ký tự đã ghi để cộng vào offset
        int written = snprintf(msgBuffer + offset, sizeof(msgBuffer) - offset, 
                              (i < 31) ? "%d," : "%d\n", 
                              lidarBuffer[startIdx + i]);
        offset += written;
        
        // Kiểm tra nếu buffer sắp đầy để tránh tràn
        if (offset >= sizeof(msgBuffer) - 1) break;
      }

      if (client.connected()) {
        // msgBuffer lúc này đã là một chuỗi char hoàn chỉnh
        client.publish("mqtt/data", msgBuffer);
      }
    }
    bufferIndex = 0; 
  }
  // Hàm bổ trợ điều khiển hướng
    void setMotorDirection(float direction) {
      if (direction == 1) { // Hướng Tiến
        digitalWrite(motor1_in1, LOW);
        digitalWrite(motor1_in2, HIGH);
      } else if (direction == -1) { // Hướng Lùi
        digitalWrite(motor1_in1, HIGH);
        digitalWrite(motor1_in2, LOW);
      } else { // Dừng cưỡng bức (Short brake)
        digitalWrite(motor1_in1, LOW);
        digitalWrite(motor1_in2, LOW);
         analogWrite(motor1_ena, 0);

      }
    }
    void setMotorDirection2(float direction2 ) {
      if (direction2 == -1) { // Hướng Tiến
        digitalWrite(motor2_in3, HIGH);
        digitalWrite(motor2_in4, LOW);
      } else if (direction2 == 1) { // Hướng Lùi
        digitalWrite(motor2_in3, LOW);
        digitalWrite(motor2_in4, HIGH);
      } else { // Dừng cưỡng bức (Short brake)s
        digitalWrite(motor2_in3, LOW);
        digitalWrite(motor2_in4, LOW);
         analogWrite(motor2_enb, 0);
      }
    }
    void updatemotorPID() {
    float dt = 0.05;
    noInterrupts();
    long currentP1 = pulseCount;
    long currentP2 = pulseCount2;
    float dP1 = (float)(currentP1 - lastPulseCount);
    float dP2 = (float)(currentP2 - lastPulseCount2);
    lastPulseCount = currentP1;
    lastPulseCount2 = currentP2;
    interrupts();

    // Xử lý Motor 1
    if (setpoint == 0) {
        output = 0;
        integral = 0;
    } else {
        error = abs(setpoint) - abs(dP1);
        integral += error * dt;
        integral = constrain(integral, -5, 5);
        output = (Kp * error) + (Ki * integral);
    }

    // Xử lý Motor 2
    if (setpoint2 == 0) {
        output2 = 0;
        integral2 = 0;
    } else {
        error2 = abs(setpoint2) - abs(dP2);
        integral2 += error2 * dt;
        integral2 = constrain(integral2, -5, 5);
        output2 = (Kp * error2) + (Ki * integral2);
        long diff = abs(currentP1) - abs(currentP2);
        if (diff > 0){
        output  -= (diff * Ksync);
        }
    }

    // XUẤT PWM - Quan trọng: Chỉ xuất khi setpoint khác 0
    if (setpoint != 0) {
        setMotorDirection(direction);
        analogWrite(motor1_ena, constrain((int)output, 0, 100));
    } else {
        digitalWrite(motor1_in1, LOW);
        digitalWrite(motor1_in2, LOW);
        analogWrite(motor1_ena, 0);
    }

    if (setpoint2 != 0) {
        setMotorDirection2(direction2);
        analogWrite(motor2_enb, constrain((int)output2, 0, 100));
    } else {
        digitalWrite(motor2_in3, LOW);
        digitalWrite(motor2_in4, LOW);
        analogWrite(motor2_enb, 0);
    }
}
  void stopmotor() {
    // 1. Dừng mục tiêu
    setpoint = 0;
    setpoint2 = 0;
    state = IDLE;

    // 2. XÓA BỘ NHỚ PID (Đây là lý do bạn không dừng được)
    integral = 0; 
    integral2 = 0;
    output = 0;
    output2 = 0;
    error = 0;
    error2 = 0;

    // 3. Ngắt điện motor ngay lập tức
    digitalWrite(motor1_in1, LOW);
    digitalWrite(motor1_in2, LOW);
    analogWrite(motor1_ena, 0);
    
    digitalWrite(motor2_in3, LOW);
    digitalWrite(motor2_in4, LOW);
    analogWrite(motor2_enb, 0);
}
  void move_forward(){
    direction = 1;
    direction2 = 1;
    setpoint=40; 
    setpoint2=40;
  } 
  void move_backward(){
    direction = -1;
    direction2 = -1;
    setpoint=-40 ;
    setpoint2=-40;
  }
  void turn_left(){
    direction = 1;
    direction2 = -1;
    setpoint=40;
    setpoint2=-40;  
  }                         
  void turn_right(){
    direction = -1;
    direction2 = 1;
    setpoint=-40;
    setpoint2=40;
  }
