This project uses Mosquitto MQTT as the communication broker between the robot and the control station. Follow these steps to set up the environment:

# 1. Mosquitto Broker Setup (Windows/Laptop)

Create "C:\Program Files\mosquitto\myconfig.conf" with 2 line following:
listener 1883 0.0.0.0
allow_anonymous true

Open a terminal and launch the Mosquitto broker with the custom configuration file to enable network listening:

cd C:\Program Files\mosquitto
mosquitto -c myconfig.conf -v

Press Win + R, type services.msc, and hit Enter.

Locate Mosquitto Broker.
Right-click and select Stop, then Start (or Restart) to ensure a fresh session.
Network Access: Ensure your firewall allows traffic on port 1883.

# 2. Launch Control & SLAM Modules
Open two separate terminals and execute the following scripts in order:

//Navigate to your  if necessary

Terminal 1: Perception & SLAM This handles LiDAR data processing and map generation.

python slam_main.py

//Navigate to your folder if necessary

Terminal 2: Robot Control This manages the movement commands and navigation logic.

python control_main.py

💡 Pro-Tips for your README:
Prerequisites: I recommend adding a section above this called ## Dependencies listing paho-mqtt, numpy, or any other libraries you used.

Troubleshooting: You might want to add: "If the scripts cannot connect, verify that the IP address in slam_main.py matches your laptop's local IP."
