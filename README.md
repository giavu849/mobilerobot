This project uses Mosquitto MQTT as the communication broker between the robot and the control station. Follow these steps to set up the environment:

1. Mosquitto Broker Setup (Windows/Laptop)
Before running the Python scripts, ensure the Mosquitto service is running correctly:

Check Service Status:

Press Win + R, type services.msc, and hit Enter.

Locate Mosquitto Broker.

Right-click and select Stop, then Start (or Restart) to ensure a fresh session.

Network Access: Ensure your firewall allows traffic on port 1883.

2. Initialize MQTT Configuration
Open a terminal and launch the Mosquitto broker with the custom configuration file to enable network listening:

Bash
# Navigate to your Mosquitto installation folder if necessary
mosquitto -c mosquitto.conf -v
(The -v flag enables verbose mode so you can see incoming messages from the robot.)

3. Launch Control & SLAM Modules
Open two separate terminals and execute the following scripts in order:

Terminal 1: Perception & SLAM This handles LiDAR data processing and map generation.

Bash
python slam_main.py
Terminal 2: Robot Control This manages the movement commands and navigation logic.

Bash
python control_main.py
💡 Pro-Tips for your README:
Prerequisites: I recommend adding a section above this called ## Dependencies listing paho-mqtt, numpy, or any other libraries you used.

Troubleshooting: You might want to add: "If the scripts cannot connect, verify that the IP address in slammain.py matches your laptop's local IP."
