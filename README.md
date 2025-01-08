# IoT_Insights
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a>
    <img src="./pictures/logo_pxl.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">
Gun shooter with AI face detection</h3>

  <p align="center">This project demonstrates an AI-powered turret system that combines face detection technology with servo motor control to track and engage targets. The turret is built using an Arduino Leonardo, a USB camera, and a PC running the detection program. The turret's movement is controlled by two servo motors, which allow it to rotate horizontally and vertically. Commands are sent from the PC to the Arduino Leonardo, which manages the servos' operation.

A USB camera mounted on the turret streams video to the PC, where a face detection program processes it in real-time using AI. When no face is detected, the turret autonomously scans its surroundings in search of a target. Once a face is detected and centered in the camera's view, the PC sends a "fire" command to the Arduino, prompting the turret to shoot a soft object at the target.

This turret is designed for safety, firing soft objects, which makes it suitable for demonstrations, games, or educational projects. The system showcases the integration of AI, robotics, and electronics into a highly interactive and engaging setup. It’s a fun and practical way to explore the possibilities of face detection and automated control systems.

Feel free to explore the code and hardware setup, customize it to your needs, and let your creativity take aim! 🚀 </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

### Built With

* [![C](https://img.shields.io/badge/C-00599C?style=for-the-badge&logo=c&logoColor=white)](https://en.wikipedia.org/wiki/C_(programming_language))
* [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Installation


<!-- USAGE EXAMPLES -->
## Usage


<!-- ROADMAP -->
## Roadmap

- [x] Setting up Arduino leonardo
- [x] Creating test script for arduino
- [x] Creating test script for AI face detection
- [x] Developing main program
    - [x] Writing connection code
    - [x] Getting frames from the camera
    - [x] Adding face detection and logic
    - [x] Optimizing Face Detection Filters

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- CONTACT -->
## Contact
Ine Beddegenoots  - Ine.Beddegenoots@student.pxl.be 

Xander Aerts -Xander.Aerts@student.pxl.be

Jonas Vanhulst - Jonas.Vanhulst@student.pxl.be


