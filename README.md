﻿# DST-FallNet

## Project Overview
This project is a real-time multimodal fall detection system that integrates visual (MoveNet, CNN) and audio (LSTM) models, utilizing Dempster-Shafer Theory (DST) for decision-level fusion. The system is designed to run on resource-constrained embedded devices such as Jetson Nano 2GB, supports GPIO output, and is suitable for long-term automated monitoring.

Project website: [https://www.mkchou.online/](https://www.mkchou.online/)

## Hardware Used
- Jetson Nano 2GB
- USB Camera
- USB Microphone
- LED, Buzzer (GPIO control)

## Project Directory Structure
```
DST-FallNet/
├── src/                # Main source code
├── models/             # ONNX model files
├── assets/             # Documentation images
├── abnormal_images/    # Abnormal images generated during runtime
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Installation
1. Install Python 3.8 or above
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
1. Enter the `src/` directory
2. Run the main program:
   ```bash
   python main.py
   ```

## Main Features
- Visual fall detection (MoveNet pose estimation, CNN action classification)
- Audio anomaly detection (LSTM)
- Dempster-Shafer Theory (DST) decision fusion
- GPIO output for alerts
- CLI dashboard for real-time system status

## System Architecture & Workflow
- Four main threads:
  1. Visual pose inference (MoveNet)
  2. Audio recognition (MFCC + LSTM)
  3. DST fusion & CNN verification
  4. GPIO control & button monitoring
- For detailed architecture and theory, please refer to the [project website](https://www.mkchou.online/)

## Model Files
- Please place the downloaded ONNX models in the `models/` directory.
- For download links, refer to the [project website](https://www.mkchou.online/) or contact the author.

## Example Screenshots
![acc_curve](assets/CNN/acc_curve.png)
![confusion_matrix](assets/CNN/confusion_matrix.png)

## References & Further Reading
- [Full project description and theory](https://www.mkchou.online/)
- For system architecture, DST fusion theory, model training details, and performance evaluation, see the respective sections on the website.

## Contact
- Author: Ming-Kun Chou
- Email: AN4096750@gs.ncku.edu.tw

## Notes
- Jetson Nano 2GB and correct GPIO connections are required
- Model files (.onnx) must be placed in the `models/` directory
- Images and temporary files generated during runtime are not recommended to be uploaded to GitHub

---

This project integrates multimodal sensing, DST fusion, ONNX deployment, and hardware control, making it suitable for embedded edge computing and long-term fall monitoring applications. For more details, please refer to the [project website](https://www.mkchou.online/).
