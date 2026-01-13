# Secure Attendance System with Face Recognition & Anti-Spoofing

A robust, automated attendance system using face recognition and anti-spoofing techniques. This project leverages computer vision and deep learning to ensure only real, authorized users are marked present, making it ideal for classrooms or secure environments.

## Features

- **Face Recognition:** Identifies students using their facial features.
- **Anti-Spoofing:** Detects and blocks spoof attempts (e.g., photos, videos).
- **GUI Application:** User-friendly interface for attendance management.
- **Dataset Capture:** Easily add new users with webcam image capture.
- **Attendance Logging:** Records attendance with timestamps and class info.
- **Timetable Integration:** Supports multiple classes and schedules.

## Project Structure

```
.
├── GUI.py                  # Main GUI application
├── encoding.py             # Face encoding script
├── comparing_Anti-Spoof.py # Real-time recognition & anti-spoofing
├── capturing_dataset/
│   ├── testcam.py          # Script to capture face images
│   ├── haarcascade_frontalface_default.xml
│   └── data/
│       ├── attendance.csv  # Attendance logs
│       ├── timetable.csv   # Class schedules
│       ├── images/         # User face images
├── *.ipynb                 # Jupyter notebooks for experiments
└── README.md
```

## How It Works

1. **Capture Dataset:** Use `capturing_dataset/testcam.py` to collect face images for each user.
2. **Encode Faces:** Run `encoding.py` to generate facial encodings.
3. **Run GUI:** Launch `GUI.py` for the main attendance interface.
4. **Mark Attendance:** The system recognizes faces in real-time, checks for spoofing, and logs attendance.

## Requirements

- Python 3.x
- OpenCV
- face_recognition
- deepface
- numpy, pandas, pillow, matplotlib, tqdm, tkinter

Install dependencies:
```bash
pip install opencv-python face_recognition deepface numpy pandas pillow matplotlib tqdm
```

## Data Files

- `attendance.csv`: Logs of attendance (name, timestamp, status, class)
- `timetable.csv`: Class schedules and enrolled students
- `images/`: Folders of user face images for training

## Usage

1. **Add a New User:** Run the dataset capture script and follow prompts.
2. **Update Encodings:** Run the encoding script after adding new users.
3. **Start Attendance:** Use the GUI to start a session and mark attendance.

## Anti-Spoofing

The system uses DeepFace to extract faces and verify liveness, reducing spoofing risks.

## License

MIT License


