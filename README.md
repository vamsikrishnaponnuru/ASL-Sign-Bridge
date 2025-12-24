ASL Sign Bridge
Real-Time American Sign Language Recognition System

📌 Abstract
ASL Sign Bridge is a real-time American Sign Language (ASL) recognition system designed to bridge the communication gap between deaf/mute individuals and non-sign language users. 
The system uses a webcam to capture hand gestures, processes them using computer vision and deep learning techniques, and converts the recognized signs into readable text or speech. 
The project also supports speech-to-sign and sign-to-speech translation, enabling two-way communication.

🎯 Problem Statement
People who rely on sign language often face communication barriers when interacting with individuals who do not understand sign language. 
Existing solutions are limited, expensive, or not accessible in real time. This project aims to provide an affordable, real-time, and user-friendly solution that translates
American Sign Language gestures into text and speech using a web-based system.

💡 Proposed Solution
ASL Sign Bridge uses a webcam-based interface combined with machine learning models to detect and classify hand gestures. 
The recognized gestures are mapped to corresponding letters or words and displayed as text. Additional modules allow conversion between speech and sign language, making the system versatile and inclusive.

🏗️ System Architecture
Webcam Input – Captures real-time hand gestures
Hand Landmark Detection – Uses MediaPipe for extracting hand features
Deep Learning Model – Classifies gestures using a trained TensorFlow model
Backend Processing – Flask handles requests and predictions
Frontend Interface – Displays recognized output in real time

🛠️ Technologies Used
Programming Languages
Python
JavaScript
HTML, CSS
Frameworks & Libraries
Flask (Backend)
TensorFlow / Keras (Deep Learning)
OpenCV (Image Processing)
MediaPipe (Hand Tracking)
NumPy
Tools
Webcam
Jupyter Notebook
Git & GitHub

✨ Features
Real-time ASL gesture recognition
Webcam-based live detection
Sign-to-text conversion
Speech-to-sign translation
Sign-to-speech conversion
User-friendly web interface
Scalable and modular design

📂 Project Structure
ASL-Sign-Bridge/
│── app.py
│── train_asl_model.py
│── augment_dataset.py
│── sign-speech.py
│── speech-sign.py
│── static/
│── templates/
│── Code/
│── README.md
│── .gitignore

🚀 How to Run the Project
1. Clone the Repository
  git clone https://github.com/vamsikrishnaponnuru/ASL-Sign-Bridge.git
  cd ASL-Sign-Bridge
 
2. Create Virtual Environment
  git clone https://github.com/vamsikrishnaponnuru/ASL-Sign-Bridge.git
  cd ASL-Sign-Bridge

3. Install Dependencies
  pip install -r requirements.txt

4. Run the Application
  python app.py

5. Open Browser
  http://127.0.0.1:5000/

📊 Dataset & Model
The dataset consists of hand gesture images representing ASL characters.
Data augmentation techniques were applied to improve accuracy.
The trained model is excluded from GitHub due to size limitations and can be hosted separately (e.g., Google Drive).

📈 Results
Accurate real-time gesture recognition
Low latency response
Effective translation between sign language and text/speech
Works efficiently under normal lighting conditions

⚠️ Limitations
Performance may vary under poor lighting
Limited to trained ASL gestures
Background noise can affect speech recognition accuracy

🔮 Future Enhancements
Sentence-level gesture recognition
Support for Indian Sign Language (ISL)
Mobile application version
Multi-language speech output
Improved accuracy using larger datasets

👨‍💻 Author
Vamsi Krishna Ponnuru
Final Year Engineering Project
GitHub: https://github.com/vamsikrishnaponnuru

📜 License
This project is developed for academic purposes and is free to use for learning and research.

✅ What to Do Now
Open README.md
Paste everything above
Save
Run:
git add README.md
git commit -m "Update README with project documentation"
git push















  

