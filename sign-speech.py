import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import pyttsx3  # Import the pyttsx3 library for text-to-speech

# Load the pre-trained model
model = load_model('sign_language_main_1.h5')

# Map class indices to letters and gestures (including the open palm gesture)
class_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing',
    28: 'space', 29: 'open_palm'
}

# Initialize the text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)
tts_engine.say("System initialized successfully.")
tts_engine.runAndWait()

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# Initialize variables for storing letters, words, and sentences
detected_letters = []
detected_word = ""
start_time = None
previous_sign = ""
previous_word = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    frame = cv2.resize(frame, (1000, 600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Hand detection rectangle setup
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
    rect_size = 250
    top_left = (frame_center[0] - rect_size // 2, frame_center[1] - rect_size // 2)
    bottom_right = (top_left[0] + rect_size, top_left[1] + rect_size)

    # Resize and preprocess the hand ROI
    hand_roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    processed_hand = cv2.resize(hand_roi, (224, 224))
    processed_hand = np.expand_dims(processed_hand, axis=0) / 255.0  # Normalize

    # Get the prediction from the model
    prediction = model.predict(processed_hand)
    predicted_class = np.argmax(prediction)
    predicted_sign = class_mapping.get(predicted_class, 'Unknown')

    print(f"Detected sign: {predicted_sign}")

    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    sign_bg_color = (255, 0, 0)
    sign_text_color = (255, 255, 255)
    cv2.rectangle(frame, (10, 10), (200, 50), sign_bg_color, -1)
    cv2.putText(frame, predicted_sign, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, sign_text_color, 2, cv2.LINE_AA)

    if predicted_sign == 'nothing':
        start_time = None
    else:
        if start_time is None:
            start_time = time.time()
        else:
            elapsed_time = time.time() - start_time
            if elapsed_time >= 3:
                if predicted_sign == 'space':
                    detected_word = ''.join(detected_letters).replace('_', '').strip()
                    print(f"Detected word: {detected_word}")
                    if detected_word and detected_word != previous_word:
                        print(f"Pronouncing: {detected_word}")
                        tts_engine.say(detected_word)
                        tts_engine.runAndWait()
                        previous_word = detected_word
                    detected_letters = []
                elif predicted_sign == 'del':
                    if detected_letters:
                        detected_letters.pop()
                elif predicted_sign != previous_sign:
                    detected_letters.append(predicted_sign)
                    previous_sign = predicted_sign

                start_time = None

    word_bg_color = (0, 0, 255)
    word_text_color = (255, 255, 255)
    cv2.rectangle(frame, (10, 60), (700, 90), word_bg_color, -1)
    cv2.putText(frame, f'Detected Word: {"".join(detected_letters)}', (20, 80 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, word_text_color, 2, cv2.LINE_AA)

    cv2.imshow('Hand Detection and Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
