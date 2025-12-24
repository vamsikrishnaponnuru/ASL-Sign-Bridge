from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import speech_recognition as sr
from googletrans import Translator
import time

# -------------------------------
# Load ASL Model
# -------------------------------
# make sure this path is correct relative to app.py
model = tf.keras.models.load_model("model/sign_language_main_1.h5")

# Class mapping (27 classes: A–Z + space)
class_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: 'space'
}

sentence = ""                 # final text
last_pred = None              # last predicted label
stable_count = 0              # how many frames same label
translator = Translator()
camera_running = False

CONF_THRESHOLD = 0.40         # min softmax prob to accept prediction
STABLE_FRAMES = 7             # min consecutive frames before adding char

app = Flask(__name__)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# -------------------------------
# ROUTES
# -------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/sign_to_text")
def sign_to_text():
    return render_template("sign_to_text.html")


@app.route("/speech_to_text")
def speech_to_text():
    return render_template("speech_to_text.html")


# -------------------------------
# SIGN → TEXT LOGIC
# -------------------------------
def gen_frames_sign_to_text():
    global sentence, camera_running, last_pred, stable_count
    camera_running = True

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    prev_time = 0

    while camera_running:
        success, frame = cap.read()
        if not success:
            break

        # Limit FPS to ~20
        curr_time = time.time()
        if curr_time - prev_time < 0.05:
            continue
        prev_time = curr_time

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        h, w, _ = frame.shape
        display_label = "No Hand"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # ----- ROI from all landmarks with padding -----
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x1 = int(min(x_coords) * w) - 40
            y1 = int(min(y_coords) * h) - 40
            x2 = int(max(x_coords) * w) + 40
            y2 = int(max(y_coords) * h) + 40

            # clamp to image
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, w)
            y2 = min(y2, h)

            hand_roi = frame[y1:y2, x1:x2]

            if hand_roi.size > 0 and (x2 > x1) and (y2 > y1):
                img = cv2.resize(hand_roi, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)

                preds = model.predict(img, verbose=0)[0]
                idx = int(np.argmax(preds))
                confidence = float(preds[idx])
                label = class_mapping.get(idx, "Unknown")

                display_label = f"{label} ({confidence:.2f})"

                if confidence >= CONF_THRESHOLD and label != "Unknown":
                    # stability filter
                    if label == last_pred:
                        stable_count += 1
                    else:
                        last_pred = label
                        stable_count = 1

                    if stable_count >= STABLE_FRAMES:
                        if label == "space":
                            sentence += " "
                        else:
                            sentence += label
                        stable_count = 0  # reset so we don't spam
                else:
                    # low confidence → reset stability
                    last_pred = None
                    stable_count = 0

                # draw ROI box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # text overlays
        if display_label == "No Hand":
            cv2.putText(
                frame,
                "No Hand Detected",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                display_label,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            sentence[-25:],  # show last few chars
            (30, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


@app.route("/video_feed_sign_to_text")
def video_feed_sign_to_text():
    return Response(
        gen_frames_sign_to_text(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/get_text")
def get_text():
    return jsonify({"text": sentence})


@app.route("/clear_text")
def clear_text():
    global sentence, last_pred, stable_count
    sentence = ""
    last_pred = None
    stable_count = 0
    return jsonify({"status": "cleared"})


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global camera_running
    camera_running = False
    return jsonify({"status": "Camera stopped"})


# -------------------------------
# SPEECH → TEXT + TRANSLATION
# -------------------------------
@app.route("/speech_to_text_translate", methods=["POST"])
def speech_to_text_translate():
    lang_code = request.form.get("lang", "en")
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=7)
            print("✅ Processing speech...")

            text = recognizer.recognize_google(audio)
            print(f"🗣 You said: {text}")

            translated = translator.translate(text, src="en", dest=lang_code)
            return jsonify(
                {
                    "original_text": text,
                    "translated_text": translated.text,
                    "language": lang_code,
                }
            )

    except sr.UnknownValueError:
        return jsonify({"error": "Speech not understood, please try again."})
    except sr.RequestError:
        return jsonify({"error": "Network error during speech recognition."})
    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("✅ Flask server started at http://127.0.0.1:5000/")
    app.run(debug=True, host="127.0.0.1", port=5000)
