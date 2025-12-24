from PIL import Image
import matplotlib.pyplot as plt
import os
import speech_recognition as sr
from gtts import gTTS
import pygame
import tempfile
import time

# Step 1: Specify the image path in the extracted directory
path = 'project/'  # Update to the correct path within the extracted folder

# Step 2: Initialize recognizer class (for recognizing the speech)
recognizer = sr.Recognizer()

# Capture the microphone input
with sr.Microphone() as source:
    # Adjust for ambient noise to improve recognition in noisy environments
    recognizer.adjust_for_ambient_noise(source, duration=1)
    
    print("Please say something (a sentence):")
    # Listen to the speech
    audio_data = recognizer.listen(source)
    print("Recognizing...")

    try:
        # Recognize the speech using Google Web API (default recognizer)
        recognized_msg = recognizer.recognize_google(audio_data)
        print(f"You said: {recognized_msg}")

        # Convert recognized text to uppercase
        recognized_msg = recognized_msg.upper()

        # Step 3: List to store images and spaces for the recognized text
        images = []

        # Loop through each character in the recognized message
        for i in recognized_msg:
            if i == ' ':  # For spaces, append None to represent a gap
                images.append(None)
                continue
            image_path = os.path.join(path, i, '0.jpg')  # Construct the image path for each character
            try:
                # Open the image and append it to the list
                img = Image.open(image_path)
                images.append(img)
            except FileNotFoundError:
                images.append(None)  # Append None if image is not found

        # Step 4: Create a figure with larger subplots for all images and spaces
        fig, axes = plt.subplots(1, len(images), figsize=(20, 10))  # Increase the figure size

        # Loop through each axis and corresponding image
        for ax, img in zip(axes, images):
            if img is not None:
                ax.imshow(img)  # Display the image
                ax.axis('off')  # Hide axes for images
            else:
                # When img is None, just set the axis background to white (for visual spacing)
                ax.set_facecolor('white')
                ax.axis('off')  # Hide axes for empty space

        # Adjust layout to ensure everything fits
        plt.tight_layout()
        plt.show()

        # Convert recognized text to speech
        tts = gTTS(text=recognized_msg, lang='en')

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
            tts.save(temp_audio_file.name)  # Save to a temporary file
            temp_audio_file_path = temp_audio_file.name  # Get the path of the temporary file

        # Initialize pygame mixer and play the audio file
        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_file_path)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            continue

        # Delay to ensure the audio playback is completely finished
        time.sleep(0.5)

        # Optionally, remove the saved file after playing
        try:
            os.remove(temp_audio_file_path)
        except PermissionError:
            pass  # If the file cannot be deleted, just skip the deletion

    except (sr.UnknownValueError, sr.RequestError):
        pass  # Do nothing if an error occurs in speech recognition


# In[6]:

from PIL import Image
import matplotlib.pyplot as plt
import os
import speech_recognition as sr
from gtts import gTTS
import pygame
import tempfile
import time

# Step 1: Specify the image path in the extracted directory
path = 'extracted_project/project/'  # Update to the correct path within the extracted folder

# Step 2: Initialize recognizer class (for recognizing the speech)
recognizer = sr.Recognizer()

# Function to calculate word accuracy
def calculate_word_accuracy(spoken_text, recognized_text):
    spoken_words = spoken_text.split()
    recognized_words = recognized_text.split()

    # Calculate the number of matching words
    matching_words = sum(1 for i in range(min(len(spoken_words), len(recognized_words))) if spoken_words[i] == recognized_words[i])

    # Calculate word accuracy as a percentage
    word_accuracy = (matching_words / len(spoken_words)) * 100
    return word_accuracy

# Capture the microphone input
with sr.Microphone() as source:
    # Adjust for ambient noise to improve recognition in noisy environments
    recognizer.adjust_for_ambient_noise(source, duration=1)
    
    print("Please say something (a sentence):")
    # Listen to the speech
    audio_data = recognizer.listen(source)
    print("Recognizing...")

    try:
        # Recognize the speech using Google Web API (default recognizer)
        recognized_msg = recognizer.recognize_google(audio_data)
        print(f"You said: {recognized_msg}")

        # Expected spoken text (for accuracy comparison)
        spoken_text = "good morning"  # Update this with the actual expected sentence

        # Calculate speech recognition accuracy
        accuracy = calculate_word_accuracy(spoken_text, recognized_msg)
        print(f"Speech Recognition Accuracy: {accuracy}%")

        # Convert recognized text to uppercase
        recognized_msg = recognized_msg.upper()

        # Step 3: List to store images and spaces for the recognized text
        images = []

        # Loop through each character in the recognized message
        for i in recognized_msg:
            if i == ' ':  # For spaces, append None to represent a gap
                images.append(None)
                continue
            image_path = os.path.join(path, i, '0.jpg')  # Construct the image path for each character
            try:
                # Open the image and append it to the list
                img = Image.open(image_path)
                images.append(img)
            except FileNotFoundError:
                images.append(None)  # Append None if image is not found

        # Step 4: Create a figure with larger subplots for all images and spaces
        fig, axes = plt.subplots(1, len(images), figsize=(20, 10))  # Increase the figure size

        # Loop through each axis and corresponding image
        for ax, img in zip(axes, images):
            if img is not None:
                ax.imshow(img)  # Display the image
                ax.axis('off')  # Hide axes for images
            else:
                # When img is None, just set the axis background to white (for visual spacing)
                ax.set_facecolor('white')
                ax.axis('off')  # Hide axes for empty space

        # Adjust layout to ensure everything fits
        plt.tight_layout()
        plt.show()

        # Convert recognized text to speech
        tts = gTTS(text=recognized_msg, lang='en')

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
            tts.save(temp_audio_file.name)  # Save to a temporary file
            temp_audio_file_path = temp_audio_file.name  # Get the path of the temporary file

        # Initialize pygame mixer and play the audio file
        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_file_path)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            continue

        # Delay to ensure the audio playback is completely finished
        time.sleep(0.5)

        # Optionally, remove the saved file after playing
        try:
            os.remove(temp_audio_file_path)
        except PermissionError:
            pass  # If the file cannot be deleted, just skip the deletion

    except (sr.UnknownValueError, sr.RequestError):
        print("Could not understand audio or request failed.")


# In[15]:


import os
import shutil

# Define your paths
test_data_dir = 'extracted_archive_11/asl_alphabet_test/asl_alphabet_test'  # Update as necessary
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Create directories for each class if they don't exist
for class_name in classes:
    class_dir = os.path.join(test_data_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

# Move your test images to their respective class folders based on your naming convention
for file_name in os.listdir(test_data_dir):
    # Assuming your files are named like 'A_test.jpg', 'B_test.jpg', etc.
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        class_name = file_name[0]  # Assuming the first character is the class
        src = os.path.join(test_data_dir, file_name)
        dst = os.path.join(test_data_dir, class_name, file_name)
        shutil.move(src, dst)


# In[1]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




