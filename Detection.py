import cv2
from fer import FER
import matplotlib.pyplot as plt

# Load the pre-trained model for emotion detection
emotion_detector = FER()

# Initialize webcam (use 0 for default camera)
cap = cv2.VideoCapture(0)

# Loop to continuously get frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect emotions from the current frame
    emotions = emotion_detector.detect_emotions(frame)

    # Loop through detected faces
    for face in emotions:
        # Get the bounding box of the face
        (x, y, w, h) = face["box"]
        # Get the most dominant emotion
        dominant_emotion = face["emotions"]
        top_emotion = max(dominant_emotion, key=dominant_emotion.get)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the emotion on the frame
        cv2.putText(frame, top_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detector', frame)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when everything is done
cap.release()
cv2.destroyAllWindows()
 