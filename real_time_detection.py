import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('asl_model.h5')
IMG_SIZE = 64

# Updated labels list to include additional classes
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

def predict_letter(frame):
    """
    Function to preprocess the frame and predict the letter.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, IMG_SIZE, IMG_SIZE, 1))
    
    # Perform prediction using trained model
    prediction = model.predict(reshaped)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index]
    
    return labels[predicted_index], confidence

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define a region of interest (ROI) for detecting the hand
    height, width, _ = frame.shape
    roi = frame[100:400, 100:400] 
    letter, confidence = predict_letter(roi)
    
    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
    
    # Display the predicted letter with confidence score
    cv2.putText(frame, f'Letter: {letter} ({confidence:.2f})', 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the video feed
    cv2.imshow('ASL Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()