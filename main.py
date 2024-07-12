from pathlib import Path
import cv2

# Correct path to the Haar cascade XML file
cascade_path = Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

# Initialize the classifier with the correct path
clf = cv2.CascadeClassifier(str(cascade_path))

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    _, frame = camera.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw rectangles around detected faces
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
    
    # Display the frame with detected faces
    cv2.imshow("Faces", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
