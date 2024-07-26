import cv2

# Load the face detection cascade classifier
face_cascade = cv2.CascadeClassifier('default_frontal_face.xml')

# Create a VideoCapture object to capture video from the default camera (index 0)
capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    success, img = capture.read()

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(img, 1.2, 4)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_region = img[y:y + h, x:x + w]

        # Apply Gaussian blur to the face region
        gaussian_blur = cv2.GaussianBlur(face_region, (91, 91), 0)

        # Replace the original face region with the blurred one
        img[y:y + h, x:x + w] = gaussian_blur

    # Check if no faces were detected
    if len(faces) == 0:
        cv2.putText(img, 'No Face Found!', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    # Display the output frame
    cv2.imshow('Face Blur', img)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the video capture object
capture.release()

# Close all OpenCV windows
cv2.destroyAllWindows()