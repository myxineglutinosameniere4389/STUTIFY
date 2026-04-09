import cv2
import os

# Load Haar cascade
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Prompt for name
person_name = input("Enter the name of the person: ").strip()

# Create folder for the person
dataset_path = "att_faces"
person_path = os.path.join(dataset_path, person_name)
os.makedirs(person_path, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting camera. Press 'c' to capture an image, 'q' to quit.")

image_count = 0
max_images = 20 # you can change this

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if image_count < max_images:
            image_path = os.path.join(person_path, f"{image_count+1}.jpg")
            cv2.imwrite(image_path, roi)
            image_count += 1
            print(f"[INFO] Saved image {image_count}/{max_images}")

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or image_count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Collected {image_count} images for {person_name}")
