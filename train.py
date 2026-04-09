import cv2
import os

# Config
size = 4
fn_dir = 'att_faces'
(im_width, im_height) = (112, 92)
count_max = 20  # number of samples to collect

# Make sure att_faces folder exists
if not os.path.isdir(fn_dir):
    os.mkdir(fn_dir)

# Ask for student name
fn_name = input("Enter student name: ").strip()
if fn_name == "":
    print("❌ Name cannot be empty")
    exit()

# Create student's folder
path = os.path.join(fn_dir, fn_name)
if not os.path.isdir(path):
    os.mkdir(path)

# Load Haar Cascade from OpenCV’s data folder
fn_haar = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(fn_haar)

# Check if cascade loaded
if haar_cascade.empty():
    print(f"❌ Failed to load Haar Cascade from {fn_haar}")
    exit()

# Start webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("❌ Cannot access webcam")
    exit()

# Find last saved image number
existing_pics = [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(path) if f.endswith(".png")]
pin = max(existing_pics, default=0) + 1

print(f"\n📸 Collecting {count_max} face samples for {fn_name}.")
print("👉 Look at the camera and move your head slightly.\n")

count, pause = 0, 0

while count < count_max:
    rval, frame = webcam.read()
    if not rval:
        print("❌ Failed to grab frame")
        break

    # Flip and grayscale
    frame = cv2.flip(frame, 1, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(gray, (gray.shape[1] // size, gray.shape[0] // size))

    # Detect faces
    faces = haar_cascade.detectMultiScale(mini)
    faces = sorted(faces, key=lambda x: x[3], reverse=True)  # largest face first

    if len(faces) > 0:
        (x, y, w, h) = [v * size for v in faces[0]]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        # Save every 5th frame
        if w * 6 >= frame.shape[1] and h * 6 >= frame.shape[0]:
            if pause == 0:
                filename = os.path.join(path, f"{fn_name}_{pin}.png")
                cv2.imwrite(filename, face_resize)
                print(f"✅ Saved {filename}")
                pin += 1
                count += 1
                pause = 1

    pause = (pause + 1) % 5
    cv2.imshow("Face Capture", frame)

    # ESC to quit early
    key = cv2.waitKey(10)
    if key == 27:
        print("❌ Interrupted by user")
        break

print("\n🎉 Done! Collected", count, "samples.")
webcam.release()
cv2.destroyAllWindows()
