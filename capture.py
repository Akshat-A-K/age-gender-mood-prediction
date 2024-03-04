import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

captured_image_path = 'captured_image.jpg'
cropped_image_path = 'cropped_face.jpg'
resized_image_path = 'resized_face.jpg'
enhanced_image_path = 'enhanced_resized_face.jpg'

cropped_face = None
resized_face_gray = None
enhanced_resized_face = None
flag = True
while flag:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cropped_face = frame[y:y + h, x:x + w]

        cv2.imshow('Real-time Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(cropped_image_path, cropped_face)
            
            resized_face_gray = cv2.resize(cropped_face, (128, 128))
            resized_face_gray = cv2.cvtColor(resized_face_gray, cv2.COLOR_BGR2GRAY)

            enhanced_resized_face = cv2.equalizeHist(resized_face_gray)

            cv2.imwrite(resized_image_path, resized_face_gray)
            cv2.imwrite(enhanced_image_path, enhanced_resized_face)

            print("Face cropped, saved, and resized!")
            print("Original Face Size:", cropped_face.shape)
            print("Resized Face Size:", resized_face_gray.shape)
            print("Enhanced Resized Face Size:", enhanced_resized_face.shape)
            flag = False
            break

cap.release()
cv2.destroyAllWindows()

if cropped_face is not None:
    cv2.imshow('Cropped Image', cropped_face)
    cv2.imshow('Resized Image', resized_face_gray)
    cv2.imshow('Enhanced Resized Image', enhanced_resized_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
