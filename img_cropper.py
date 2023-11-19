## Looks for faces in known_faces and generates a cropped version to cropped_faces

import cv2, os

data_path = 'known_faces'
output_path = 'cropped_faces'

face_detector = cv2.CascadeClassifier('lib/haarcascade_frontalface_default.xml')

for dir_name in os.listdir(data_path):
    dir_path = os.path.join(data_path, dir_name)
    if not os.path.isdir(dir_path):
        continue

    output_dir_path = os.path.join(output_path, dir_name)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if not os.path.isfile(file_path):
            continue

        try:
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            print(f"Error reading file {file_path}")
            continue
        
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            face_file_path = os.path.join(output_dir_path, f"{file_name}")
            cv2.imwrite(face_file_path, face)
            print(f'Successfully cropped & wrote file {file_path}') 