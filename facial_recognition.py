## Ignore, old version

import numpy as np, cv2, os
from imutils import paths
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from keras.utils import to_categorical

data_path = 'cropped_faces'
model_path = 'face_model.h5'
label_path = 'fsace_labels.npy'

face_cascade = cv2.CascadeClassifier('lib/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('lib/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('lib/haarcascade_smile.xml')
nose_cascade = cv2.CascadeClassifier('lib/nose.xml')

if os.path.exists(model_path) and os.path.exists(label_path):
    print('Loading pre-trained model and label encoder...')
    model = load_model(model_path)
    le = LabelEncoder()
    le.classes_ = np.load(label_path)

    frame = cv2.imread('known_faces/Yuna/220913 Yuna 17.jpeg') # set this to the path of the image you wish to check

    input_type = 'c' # Set this to 'c' if you want to use your webcam

    if input_type.lower() == 'c':
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=30, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y+h, x:x+w], (64, 64))
            roi = np.expand_dims(roi, axis=-1)

            preds = model.predict(np.array([roi]))
            label = le.inverse_transform(np.array([np.argmax(preds)]))[0]
            prob = np.max(preds)

            text = f"{label}: {prob:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.putText(frame, str(round(prob*100)) + '%', (x+w-30, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            print(f'AI Confidence [{label} - {prob*100:.2f}%]')

            eyes = eyes_cascade.detectMultiScale(gray, minNeighbors=20)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

            nose = nose_cascade.detectMultiScale(gray, minNeighbors=20)
            for (nx, ny, nw, nh) in nose:
                cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (102, 230, 255), 2)

            smile = smile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=300)
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 255, 255), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print('Training new model...')
    data, labels = [], []
    for person in os.listdir(data_path):
        image_paths = list(paths.list_images(os.path.join(data_path, person)))
        if len(image_paths) == 0:
            print(f"No images found for {person}")
            continue
        for image_path in image_paths:
            gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30))
            if len(faces) == 0:
                print(f"No faces found in {image_path}")
                continue
            for (x, y, w, h) in faces:
                roi = np.expand_dims(cv2.resize(gray[y:y+h, x:x+w], (64, 64)), axis=-1)
                data.append(roi)
                labels.append(person)
    data, labels = np.array(data), np.array(labels)
    le = LabelEncoder()
    labels = to_categorical(le.fit_transform(labels))
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

    missing_labels = set(le.inverse_transform(np.argmax(y_test, axis=1))) - set(le.inverse_transform(np.argmax(y_train, axis=1)))
    if missing_labels:
        print(f"Warning: The  following labels are missing from the training set and will be removed from the test set: {missing_labels}")
        mask = np.isin(le.inverse_transform(np.argmax(y_test, axis=1)), list(missing_labels), invert=True)
        X_test, y_test = X_test[mask], y_test[mask]

    datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(le.classes_), activation='softmax')
    ])
    adam = Adam(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(datagen.flow(X_train, y_train, batch_size=512), epochs=20, validation_data=(X_test, y_test), verbose=1)

    if not os.path.exists(model_path) or not os.path.exists(label_path):
        model.save(model_path)
        np.save(label_path, le.classes_)

    predictions = model.predict(X_test, batch_size=512)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

    precision = precision_score(y_test.argmax(axis=1), predictions.argmax(axis=1), average='weighted', zero_division=1)
    recall = recall_score(y_test.argmax(axis=1), predictions.argmax(axis=1), average='weighted', zero_division=1)
    f1 = f1_score(y_test.argmax(axis=1), predictions.argmax(axis=1), average='weighted', zero_division=1)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    print("Model training and evaluation complete. Model saved as 'face_model.h5'.")
