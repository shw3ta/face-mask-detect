from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from keras.utils.vis_utils import plot_model
import imutils
from imutils.video import VideoStream
import numpy as np
import time
import cv2
import cv2.dnn


def detect_face_mask_from_vid(frame, face_detector, mask_detector):
    (h, w) = frame.shape[:2]
    # (104,177,123) are numbers used to normalize the image colors so that blobs can be \
    # detected better
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    probably_faces = face_detector.forward()

    faces = []
    coordinates = []
    predictions = []

    for i in range(0, probably_faces.shape[2]):
        confidence = probably_faces[0, 0, i, 2]
        if confidence > min_confidence:
            box = probably_faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            (x1, y1) = (max(0, x1), max(0, y1))
            (x2, y2) = (min(w - 1, x2), min(h - 1, y2))

            face = frame[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224)) # these are the dimensions our mask_detector was trained on
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            coordinates.append((x1, y1, x2, y2))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        predictions = mask_detector.predict(faces, batch_size=32)

    return coordinates, predictions


min_confidence = 0.7

# face_detector is a residual net from OpenCV's dnn repo, it's a modified resnet architecture
face_detector = cv2.dnn.readNet("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
# plot_model(model=face_detector, to_file='face_detector-architecture.png')

# mask_detector is a network that we separately trained and saved on top of face_detector
# the colab notebook for the same is available.
mask_detector = load_model("face_mask_model.h5")
# plot_model(model=mask_detector, to_file='mask_detector-architecture.png')

video_stream = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = video_stream.read()
    frame = imutils.resize(frame, width=400)
    (coordinates, predictions) = detect_face_mask_from_vid(frame, face_detector, mask_detector)

    for box, pred in zip(coordinates, predictions):
        x1, y1, x2, y2 = box
        masks, noMasks = pred
        label = "Mask on" if masks > noMasks else "Mask off"
        color = (0, 255, 0) if label == "Mask on" else (0, 0, 255)
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(masks, noMasks) * 100)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Hello, Chanda!", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
video_stream.stop()





