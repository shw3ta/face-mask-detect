from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cv2.dnn

prototxt = "deploy.prototxt.txt"
weights = "res10_300x300_ssd_iter_140000.caffemodel"
facenet = cv2.dnn.readNet(prototxt, weights)

model = load_model("face_mask_model2.h5")

image = cv2.imread("path/to/image.jpg")
(h, w) = image.shape[:2]
# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
facenet.setInput(blob)
detections = facenet.forward()

for i in range(0, detections.shape[2]):
    # extract the confidence associated with
    # the detection
    confidence = detections[0, 0, i, 2]
    
    if confidence > 0.15:
        # compute the (x, y)-coordinates of the bounding box for
        # the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        # ensure the bounding boxes fall within the dimensions of
        # the frame
        (x1, y1) = (max(0, x1), max(0, y1))
        (x2, y2) = (min(w - 1, x2), min(h - 1, y2))
        face = image[y1:y2, x1:x2]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        # pass the face through the model to determine if the face
        # has a mask or not
        (mask, withoutMask) = model.predict(face)[0]
        label = "Mask on" if mask > withoutMask else "Mask off"
        color = (0, 255, 0) if label == "Mask on" else (0, 0, 255)
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
# show the output image
cv2.imshow("Face Mask Detector", image)
cv2.waitKey(0)

