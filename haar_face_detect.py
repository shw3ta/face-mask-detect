import cv2
import imutils
from imutils.video import VideoStream
import time

clf = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_stream = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = video_stream.read()
    frame = imutils.resize(frame, width=400)
    bboxes = clf.detectMultiScale(frame, 1.05, 6)

    for box in bboxes:
        x, y, w, h = box
        x2, y2 = x + w, y + h

        cv2.putText(frame, "DONKEY", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,255,0), 1)
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("face detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
video_stream.stop()