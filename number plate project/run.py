import os
from ultralytics import YOLO
import cv2


video_path = 'video.mp4'
video_path_out = 'frame.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = 'best (1).pt'

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.2
count = 0

while ret:

    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    if count%10==0:
        cv2.imwrite(f"{count}.jpg",frame)
    out.write(frame)
    ret, frame = cap.read()
    count+=1

cap.release()
out.release()
cv2.destroyAllWindows()