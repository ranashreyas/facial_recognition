from ultralytics import YOLO
import cv2
import supervision as sv


# model = YOLO('yolov8n')
# model.train(data='coco128.yaml', epochs=3)

def process_video(video_path):

    cap = cv2.VideoCapture(0)

    # cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     print("Error opening video file.")
    #     return

    # fps = cap.get(cv2.CAP_PROP_FPS)
    # wait_ms = int(1000/fps)
    # print('FPS:', fps)

    while cap.isOpened():
        ret, frame = cap.read()
        
        
        if not ret:
            break
        # results = model(frame)[0]

        # detections = sv.Detections.from_ultralytics(results)
        # labels = [
        #     f"{model.model.names[class_id]} {confidence:0.2f}"
        #     for _, _, confidence, class_id, _
        #     in detections
        # ]

        # frame = box_annotator.annotate(
        #     scene=frame,
        #     detections=detections,
        #     labels=labels
        # )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        print('Number of detected faces:', len(faces))

        # loop over the detected faces
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # detects eyes of within the detected face area (roi)
            eyes = eye_cascade.detectMultiScale(roi_gray)
            print("    Number of Eyes: " + str(len(eyes)))
            
            # draw a rectangle around eyes
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)

        # display the image with detected eyes
        cv2.imshow('Eyes Detection',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

process_video('sample_study_video.mov')
