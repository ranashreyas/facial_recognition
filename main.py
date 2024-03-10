from ultralytics import YOLO  # Import the YOLO class from the ultralytics package
import cv2
import supervision as sv

# Load a pre-trained YOLO model
model = YOLO('yolov8n')  # This loads the YOLOv8n model, adjust the model name based on your needs
# model.train(data='coco128.yaml', epochs=3)

# Function to process and display video
def process_video(video_path):
    # Capture video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Define the codec and create VideoWriter object to save the output
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_ms = int(1000/fps)
    print('FPS:', fps)

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        text_padding=2
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]

        detections = sv.Detections.from_ultralytics(results)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        cv2.imshow('frame',frame)

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break


    # Release everything
    cap.release()
    # out.release()
    cv2.destroyAllWindows()

# Process your video
process_video('sample_study_video.mov')
