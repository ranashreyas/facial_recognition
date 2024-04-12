from ultralytics import YOLO
import cv2
import numpy as np

def draw_threshold_line(image, start_point, end_point, color, thickness):
    start_point = (int(start_point[0]), int(start_point[1]))
    end_point = (int(end_point[0]), int(end_point[1]))
    cv2.line(image, start_point, end_point, color, thickness)


def process_video(video_path, horizontal_threshold = 3, vertical_threshold = 3):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]

                _, threshold = cv2.threshold(eye_roi_gray, 30, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                for cnt in contours[:1]:
                    (px, py), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(px), int(py))
                    radius = int(radius)

                    # cv2.circle(eye_roi_color, center, radius, (0, 0, 255), 2)
                    cv2.circle(eye_roi_color, center, 3, (0, 0, 255), 2)

                    eye_center = (ex + ew // 2, ey + eh // 2)  # The center of the eye bounding box

                    dx = center[0] - eye_center[0]  # Difference in x from the eye center
                    dy = center[1] - eye_center[1]  # Difference in y from the eye center

                    # Use a portion of the eye width and height as thresholds for determining direction
                    horizontal_threshold_px = ew / horizontal_threshold  # This will be a float
                    vertical_threshold_px = eh / vertical_threshold  # This will be a float

                    # Draw horizontal threshold lines
                    draw_threshold_line(roi_color, 
                                        (ex + int(horizontal_threshold_px), ey), 
                                        (ex + int(horizontal_threshold_px), ey + eh), 
                                        (0, 255, 255), 1)
                    draw_threshold_line(roi_color, 
                                        (ex + ew - int(horizontal_threshold_px), ey), 
                                        (ex + ew - int(horizontal_threshold_px), ey + eh), 
                                        (0, 255, 255), 1)

                    # Draw vertical threshold lines
                    draw_threshold_line(roi_color, 
                                        (ex, ey + int(vertical_threshold_px)), 
                                        (ex + ew, ey + int(vertical_threshold_px)), 
                                        (0, 255, 255), 1)
                    draw_threshold_line(roi_color, 
                                        (ex, ey + eh - int(vertical_threshold_px)), 
                                        (ex + ew, ey + eh - int(vertical_threshold_px)), 
                                        (0, 255, 255), 1)

                    # Adjust the direction check accordingly
                    direction = ""
                    if abs(dx) > horizontal_threshold_px:
                        if dx < 0:
                            direction += "Left"
                        else:
                            direction += "Right"
                    if abs(dy) > vertical_threshold_px:
                        if dy < 0:
                            direction += "Up" 
                        else:
                            direction += "Down"

                    # If no direction is determined, it's center
                    if direction == "":
                        direction = "Center"

                    cv2.putText(roi_color, direction, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)

        cv2.imshow('Pupil Direction with Thresholds', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# If you're testing with a webcam, you can call it with 0
process_video(0, horizontal_threshold=2.5, vertical_threshold=2.5)

# For a video file, replace 0 with the path to your video file
# process_video('path_to_video_file')
