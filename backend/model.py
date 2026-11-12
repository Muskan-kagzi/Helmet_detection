
# from ultralytics import YOLO
# import os
# import sys
# import cv2

# MODEL_PATH = os.path.join(
#     os.path.dirname(__file__),
#     "models",
#     "best.pt"
# )

# model = YOLO(MODEL_PATH)

# print("YOLO Model Loaded Successfully!")
# print("Classes:", model.names)


# def calculate_iou(box1, box2):
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     intersection = max(0, x2 - x1) * max(0, y2 - y1)

#     area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

#     union = area1 + area2 - intersection

#     return intersection / union if union > 0 else 0


# def detect_objects(image_path):

#     results = model.predict(
#         source=image_path,
#         conf=0.25,
#         save=False
#     )

#     for result in results:

#         boxes = result.boxes

#         detections = []

#         for box in boxes:
#             class_id = int(box.cls[0])
#             confidence = float(box.conf[0])
#             coordinates = box.xyxy[0].tolist()

#             detections.append({
#                 "class_id": class_id,
#                 "confidence": confidence,
#                 "box": coordinates
#             })

#         # Highest confidence detections first
#         detections.sort(
#             key=lambda x: x["confidence"],
#             reverse=True
#         )

#         filtered = []

#         for detection in detections:

#             conflict = False

#             for previous in filtered:

#                 # Helmet = 1, NoHelmet = 2
#                 is_opposite_class = (
#                     {detection["class_id"], previous["class_id"]}
#                     == {1, 2}
#                 )

#                 if is_opposite_class:
#                     iou = calculate_iou(
#                         detection["box"],
#                         previous["box"]
#                     )

#                     if iou >= 0.30:
#                         conflict = True
#                         break

#             if not conflict:
#                 filtered.append(detection)

#         # Draw filtered detections
#         image = cv2.imread(image_path)

#         print("\nFinal Detection Results:")

#         for detection in filtered:

#             class_id = detection["class_id"]
#             confidence = detection["confidence"]
#             x1, y1, x2, y2 = map(int, detection["box"])

#             class_name = model.names[class_id]

#             print(
#                 f"{class_name} - Confidence: {confidence:.2f}"
#             )

#             color = (0, 255, 0)

#             cv2.rectangle(
#                 image,
#                 (x1, y1),
#                 (x2, y2),
#                 color,
#                 2
#             )

#             label = f"{class_name} {confidence:.2f}"

#             cv2.putText(
#                 image,
#                 label,
#                 (x1, max(y1 - 10, 20)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 color,
#                 2
#             )

#         output_path = os.path.join(
#             os.path.dirname(__file__),
#             "results",
#             "detected.jpg"
#         )

#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         cv2.imwrite(output_path, image)

#         print("\nDetected image saved at:", output_path)


# if __name__ == "__main__":

#     if len(sys.argv) < 2:
#         print("Usage: python backend/model.py image_path")
#     else:
#         detect_objects(sys.argv[1])






from ultralytics import YOLO
import os
import cv2
import sys

# -------------------------------
# 1. Load YOLO Model
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best.pt"
)

model = YOLO(MODEL_PATH)

print("YOLO Model Loaded Successfully!")
print("Classes:", model.names)


# -------------------------------
# 2. Calculate IoU
# -------------------------------

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


# -------------------------------
# 3. Detect Objects
# -------------------------------

def detect_objects(image_path):

    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False
    )

    all_detections = []

    for result in results:

        detections = []

        # Extract detections
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coordinates = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "confidence": confidence,
                "box": coordinates
            })

        # Highest confidence first
        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        # -------------------------------
        # 4. Remove Helmet/NoHelmet conflicts
        # -------------------------------

        filtered = []

        for detection in detections:

            conflict = False

            for previous in filtered:

                is_opposite_class = (
                    {detection["class_id"], previous["class_id"]}
                    == {1, 2}
                )

                if is_opposite_class:

                    iou = calculate_iou(
                        detection["box"],
                        previous["box"]
                    )

                    if iou >= 0.30:
                        conflict = True
                        break

            if not conflict:
                filtered.append(detection)

        # -------------------------------
        # 5. Draw Bounding Boxes
        # -------------------------------

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Unable to read uploaded image")

        print("\nFinal Detection Results:")

        for detection in filtered:

            class_id = detection["class_id"]
            confidence = detection["confidence"]

            x1, y1, x2, y2 = map(
                int,
                detection["box"]
            )

            class_name = model.names[class_id]

            print(
                f"{class_name} - Confidence: {confidence:.2f}"
            )

            color = (0, 255, 0)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            all_detections.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "box": [
                    round(float(value), 2)
                    for value in detection["box"]
                ]
            })

        # -------------------------------
        # 6. Save Detected Image
        # -------------------------------

        result_dir = os.path.join(
            BASE_DIR,
            "results"
        )

        os.makedirs(result_dir, exist_ok=True)

        filename = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        output_path = os.path.join(
            result_dir,
            filename + "_detected.jpg"
        )

        cv2.imwrite(output_path, image)

        print("\nDetected image saved at:", output_path)

    # -------------------------------
    # 7. Return Results to FastAPI
    # -------------------------------

    return {
        "detections": all_detections,
        "output_path": output_path
    }


# -------------------------------
# 8. Run Directly for Testing
# -------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage: python backend/model.py image_path")

    else:

        result = detect_objects(sys.argv[1])

        print("\nReturned Detection Data:")
        print(result["detections"])
