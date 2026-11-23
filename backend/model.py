
from ultralytics import YOLO
import easyocr
import os
import cv2
import sys
import re

# ---------------------------------
# 1. Load YOLO and EasyOCR
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best.pt"
)

model = YOLO(MODEL_PATH)

print("YOLO Model Loaded Successfully!")
print("Classes:", model.names)

reader = easyocr.Reader(["en"], gpu=False)

print("EasyOCR Loaded Successfully!")

OCR_THRESHOLD = 0.50


# ---------------------------------
# 2. Calculate IoU
# ---------------------------------

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


# ---------------------------------
# 3. OCR Function
# ---------------------------------

def read_number_plate(plate_image):

    if plate_image is None or plate_image.size == 0:
        return "", 0.0

    gray = cv2.cvtColor(
        plate_image,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    gray = cv2.resize(
        gray,
        (max(1, width * 4), max(1, height * 4)),
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        2
    )

    sharpened = cv2.addWeighted(
        enhanced, 1.5,
        blurred, -0.5,
        0
    )

    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    processed_images = [
        enhanced,
        sharpened,
        adaptive,
        otsu
    ]

    best_text = ""
    best_confidence = 0.0
    best_score = 0.0

    for processed_image in processed_images:

        results = reader.readtext(
            processed_image,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        if not results:
            continue

        results.sort(
            key=lambda item: min(
                point[0] for point in item[0]
            )
        )

        text_parts = []
        confidence_values = []

        for item in results:

            text = re.sub(
                r"[^A-Z0-9]",
                "",
                item[1].upper()
            )

            confidence = float(item[2])

            if text:
                text_parts.append(text)
                confidence_values.append(confidence)

        combined_text = "".join(text_parts)

        if not combined_text:
            continue

        average_confidence = (
            sum(confidence_values) /
            len(confidence_values)
        )

        score = average_confidence * min(
            len(combined_text) / 8,
            1.0
        )

        if score > best_score:

            best_score = score
            best_text = combined_text
            best_confidence = average_confidence

    return best_text, best_confidence


# ---------------------------------
# 4. Find Which Bike Owns a Detection
# ---------------------------------

def find_associated_bike(detection, bikes):

    x1, y1, x2, y2 = detection["box"]

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    matching_bikes = []

    for index, bike in enumerate(bikes):

        bx1, by1, bx2, by2 = bike["box"]

        if (
            bx1 <= center_x <= bx2
            and by1 <= center_y <= by2
        ):

            area = max(
                1,
                (bx2 - bx1) * (by2 - by1)
            )

            matching_bikes.append(
                (area, index)
            )

    if not matching_bikes:
        return None

    # If boxes overlap, use the smallest enclosing bike box
    matching_bikes.sort(key=lambda item: item[0])

    return matching_bikes[0][1]


# ---------------------------------
# 5. Main Detection Function
# ---------------------------------

def detect_objects(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read uploaded image")

    image_height, image_width = image.shape[:2]

    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False
    )

    all_detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coordinates = box.xyxy[0].tolist()

            all_detections.append({
                "class_id": class_id,
                "confidence": confidence,
                "box": coordinates
            })

    # ---------------------------------
    # 6. Remove Helmet / NoHelmet Conflicts
    # ---------------------------------

    all_detections.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    filtered = []

    for detection in all_detections:

        conflict = False

        for previous in filtered:

            opposite_classes = (
                {detection["class_id"], previous["class_id"]}
                == {1, 2}
            )

            if opposite_classes:

                if calculate_iou(
                    detection["box"],
                    previous["box"]
                ) >= 0.30:

                    conflict = True
                    break

        if not conflict:
            filtered.append(detection)

    # ---------------------------------
    # 7. Separate Bike/Rider Detections
    # ---------------------------------

    bikes = [
        detection.copy()
        for detection in filtered
        if detection["class_id"] == 0
    ]

    other_detections = [
        detection
        for detection in filtered
        if detection["class_id"] != 0
    ]

    riders = []

    for index, bike in enumerate(bikes):

        riders.append({
            "rider_id": index + 1,
            "bike_confidence": round(
                bike["confidence"], 4
            ),
            "bike_box": bike["box"],
            "helmet_status": "Unknown",
            "violation": False,
            "number_plate": "",
            "number_plate_status": "OCR Not Required",
            "ocr_confidence": 0.0
        })

    # ---------------------------------
    # 8. Associate Helmet and NumberPlate
    # ---------------------------------

    assigned = {
        index: {
            "helmet": [],
            "plates": []
        }
        for index in range(len(bikes))
    }

    for detection in other_detections:

        class_id = detection["class_id"]

        bike_index = find_associated_bike(
            detection,
            bikes
        )

        if bike_index is None:
            continue

        if class_id == 1:
            assigned[bike_index]["helmet"].append(
                detection
            )

        elif class_id == 2:
            assigned[bike_index]["helmet"].append(
                detection
            )

        elif class_id == 3:
            assigned[bike_index]["plates"].append(
                detection
            )

    # ---------------------------------
    # 9. Process Each Rider
    # ---------------------------------

    print("\nRider-wise Detection Results:")

    for index, rider in enumerate(riders):

        helmet_detections = assigned[index]["helmet"]

        if helmet_detections:

            best_helmet = max(
                helmet_detections,
                key=lambda item: item["confidence"]
            )

            helmet_class = best_helmet["class_id"]

            if helmet_class == 1:
                rider["helmet_status"] = "Helmet"
                rider["violation"] = False

            elif helmet_class == 2:
                rider["helmet_status"] = "NoHelmet"
                rider["violation"] = True

        # OCR is performed ONLY for NoHelmet riders
        if rider["violation"]:

            plate_detections = assigned[index]["plates"]

            if not plate_detections:

                rider["number_plate_status"] = (
                    "Number Plate Not Detected"
                )

            else:

                # Select the highest-confidence plate
                best_plate = max(
                    plate_detections,
                    key=lambda item: item["confidence"]
                )

                px1, py1, px2, py2 = map(
                    int,
                    best_plate["box"]
                )

                px1 = max(0, min(px1, image_width))
                px2 = max(0, min(px2, image_width))
                py1 = max(0, min(py1, image_height))
                py2 = max(0, min(py2, image_height))

                pad_x = int((px2 - px1) * 0.05)
                pad_y = int((py2 - py1) * 0.15)

                crop = image[
                    max(0, py1 - pad_y):
                    min(image_height, py2 + pad_y),
                    max(0, px1 - pad_x):
                    min(image_width, px2 + pad_x)
                ]

                plate_text, ocr_confidence = read_number_plate(
                    crop
                )

                rider["number_plate"] = plate_text
                rider["ocr_confidence"] = round(
                    ocr_confidence, 4
                )

                if not plate_text:

                    rider["number_plate_status"] = (
                        "Text Not Readable"
                    )

                elif ocr_confidence < OCR_THRESHOLD:

                    rider["number_plate_status"] = (
                        "OCR Result Unverified"
                    )

                else:

                    rider["number_plate_status"] = (
                        "Number Plate Successfully Read"
                    )

        print(
            f"Rider {rider['rider_id']} | "
            f"Helmet: {rider['helmet_status']} | "
            f"Violation: {rider['violation']} | "
            f"Plate: {rider['number_plate']} | "
            f"Status: {rider['number_plate_status']}"
        )

    # ---------------------------------
    # 10. Draw Bike Boxes and Rider Labels
    # ---------------------------------

    for rider in riders:

        x1, y1, x2, y2 = map(
            int,
            rider["bike_box"]
        )

        x1 = max(0, min(x1, image_width))
        x2 = max(0, min(x2, image_width))
        y1 = max(0, min(y1, image_height))
        y2 = max(0, min(y2, image_height))

        if rider["violation"]:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        label = (
            f"Rider {rider['rider_id']} - "
            f"{rider['helmet_status']}"
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        if rider["violation"] and rider["number_plate"]:

            plate_label = rider["number_plate"]

            if rider["number_plate_status"] == "OCR Result Unverified":
                plate_label += " (Unverified)"

            cv2.putText(
                image,
                plate_label,
                (x1, min(y2 + 25, image_height - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ---------------------------------
    # 11. Save Evidence Image
    # ---------------------------------

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

    total_riders = len(riders)

    violations = [
        rider for rider in riders
        if rider["violation"]
    ]

    total_violations = len(violations)

    # ---------------------------------
    # 12. Return Results
    # ---------------------------------

    return {
        "riders": riders,
        "total_riders": total_riders,
        "total_violations": total_violations,
        "output_path": output_path
    }


# ---------------------------------
# 13. Direct Testing
# ---------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage: python model.py image_path")

    else:

        result = detect_objects(sys.argv[1])

        print("\nFinal Result:")
        print(result)
