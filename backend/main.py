
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import uuid
import json

from model import detect_objects
from database import init_db, save_detection, get_history


BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Helmet and Number Plate Detection API",
    description="YOLO based detection system",
    version="1.0"
)

app.mount(
    "/results",
    StaticFiles(directory=str(RESULT_DIR)),
    name="results"
)

init_db()


@app.get("/")
def home():
    return {
        "message": "Helmet Detection API is running!"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image"
        )

    extension = Path(file.filename or "").suffix.lower()

    if extension not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid image extension"
        )

    unique_name = f"{uuid.uuid4().hex}{extension}"
    image_path = UPLOAD_DIR / unique_name

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty"
            )

        image_path.write_bytes(image_bytes)

        result = detect_objects(str(image_path))

        output_filename = Path(result["output_path"]).name

        save_detection(
            file.filename,
            result["detections"],
            output_filename
        )

        return {
            "message": "Detection successful",
            "filename": file.filename,
            "detections": result["detections"],
            "output_image": f"/results/{output_filename}"
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(error)}"
        )


@app.get("/history")
def history():

    records = get_history()

    for record in records:
        record["detections"] = json.loads(
            record["detections"]
        )

    return {
        "total": len(records),
        "history": records
    }
