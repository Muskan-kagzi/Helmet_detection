
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import uuid

from model import detect_objects
from database import (
    init_db,
    save_violation,
    get_history,
    get_violation_count
)


BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


app = FastAPI(
    title="Helmet Violation Detection API",
    description="Multiple Rider-wise Helmet Violation Detection with OCR",
    version="2.0"
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
        "message": "Helmet Violation Detection API is running!"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]

    if file.content_type not in allowed_types:

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

        # Run YOLO + rider association + OCR
        result = detect_objects(str(image_path))

        output_filename = Path(
            result["output_path"]
        ).name

        # Save ONLY violating riders
        saved_records = 0

        for rider in result["riders"]:

            if rider["violation"]:

                save_violation(
                    image_filename=file.filename,
                    rider=rider,
                    output_image=output_filename
                )

                saved_records += 1

        return {
            "message": "Detection completed successfully",

            "filename": file.filename,

            "total_riders": result["total_riders"],

            "total_violations": result["total_violations"],

            "saved_records": saved_records,

            "riders": result["riders"],

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

    return {
        "total_violations": len(records),
        "history": records
    }


@app.get("/stats")
def stats():

    return {
        "total_violations": get_violation_count()
    }
