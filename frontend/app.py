
import streamlit as st
import requests

from PIL import Image
from io import BytesIO


# ---------------------------------
# 1. Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Helmet Violation Detection",
    page_icon="🏍️",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"


# ---------------------------------
# 2. Styling
# ---------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    font-size: 32px;
    font-weight: bold;
    color: #17365D;
}

.subtitle {
    color: #64748b;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------
# 3. Header
# ---------------------------------

st.markdown(
    '<div class="title">🏍️ Helmet Violation Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Multiple rider detection with number plate recognition'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------
# 4. Sidebar Navigation
# ---------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Violation History",
        "About Project"
    ]
)


# ---------------------------------
# 5. Home Page
# ---------------------------------

if page == "Home":

    st.subheader("Upload Traffic Image")

    st.write(
        "Upload an image containing one or multiple motorcycle riders."
    )

    uploaded_file = st.file_uploader(
        "Choose Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        original_image = Image.open(uploaded_file)

        st.image(
            original_image,
            caption="Original Image",
            use_container_width=True
        )

        if st.button(
            "Start Detection",
            type="primary"
        ):

            with st.spinner(
                "Detecting riders and checking helmet violations..."
            ):

                try:

                    uploaded_file.seek(0)

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    }

                    response = requests.post(
                        f"{API_URL}/detect",
                        files=files,
                        timeout=180
                    )

                    response.raise_for_status()

                    result = response.json()

                    st.session_state["last_result"] = result

                    st.success(
                        "Detection completed successfully!"
                    )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Backend is not running. "
                        "Please start FastAPI first."
                    )

                except requests.exceptions.RequestException as error:

                    st.error(f"API Error: {error}")

                except Exception as error:

                    st.error(f"Something went wrong: {error}")

        # Display latest result
        if "last_result" in st.session_state:

            result = st.session_state["last_result"]

            st.divider()

            st.subheader("Detection Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Riders",
                result["total_riders"]
            )

            col2.metric(
                "Total Violations",
                result["total_violations"]
            )

            col3.metric(
                "Records Saved",
                result["saved_records"]
            )

            st.divider()

            st.subheader("Rider-wise Results")

            for rider in result["riders"]:

                with st.container(border=True):

                    st.markdown(
                        f"### Rider {rider['rider_id']}"
                    )

                    if rider["helmet_status"] == "NoHelmet":

                        st.error("Helmet Violation Detected")

                        st.write(
                            "**Helmet Status:** No Helmet"
                        )

                        st.write(
                            "**Violation:** Yes"
                        )

                        st.write(
                            "**Number Plate:**",
                            rider["number_plate"]
                            if rider["number_plate"]
                            else "Not detected / not readable"
                        )

                        st.write(
                            "**Plate Status:**",
                            rider["number_plate_status"]
                        )

                        st.write(
                            "**OCR Confidence:**",
                            f"{rider['ocr_confidence'] * 100:.2f}%"
                        )

                    elif rider["helmet_status"] == "Helmet":

                        st.success(
                            "Helmet detected - No violation"
                        )

                        st.write(
                            "**Helmet Status:** Helmet"
                        )

                        st.write(
                            "**Database:** Record not saved"
                        )

                    else:

                        st.warning(
                            "Helmet status could not be determined"
                        )

                        st.write(
                            "**Database:** Record not saved"
                        )

                    st.write(
                        "**Bike Detection Confidence:**",
                        f"{rider['bike_confidence'] * 100:.2f}%"
                    )

            # Display annotated image
            st.divider()

            st.subheader("Annotated Detection Image")

            try:

                output_url = (
                    API_URL + result["output_image"]
                )

                image_response = requests.get(
                    output_url,
                    timeout=30
                )

                image_response.raise_for_status()

                detected_image = Image.open(
                    BytesIO(image_response.content)
                )

                st.image(
                    detected_image,
                    caption="YOLO Detection Result",
                    use_container_width=True
                )

            except Exception as error:

                st.warning(
                    f"Unable to load result image: {error}"
                )


# ---------------------------------
# 6. Violation History
# ---------------------------------

elif page == "Violation History":

    st.subheader("📋 Helmet Violation History")

    if st.button("Refresh History"):

        st.rerun()

    try:

        response = requests.get(
            f"{API_URL}/history",
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        history = data.get("history", [])

        st.metric(
            "Total Violation Records",
            data.get("total_violations", 0)
        )

        if history:

            for record in history:

                with st.expander(
                    f"Violation #{record['id']} | "
                    f"Rider {record['rider_id']} | "
                    f"{record['image_filename']}"
                ):

                    st.write(
                        "**Date:**",
                        record["created_at"]
                    )

                    st.write(
                        "**Rider ID:**",
                        record["rider_id"]
                    )

                    st.write(
                        "**Helmet Status:**",
                        record["helmet_status"]
                    )

                    st.write(
                        "**Number Plate:**",
                        record["number_plate"]
                        if record["number_plate"]
                        else "Not detected / not readable"
                    )

                    st.write(
                        "**Plate Status:**",
                        record["number_plate_status"]
                    )

                    st.write(
                        "**Bike Confidence:**",
                        f"{record['bike_confidence'] * 100:.2f}%"
                    )

                    st.write(
                        "**OCR Confidence:**",
                        f"{record['ocr_confidence'] * 100:.2f}%"
                    )

                    output_filename = record.get(
                        "output_image"
                    )

                    if output_filename:

                        image_url = (
                            f"{API_URL}/results/{output_filename}"
                        )

                        image_response = requests.get(
                            image_url,
                            timeout=30
                        )

                        if image_response.ok:

                            st.image(
                                Image.open(
                                    BytesIO(image_response.content)
                                ),
                                caption="Violation Evidence",
                                use_container_width=True
                            )

        else:

            st.info(
                "No helmet violation records available yet."
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Backend is not running. Please start FastAPI."
        )

    except Exception as error:

        st.error(
            f"Unable to load history: {error}"
        )


# ---------------------------------
# 7. About Project
# ---------------------------------

elif page == "About Project":

    st.subheader("About Our Project")

    st.write("""
    This project is an AI-powered Helmet Violation Detection System.

    It uses YOLO and EasyOCR to identify motorcycle riders,
    detect helmet violations and read number plates of riders
    who are not wearing helmets.

    Main Features:

    - Multiple Bike and Rider Detection
    - Helmet and NoHelmet Classification
    - Rider-wise Detection Association
    - Number Plate Detection
    - OCR-based Number Plate Recognition
    - Separate Violation Records
    - SQLite Database
    - Detection Evidence Images

    Technologies Used:

    - Python
    - YOLO / Ultralytics
    - OpenCV
    - EasyOCR
    - FastAPI
    - Streamlit
    - SQLite
    """)

    st.info(
        "Only riders classified as NoHelmet are saved "
        "in the violation database."
    )


# ---------------------------------
# 8. Footer
# ---------------------------------

st.divider()

st.caption(
    "Helmet Violation Detection System | "
    "Powered by YOLO, EasyOCR, FastAPI and SQLite"
)
