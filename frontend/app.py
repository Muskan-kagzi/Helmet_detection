
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# -------------------------------
# 1. Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Helmet & Number Plate Detection",
    page_icon="🏍️",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"


# -------------------------------
# 2. Custom Styling
# -------------------------------

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


# -------------------------------
# 3. Header
# -------------------------------

st.markdown(
    '<div class="title">🏍️ Helmet & Number Plate Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered road safety monitoring system using YOLO</div>',
    unsafe_allow_html=True
)

st.divider()


# -------------------------------
# 4. Sidebar
# -------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "Detection History", "About Project"]
)


# -------------------------------
# 5. Home Page
# -------------------------------

if page == "Home":

    st.subheader("Upload Image")

    st.write(
        "Upload a motorcycle image to detect riders, helmets, "
        "no-helmet cases and number plates."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### Detection Result")

            if st.button("Start Detection", type="primary"):

                with st.spinner("Detecting objects... Please wait"):

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
                            timeout=120
                        )

                        response.raise_for_status()

                        result = response.json()

                        st.success("Detection completed successfully!")

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
                            caption="Detected Image",
                            use_container_width=True
                        )

                        st.session_state["last_result"] = result

                    except requests.exceptions.ConnectionError:
                        st.error(
                            "Backend is not running. "
                            "Please start FastAPI first."
                        )

                    except requests.exceptions.RequestException as error:
                        st.error(f"API Error: {error}")

                    except Exception as error:
                        st.error(f"Something went wrong: {error}")

            # Show detection details
            if "last_result" in st.session_state:

                result = st.session_state["last_result"]

                st.markdown("### Detection Details")

                detections = result.get("detections", [])

                if detections:

                    for item in detections:

                        class_name = item["class_name"]
                        confidence = item["confidence"]

                        st.write(
                            f"**{class_name}** — "
                            f"{confidence * 100:.2f}% confidence"
                        )

                else:
                    st.warning("No objects detected.")


# -------------------------------
# 6. Detection History Page
# -------------------------------

elif page == "Detection History":

    st.subheader("📋 Detection History")

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
            "Total Images Processed",
            data.get("total", 0)
        )

        if history:

            for record in history:

                with st.expander(
                    f"Image: {record['filename']} | "
                    f"ID: {record['id']}"
                ):

                    st.write(
                        "**Date:**",
                        record["created_at"]
                    )

                    st.write("**Detections:**")

                    for item in record["detections"]:

                        st.write(
                            f"- {item['class_name']} "
                            f"({item['confidence'] * 100:.2f}%)"
                        )

                    output_filename = record.get("output_image")

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
                                caption="Detection Result",
                                use_container_width=True
                            )

        else:

            st.info("No detection history available yet.")

    except requests.exceptions.ConnectionError:

        st.error("Backend is not running. Please start FastAPI.")

    except Exception as error:

        st.error(f"Error loading history: {error}")


# -------------------------------
# 7. About Project Page
# -------------------------------

elif page == "About Project":

    st.subheader("About Our Project")

    st.write("""
    This project is an AI-based Helmet and Number Plate Detection System.

    It uses a trained YOLO model to identify:

    - BikeWithRider
    - Helmet
    - NoHelmet
    - NumberPlate

    Technology Used:

    - Python
    - YOLO / Ultralytics
    - OpenCV
    - FastAPI
    - Streamlit
    - SQLite Database
    """)

    st.info(
        "The system detects objects in uploaded images "
        "and stores detection history in SQLite."
    )


# -------------------------------
# 8. Footer
# -------------------------------

st.divider()

st.caption(
    "Helmet & Number Plate Detection System | "
    "Powered by YOLO, FastAPI and Streamlit"
)
