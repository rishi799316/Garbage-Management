import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from gradcam import generate_gradcam, overlay_gradcam

st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Waste Management Classifier")
st.markdown("Upload an image to classify waste as **Organic** or **Recyclable**")

# Load model with caching to avoid reloading on every interaction
@st.cache_resource
def load_waste_model():
    try:
        model = load_model("best_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_waste_model()

if model is None:
    st.stop()

st.success("✓ Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image of waste",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    try:
        # Load original image
        original_img_pil = Image.open(uploaded_file).convert("RGB")

        # Create two columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(original_img_pil, use_container_width=True)

        # Preprocess image for model
        img_resized = original_img_pil.resize((150, 150))
        img_array = image.img_to_array(img_resized)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        with st.spinner("Analyzing image..."):
            pred = model.predict(img_array, verbose=0)[0][0]

        # Display results in second column
        with col2:
            st.subheader("Analysis Results")

            # Show confidence scores for both classes
            organic_conf = (1 - pred) * 100
            recyclable_conf = pred * 100

            st.metric("Organic Confidence", f"{organic_conf:.1f}%")
            st.metric("Recyclable Confidence", f"{recyclable_conf:.1f}%")

            

        st.divider()

        # Classification result
        UNCERTAINTY_THRESHOLD_LOW = 0.40
        UNCERTAINTY_THRESHOLD_HIGH = 0.60

        if UNCERTAINTY_THRESHOLD_LOW <= pred <= UNCERTAINTY_THRESHOLD_HIGH:
            st.warning("⚠️ Model is unsure about this classification. Please help us learn!")
            user_choice = st.radio(
                "What type of waste is this?",
                ["Organic", "Recyclable"],
                help="Your feedback helps improve the model"
            )
            if user_choice:
                st.info(f"✓ You selected: **{user_choice}**")
                st.caption("Thank you for your feedback!")
        else:
            # Confident prediction
            if pred < 0.5:
                label = "Organic"
                icon = "♻️"
                color = "green"
            else:
                label = "Recyclable"
                icon = "🗑️"
                color = "blue"

            st.success(f"Prediction: **{label} {icon}**")

            # Disposal instructions
            with st.expander("ℹ️ Disposal Instructions"):
                if label == "Organic":
                    st.markdown("""
                    **Organic Waste** includes:
                    - Food scraps
                    - Garden waste
                    - Paper products (non-glossy)

                    **Disposal:** Compost bin or organic waste collection
                    """)
                else:
                    st.markdown("""
                    **Recyclable Waste** includes:
                    - Plastic containers
                    - Glass bottles
                    - Metal cans
                    - Cardboard

                    **Disposal:** Recycling bin (clean and dry)
                    """)

        st.divider()

    except Exception as e:
        st.error(f"Error processing the image: {e}")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload an image to get started")

    with st.expander("📖 How to use this app"):
        st.markdown("""
        1. **Upload** an image of waste material
        2. **Wait** for the model to analyze it
        3. **Review** the classification and confidence scores
        4. **Check** the attention map to see what the model focused on
        5. If the model is uncertain, **provide feedback** to help it learn
        """)

# Footer
st.divider()
st.caption("Powered by TensorFlow & Streamlit | Helping make waste management smarter ♻️")
