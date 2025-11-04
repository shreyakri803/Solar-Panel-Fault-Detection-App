# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from datetime import datetime

st.set_page_config(layout="wide", page_title="Solar Panel Fault Detection")

# ------------ Utilities ------------
def load_model():
    interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

def load_labels(path="labels.txt"):
    """
    Robustly parse labels lines such as:
      '0 Clean'  OR  'Clean'
    Returns a clean list like ['Clean','Physical','Electrical'].
    """
    parsed = []
    with open(path, "r") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split()
            label = parts[-1] if parts[0].isdigit() else raw
            parsed.append(label)
    return parsed

def process_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

def predict(interpreter, image_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0] 

DISPLAY_MAP = {
    "Clean": "✅ Clean / Healthy",
    "Physical": "🧩 Physical Damage",
    "Electrical": "⚡ Electrical Damage"
}
def format_condition(label: str) -> str:
    return DISPLAY_MAP.get(label, label)

def build_prediction_row(filename, probs_vec, labels, top_idx,
                         site_id="", array_id="", lat="", lon=""):
    probs = {labels[i]: float(probs_vec[i]) for i in range(len(labels))}
    top_label = labels[top_idx]
    top_conf = float(probs_vec[top_idx])
    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "filename": filename,
        "top_label": top_label,
        "top_label_display": format_condition(top_label),
        "top_confidence": round(top_conf, 4),
        "site_id": site_id,
        "array_id": array_id,
        "lat": lat,
        "lon": lon,
    }

    for k in sorted(probs.keys()):
        row[f"prob_{k}"] = round(probs[k], 4)
    return row

def main():
    st.title("Solar Panel Fault Detection")

    try:
        interpreter = load_model()
        labels = load_labels("labels.txt")
        num_classes = len(labels)
        if num_classes == 0:
            st.error("No labels found in labels.txt")
            return


        st.sidebar.header("Analysis Settings")
        min_conf = st.sidebar.slider("Minimum confidence to flag issues",
                                     min_value=0.0, max_value=1.0, value=0.70, step=0.05)
        st.sidebar.markdown("**Optional metadata (included in CSV):**")
        site_id = st.sidebar.text_input("Site ID", value="")
        array_id = st.sidebar.text_input("Array / String ID", value="")
        lat = st.sidebar.text_input("Latitude", value="")
        lon = st.sidebar.text_input("Longitude", value="")

        tab1, tab2 = st.tabs(["Single Image Analysis", "Batch Analysis"])

        with tab1:
            uploaded_file = st.file_uploader("Upload a solar panel image", type=['jpg', 'png', 'jpeg'], key="single")
            if uploaded_file:
                c1, c2 = st.columns(2)
                with c1:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                with c2:
                    if st.button("Analyze Image", type="primary"):
                        with st.spinner("Analyzing..."):
                            x = process_image(image)
                            preds = predict(interpreter, x)
                            top_idx = int(np.argmax(preds))
                            top_label = labels[top_idx]
                            top_conf = float(preds[top_idx])

                        
                            if top_label == "Clean":
                                st.success(f"{format_condition(top_label)} • Confidence: {top_conf:.2%}")
                            else:
                                if top_conf >= min_conf:
                                    st.error(f"{format_condition(top_label)} • Confidence: {top_conf:.2%}")
                                else:
                                    st.warning(f"{format_condition(top_label)} • Low confidence ({top_conf:.2%}). Review manually.")

                            st.subheader("Per-class probabilities")
                            for i in range(num_classes):
                                lab = labels[i]
                                disp = format_condition(lab)
                                pct = float(preds[i]) * 100.0
                                st.write(f"- **{disp}**: {pct:.1f}%")
                                st.progress(int(round(pct)))

                
                            row = build_prediction_row(
                                filename=getattr(uploaded_file, "name", "single_image"),
                                probs_vec=preds,
                                labels=labels,
                                top_idx=top_idx,
                                site_id=site_id, array_id=array_id, lat=lat, lon=lon
                            )
                            df_single = pd.DataFrame([row])
                            st.download_button(
                                "Download CSV for this image",
                                df_single.to_csv(index=False).encode("utf-8"),
                                "result_single.csv",
                                "text/csv"
                            )

        with tab2:
            uploaded_files = st.file_uploader("Upload multiple images",
                                              type=['jpg', 'png', 'jpeg'],
                                              accept_multiple_files=True,
                                              key="batch")
            if uploaded_files:
                if st.button("Analyze All Images", type="primary"):
                    results = []
                    progress = st.progress(0)
                    status = st.empty()

                    cols = st.columns(3)
                    grid_i = 0

                    for idx, file in enumerate(uploaded_files):
                        status.text(f"Processing image {idx + 1} of {len(uploaded_files)}")
                        try:
                            img = Image.open(file).convert('RGB')
                            x = process_image(img)
                            preds = predict(interpreter, x)
                            top_idx = int(np.argmax(preds))
                            top_label = labels[top_idx]
                            top_conf = float(preds[top_idx])

                            with cols[grid_i % 3]:
                                st.image(img, caption=file.name, use_container_width=True)
                                if top_label == "Clean":
                                    st.caption(f"✅ Clean ({top_conf:.0%})")
                                else:
                                    badge = "🧩" if top_label == "Physical" else ("⚡" if top_label == "Electrical" else "⚠️")
                                    if top_conf >= min_conf:
                                        st.caption(f"{badge} {top_label} ({top_conf:.0%}) – flagged")
                                    else:
                                        st.caption(f"⚠️ {top_label} ({top_conf:.0%}) – low conf")
                            grid_i += 1

                            row = build_prediction_row(
                                filename=file.name,
                                probs_vec=preds,
                                labels=labels,
                                top_idx=top_idx,
                                site_id=site_id, array_id=array_id, lat=lat, lon=lon
                            )
                            results.append(row)

                        except Exception as e:
                            results.append({
                                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                "filename": file.name,
                                "top_label": "Error",
                                "top_label_display": "Error",
                                "top_confidence": 0.0,
                                "site_id": site_id, "array_id": array_id, "lat": lat, "lon": lon,
                                "error": str(e)
                            })

                        progress.progress(int(round((idx + 1) / len(uploaded_files) * 100)))

                    status.text("Analysis Complete!")

                    if results:
                        df = pd.DataFrame(results)
                        st.subheader("Summary")
                        total = len(df)
                        issues = df[(df["top_label"] != "Clean") & (df["top_confidence"] >= min_conf)]
                        clean = df[df["top_label"] == "Clean"]
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Panels", total)
                        c2.metric("Issues Found (high-conf)", len(issues))
                        c3.metric("Clean Panels", len(clean))

                        st.subheader("Detailed Results")
                        show_only_flags = st.checkbox("Show only high-confidence non-clean issues", value=False)
                        df_view = df.copy()
                        if show_only_flags:
                            df_view = df_view[(df_view["top_label"] != "Clean") & (df_view["top_confidence"] >= min_conf)]
                        st.dataframe(df_view, use_container_width=True)

                        st.download_button(
                            "Download Results (CSV)",
                            df.to_csv(index=False).encode("utf-8"),
                            "solar_panel_analysis.csv",
                            "text/csv"
                        )

    except FileNotFoundError:
        st.error("Model or labels file not found!")
        st.info("Ensure 'converted_model.tflite' and 'labels.txt' are in the app directory.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
