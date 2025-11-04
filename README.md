# Solar-Panel-Fault-Detection-App

This project is a computer vision–based solar PV fault detection tool designed to assist in automated solar asset inspection. It classifies solar panel images into three categories — **Clean**, **Physical Damage**, and **Electrical Fault** — using a TensorFlow Lite model and an interactive Streamlit interface.

The system simulates real-world workflows used in drone-based solar inspections and renewable energy analytics. It supports both single-image and batch processing modes, and exports structured CSV reports suitable for field diagnostics and reporting.

---

## 🚀 Features

* ✅ Solar PV fault classification using deep learning
* ✅ Categories: *Clean*, *Physical Damage*, *Electrical Fault*
* ✅ User-friendly Streamlit web interface
* ✅ Supports **single & bulk image uploads**
* ✅ Real-time predictions with confidence scores
* ✅ CSV report export for field teams
* 🎯 Designed to mimic real-world drone inspection workflows

---

## 🧠 Tech Stack

| Component        | Technology                |
| ---------------- | ------------------------- |
| Model            | TensorFlow Lite           |
| UI               | Streamlit                 |
| Image Processing | PIL, NumPy                |
| Data Handling    | Pandas                    |
| Environment      | Python (3.10 recommended) |

---

## 📁 Project Structure

```
├── app.py

├── converted_model.tflite

├── labels.txt

├── requirements.txt

└── sample_images/
```

---

## 📦 Input Requirements

* Solar panel images in `.jpg` / `.jpeg` / `.png`
* Works with drone / mobile / thermal-converted images

---

## 📊 Output

The tool generates:

* Predicted class (Clean / Physical / Electrical Fault)
* Confidence score
* CSV report (for bulk runs)

Example CSV Columns:

```
filename, top_label, top_confidence, prob_Clean, prob_Physical, prob_Electrical
```
----
🧠 Codebase Overview & Core Logic

This project is implemented in Python with a clean, modular structure designed for clarity, maintainability, and efficient computer‑vision inference.

## Core Files
File --->  Purpose
app.py	 ---> Main application entry point; handles UI, preprocessing, inference, results UI & export
converted_model.tflite	 ---> Optimized TensorFlow Lite model for lightweight, fast inference
labels.txt	 ---> Label mapping corresponding to model output indices


## Key Functional Components

Component	 ---> Responsibility


Model  ---> Loader	Initializes TensorFlow Lite interpreter and allocates tensors


Preprocessing Pipeline	 ---> Image resizing (224×224), normalization, tensor formatting


Inference Engine	 ---> Runs model inference and retrieves softmax probability scores


Post‑processing	 ---> Extracts top prediction, interprets class labels, formats confidence


UI Logic	 ---> Streamlit interface for image upload, visualization, metrics, batch progress


Reporting Engine	 ---> Generates structured CSV outputs for single & batch modes


## Design Principles


Lightweight Deployment: TFLite ensures fast inference even on low‑power devices


Separation of Concerns: Preprocessing, inference, and UI layers are logically split


Scalable Input Modes: Supports both single‑image and multi‑image workflows


Practical Field Utility: CSV outputs mirror real inspection reporting formats



## Execution Flow

Load Model → Upload Image(s) → Preprocess → Infer → Score & Classify → Display Results → Export CSV

This structure aligns with modern renewable‑AI inspection systems—bridging ML inference with practical solar O&M workflows.
