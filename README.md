# Solar Panel Fault Detection – TensorFlow Lite + Streamlit

A lightweight, real-time **solar PV fault detection** system designed to support automated solar inspection workflows. This project classifies solar panel images into three operational states and generates structured maintenance reports suitable for field operations.

> ✅ Built for edge efficiency · ✅ Streamlit UI · ✅ Single & batch inference · ✅ CSV reporting · ✅ Clean engineering & traceable logic

---

## 🚀 Project Overview

Manual inspection of large-scale solar farms is time-consuming and prone to human error. This project automates visual fault detection to assist maintenance teams and drone-inspection pipelines.

### **Core Objectives**

* Detect panel health conditions from images
* Enable real-time and batch inspection modes
* Generate structured maintenance logs for field workflows
* Support deployment on CPU-only edge devices

---

## 🎯 Capabilities

| Feature       | Description                                     |
| ------------- | ----------------------------------------------- |
| Fault Classes | Clean, Physical Damage, Electrical Damage       |
| UI            | Streamlit-based web interface                   |
| Model Format  | TensorFlow Lite (CPU-optimized)                 |
| Modes         | Single image & bulk processing                  |
| Output        | CSV report with probabilities & metadata        |
| Metadata      | Site ID, Array ID, GPS fields for field mapping |

---

## 🧠 Architecture

```text
User Uploads Image(s)
        ↓
Preprocessing (224×224, Normalization)
        ↓
TFLite Inference (CPU)
        ↓
Top-Class & Probability Extraction
        ↓
Streamlit UI + CSV Export
```

---

## 💻 Tech Stack

* Python
* TensorFlow Lite
* Streamlit
* NumPy, Pillow, Pandas

---

## 📂 Repository Structure

```plaintext
📦 solar-pv-fault-detection
 ┣ app.py                  # Streamlit application
 ┣ converted_model.tflite  # Optimized inference model
 ┣ labels.txt              # Class labels
 ┗ requirements.txt        # Dependencies
 
```

---

## ⚙️ How It Works

**Input Processing**

* Images auto-resized to **224×224**
* Normalized to **[-1, 1]** tensor

**Inference**

* TFLite interpreter loads model to CPU
* Outputs probability for each class

**Decision Logic**

* Top label selected via softmax
* Confidence threshold configurable (default **0.70**)

**Batch Mode**

* Sequential inference with progress UI
* Displays preview & flag results

---

## 📊 Output Format (CSV)

| Column         | Description              |
| -------------- | ------------------------ |
| timestamp      | UTC time                 |
| filename       | Image name               |
| top_label      | Predicted class          |
| top_confidence | Confidence score         |
| metadata       | Site ID, Array ID, GPS   |
| prob_*         | Class-wise probabilities |

---

## 🛠️ Installation & Usage

```bash
git clone <repo-url>
cd solar-pv-fault-detection
pip install -r requirements.txt
streamlit run app.py
```

Access at: **[http://localhost:8501](http://localhost:8501)**

---

## 📌 Practical Use Cases

* Drone-based solar inspections
* PV maintenance workflows
* Renewable asset health scoring
* Edge-AI deployment prototypes

---

## 📥 Future Enhancements

* Thermal image support
* YOLO-based hotspot/defect localization
* ONNX export for embedded boards
* MongoDB / cloud report sync

---

## 👤 Author

**Shreya Kumari**

LinkedIn: *https://www.linkedin.com/in/shreya-k-986a8321b*

*For recruiters & engineers reviewing this repo: this implementation focuses on inference & deployment workflow rather than model training — enabling easy integration into real solar asset inspection systems.*
