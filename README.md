# OCR-optical-character-recognition-using-easyOCR
Perfect ✅ — here’s the **final detailed README.md** version for your mini project **“Optical Character Recognition using EasyOCR”**, including:

* A professional GitHub-style layout
* A **requirements installation section**
* Commands to install each library
* Ready-to-use format for submission or upload

---

# 🧠 Optical Character Recognition (OCR) using EasyOCR

## 📘 Overview

This mini-project demonstrates **Optical Character Recognition (OCR)** — a computer vision technique that extracts readable text from images.
Using the **EasyOCR** library (powered by **PyTorch**), this project detects and recognizes characters from various image types such as screenshots, scanned pages, and signboards.

The goal is to automate text extraction from images using a deep learning–based OCR engine.

---

## 🧩 Features

* Reads and extracts printed or handwritten text from images.
* Works efficiently on both **CPU** and **GPU**.
* Supports **multiple languages** (default: English).
* Uses **OpenCV** for image handling and **Matplotlib** for visualization.
* Simple and easy-to-run implementation in Jupyter Notebook or Google Colab.

---

## 🛠️ Tech Stack

| Category                | Tool / Library                  |
| ----------------------- | ------------------------------- |
| Programming Language    | Python 3                        |
| OCR Engine              | EasyOCR                         |
| Deep Learning Framework | PyTorch                         |
| Image Processing        | OpenCV                          |
| Visualization           | Matplotlib                      |
| Environment             | Jupyter Notebook / Google Colab |

---

## 📂 Project Structure

```
Optical_Character_Recognition/
│
├── Optical_Character_Recognition.ipynb   # Main project notebook
├── requirements.txt                       # List of dependencies
├── sample_images/                         # Folder containing test images
└── README.md                              # Project documentation
```

---

## ⚙️ Installation and Setup

### Step 1️⃣: Clone the Repository

```bash
git clone https://github.com/<your-username>/Optical_Character_Recognition.git
cd Optical_Character_Recognition
```

### Step 2️⃣: Install Dependencies

You can either install manually or use the `requirements.txt` file.

#### Option 1 — Install manually:

```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install easyocr
```

#### Option 2 — Using requirements.txt:

```bash
pip install -r requirements.txt
```

---

## 🧾 requirements.txt

Here’s the content you can save in your **requirements.txt** file:

```
opencv-python
numpy
matplotlib
torch
torchvision
torchaudio
easyocr
```

---

## 🚀 How to Run the Project

1. **Open the notebook** in Jupyter or Google Colab:

   ```
   Optical_Character_Recognition.ipynb
   ```

2. **Set the image path**:

   ```python
   image_path = '/content/sample_data/image.jpg'
   ```

3. **Run the OCR model**:

   ```python
   import easyocr
   reader = easyocr.Reader(['en'])
   results = reader.readtext(image_path)
   ```

4. **Display recognized text**:

   ```python
   for detection in results:
       text = detection[1]
       print("Detected text:", text)
   ```

5. **Visualize results**:

   ```python
   import cv2, matplotlib.pyplot as plt
   img = cv2.imread(image_path)
   plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
   plt.axis('off')
   plt.show()
   ```

---

## 🖼️ Example Output

| Input Image                                                       | OCR Output                |
| ----------------------------------------------------------------- | ------------------------- |
| <img width="400" height="400" alt="Screenshot 2025-10-27 122427" src="https://github.com/user-attachments/assets/60da8a99-9a93-4954-958e-5823e6a14832" />|<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/f14f3611-d73c-4273-8f35-6ace9a5c1f83" /> |

---

## 📊 Results

* Extracted printed text with high accuracy on clear images.
* EasyOCR’s deep learning model handled complex fonts and lighting variations.
* Processing time was minimal on both CPU and GPU setups.

---

## 🔮 Future Enhancements

* Add **real-time OCR** using a webcam with OpenCV.
* Enable **multilingual support** (e.g., English, Hindi, Tamil, Kannada).
* Integrate a **Graphical User Interface (GUI)** for easier image uploads.
* Implement **spell correction** for OCR outputs.

---

## 👩‍💻 Author

**Preetham N D**
Mini Project — *Optical Character Recognition using EasyOCR*
Department of Computer Science

---

Would you like me to now generate a **ready-to-download `requirements.txt` file** and the **final README.md** (as files you can attach or upload to GitHub)?
