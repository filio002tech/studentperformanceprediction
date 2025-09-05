
---

```markdown
# 🎓 Student Performance Predictor (SDU-Ozoro)

A web-based machine learning tool that assists academic advisors at **Southern Delta University (SDU), Ozoro** in assessing student performance risks.  
The system uses **study habits, support indicators, and extracurricular data** to predict student performance categories (`Failing`, `Average`, `Passing`, `Excellent`).

This project is built with **Python, Streamlit, and Scikit-Learn**.

---

## 🚀 Features
- Predicts student performance category based on input data
- Interactive web UI built with **Streamlit**
- Uses **Random Forest Classifier** and **Standard Scaler**
- Model training script included for reproducibility
- Data visualization with **Plotly**
- Lightweight and easy to deploy (Streamlit Cloud / Anaconda / local server)

---

## 📂 Project Structure
```

Fortunate\_Email\_Spam\_Detector/
│── student\_app.py                # Main Streamlit app
│── train\_model.py                # Script to train & save model
│── requirements.txt              # Dependencies
│── README.md                     # Project documentation
│
└── models/
├── best\_random\_forest\_model.pkl   # Trained model
└── scaler.pkl                     # StandardScaler object
└── assets/
└── sdu\_logo.png                   # University logo

````

---

## ⚙️ Installation & Setup

### 1. Clone this repository
```bash
git clone https://github.com/filio002tech/studentperformanceprediction.git
cd SDU-Student-Predictor
````

### 2. Create & activate environment

Using **Anaconda**:

```bash
conda create -n sdu_predictor python=3.10
conda activate sdu_predictor
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (optional, only if models are missing)

```bash
python train_model.py
```

### 5. Run the Streamlit app

```bash
streamlit run student_app.py
```

---

## 📊 Example Input

* **Age**: 18
* **Gender**: Male
* **Parental Education**: Tertiary
* **Study Time (hrs/week)**: 12
* **Absences**: 5
* **Extracurricular Activities**: Yes

➡️ Prediction: **Passing** 🎉

---

## 🛠️ Tech Stack

* **Python 3.10**
* **Streamlit**
* **Scikit-Learn**
* **Plotly**
* **Pandas / NumPy**

---

## 📬 Contacts

* **Student Affairs (SDU, Ozoro)**
  📧 [sdu.studentaffairs@example.edu.ng](mailto:sdu.studentaffairs@dsust.edu.ng)

* **Dept. of Computer Science**
  📧 [compsci@example.edu.ng](mailto:compsci@dsust.edu.ng)

---

## 📜 License

This project is for **academic research and educational purposes only**.
Not intended for commercial use.

```

---
