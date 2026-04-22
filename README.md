# 🛒 Online Retail MLOps Project

This project demonstrates an end-to-end **MLOps pipeline applied to an Online Retail dataset**, focusing on building, versioning, and deploying machine learning workflows in a production-like environment.

---

## 📖 Overview

MLOps combines machine learning with DevOps practices to automate and streamline the lifecycle of ML models, including data processing, training, testing, and deployment.

In this project, I built a pipeline to:
- 📊 Analyze online retail data
- 🧠 Train machine learning models
- 🔄 Track experiments and data versions
- ⚙️ Ensure reproducibility of workflows

> ⚠️ Note: The experimentation and model development were performed using **Google Colab**, while the pipeline and versioning are managed locally using MLOps tools.

---

## ✨ Key Features

- 📦 Data versioning using DVC  
- 🔁 Reproducible ML pipeline (`dvc.yaml`)  
- 🧠 Machine learning model training  
- 📊 Data preprocessing & feature engineering  
- 🧪 Testing using pytest & tox  
- ⚙️ Modular project structure  
- 🚀 CI/CD ready workflow  

---

## 🛠️ Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-learn  
- DVC (Data Version Control)  
- Git & GitHub  
- Pytest / Tox  
- Google Colab (for experimentation)  

---

## 📂 Project Structure
online-retail-mlops/
│── data/ # Dataset (tracked using DVC)
│── .dvc/ # DVC metadata
│── dvc.yaml # Pipeline definition
│── params.yaml # Model parameters
│── src/ # Source code
│── notebooks/ # Colab / experimentation files
│── requirements.txt
│── setup.py
│── tox.ini
│── .github/workflows/ # CI/CD pipelines
│── README.md
