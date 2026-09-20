# 🫀Heart Disease Web-based Prediction System

> A machine learning-powered web application that predicts the likelihood 
> of heart disease from patient clinical data — fast, accessible, and non-invasive.

![Python](https://img.shields.io/badge/Python-3.14.6-blue)
![Flask](https://img.shields.io/badge/Flask-3.1.3-lightgrey)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange)
![SQLite](https://img.shields.io/badge/Database-SQLite-green)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple)

---

## 📋Table of Contents

- [Project Overview](#-project-overview)
- [Screenshots](#-screenshots)
- [Features](#-features)
- [Machine Learning Results](#-machine-learning-results)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Academic Context](#-academic-context)
- [Disclaimer](#-disclaimer)

---

## 📌Project Overview

The **Heart Disease Web-based Prediction System** is a full-stack web application 
developed as part of my Computer Science Final Year Project. It uses supervised 
machine learning to analyse 13 patient clinical features and predict the 
likelihood of heart disease, delivering results with a confidence score 
through a user-friendly browser interface.

Six machine learning algorithms were trained and compared on the 
**UCI Cleveland Heart Disease Dataset**. The best-performing model — 
**Logistic Regression (87.78% accuracy)** — was deployed into the Flask 
web application as the prediction engine.

The system supports two user roles:
- **User / Patient** — enter clinical data and receive predictions
- **Administrator** — manage users, monitor performance, and manage the dataset

---

## 📸Screenshots

| Home Page | Prediction Form |
|-----------|----------------|
| ![Home](screenshots/home_page.png) | ![Form](screenshots/prediction_form.png) |

| Result — Disease Detected | Result — No Disease |
|--------------------------|---------------------|
| ![Disease](screenshots/result_disease.png) | ![No Disease](screenshots/result_no_disease.png) |

| User Dashboard | Prediction History |
|---------------|-------------------|
| ![Dashboard](screenshots/dashboard.png) | ![History](screenshots/prediction_history.png) |

| Admin Dashboard | System Performance |
|----------------|-------------------|
| ![Admin](screenshots/admin_dashboard.png) | ![Performance](screenshots/admin_performance.png) |

---

## ✨Features

### User / Patient Side
- 🔐 Secure registration and login with password hashing
- 📋 13-field clinical data input form with tooltips and validation
- 🤖 Instant heart disease risk prediction powered by Logistic Regression
- 📊 Confidence score and clinical recommendation on result page
- 🕐 Persistent prediction history across sessions
- 📈 Algorithm accuracy comparison chart on dashboard

### Administrator Side
- 🛡️ Role-based admin panel (separate from user interface)
- 👥 Manage all registered user accounts
- 📋 View all system-wide predictions from all users
- 📊 Monitor model performance — accuracy table, bar chart, confusion matrices
- 🗄️ Dataset management — view dataset info and status

---

## 🤖Machine Learning Results

Six supervised learning algorithms were trained on the UCI Cleveland 
Heart Disease Dataset (297 records, 70/30 train-test split):

| Algorithm | Accuracy | Precision | Recall | F1-Score | Specificity |
|-----------|----------|-----------|--------|----------|-------------|
| **Logistic Regression** ✅ | **87.78%** | **94.29%** | **78.57%** | **85.71%** | **95.83%** |
| K-Nearest Neighbors | 86.67% | 94.12% | 76.19% | 84.21% | 95.83% |
| Naive Bayes | 86.67% | 94.12% | 76.19% | 84.21% | 95.83% |
| Support Vector Machine | 86.67% | 94.12% | 76.19% | 84.21% | 95.83% |
| Random Forest | 83.33% | 88.57% | 73.81% | 80.52% | 91.67% |
| Decision Tree | 66.67% | 65.00% | 61.90% | 63.41% | 70.83% |

✅ **Logistic Regression** was selected for deployment based on highest 
overall accuracy, precision, F1-score, and lowest false negative count.

---

## 🛠 Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.14.6 |
| Web Framework | Flask 3.1.3 |
| Machine Learning | Scikit-Learn 1.9.0 |
| Database | SQLite (via Flask-SQLAlchemy 3.1.1) |
| Authentication | Flask-Login 0.6.3 + Werkzeug 3.1.8 |
| Frontend | HTML5, CSS3, Bootstrap 5.3, Font Awesome 6 |
| Data Processing | Pandas 3.0.3, NumPy 2.5.1 |
| Visualisation | Matplotlib 3.11.1, Seaborn 0.13.2 |
| IDE | Visual Studio Code |
| OS | Windows 11 |

---

## 🏗 System Architecture

The system follows a **client-server architecture** with two parallel methodologies:
- **OOADM (Object-Oriented Analysis & Design Methodology)** — for the web application
- **CRISP-DM** — for the machine learning pipeline
