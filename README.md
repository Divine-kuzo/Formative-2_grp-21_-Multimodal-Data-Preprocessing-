# Group 21 – Multimodal Data Preprocessing

## 1. Project Overview

This project focuses on building a multimodal data preprocessing and machine learning system.

The system combines different types of data, including customer information, images, audio recordings, and product-related data, to support customer identification, voice verification, and product recommendations.

The project demonstrates how different data modalities can be processed independently and integrated into a single workflow.

### Main Objectives

- Preprocess and explore customer data.
- Process images and extract relevant visual features.
- Process audio recordings and extract voice features.
- Train machine learning models for voice identification and product recommendation.
- Integrate the different components into a working application.

---

## 2. Team Contributions

| Member | Responsibility | Description |
|---|---|---|
| Member 1 | Customer Data Preprocessing | Merge, clean, and explore customer data. |
| Member 2 | Image Preprocessing | Process images and implement facial recognition. |
| Member 3 – Constantine | Audio Preprocessing and Voice Verification | Process audio recordings, extract audio features, apply augmentation, and train a voice classification model. |
| Member 4 | Product Recommendation and Integration | Develop the product recommendation component and integrate the system. |

---

# 3. Member 3 – Audio Preprocessing and Voice Verification

## 3.1 Overview

The audio component processes voice recordings from four group members.

Its purpose is to extract meaningful numerical features from audio recordings and use machine learning to identify the speaker.

The implementation includes:

- Audio loading and preprocessing.
- Audio augmentation.
- Feature extraction.
- Audio dataset construction.
- Machine learning model training and evaluation.
- Voice verification functionality.

### Technologies Used

- Python
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## 3.2 Audio Dataset Structure

The original recordings are organized by speaker.

Each speaker provides two recordings:

1. `confirm_transaction.wav`
2. `yes_approve.wav`

The original audio directory contains eight recordings from four speakers.

```text
audio/
├── member_01/
│   ├── confirm_transaction.wav
│   └── yes_approve.wav
├── member_02/
│   ├── confirm_transaction.wav
│   └── yes_approve.wav
├── member_03/
│   ├── confirm_transaction.wav
│   └── yes_approve.wav
└── member_04/
    ├── confirm_transaction.wav
    └── yes_approve.wav