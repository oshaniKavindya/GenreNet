# 🎵 GenreNet

**Deep Learning Music Genre Classification using CNNs**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

**GenreNet** is a deep learning project that automatically classifies music tracks into 10 different genres using Convolutional Neural Networks (CNNs). The system analyzes audio signals and identifies whether a song belongs to blues, classical, country, disco, hip hop, jazz, metal, pop, reggae, or rock.



## 🎯 Overview

**GenreNet** implements a CNN-based approach for automatic music genre classification, a fundamental task in Music Information Retrieval (MIR). The system converts audio signals into mel-spectrograms (visual representations of sound) and uses deep learning to automatically learn genre-discriminative features without manual feature engineering.

### **Why This Matters**

Music streaming platforms like Spotify and Apple Music use similar systems to:
- Organize millions of tracks in their catalogs
- Generate personalized playlists
- Power recommendation engines
- Enable efficient music discovery

### **The Problem**

Traditional approaches required manual feature engineering (extracting MFCCs, spectral features, etc.) and relied on handcrafted audio descriptors. **GenreNet** demonstrates how CNNs can automatically learn better features directly from mel-spectrograms, eliminating the need for manual feature design.

---

## ✨ Key Features

- **🎵 10 Genre Classification:** Blues, Classical, Country, Disco, Hip Hop, Jazz, Metal, Pop, Reggae, Rock
- **🧠 Automatic Feature Learning:** No manual feature engineering required
- **🎨 Mel-Spectrogram Processing:** Converts audio to visual time-frequency representations
- **⚡ Fast Inference:** Real-time prediction capability
- **📊 Comprehensive Analysis:** Confusion matrix, per-genre performance metrics

---

## 📊 Dataset

### **GTZAN Genre Collection**

- **Total Tracks:** 1,000 audio files (WAV format, 22,050 Hz)
- **Duration:** 30 seconds per track
- **Genres:** 10 balanced classes (100 tracks each)
- **Augmentation:** Each track split into 10 × 3-second segments → 9,990 training samples
- **Data Split:** 80% training / 10% validation / 10% test (stratified)

**Download Dataset:**
```bash
# Kaggle
[https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification]
```

**Genre Distribution:**
```
Blues:     100 tracks
Classical: 100 tracks
Country:   100 tracks
Disco:     100 tracks
Hip Hop:   100 tracks
Jazz:      100 tracks
Metal:     100 tracks
Pop:       100 tracks
Reggae:    100 tracks
Rock:      100 tracks
```

---

## 🏗️ Model Architecture

### **CNN Architecture Overview**

```
Input: Mel-Spectrogram (128 × 130 × 1)
    ↓
┌─────────────────────────────┐
│  Convolutional Block 1      │
│  - Conv2D (32 filters, 3×3) │
│  - ReLU Activation          │
│  - Batch Normalization      │
│  - MaxPooling (2×2)         │
│  - Dropout (0.25)           │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Convolutional Block 2      │
│  - Conv2D (64 filters, 3×3) │
│  - ReLU + BatchNorm         │
│  - MaxPooling + Dropout     │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Convolutional Block 3      │
│  - Conv2D (128 filters)     │
│  - ReLU + BatchNorm         │
│  - MaxPooling + Dropout     │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Convolutional Block 4      │
│  - Conv2D (256 filters)     │
│  - ReLU + BatchNorm         │
│  - MaxPooling + Dropout     │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Classification Head        │
│  - Global Average Pooling   │
│  - Dense (256 units)        │
│  - ReLU + BatchNorm         │
│  - Dropout (0.5)            │
│  - Dense (10 units)         │
│  - Softmax Activation       │
└─────────────────────────────┘
    ↓
Output: 10 Genre Probabilities
```

### **Key Design Choices**

1. **Hierarchical Learning:** Progressive filter increase (32→64→128→256) enables learning from simple patterns to complex genre-specific features
2. **Regularization:** Dropout (0.25-0.5) and Batch Normalization prevent overfitting
3. **Data Augmentation:** Temporal segmentation increases dataset 10× (1,000 → 9,990 samples)
4. **Mel-Spectrograms:** 128 mel-frequency bands capture human-like audio perception

---

## 🎯 What GenreNet Can Do

GenreNet analyzes audio files and predicts the most likely genre based on learned patterns in mel-spectrograms. The system provides:

### **Classification Capabilities**
- Processes 30-second audio clips in WAV format
- Converts audio to mel-spectrogram representations
- Applies convolutional neural network analysis
- Outputs probability distribution across all 10 genres
- Identifies the most likely genre classification

### **Training Process**
- Uses stratified data splitting (80% train / 10% validation / 10% test)
- Implements data augmentation through temporal segmentation
- Applies regularization techniques (dropout, batch normalization)
- Monitors training with early stopping and model checkpointing
- Adaptively adjusts learning rate during training

### **Model Characteristics**
- Learns hierarchical features automatically from spectrograms
- Early layers detect basic audio patterns
- Deeper layers learn genre-specific characteristics
- No manual feature engineering required
- Real-time inference capability

---

## 🔧 Technical Details

### **Audio Preprocessing**

```python
# Mel-Spectrogram Parameters
SAMPLE_RATE = 22050 Hz
DURATION = 3 seconds per segment
N_MELS = 128 mel-frequency bands
N_FFT = 2048 samples
HOP_LENGTH = 512 samples
OUTPUT_SHAPE = (128, 130, 1)
```

### **Data Augmentation Strategy**

- Each 30-second track → 10 × 3-second segments
- Increases dataset from 1,000 → 9,990 samples
- Captures different temporal positions (intro, verse, chorus, etc.)

### **Training Configuration**

```python
OPTIMIZER = Adam(lr=0.001)
LOSS = Categorical Crossentropy
BATCH_SIZE = 32
EPOCHS = 50 (with early stopping)
CALLBACKS = [
    EarlyStopping(monitor='val_loss', patience=15),
    ModelCheckpoint(save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

---
```

## ⚠️ Limitations

### **Dataset Limitations**

1. **Temporal Bias:** GTZAN contains only music from 1980-2000
2. **Geographic Bias:** Focuses exclusively on Western music styles
3. **Quality Issues:** Contains mislabeled tracks, duplicates, distortions (Sturm, 2013)
4. **Genre Coverage:** Missing contemporary genres (trap, EDM, K-pop, etc.)

### **Model Limitations**

1. **Black Box:** Difficult to interpret why specific classifications are made
2. **Single-Label:** Cannot handle genre-blending music (e.g., "jazz-fusion")
3. **Generalization:** Performance degrades on music styles not in training data
4. **Fixed Input:** Requires audio preprocessing to specific format

### **Ethical Considerations**

- Genre classifications may affect artist revenue on streaming platforms
- Cultural bias toward Western music raises fairness concerns
- Lack of explainability limits accountability for classification decisions

---



<div align="center">

**GenreNet - Teaching Machines to Hear Music** 🎵🤖


[⬆ Back to Top](#-genrenet)

</div>
