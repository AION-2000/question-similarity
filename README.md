# 🔍 Question Pair Similarity Classification

> An end-to-end machine learning solution for identifying semantically similar questions using Siamese LSTM networks

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This project tackles the challenge of classifying pairs of questions from Quora as either **duplicate** (semantically similar) or **not duplicate**. It's a complete, production-ready machine learning solution that demonstrates best practices from raw data analysis to a deployable predictive model.

**Why it matters:** Determining if two questions share the same intent is a core problem in Natural Language Understanding (NLU) with direct applications in search engines, forums, and customer support systems.

### ✨ Key Features

- 📊 Comprehensive data analysis and visualization
- 🧠 Siamese LSTM architecture for semantic comparison
- 🎯 **78.71%** accuracy with strong F1-Score performance
- 🚀 Production-ready prediction pipeline
- 📝 Detailed documentation and reproducible workflow

## 📁 Project Structure

```
question-similarity/
├── 📂 data/
│   ├── train.csv                    # Raw training data
│   └── 📂 processed/                # Tokenizer & preprocessing artifacts
├── 📂 models/
│   └── best_siamese_model.keras     # Trained model
├── 📂 notebooks/
│   └── final_submission.ipynb       # Complete analysis workflow
├── 📂 results/
│   └── submission.csv               # Prediction outputs
├── 📂 src/
│   └── predict.py                   # Standalone prediction script
├── 📄 performance_report.txt        # Detailed methodology & results
├── 📄 requirements.txt              # Python dependencies
└── 📄 README.md                     # You are here!
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/question-similarity.git
   cd question-similarity
   ```

2. **Create a virtual environment**
   ```bash
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Complete Analysis

Open and execute the main notebook to reproduce the entire workflow:

```bash
jupyter notebook notebooks/final_submission.ipynb
```

The notebook walks through:
- Data loading and exploration
- Text preprocessing and feature engineering
- Model training and evaluation
- Performance analysis with visualizations

### Making Predictions

Use the trained model on new data:

```bash
python src/predict.py
```

Input: Place your test data file in the `data/` directory  
Output: Predictions saved to `results/submission.csv`

## 🧪 Methodology

### 1. Exploratory Data Analysis
- Analyzed class distribution: **63% non-duplicates** vs **37% duplicates**
- Identified class imbalance, establishing **F1-Score** as the primary evaluation metric
- Explored question length distributions and common patterns

### 2. Text Preprocessing Pipeline
- Text normalization (lowercase, special character removal)
- Tokenization using NLTK
- Stopword removal and lemmatization
- Sequence padding to uniform length (25 tokens)

### 3. Model Architecture

**Siamese LSTM Network** — designed specifically for comparing two text sequences:

```
Input: Question 1        Input: Question 2
      ↓                        ↓
  Embedding (shared)      Embedding (shared)
      ↓                        ↓
   LSTM (shared)           LSTM (shared)
      ↓                        ↓
      └────── Concatenate ──────┘
                 ↓
            Dense Layers
                 ↓
          Binary Output
```

**Key Components:**
- Shared embedding layer for consistent word representations
- Shared LSTM layer to capture contextual information
- Concatenation layer to merge question representations
- Dense layers with dropout for final classification

**Training Details:**
- Optimizer: Adam
- Loss function: Binary cross-entropy
- Early stopping to prevent overfitting

### 4. Evaluation Strategy
Comprehensive evaluation on held-out validation set using multiple metrics to account for class imbalance.

## 📊 Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 78.71% |
| **Precision** | 71.06% |
| **Recall** | 71.45% |
| **F1-Score** | 71.25% |
| **AUC-ROC** | 86.13% |

**Highlights:**
- Strong AUC-ROC (86.13%) indicates excellent discriminative ability
- Balanced precision and recall despite class imbalance
- Confusion matrix and ROC curve available in the notebook

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.11 |
| **Data Processing** | Pandas, NumPy, NLTK |
| **ML Framework** | TensorFlow 2.x / Keras |
| **Evaluation** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |

## 📈 Future Improvements

- [ ] Experiment with transformer-based models (BERT, RoBERTa)
- [ ] Implement ensemble methods for improved accuracy
- [ ] Add real-time prediction API with Flask/FastAPI
- [ ] Expand to multi-language support
- [ ] Deploy as a web application

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the https://github.com/AION-2000
## 📬 Contact

**Questions or suggestions?** Let's connect!

- 🐙 GitHub: (https://github.com/your-username)](https://github.com/AION-2000)
- 📧 Email: aionshihabshahriar@gmail.com

---

<div align="center">
Made with ❤️ and Python | ⭐ Star this repo if you found it helpful!
</div>
