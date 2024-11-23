## Files

### NLP Projects
This folder contains multiple NLP works from scratch. 
- **Project 1: Neural POS Tagging**  
  Developed a BiLSTM-based Part-of-Speech (POS) tagger for grammatical structure prediction. Optimized performance compared to traditional HMM-based models.

- **Project 2: English-German Translation**  
  Implemented two models for machine translation: N-gram and Seq2Seq LSTM with attention. The Seq2Seq model outperformed N-gram, especially for handling long-range dependencies.

- **Project 3: POS Tagging and Constituency Parsing with Transformers**  
  Implemented a Transformer-based model for Part-of-Speech (POS) tagging and constituency parsing. The POS tagging achieved a best validation accuracy of **96.1%**, with the model checkpoint saved as `tagging_model.pt`. For parsing, the model `parsing_model.pt` predicts labeled spans and uses the CKY algorithm to convert predictions into full parse trees, reaching a validation F1-score of **87.01%**.

- **NLP for Social Science Project:**
  - **Sentiment Classification:** Fine-tuned a BERT model for sentiment classification using domain-specific data, achieving superior accuracy compared to traditional LSTM and CNN models.
  - **Job Seniority Prediction:** Built a BERT-based classifier to predict job seniority levels from job descriptions with high precision.

### DeepAR M5 Retail Sales Forecasting
- **Project Overview:** PyTorch-based implementation of DeepAR for multi-store sales prediction. Training on store-category level data with 90-day encoder, 28-day prediction window. 
- **Model Features & Configuration:** LSTM architecture with 2 RNN layers and NormalDistributionLoss for probabilistic forecasting.
- **Data Source:** [The M5 Competition Dataset (2011-2016)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

**Tools Used:** TensorFlow, PyTorch, Hugging Face

### Random Forest.ipynb
- **Objective:** Predict flight disruptions (cancellations, diversions, delays) using flight data from 2018 to 2022 in the USA.
- **Methods:** Built a Random Forest model to predict flight disruption outcomes.
- **Data Source:** [Flight Delay Dataset (2018-2022)](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022)

### Crab Age.Rmd
- **Objective:** Predict crab age using dimensionality reduction and clustering techniques.
- **Methods:** Employed PCA, k-means clustering, and decision tree modeling to analyze the dataset.
- **Data Source:** [Crab Age Prediction Dataset](https://www.kaggle.com/datasets/sidhus/crab-age-prediction/data)

---



