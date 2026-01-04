# Text Classification Model using NLP

##  Project Overview
This project implements a Text Classification Model using Natural Language Processing (NLP) and Machine Learning techniques.  
The model is trained to classify text into multiple categories such as news, spam, positive, and negative.

It demonstrates the complete NLP pipeline including  text preprocessing, TF-IDF feature extraction, model training, evaluation, and visualization.

---

## Objectives
- Convert raw text data into numerical features using TF-IDF
- Train a machine learning model for text classification
- Evaluate the model using precision, recall, and F1-score
- Visualize performance using a confusion matrix


##  Tech Stack
- Python
- Scikit-learn
- Pandas
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Matplotlib
- Seaborn

##  Methodology
1. Load dataset from CSV file  
2. Split data into training and testing sets  
3. Transform text data using TF-IDF Vectorizer  
4. Train a Naive Bayes classification model  
5. Evaluate performance using:
   - Precision
   - Recall
   - F1-score  
6. Plot Confusion Matrix for visualization  


##  How to Run the Project

###  Install dependencies
```bash
pip install -r requirements.txt
