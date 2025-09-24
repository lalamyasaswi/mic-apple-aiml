# IMDb Movie Review Sentiment Classification

## Dataset
- **Source:** Kaggle - IMDb Dataset  
- **Number of reviews:** 50,000  
- **Columns:** `review`, `sentiment`  
- **Class distribution:** 25,000 positive, 25,000 negative  
- **Balanced dataset** with no missing values  

## Preprocessing
- Converted all text to **lowercase**  
- Removed **HTML tags** (`<br />`), punctuation, and numbers  
- Removed extra spaces  
- Encoded sentiment labels: **positive → 1**, **negative → 0**  
- Calculated review lengths (chars & words) for exploration  

## Exploratory Analysis
- Average review length: ~231 words  
- Top 20 words (excluding stopwords): `br`, `movie`, `film`, `like`, `just`, `good`, `time`, `story`, `really`, `bad`, `people`, `great`, `don`, `make`, `way`, `movies`, `characters`, `think`, `watch`, `character`  
- Class distribution is **balanced**  
- Histogram shows most reviews have 100–300 words  

## Models Used
1. **Logistic Regression** (max_iter=200)  
2. **Linear SVM**  

## Evaluation Metrics

| Model                | Accuracy | Precision | Recall  | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.8945   | 0.8864    | 0.9050 | 0.8956   |
| Linear SVM           | 0.8872   | 0.8850    | 0.8900 | 0.8875   |

- Both models perform well on the test set (10,000 reviews).  
- Logistic Regression slightly outperforms Linear SVM in accuracy and F1-score.  

## Visualizations
- **Confusion Matrices** show most predictions are correct, with misclassifications evenly distributed across positive/negative reviews.  
- **Bar chart** compares Accuracy, Precision, Recall, F1-score for both models.  

## Observations
- TF-IDF vectorization with unigrams captures most sentiment-relevant words.  
- Both models are robust and fast for high-dimensional sparse data.  
- Logistic Regression slightly edges out Linear SVM in overall performance.  

## Potential Improvements
- Hyperparameter tuning (`C` for Logistic Regression/SVM)  
- Try **TF-IDF bigrams or trigrams** to capture multi-word expressions  
- Lemmatization or stemming to reduce word variations  
- Experiment with **deep learning models** (LSTM, BERT) for better context understanding  

## References
- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)  
- [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)  
- [Confusion Matrix Visualization](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)  

