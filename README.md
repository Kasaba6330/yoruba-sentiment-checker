# Yoruba-sentiment-checker
Yorùbá Sentiment Checker
A hybrid (Rule-Based + Machine Learning) sentiment analysis tool for classifying Yorùbá text into Positive, Negative, or Neutral categories. This project addresses the challenge of building NLP tools for low-resource languages by combining the precision of a curated lexicon with the contextual understanding of a statistical model.

**Features**
Hybrid Architecture: Integrates a comprehensive, hand-curated Yorùbá sentiment lexicon with a Logistic Regression classifier for robust sentiment prediction.

**Web Application:** A user-friendly Streamlit web app for real-time sentiment analysis of text or uploaded files. https://yoruba-sentiment-checker.streamlit.app/

**Public Resources:** Provides a valuable public sentiment lexicon and sentiment analysis language model for Yorùbá to support further NLP research.

**Reproducible Research:** Complete code and methodology are provided for full transparency and reproducibility.

**Installation & Usage**
This demo assumes you are already familiar with python!
Using the wildcard ('*') imports the following, to import one or more specific function, you can just call from any of the below mentioned:
- vectorizer (This is the actual vectorizer)
- app_pred (This is the function that does the sentiment analysis. It takes a string!)
- sentiment_model (This is the trained model)
- preprocess_text (This is a preprocessing function for Yorùbá texts.)
- yoruba_stopwords (This is an iterable of stop words identified in the Yorùbá language)
- positive_words (This is an iterable of words with positive connotations in the Yorùbá language)
- negative_words (This is an iterable of words with negative connotations in the Yorùbá language)
- neutral_words (This is an iterable of words with neutral connotations in the Yorùbá language)

`pip install yorsent`

`from yorsent import *`
`text = 'Òru là ń ṣ'èkà, ẹni tí ó bá ṣe é lọ́sàn-án ò ní fi ara ire lọ.'`
`sent = app_pred(text)`
`print(sent)`



**Clone the repository:**

bash

git clone https://github.com/Kasaba6330/yoruba-sentiment-checker.git

cd yoruba-sentiment-checker

**Run the web application:**

bash

streamlit run app.py

**Data**

The model is trained on an extended dataset built upon the Yorùbá portion of the AfriSenti-SemEval dataset.

**Contributing**

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

**License**

This project is licensed under the Apache-2.0 License.
