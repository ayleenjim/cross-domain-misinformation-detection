Cross-Domain Misinformation Detection

Senior capstone project exploring cross-domain misinformation detection using fusion-based Natural Language Processing (NLP) pipelines.

This project analyzes short-form text using sentiment, rhetorical, semantic, and lexical features to identify linguistic patterns commonly associated with misinformation. The system evaluates how misinformation detection models transfer across multiple domains, including political, health, and emotionally framed content.

Important:
This project does not perform direct fact-checking or verify factual truth.
Instead, it analyzes linguistic and rhetorical patterns associated with misinformation-style content.

⸻

Project Contributions

* Created a unified cross-domain misinformation dataset spanning political, health, and emotional domains
* Developed and evaluated TF-IDF, sentiment/rhetorical, semantic, and fusion-based NLP pipelines
* Built an interactive Streamlit web application for real-time misinformation analysis and explainability
* Performed cross-domain transfer analysis to evaluate how training-domain composition influenced model generalization

⸻

Tech Stack

* Python
* Streamlit
* scikit-learn
* Hugging Face Transformers
* PyTorch
* VADER Sentiment Analysis
* LIME Explainability
* pandas / NumPy

⸻

Repository Structure

cross-domain-misinformation-detection/
app/
    streamlit_app.py
models/
    combined_fusion_logreg.joblib
    bert_combined/   <-- downloaded separately
notebooks/
    01_preprocessing.ipynb
    02_tfidf_baseline.ipynb
    03_sentiment_rhetorical.ipynb
    04_fusion_pipeline.ipynb
    05_visualizations.ipynb
requirements.txt
run.sh
README.md

⸻

Running the Application

1. Clone the repository

git clone https://github.com/ayleenjim/cross-domain-misinformation-detection.git
cd cross-domain-misinformation-detection

⸻

2. Download the BERT model folder

The pretrained BERT model weights are not included in this repository due to GitHub file size limitations.

Download the bert_combined folder separately and place it inside:

models/bert_combined/

The folder should contain files such as:

config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json

⸻

3. Run the application

./run.sh

The script will automatically:

* Create a virtual environment (if needed)
* Install required dependencies (first run only)
* Launch the Streamlit application

⸻

Example Features Used

The fusion pipeline combines multiple feature types:

Sentiment Features

* Positive / negative polarity
* Compound sentiment score

Rhetorical Features

* ALL CAPS usage
* Exclamation/question frequency
* Urgency and certainty language
* Second-person pronouns

Semantic Features

* BERT-based semantic probabilities

Lexical Features

* TF-IDF representations

⸻

Cross-Domain Evaluation

Models were evaluated across multiple misinformation domains:

* Political fact-check statements (LIAR)
* COVID/health misinformation
* Emotional and sensational headline-style content

The project explored how:

* training-domain composition
* linguistic similarity
* domain-specific framing

influenced model transfer performance across domains.

⸻

Streamlit Application

The Streamlit interface allows users to:

* Enter short-form text
* Generate misinformation predictions
* View confidence scores
* Explore LIME explainability outputs
* Analyze influential sentiment and semantic features

⸻

Author

Ayleen Jimenez
University of Texas Rio Grande Valley
B.S. Computer Science – Senior Capstone Project
