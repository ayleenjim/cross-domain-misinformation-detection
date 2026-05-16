# from google.colab import drive
# drive.mount('/content/drive')

import streamlit as st

st.set_page_config(page_title = 'Misinformation Detector')
placeholder = st.empty()

with placeholder.container():
    with st.container(vertical_alignment='center'):
        st.title(":red[Misinformation Detection Tool]")

        # using fabians lime function -----------------------------------------------------------------------
        def predict_proba_bert(texts, batch_size=16, max_length=128):
            all_probs = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                encodings = token(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )

                encodings = {k: v.to(device) for k, v in encodings.items()}

                with torch.no_grad():
                    outputs = bert_model(**encodings)
                    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

                all_probs.append(probs)

            return np.vstack(all_probs)

        def build_brief_explanation(exp, pred_label, top_k=3):
            if pred_label == 0:
                return "Flagged as REAL based on the overall wording and context."

            try:
                terms = exp.as_list(label=1)
                positive_terms = [term for term, weight in terms if weight > 0]
                key_terms = positive_terms[:top_k]

                if len(key_terms) == 0:
                    return "Flagged as FAKE based on the overall wording and context."

                joined = ", ".join(key_terms)
                return f"Flagged as FAKE because words like {joined} pushed the prediction toward fake."

            except Exception:
                return "Flagged as FAKE based on the overall wording and context."
#------------------------------------------------------------------------------------------------------

        with st.spinner("Loading models..."):
          @st.cache_resource(show_spinner= False)
          def load_models():
              #import standard tools
              import time
              import pandas as pd
              import joblib
              import nltk
              from transformers import AutoModelForSequenceClassification, AutoTokenizer #hugging face library

              time.sleep(1)
              bert_path = "models/bert_combined"
              model = AutoModelForSequenceClassification.from_pretrained(bert_path) #load weights
              tokenizer = AutoTokenizer.from_pretrained(bert_path)
              fusion_model = joblib.load("models/combined_fusion_logreg.joblib")
              return model, tokenizer, fusion_model

          with st.spinner("Loading models..."):
              bert_model, token, fusion_combined_model = load_models()

st.header(":red[Misinformation Detection Tool]", divider = "gray")

import os
import numpy as np
import pandas as pd
import torch
import streamlit.components.v1 as components

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lime.lime_text import LimeTextExplainer

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None


vader_scores = SentimentIntensityAnalyzer() #create vader object

model = bert_model
tokenizer = token
device = torch.device("cpu")

placeholder.empty()

# API helper
def get_secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)


def get_top_vader_tokens(text, top_n=3):
    words = text.split()
    scored_words = []

    for word in words:
        cleaned_word = word.lower().strip(".,!?;:\"'()[]{}")

        if cleaned_word in vader_scores.lexicon:
            vader_weight = vader_scores.lexicon[cleaned_word]
            scored_words.append((cleaned_word, vader_weight))

    scored_words = sorted(
        scored_words,
        key=lambda pair: abs(pair[1]),
        reverse=True
    )

    return scored_words[:top_n]


def generate_api_explanation(text, label, prob_real, prob_fake, top_bert_tokens, top_vader_tokens):
    if AzureOpenAI is None:
        return "API explanation unavailable because the OpenAI package is not installed."

    endpoint = get_secret("AZURE_OPENAI_ENDPOINT")
    api_key = get_secret("AZURE_OPENAI_API_KEY")
    api_version = get_secret("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    deployment = get_secret("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    if not endpoint or not api_key:
        return "API explanation unavailable because Azure/OpenAI credentials are missing."

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key
    )

    bert_token_text = ", ".join(
        [f"{token}: {weight:.4f}" for token, weight in top_bert_tokens]
    )

    vader_token_text = ", ".join(
        [f"{token}: {weight:.4f}" for token, weight in top_vader_tokens]
    )

    prompt = f"""
Text:
{text}

Prediction:
{label}

Probability Not Misinformation:
{prob_real:.6f}

Probability Might be Misinformation:
{prob_fake:.6f}

Top 3 BERT LIME tokens:
{bert_token_text}

Top 3 VADER sentiment tokens:
{vader_token_text}

Write 2-3 sentences explaining why the model gave this prediction.
Use the BERT LIME tokens and VADER sentiment tokens.
Do not claim the text is factually true or false.
Explain the model's reasoning and wording/rhetorical patterns.
"""

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You explain misinformation detection model outputs using concise rhetorical analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=200
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"API explanation error: {e}"

# User interface
# user input
Content_Text = st.text_area("Paste short text below to see if it is Misinformation", height=150)


#see if the length is too long ot text is empty
#the length is based on the number of tokens bert can handle
is_empty = len(Content_Text.strip()) == 0
token_count = len(token.encode(Content_Text))
limit = 128

#Check that text box meets requirements
if token_count > limit:
    st.error(f"Text is too long")
elif is_empty:
    st.error("Error: Please enter the text")

else:
    if st.button('Analyze'):
        with st.spinner('Analyzing text...'):
            if Content_Text:
                # get vader scores
                vs = vader_scores.polarity_scores(Content_Text)

                # handle bert
                # get bert_prob_fake to match fusion model
                inputs = token(Content_Text, return_tensors = "pt", truncation = True, padding =True, max_length = 128)
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim =1).cpu().numpy()
                bert_prob_fake = float(probs[0][1])

                # rhetoric features
                words = Content_Text.split()
                word_count = len(words)
                char_count = len(Content_Text)
                exclamation_count = Content_Text.count('!')
                question_count = Content_Text.count('?')
                all_caps_count = sum(1 for word in words if word.isupper() and len(word) > 1)

                #keywords from final_combined
                certainty_words = certainty_words = ["always", "never", "definitely", "proven", "guaranteed", "undeniable", "everyone", "nobody", "all", "none"]
                urgency_words = urgency_words = ["breaking", "urgent", "now", "immediately", "alert","warning", "before", "latest"]
                attack_words = attack_words = ["fake", "lie", "lies", "liar", "corrupt", "scam","fraud", "hoax", "attack", "exposed"]
                second_person =  ["you", "your", "yours"]

                certainty_count = sum(1 for w in words if w in certainty_words)
                urgency_count = sum(1 for w in words if w in urgency_words)
                attack_count = sum(1 for w in words if w in attack_words)
                second_person_count = sum(1 for w in words if w in second_person)

                #get the total
                total = [bert_prob_fake, vs['neg'],vs['neu'], vs['pos'], vs['compound'],
                        word_count, char_count, exclamation_count, question_count,
                        all_caps_count, certainty_count, urgency_count,
                        attack_count, second_person_count]

                #get prediction
                prediction = fusion_combined_model.predict([total])[0]
                probability =  fusion_combined_model.predict_proba([total])[0]
                score = probability[prediction]

                label = "Might be Misinformation" if prediction == 1 else "Not Misinformation"
                st.success(f"Prediction: {label}")
                st.info(f"How confident am I?: {score: .0%}")

                # Run BERT LIME explainer
                explainer = LimeTextExplainer(
                  class_names=["Not Misinformation", "Might be Misinformation"],
                  random_state=42
                  )

                exp = explainer.explain_instance(
                  Content_Text,
                  predict_proba_bert,
                  num_features=10,
                  num_samples=100,
                  labels=[int(prediction)]
                  )

                bert_token_weights = exp.as_list(label=int(prediction))
                top_bert_tokens = sorted(
                  bert_token_weights,
                  key=lambda pair: abs(pair[1]),
                  reverse=True
                )[:3]

                # Get top VADER words
                top_vader_tokens = get_top_vader_tokens(Content_Text, top_n=3)

                # Generate API explanation
                api_explanation = generate_api_explanation(
                  text=Content_Text,
                  label=label, # FIX: Changed 'abel' to 'label'
                  prob_real=float(probability[0]),
                  prob_fake=float(probability[1]),
                  top_bert_tokens=top_bert_tokens,
                  top_vader_tokens=top_vader_tokens
                )

                st.subheader("Explainability")
                st.write(api_explanation)

                st.subheader("Top BERT Tokens")
                bert_df = pd.DataFrame(
                  top_bert_tokens,
                  columns=["Token", "BERT LIME Weight"]
                )
                st.dataframe(bert_df, use_container_width=True)

                st.subheader("Top VADER Tokens")
                vader_df = pd.DataFrame(
                  top_vader_tokens,
                  columns=["Token", "VADER Sentiment Weight"]
                )
                st.dataframe(vader_df, use_container_width=True)

                st.subheader("Word analysis")
                components.html(exp.as_html(), height=400, scrolling=True)
            else:
                st.warning("Error: Please enter the text")
