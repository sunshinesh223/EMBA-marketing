from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load pre-trained BERT model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Create a pipeline for sentiment analysis
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Define your customer reviews
reviews = [
    "I love this product! It works great and I use it every day.",
    "This is the worst purchase I've ever made. It broke after one use.",
    "It's okay, does the job but nothing special.",
    "Absolutely fantastic! Highly recommend to everyone.",
    "Not worth the money. Very disappointing."
]

# Analyze the sentiment of each review
for review in reviews:
    result = nlp(review)[0]
    print(f"Review: {review}\nSentiment: {result['label']}, Score: {result['score']:.2f}\n")