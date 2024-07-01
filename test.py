# Import necessary libraries
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example reviews
reviews = [
    "I love this product! It works great and I use it every day.",
    "This is the worst purchase I've ever made. It broke after one use.",
    "It's okay, does the job but nothing special.",
    "Absolutely fantastic! Highly recommend to everyone.",
    "Not worth the money. Very disappointing."
]

# Step 1: Preprocess the Reviews
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

processed_reviews = [preprocess_text(review) for review in reviews]

# Step 2: Extract Keywords and Phrases
# Extract key phrases using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.85, stop_words='english')
X = vectorizer.fit_transform(processed_reviews)
features = vectorizer.get_feature_names_out()

# Convert to DataFrame for easy manipulation
df = pd.DataFrame(X.toarray(), columns=features)

# Step 3: Categorize and Weight
# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to('cuda')

def get_bert_embeddings(texts):
    all_embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# Get embeddings for key phrases
embeddings = get_bert_embeddings(features)

# Perform clustering to group similar phrases
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
labels = kmeans.labels_

# Create a DataFrame to map phrases to clusters
phrase_clusters = pd.DataFrame({'phrase': features, 'cluster': labels})

# Calculate frequency and sentiment scores for each cluster
phrase_counts = df.sum(axis=0)
cluster_summary = phrase_clusters.groupby('cluster')['phrase'].apply(list).reset_index()
cluster_summary['frequency'] = cluster_summary['phrase'].apply(lambda phrases: phrase_counts[phrases].sum())

# Step 4: Visualize the Results
# Plot the frequency of each cluster
plt.figure(figsize=(10, 6))
sns.barplot(x='cluster', y='frequency', data=cluster_summary)
plt.title('Frequency of Key Aspects in Customer Reviews')
plt.xlabel('Aspect Cluster')
plt.ylabel('Frequency')
plt.show()
# Print out phrases in each cluster
for cluster_id in cluster_summary['cluster']:
    print(f"Cluster {cluster_id}:")
    phrases = cluster_summary[cluster_summary['cluster'] == cluster_id]['phrase'].values[0]
    for phrase in phrases:
        print(f"  - {phrase}")
    print("\n")