import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import KMeans
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from collections import defaultdict

# Define the source files and their labels
files = {
    'AIDA': 'Data/AIDA_trustpilot_10411_reviews_translated.csv',
    'ColorLine': 'Data/colorline_trustpilot_177_reviews_translated.csv',
    'DFDS': 'Data/DFDS_trustpilot_12437_reviews_translated.csv',
    'StenaLine': 'Data/Stena Line_trustpilot_558_reviews_translated.csv'
}

# Load and concatenate all CSV files, adding a 'source' column to each
dfs = []
for source, file in files.items():
    df = pd.read_csv(file)
    df['source'] = source  # Add source column
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)


# Preprocess the translated reviews
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


combined_df['translated_review'] = combined_df['translated_review'].apply(preprocess_text)

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vectorize the translated reviews using BERT
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name).to(device)
review_embeddings = model.encode(combined_df['translated_review'].tolist(), show_progress_bar=True, device=device)

# Define the number of clusters
num_clusters = 4

# Fit KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(review_embeddings)
labels = kmeans.labels_

# Add cluster labels to the dataframe
combined_df['cluster'] = labels
combined_df.to_csv('Data/combined_clustered_reviews.csv', index=False)

# Debug print to check the combined DataFrame
print(combined_df.head())
print(combined_df['cluster'].value_counts())

# Create a directory to store the cluster files
os.makedirs('Data/Clusters', exist_ok=True)

# Save reviews in each cluster to separate files
for cluster in range(num_clusters):
    cluster_reviews = combined_df[combined_df['cluster'] == cluster]
    file_path = f'Data/Clusters/cluster_{cluster}.txt'
    with open(file_path, 'w', encoding='utf-8') as f:
        for review in cluster_reviews['translated_review'].tolist():
            f.write(review + '\n')

# Check for 'score' column and calculate sentiment if missing
if 'score' not in combined_df.columns or combined_df['score'].isnull().all():
    print("Calculating sentiment scores...")
    analyzer = SentimentIntensityAnalyzer()


    def calculate_sentiment(text):
        sentiment = analyzer.polarity_scores(text)
        return sentiment['compound']


    combined_df['score'] = combined_df['translated_review'].apply(calculate_sentiment)

# Debug print to check the updated DataFrame with sentiment scores
print(combined_df.head())

# Calculate average sentiments for each cluster
aspect_scores = []
for cluster in range(num_clusters):
    cluster_reviews = combined_df[combined_df['cluster'] == cluster]
    if not cluster_reviews.empty:
        avg_sentiment = cluster_reviews['score'].mean()
        aspect_scores.append((cluster, avg_sentiment))
    else:
        print(f"Warning: No reviews found for cluster {cluster}")

aspect_scores_df = pd.DataFrame(aspect_scores, columns=['Aspect', 'Average Sentiment'])
aspect_scores_df.to_csv('Data/aspect_scores.csv', index=False)

# Ensure 'Aspect' column in aspect_scores_df is int
aspect_scores_df['Aspect'] = aspect_scores_df['Aspect'].astype(int)

# Debug print to check the aspect scores DataFrame
print(aspect_scores_df)

# Create a dictionary to store summaries
aspect_summary = {}

# Manually input summaries for each cluster using triple quotes for multi-line strings
for cluster in range(num_clusters):
    if cluster == 0:
        summary = """
Punctuality and adherence to schedules
Handling of disabled passengers and their needs
Treatment and criminalization of customers
Customer service responsiveness and helpfulness
Cleanliness and condition of facilities and cabins
Overall customer experience with booking and refunds
Transparency and fairness in pricing and booking policies
Quality of food and drink options
Entertainment options and availability
Staff professionalism and friendliness
Handling of complaints and issues during travel
Internet and onboard amenities quality
Safety and security measures
        """
    elif cluster == 1:
        summary = """
Suitability for families with children
Length and quality of excursions
Crowd management and accommodation of different passenger needs
Entertainment and activity options for children
Availability and variety of food and drinks
Efficiency of arrival and departure processes
Customer service and issue resolution on board
Transparency in pricing, especially for drink packages
Cleanliness and hygiene measures, particularly during health crises
Overall passenger satisfaction with staff and service
        """
    elif cluster == 2:
        summary = """
Variety and quality of food and drinks
Entertainment options and their quality
Cleanliness and comfort of cabins and common areas
Staff friendliness and professionalism
Overall organization and logistics of the trips
Customer service efficiency and helpfulness
Handling of health and safety concerns
Value for money and pricing transparency
Onboard amenities and their condition
Internet connectivity and quality
Experience of special events and occasions on board
Flexibility and responsiveness to individual passenger needs
        """
    elif cluster == 3:
        summary = """
Friendly and helpful staff
Cleanliness and maintenance of facilities
Organization and smooth operations
Variety and quality of food and entertainment
Customer service responsiveness and helpfulness
Overall passenger experience and satisfaction
Handling of special requests and individual needs
Value for money and pricing transparency
Accessibility and support for disabled passengers
Safety and security measures
Handling of complaints and issues
Internet connectivity and onboard amenities quality
Flexibility in booking and cancellation processes
        """
    aspect_summary[cluster] = summary.strip()

# Save the summary to a JSON file for reference
summaries_file_path = 'Data/Clusters/summaries.json'
with open(summaries_file_path, 'w', encoding='utf-8') as f:
    json.dump(aspect_summary, f)


# Load summaries from a JSON file
def load_summaries(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
    return summaries


# Load the summaries back into the program
aspect_summary = load_summaries(summaries_file_path)

# Display the loaded summaries
for cluster, summary in aspect_summary.items():
    print(f"Cluster {cluster} Summary: {summary}")

bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
kw_model = KeyBERT()
def extract_keywords(summary, num_keywords=10):
    keywords = kw_model.extract_keywords(summary, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=num_keywords)
    return [kw[0] for kw in keywords]
# Automatically categorize clusters based on the summaries using LDA
def generate_categories(aspect_summary):
    summaries = list(aspect_summary.values())

    # Generate BERT embeddings for summaries
    summary_embeddings = bert_model.encode(summaries)

    # Vectorize the aspect summaries using BERT
    categories = defaultdict(list)
    topic_keywords_dict = {}

    for cluster, summary in aspect_summary.items():
        cluster_embedding = bert_model.encode([summary])

        # Compute cosine similarity between cluster summary and all summaries
        similarities = cosine_similarity(cluster_embedding, summary_embeddings).flatten()

        # Find the index of the most similar summary
        most_similar_index = similarities.argmax()

        topic_keywords = extract_keywords(summaries[most_similar_index])
        topic_keywords_dict[most_similar_index] = topic_keywords
        print(f"Matching cluster {cluster} with summary '{summary.strip()}' to topic keywords {topic_keywords}")

        categories[f"Topic {most_similar_index}"].append(cluster)
    return categories


# Generate categories from summaries
categories = generate_categories(aspect_summary)
print(categories)

# Plot the average sentiment for each generated category
# Create a DataFrame for plotting
plot_data = []
for category, clusters in categories.items():
    for cluster in clusters:
        # Handle cases where the cluster might not be found in aspect_scores_df
        matching_rows = aspect_scores_df[aspect_scores_df['Aspect'] == int(cluster)]
        if not matching_rows.empty:
            score = matching_rows['Average Sentiment'].values[0]
            plot_data.append((category, score))
        else:
            print(f"Warning: Cluster {cluster} not found in aspect_scores_df")

plot_df = pd.DataFrame(plot_data, columns=['Category', 'Average Sentiment'])

# Visualize
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Average Sentiment', data=plot_df)
plt.title('Average Sentiment by Category')
plt.xlabel('Category')
plt.ylabel('Average Sentiment')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Save the plot
plt.savefig('/mnt/data/average_sentiment_by_category.png')
