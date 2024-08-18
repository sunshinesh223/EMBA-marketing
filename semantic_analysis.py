import pandas as pd
from transformers import pipeline, AutoTokenizer
import matplotlib.pyplot as plt
import itertools
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk


# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
# Load your real data
file_path = 'Data/combined_clustered_reviews.csv'  # Update with your actual file path
reviews_df = pd.read_csv(file_path)

# We will use the 'translated_review' column for analysis
reviews_df['Review'] = reviews_df['translated_review']

invalid_reviews = reviews_df[reviews_df['Review'].isna() | ~reviews_df['Review'].apply(lambda x: isinstance(x, str))]

# Display the problematic rows, if any
if not invalid_reviews.empty:
    print("Found invalid reviews. Deleting them...")
    # Remove the invalid reviews from the dataset
    reviews_df = reviews_df.drop(invalid_reviews.index)


reviews_df['Review'] = reviews_df['translated_review'].astype(str)

# Define the attributes for analysis
attribute_descriptions = {
    'price comparison': "This review discusses pricing comparisons between different options.",
    'service quality': "This review discusses the quality of service provided.",
    'cabin experience': "This review discusses the experience of staying in the cabins."
}

# Initialize the LLM pipeline for classification (replace with your model or API)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0,tokenizer=AutoTokenizer.from_pretrained("facebook/bart-large-mnli"))
#classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True,device=0)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    tokenizer=AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment"),
    max_length=512,
    truncation=True,
    device=0
)
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def truncate_review(review):
    inputs = tokenizer(review, max_length=512, truncation=True, return_tensors="pt")
    return tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

# Function to determine relevance using the LLM
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def determine_relevance(review, attribute):
    truncated_review = truncate_review(review)

    # Refined prompt with more context to help the model distinguish better
    prompt = "In the context of a cruise experience, how relevant is this review to {}?"

    # Get the relevance score for the attribute
    result = classifier(truncated_review, candidate_labels=[attribute], hypothesis_template=prompt)
    relevance_score = result['scores'][0]  # Extract the score for the relevant attribute
    return relevance_score

def analyze_sentiment_if_relevant(review, attribute):
    truncated_review = truncate_review(review)
    relevance = determine_relevance(truncated_review, attribute)
    if relevance > 0.5:  # Only perform sentiment analysis if relevance is non-zero
        sentiment_result = sentiment_analyzer(truncated_review)[0]
        sentiment_score = sentiment_result['label']  # Extract the sentiment label (e.g., "1 star", "5 stars")

        # Convert the sentiment label to a numerical score
        sentiment_mapping = {
            "1 star": -1.0,
            "2 stars": -0.5,
            "3 stars": 0.0,
            "4 stars": 0.5,
            "5 stars": 1.0
        }
        sentiment = sentiment_mapping.get(sentiment_score, 0.0)
        return sentiment * relevance  # Weight sentiment by relevance
    return None


# Apply relevance filtering and sentiment analysis
for attribute in attributes:
    sentiment_scores = []
    for review in reviews_df['Review']:
        sentiment = analyze_sentiment_if_relevant(review, attribute)
        sentiment_scores.append(sentiment)
    reviews_df[attribute + '_sentiment'] = sentiment_scores
# Extract numerical rating from 'rating_raw' and normalize it
reviews_df['Review_Score'] = reviews_df['rating_raw'].str.extract('(\d)').astype(float)
reviews_df['Review_Score'] = reviews_df['Review_Score'] / 5.0  # Normalize the rating to a scale of 0-1

# Combine sentiment score and review score to generate final scores
for attribute in attributes:
    attribute_scores = []
    for i in range(len(reviews_df)):
        sentiment_score = reviews_df[attribute + '_sentiment'].iloc[i]
        review_score = reviews_df['Review_Score'].iloc[i]
        if sentiment_score is not None:
            combined_score = (sentiment_score * 0.9) + (review_score * 0.1)
        else:
            combined_score = None
        attribute_scores.append(combined_score)
    reviews_df[attribute + '_score'] = attribute_scores

# Filter out rows with NaN scores (no sentiment found for any attributes)
reviews_df_filtered = reviews_df.dropna(subset=[attr + '_score' for attr in attributes])

# Group by source (company) and calculate average scores for each attribute
grouped_df = reviews_df_filtered.groupby('source')[[attr + '_score' for attr in attributes]].mean().reset_index()

# Generate pairwise combinations for plotting
attribute_combinations = list(itertools.combinations(attributes, 2))

# Plotting the combinations
fig, axs = plt.subplots(1, len(attribute_combinations), figsize=(18, 6))

for i, (attr1, attr2) in enumerate(attribute_combinations):
    for company in grouped_df['source'].unique():
        subset = grouped_df[grouped_df['source'] == company]
        axs[i].scatter(subset[attr1 + '_score'], subset[attr2 + '_score'], s=200, label=company)
        for _, row in subset.iterrows():
            axs[i].text(row[attr1 + '_score'] + 0.02, row[attr2 + '_score'] + 0.02, company, fontsize=10)
    axs[i].set_xlabel(attr1.capitalize(), fontsize=14)
    axs[i].set_ylabel(attr2.capitalize(), fontsize=14)
    axs[i].set_title(f'{attr1.capitalize()} vs {attr2.capitalize()}', fontsize=16)
    axs[i].grid(True)

plt.legend()
plt.tight_layout()
plt.show()
