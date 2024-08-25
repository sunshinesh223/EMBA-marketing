import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import re
from transformers import pipeline, AutoTokenizer


# Define the URL for the local LLaMA server
url = "http://localhost:11434/api/chat"


# Function to call the LLaMA model via Ollama's API for a batch of reviews
def llama_batch(prompts):
    data = {
        "model": "llama3.1",
        "messages": [{"role": "user", "content": prompt} for prompt in prompts],
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_json = response.json()

        # Extract the content from the response and parse the scores
        if "message" in response_json and "content" in response_json["message"]:
            content = response_json["message"]["content"]

            # Use regex to extract relevance scores from the content
            scores = re.findall(r"Relevance score:\s*([\d.]+)", content)
            return [float(score) for score in scores]

        else:
            print("Unexpected response structure:", response_json)
            return [0.0] * len(prompts)  # Return a list of default scores if structure is unexpected

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return [0.0] * len(prompts)  # Return a list of default scores in case of a failure


# Load your real data
file_path = 'Data/combined_clustered_reviews.csv'  # Update with your actual file path
reviews_df = pd.read_csv(file_path)

# We will use the 'translated_review' column for analysis
reviews_df['Review'] = reviews_df['translated_review']

# Clean invalid reviews
invalid_reviews = reviews_df[reviews_df['Review'].isna() | ~reviews_df['Review'].apply(lambda x: isinstance(x, str))]

if not invalid_reviews.empty:
    print("Found invalid reviews. Deleting them...")
    reviews_df = reviews_df.drop(invalid_reviews.index)

reviews_df['Review'] = reviews_df['translated_review'].astype(str)

# Define the attributes for analysis
attributes = ['price comparison', 'service quality', 'cabin experience']


# Function to determine relevance using only LLaMA in batches
def determine_relevance_batch(reviews, attribute):
    prompts = [
        f"""
        I am analyzing customer reviews for cruise experiences. Please evaluate the relevance of the following review for the given attribute. The possible attributes are:

        Price: Discusses pricing, costs, or value for money.
        Service: Discusses the quality of service provided during the cruise.
        Cabins: Discusses the experience of staying in the cabins (comfort, cleanliness, amenities, etc.).

        Review:
        "{review}"

        Attribute: {attribute}

        Please return a relevance score from 0 to 1, where:

        0 means the review is not relevant to the attribute.
        anything in between reflects the confidence level you have with the relevance.
        1 means the review is highly relevant to the attribute.

        Only output a float with the relevance score, nothing else.
        """
        for review in reviews
    ]
    llama_responses = llama_batch(prompts)

    relevance_scores = []
    for response in llama_responses:
        try:
            # Check if the response is already a float
            if isinstance(response, float):
                relevance_score = response
            else:
                relevance_score = float(response.strip())  # Handle if it's a string that needs conversion
        except ValueError:
            relevance_score = 0.0  # Default to 0 if parsing fails
        relevance_scores.append(relevance_score)

    # Ensure the length matches the input
    if len(relevance_scores) != len(reviews):
        print(f"Warning: Mismatch in length. Expected {len(reviews)}, got {len(relevance_scores)}.")

    return relevance_scores


def analyze_sentiment_if_relevant_batch(reviews, attribute):
    relevance_scores = determine_relevance_batch(reviews, attribute)
    relevant_reviews = [
        review for review, score in zip(reviews, relevance_scores) if score > 0.5
    ]

    if relevant_reviews:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            tokenizer=AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment"),
            max_length=512,
            truncation=True,
            device=0
        )
        sentiment_results = sentiment_analyzer(relevant_reviews)

        sentiment_mapping = {
            "1 star": -1.0,
            "2 stars": -0.5,
            "3 stars": 0.0,
            "4 stars": 0.5,
            "5 stars": 1.0
        }

        sentiment_scores = []
        for sentiment_result in sentiment_results:
            sentiment_score = sentiment_mapping.get(sentiment_result['label'], 0.0)
            relevant_review_index = sentiment_results.index(sentiment_result)
            relevance_score = relevance_scores[relevant_review_index]
            sentiment_scores.append(sentiment_score * relevance_score)
    else:
        sentiment_scores = [None] * len(reviews)  # If no relevant reviews, return None for all

    return sentiment_scores


def process_attribute(attribute, reviews_df, batch_size=5):
    sentiment_scores = []
    total_batches = len(reviews_df) // batch_size + 1
    with tqdm(total=total_batches, desc=f"Processing {attribute.capitalize()}") as pbar:
        for i in range(0, len(reviews_df), batch_size):
            reviews_batch = reviews_df['Review'][i:i + batch_size]
            sentiment_scores.extend(analyze_sentiment_if_relevant_batch(reviews_batch, attribute))
            pbar.update(1)

    # Ensure the length matches the input
    if len(sentiment_scores) != len(reviews_df):
        print(f"Warning: Mismatch in length. Expected {len(reviews_df)}, got {len(sentiment_scores)}.")

    return sentiment_scores


def process_all_attributes():
    for attribute in attributes:
        reviews_df[attribute + '_sentiment'] = process_attribute(attribute, reviews_df)


# Run the processing
process_all_attributes()

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
