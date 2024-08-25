from datetime import datetime
import os
import numpy as np
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
def llama_batch(prompts, max_retries=2):
    data = {
        "model": "llama3.1",
        "messages": [{"role": "user", "content": prompt} for prompt in prompts],
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}


    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raise an exception for HTTP errors
    response_json = response.json()

    # Extract the content from the response and parse the scores
    if "message" in response_json and "content" in response_json["message"]:
        content = response_json["message"]["content"]
        scores = re.findall(r"Review \d+:\s*([\d.]+)", content)


        return [float(score) for score in scores]


    # If all retries fail, return a list of default scores


# Load your real data
file_path = 'Data/combined_clustered_reviews.csv'  # Update with your actual file path
reviews_df = pd.read_csv(file_path)
'''
# Set the desired number of reviews per company
desired_reviews_per_company = 10000 // reviews_df['source'].nunique()

# Sample reviews from each company, ensuring equal representation
reviews_df_balanced = reviews_df.groupby('source', group_keys=False).apply(
    lambda x: x.sample(min(len(x), desired_reviews_per_company), random_state=42)
)
# If the total sample size is less than 500 due to companies with fewer reviews, you can adjust accordingly:
reviews_df_balanced = reviews_df_balanced.sample(n=min(len(reviews_df_balanced), 10000), random_state=42).reset_index(drop=True)

reviews_df=reviews_df_balanced

'''
# We will use the 'translated_review' column for analysis
reviews_df['Review'] = reviews_df['translated_review']

# Clean invalid reviews
invalid_reviews = reviews_df[reviews_df['Review'].isna() | ~reviews_df['Review'].apply(lambda x: isinstance(x, str))]

if not invalid_reviews.empty:
    print("Found invalid reviews. Deleting them...")
    reviews_df = reviews_df.drop(invalid_reviews.index)

reviews_df['Review'] = reviews_df['translated_review'].astype(str)

# Define the attributes for analysis
attributes = ['cabin experience', 'Dining experience']


# Function to determine relevance using only LLaMA in batches
# Function to determine relevance using LLaMA in batches with combined prompts
def determine_relevance_batch(reviews, attribute, batch_size=3):
    # Group reviews into batches
    combined_prompts = []
    for i in range(0, len(reviews), batch_size):
        reviews_batch = reviews[i:i + batch_size]
        combined_review_text = "\n".join([f"Review {j + 1}: \"{review}\"" for j, review in enumerate(reviews_batch)])

        prompt = f"""
        I am analyzing customer reviews for cruise experiences. Please evaluate the relevance of each of the following reviews for the given attribute. The possible attributes are:

        Price: Discusses pricing, costs, or value for money.
        Service: Discusses the quality of service provided during the cruise.
        Cabins: Discusses the experience of staying in the cabins (comfort, cleanliness, amenities, etc.).

        Reviews:
        {combined_review_text}

        Attribute: {attribute}
        
        i need to you to dertermin the relevant score for each review.
        Please return a relevance score from 0 to 1, where:

        0 means the review is not relevant to the attribute.
        anything in between reflects the confidence level you have with the relevance.
        1 means the review is highly relevant to the attribute.
        
        Please return the relevance scores for each review in this format:
        Relevance Scores:
        Review 1: [score]
        Review 2: [score]
        Review 3: [score]
        and so on...

        Only output the relevance scores in the exact format requested.
        """
        combined_prompts.append(prompt)

    # Call LLaMA API for the combined prompts
    relevance_scores = []
    for combined_prompt in combined_prompts:
        llama_responses = llama_batch([combined_prompt])

        if llama_responses:
            # Extract relevance scores using regex with the updated structure
            relevance_scores.extend([float(score) for score in llama_responses])

    # Ensure the length matches the input
    if len(relevance_scores) != len(reviews):
        print(f"Warning: Mismatch in length. Expected {len(reviews)}, got {len(relevance_scores)}.")

    return relevance_scores


sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    tokenizer=AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment"),
    max_length=512,
    truncation=True,
    device=0
)

def analyze_sentiment_if_relevant_batch(reviews_df, attribute,sentiment_analyzer=sentiment_analyzer):
    relevance_scores = determine_relevance_batch(reviews_df['Review'].tolist(), attribute)

    # Update relevance scores directly in the DataFrame
    try:
        reviews_df[f'{attribute}_relevance'] = relevance_scores
    except:
        reviews_df[f'{attribute}_relevance'] = np.nan
        relevance_scores = determine_relevance_batch(reviews_df['Review'].tolist(), attribute)
        reviews_df[f'{attribute}_relevance'] = relevance_scores
        reviews_df[f'{attribute}_relevance'] = np.nan

    # Filter only relevant reviews (those with relevance score >= 0.5)
    relevant_reviews_df = reviews_df[reviews_df[f'{attribute}_relevance'] >= 0.5]

    if not relevant_reviews_df.empty:

        sentiment_results = sentiment_analyzer(relevant_reviews_df['Review'].tolist())

        sentiment_mapping = {
            "1 star": -1.0,
            "2 stars": -0.5,
            "3 stars": 0.0,
            "4 stars": 0.5,
            "5 stars": 1.0
        }

        # Update the sentiment scores directly in the DataFrame using the original indices
        for idx, sentiment_result in zip(relevant_reviews_df.index, sentiment_results):
            sentiment_score = sentiment_mapping.get(sentiment_result['label'], 0.0)
            relevance_score = reviews_df.at[idx, f'{attribute}_relevance']
            reviews_df.at[idx, f'{attribute}_sentiment'] = sentiment_score * relevance_score

    else:
        # If no relevant reviews, fill with NaN for the sentiment column
        reviews_df[f'{attribute}_sentiment'] = np.nan

    return reviews_df[f'{attribute}_sentiment'].tolist()


def process_attribute(attribute, reviews_df, batch_size=3):
    if attribute + '_sentiment' in reviews_df.columns:
        sentiment_scores = reviews_df[attribute + '_sentiment'].tolist()
    else:
        total_batches = len(reviews_df) // batch_size + 1
        with tqdm(total=total_batches, desc=f"Processing {attribute.capitalize()}") as pbar:
            for i in range(0, len(reviews_df), batch_size):
                reviews_batch_df = reviews_df.iloc[i:i + batch_size]
                batch_scores = analyze_sentiment_if_relevant_batch(reviews_batch_df, attribute)
                reviews_df.loc[i:i + batch_size - 1,
                attribute + '_sentiment'] = batch_scores  # Update the DataFrame with batch results
                pbar.update(1)
                reviews_df.to_csv(file_path, index=False)  # Save the file after each batch

    return reviews_df[attribute + '_sentiment'].tolist()


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
#reviews_df_filtered = reviews_df.dropna(subset=[attr + '_score' for attr in attributes])

# Group by source (company) and calculate average scores for each attribute
grouped_df = reviews_df.groupby('source')[[attr + '_score' for attr in attributes]].mean().reset_index()

# Generate pairwise combinations for plotting
if not os.path.exists('plottings'):
    os.makedirs('plottings')

# Generate pairwise combinations for plotting
attribute_combinations = list(itertools.combinations(attributes, 2))

# Create a separate plot for each attribute combination
for attr1, attr2 in attribute_combinations:
    fig, ax = plt.subplots(figsize=(10, 8))
    for company in grouped_df['source'].unique():
        subset = grouped_df[grouped_df['source'] == company]
        x = subset[f'{attr1}_score'].values[0]
        y = subset[f'{attr2}_score'].values[0]
        ax.scatter(x, y, s=200, alpha=0.6, label=company)
        ax.annotate(company, xy=(x, y), xytext=(x-0.1,y+0.1))
    ax.set_xlabel(attr1.capitalize(), fontsize=12)
    ax.set_ylabel(attr2.capitalize(), fontsize=12)
    ax.set_title(f'{attr1.capitalize()} vs {attr2.capitalize()}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_yticks(np.arange(-1, 1.1, 0.5))
    plt.tight_layout()
    plt.show()


    # Generate filename with attributes and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plottings/{attr1}_{attr2}_{timestamp}.png"

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
