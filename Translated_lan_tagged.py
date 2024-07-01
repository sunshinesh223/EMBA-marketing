import sys

import pandas as pd
import requests


api_key = ''  # Replace with your actual API key

#load all four review pkls and translate them, save the translated results to a new pkl
file_paths = ["./Data/AIDA_trustpilot_10411_reviews_raw.xlsx","./Data/colorline_trustpilot_177_reviews_raw.xlsx","./Data/DFDS_trustpilot_12437_reviews_raw.xlsx","./Data/Stena Line_trustpilot_558_reviews_raw.xlsx"]
translated_file_paths = ["./Data/AIDA_trustpilot_10411_reviews_translated.xlsx","./Data/colorline_trustpilot_177_reviews_translated.xlsx","./Data/DFDS_trustpilot_12437_reviews_translated.xlsx","./Data/Stena Line_trustpilot_558_reviews_translated.xlsx"]

# Load the reviews into DataFrames
dfs = [pd.read_excel(file_path) for file_path in file_paths]



# Print the columns in each DataFrame to ensure we have the correct review column
for i, df in enumerate(dfs):
    print(f"Columns in file {i+1}: {df.columns}")


def translate_text(text, target_language='en'):
    url = f"https://translation.googleapis.com/language/translate/v2"
    params = {
        'q': text,
        'target': target_language,
        'key': api_key
    }
    response = requests.get(url, params=params)
    result = response.json()
    translated_text = result['data']['translations'][0]['translatedText']
    detected_language = result['data']['translations'][0]['detectedSourceLanguage']
    return translated_text, detected_language

# Translate reviews to English and mark the original language for each DataFrame
for df in dfs:
    reviews = df['body_text'].tolist()  # Replace 'review_column_name' with the actual column name
    translated_reviews = []
    original_languages = []

    for review in reviews:
        translated_text, detected_language = translate_text(review)
        translated_reviews.append(translated_text)
        original_languages.append(detected_language.upper())

    df['translated_review'] = translated_reviews
    df['original_language'] = original_languages

# File paths for the translated pickle files


# Save the DataFrames with translated reviews and original languages
for df, translated_file_path in zip(dfs, translated_file_paths):
    df.to_pickle(translated_file_path)
    df.to_csv(translated_file_path.replace('.pkl', '.csv'), index=False)  # Optionally save as CSV

# Check the translated DataFrames
for i, df in enumerate(dfs):
    print(f"Sample data from translated file {i+1}:")
    print(df.head())