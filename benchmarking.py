import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the combined data
combined_df = pd.read_csv('Data/combined_clustered_reviews.csv')
combined_df['translated_review'] = combined_df['translated_review'].astype(str).fillna('')


# Check for 'score' column and calculate sentiment if missing
if 'score' not in combined_df.columns or combined_df['score'].isnull().all():
    print("Calculating sentiment scores...")
    analyzer = SentimentIntensityAnalyzer()

    def calculate_sentiment(text):
        sentiment = analyzer.polarity_scores(text)
        return sentiment['compound']

    combined_df['score'] = combined_df['translated_review'].apply(calculate_sentiment)

# Calculate average sentiment for each company by topic
company_scores = combined_df.groupby(['source', 'cluster'])['score'].mean().reset_index()
company_scores.columns = ['Company', 'Topic', 'Average Sentiment']

# Calculate industry average sentiment by topic
industry_scores = combined_df.groupby('cluster')['score'].mean().reset_index()
industry_scores.columns = ['Topic', 'Industry Average Sentiment']

# Merge company scores with industry scores for comparison
comparison_df = pd.merge(company_scores, industry_scores, on='Topic', how='left')
comparison_df.to_csv('Data/company_vs_industry_scores.csv', index=False)

# Debug print to check the comparison DataFrame
print("Comparison DataFrame:")
print(comparison_df)

# Plot the comparison of average sentiment for each company against industry average
plt.figure(figsize=(12, 8))

# Plot for each company
companies = comparison_df['Company'].unique()
for company in companies:
    company_data = comparison_df[comparison_df['Company'] == company]
    plt.plot(company_data['Topic'], company_data['Average Sentiment'], marker='o', label=company)

# Plot industry average
plt.plot(comparison_df['Topic'].unique(), comparison_df.groupby('Topic')['Industry Average Sentiment'].mean(), marker='x', linestyle='--', color='k', label='Industry Average')

plt.title('Average Sentiment by Topic: Company vs Industry')
plt.xlabel('Topic')
plt.ylabel('Average Sentiment')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Save the plot
plt.savefig('average_sentiment_comparison.png')
