import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class ItemRecommender:
    def __init__(self, styles_csv_path, images_folder_path):
        self.styles_csv_path = styles_csv_path
        self.images_folder_path = images_folder_path
        self.styles_df = pd.read_csv("updated_styles.csv")
        self.vectorizer = TfidfVectorizer()

    def recommend_similar_items(self, current_item_id):
        # Get the details of the current item
        current_item = self.styles_df[self.styles_df['id'] == current_item_id]
        if current_item.empty:
            return pd.DataFrame()  # Return an empty DataFrame if the item is not found

        current_item = current_item.iloc[0]

        # Filter the dataset based on the gender of the current item
        filtered_df = self.styles_df[self.styles_df['gender'] == current_item['gender']]

        # Combine relevant features into a single string for vectorization
        filtered_df['combined_features'] = filtered_df.apply(lambda row: f"{row['subCategory']} {row['articleType']} {row['baseColour']} {row['season']} {row['usage']}", axis=1)
        current_item_combined_features = f"{current_item['subCategory']} {current_item['articleType']} {current_item['baseColour']} {current_item['season']} {current_item['usage']}"

        # Vectorize the combined features
        tfidf_matrix = self.vectorizer.fit_transform(filtered_df['combined_features'])
        current_item_vector = self.vectorizer.transform([current_item_combined_features])

        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(current_item_vector, tfidf_matrix).flatten()

        # Get the indices of the most similar items
        similar_indices = cosine_similarities.argsort()[-6:-1][::-1]  # Get top 5 similar items

        # Return the most similar items
        return filtered_df.iloc[similar_indices]

# Example usage:
# recommender = ItemRecommender(styles_csv_path="updated_styles.csv", images_folder_path="output_images")