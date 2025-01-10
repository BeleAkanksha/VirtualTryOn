import pandas as pd

# Load your dataset
styles_df = pd.read_csv("styles.csv")

# Clean the 'masterCategory' column by stripping spaces and converting to lowercase
styles_df['masterCategory'] = styles_df['masterCategory'].str.strip().str.lower()

# Now, filter the dataset to only include rows where 'masterCategory' is 'apparel'
filtered_df = styles_df[styles_df['masterCategory'] == 'apparel']

# Save the filtered dataset to a new CSV file
filtered_df.to_csv("filtered_styles.csv", index=False)

# Print the number of rows removed (for reference)
removed_count = styles_df.shape[0] - filtered_df.shape[0]
print(f"Number of rows removed: {removed_count}")
