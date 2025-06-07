import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def preprocess_text(text):
    """
    Preprocesses text by tokenizing and removing stop words.
    
    Args:
        text (str): The text to preprocess.
    
    Returns:
        list: A list of tokens after removing stop words.
    """
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    return filtered_tokens

def load_and_analyze_data(file_path):
    """
    Loads data from a CSV file and performs descriptive analysis.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        None
    """
    # Load the data into a pandas DataFrame
    data = pd.read_csv(file_path)
    
    # Display the first few rows of the DataFrame
    print(f"\nAnalyzing file: {file_path}")
    print("First 5 rows of the dataset:")
    print(data.head())
    
    # Preprocess text columns (if any)
    for column in data.select_dtypes(include=['object']).columns:
        print(f"\nPreprocessing text in column: {column}")
        data[column] = data[column].astype(str).apply(preprocess_text)
        print(data[column].head())
    
    # Display basic information about the DataFrame
    print("\nDataset Info:")
    print(data.info())
    
    # Display summary statistics for numerical columns
    print("\nSummary Statistics:")
    print(data.describe())

# Example usage
if __name__ == "__main__":
    # List of CSV file paths
    file_paths = [
        "ml-32m/ml-32m/ratings.csv",
        "ml-32m/ml-32m/tags.csv",
        "ml-32m/ml-32m/movies.csv",
        "ml-32m/ml-32m/links.csv"
    ]
    
    for file_path in file_paths:
        try:
            load_and_analyze_data(file_path)
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
        except Exception as e:
            print(f"An error occurred while processing '{file_path}': {e}")