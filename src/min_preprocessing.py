import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class FinancialTweetPreprocessor:
    def __init__(self, preserve_entities=True):
        self.tokenizer = TweetTokenizer(preserve_case=False,
                                       strip_handles=True,
                                       reduce_len=True)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Financial-specific terms to preserve
        self.financial_terms = {'bull', 'bear', 'long', 'short', 'call', 'put',
                               'bid', 'ask', 'ipo', 'eps', 'pe', 'rsi', 'macd'}

        # Remove common stopwords that might be important in finance
        self.stop_words -= {'up', 'down', 'above', 'below', 'under', 'over',
                           'before', 'after', 'against', 'between', 'into',
                           'through', 'during', 'until', 'while', 'about'}

    def clean_tweet(self, text):
        """Clean and preprocess a single tweet"""

        # Convert emojis to text descriptions
        text = emoji.demojize(text, delimiters=(" ", " "))

        # Get tickers
        tickers = re.findall(r'\$[A-Z]+', text)

        # Handle @mentions 
        mention_count = len(re.findall(r'@\w+', text))
        text = re.sub(r'@\w+', '@USER', text)

        # Handle URLs
        url_count = len(re.findall(r'http[s]?://\S+|www\.\S+', text))
        text = re.sub(r'http[s]?://\S+|www\.\S+', 'URL', text)

        # Normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        tokens = self.tokenizer.tokenize(text)

        # Remove stopwords but preserve financial terms
        tokens = [token for token in tokens
                 if token not in self.stop_words or token in self.financial_terms]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Add metadata features
        metadata = {
            'ticker_count': len(tickers),
            'mention_count': mention_count,
            'url_count': url_count,
            'token_count': len(tokens),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?')
        }

        return text, metadata

    def preprocess_dataset(self, df, text_column='text'):
        """Preprocess entire dataset"""
        processed_texts = []
        metadata_list = []

        for text in df[text_column]:
            cleaned_text, metadata = self.clean_tweet(text)
            processed_texts.append(cleaned_text)
            metadata_list.append(metadata)

        # Add processed text
        df['processed_text'] = processed_texts

        # Add metadata features
        metadata_df = pd.DataFrame(metadata_list)
        df = pd.concat([df, metadata_df], axis=1)

        return df