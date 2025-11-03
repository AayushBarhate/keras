# keras

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import collections
import nltk

# Import NLTK components
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# Import visualization tools
from wordcloud import WordCloud

# Import scikit-learn components
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import TensorFlow and Keras components
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Download NLTK Data ---
# (Run this once)
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
print("NLTK data downloaded.")

# --- Step 1: Load Data and Divide into Train-Validation Sets ---
print("\n--- Step 1: Loading and Splitting Data ---")
try:
    df = pd.read_csv('IMDB Dataset.csv')
except FileNotFoundError:
    print("Error: 'IMDB Dataset.csv' not found.")
    print("Please download it from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    exit()

# Map sentiment labels to integers
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Divide the data
X = df['review']
y = df['sentiment']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")


# --- Step 2: Perform Text Pre-processing ---
print("\n--- Step 2: Pre-processing Text ---")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Applies all preprocessing steps to a single text string.
    """
    text = text.lower() # Lowercase
    text = re.sub(r'<[^>]+>', '', text) # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    
    # Remove stop words
    text_tokens = text.split()
    text = ' '.join([word for word in text_tokens if word not in stop_words])
    
    # Optional: Spelling correction - VERY SLOW!
    # from textblob import TextBlob
    # text = str(TextBlob(text).correct())
    
    return text

# Apply cleaning
print("Cleaning training data (this may take a moment)...")
X_train_clean = X_train.apply(clean_text)
print("Cleaning validation data...")
X_val_clean = X_val.apply(clean_text)

print("\nExample of cleaned text:")
print("Original:", X_train.iloc[0][:200])
print("Cleaned:", X_train_clean.iloc[0][:200])


# --- Step 3: Perform Tokenization and Lemmatization ---
print("\n--- Step 3: Tokenizing and Lemmatizing ---")
lemmatizer = WordNetLemmatizer()

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas

# Apply
print("Tokenizing and lemmatizing training data (this may take a moment)...")
X_train_lemmatized = X_train_clean.apply(tokenize_and_lemmatize)
print("Tokenizing and lemmatizing validation data...")
X_val_lemmatized = X_val_clean.apply(tokenize_and_lemmatize)

print("\nExample of lemmatized tokens:")
print(X_train_lemmatized.iloc[0][:15])


# --- Step 4: Visualize Frequent Words and Bigrams ---
print("\n--- Step 4: Visualizing Data ---")

# --- 4a. Most Frequent Words (Word Cloud) ---
print("Generating most frequent words word cloud...")
all_words_text = ' '.join([' '.join(tokens) for tokens in X_train_lemmatized])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Training Data')
plt.show() # Close this window to continue

# --- 4b. Most Frequent Bigrams (Bar Chart) ---
print("Generating frequent bigrams bar chart...")
all_tokens = [token for sublist in X_train_lemmatized for token in sublist]
bigram_counts = collections.Counter(nltk.bigrams(all_tokens))
top_20_common_bigrams = bigram_counts.most_common(20)

df_bigrams = pd.DataFrame(top_20_common_bigrams, columns=['bigram', 'count'])
df_bigrams['bigram'] = df_bigrams['bigram'].apply(lambda x: ' '.join(x))

plt.figure(figsize=(12, 8))
plt.barh(df_bigrams['bigram'], df_bigrams['count'])
plt.title('Top 20 Most Common Bigrams')
plt.xlabel('Frequency')
plt.gca().invert_yaxis()
plt.show() # Close this window to continue

# --- 4c. Positive and Negative Words (Word Clouds) ---
df_train_cleaned = pd.DataFrame({'review': X_train_clean, 'sentiment': y_train})

positive_text = ' '.join(df_train_cleaned[df_train_cleaned['sentiment'] == 1]['review'])
negative_text = ' '.join(df_train_cleaned[df_train_cleaned['sentiment'] == 0]['review'])

print("Generating positive sentiment word cloud...")
positive_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Practical Words - Positive Sentiment')
plt.show() # Close this window to continue

print("Generating negative sentiment word cloud...")
negative_wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)
plt.figure(figsize=(10, 5))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Practical Words - Negative Sentiment')
plt.show() # Close this window to continue


# --- Step 5: Create Embedding Layer and Build Models ---
print("\n--- Step 5: Preparing Data and Building Models ---")

# --- 5a. Prepare Data for Keras Models ---
VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 100

# We use the cleaned (but not lemmatized) text for the Keras Tokenizer
X_train_final = X_train_clean
X_val_final = X_val_clean

# Tokenize
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_final)
word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens.")

# Convert to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train_final)
X_val_seq = tokenizer.texts_to_sequences(X_val_final)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

# Convert labels
y_train_final = np.array(y_train)
y_val_final = np.array(y_val)

print("Shape of padded training data:", X_train_pad.shape)
print("Shape of padded validation data:", X_val_pad.shape)

# --- 5b. Model 1: 3-Layer LSTM ---
print("\nBuilding 3-Layer LSTM model...")
model_lstm = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=True),
    Dropout(0.2),
    LSTM(16),
    Dense(1, activation='sigmoid')
])
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.summary()

# --- 5c. Model 2: 5-Layer Bidirectional RNN (using GRU) ---
# Note: This model is very deep and complex as requested.
print("\nBuilding 5-Layer Bidirectional GRU model...")
model_bi_gru = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
    Bidirectional(GRU(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(GRU(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(GRU(32, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(GRU(32, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(GRU(16)),
    Dense(1, activation='sigmoid')
])
model_bi_gru.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_bi_gru.summary()

# --- 5d. Train the 3-Layer LSTM Model ---
# We will train the simpler LSTM model.
print("\nTraining the 3-Layer LSTM model...")
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

history_lstm = model_lstm.fit(
    X_train_pad, y_train_final,
    epochs=10,
    batch_size=64,
    validation_data=(X_val_pad, y_val_final),
    callbacks=[early_stopping],
    verbose=1
)
print("Training finished.")


# --- Step 6: Test on Custom Dataset and Tabulate Accuracy ---
print("\n--- Step 6: Testing on Custom Dataset ---")

# 1. Build test dataset
my_test_reviews = [
    # Positive (5)
    "This was an absolutely fantastic movie. The acting was superb and the plot was gripping.", # 1
    "I loved every minute of it. A masterpiece of cinema.", # 1
    "Wonderful, heartwarming, and beautifully shot. I will watch it again.", # 1
    "The best film I've seen all year! Brilliant performances.", # 1
    "A truly uplifting and delightful experience.", # 1
    # Negative (5)
    "What a complete waste of time. The plot was non-existent and the acting was terrible.", # 0
    "I fell asleep halfway through. Incredibly boring.", # 0
    "This is probably the worst movie ever made. Avoid at all costs.", # 0
    "Horrible, just horrible. I want my money back.", # 0
    "I'm shocked this film even got made. It was a complete mess from start to finish." # 0
]
my_test_labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# 2. Preprocessing function for new data
def preprocess_new_reviews(reviews, tokenizer, max_length):
    cleaned_reviews = [clean_text(review) for review in reviews]
    sequences = tokenizer.texts_to_sequences(cleaned_reviews)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences

print("Preprocessing custom test dataset...")
X_custom_test = preprocess_new_reviews(my_test_reviews, tokenizer, MAX_LENGTH)

# 3. Predict
print("Predicting sentiment on custom test dataset...")
predictions = model_lstm.predict(X_custom_test)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# 4. Tabulate Accuracy
accuracy = accuracy_score(my_test_labels, predicted_classes)
print(f"\n--- Accuracy on Custom 10-Review Test Set ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\n--- Classification Report ---")
print(classification_report(my_test_labels, predicted_classes, target_names=['Negative', 'Positive']))

# 5. Create results table
results_df = pd.DataFrame({
    'Review': my_test_reviews,
    'Actual Sentiment': my_test_labels,
    'Predicted Sentiment': predicted_classes
})
results_df['Actual Sentiment'] = results_df['Actual Sentiment'].map({1: 'Positive', 0: 'Negative'})
results_df['Predicted Sentiment'] = results_df['Predicted Sentiment'].map({1: 'Positive', 0: 'Negative'})

print("\n--- Tabulated Results ---")
print(results_df)

print("\n--- SCRIPT COMPLETE ---")```
