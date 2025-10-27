import sqlite3
import string
import re
import nltk
import pandas as pd
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def preprocess(text):
    if isinstance(text, str):
            text = text.lower()
            # remove links
            url = rf"/(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})/gi"
            text = re.sub(url, '', text, flags=re.MULTILINE)
            matches = re.finditer(r'(\S+)\.([^/\s]+)', text) # find '.' and ignore part after '/'
            for m in matches:
                left = m.group(1)
                right = m.group(2)
                # assume if left side of the dot is > 3 and right side < 6, it's probably a link
                if len(left) > 3 and len(right) < 6 and not right.isnumeric() and not right[0].isupper():
                    idx = text.find(m.group(0))
                    if idx != -1:
                        nextidx = text.find(' ', idx)
                        if nextidx == -1:
                            nextidx = len(text)
                        text = text[:idx] + " " + text[nextidx:]
            # remove possessives
            text = re.sub(r"'s\b", '', text) 
            # remove punctuation
            text = re.sub(r'[^\w\s]', ' ', text)
            # remove short words
            text = re.sub(r'\b\w{1,2}\b', '', text)
            # stopword removal
            stop_words = stopwords.words('english')
            # custom stop words
            stop_words.extend(['get', 'like', 'would', 'also', 'one', 'know', 'think', 'time', 'see', 'could', 'make', 'even', 'really', 'going', 'want', 'need', 'way', 'new', 'good', 'much', 'still', 'take', 'lot', 'let', 'put', 'try', 'well', 'sure', 'thing', 'things', 'might', 'last', 'wow', 'hey', 'feel', 'new', 'keep', 'see'])
            processed = nltk.word_tokenize(text)
            processed = [word for word in processed if word not in stop_words]
            # lemmatization
            lemmatizer = WordNetLemmatizer()
            processed = [lemmatizer.lemmatize(word) for word in processed]   
            return processed
    else:
            return []

def main():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    nltk.download('omw-1.4')

    # connect to the SQLite database
    conn = sqlite3.connect('database.sqlite')
    cursor = conn.cursor()
    # read contents from the database
    posts = pd.read_sql_query("SELECT * FROM Posts", conn)
    comments = pd.read_sql_query("SELECT * FROM Comments", conn)
    # close the database connection
    conn.close()

    contents = pd.concat([posts['content'], comments['content']], ignore_index=True)
    # preprocess the contents
    processed = contents.apply(preprocess)

    # bag of words representation
    bow_list = []

    for tokens in processed:
            bow_list.append(tokens)

    dictionary = Dictionary(bow_list)
    dictionary.filter_extremes(no_below=2, no_above=0.3)
    corpus = [dictionary.doc2bow(tokens) for tokens in bow_list]

    optimal_coherence = -100
    optimal_lda = None
    optimal_k = 0

    coherence_values = []
    k_values = []

    for k in range(10, 50):
        lda_model = gensim.models.LdaModel(corpus=corpus,
                                        id2word=dictionary,
                                        num_topics=k,
                                        random_state=2,
                                        passes=10,
                                        alpha='auto',
                                        per_word_topics=True)

        coherence_model_lda = CoherenceModel(model=lda_model, texts=bow_list, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model_lda.get_coherence()
        coherence_values.append(coherence_score)
        k_values.append(k)

        if(coherence_score > optimal_coherence):
            print(f'Trained LDA with {k} topics. Average topic coherence (higher is better): {coherence_score} which is the best so far!')
            optimal_coherence = coherence_score
            optimal_lda = lda_model
            optimal_k = k
        else: 
            print(f'Trained LDA with {k} topics. Average topic coherence (higher is better): {coherence_score} which is not very good.')

    print(f'These are the words most representative of each of the {optimal_k} topics:')
    for i, topic in optimal_lda.print_topics(num_topics=50, num_words=5):
        print(f"Topic {i}: {topic}")

    # Count the dominant topic for each document
    topic_counts = [0] * optimal_k  # one counter per topic
    for bow in corpus:
        topic_dist = optimal_lda.get_document_topics(bow)  # list of (topic_id, probability)
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0] # find the top probability
        topic_counts[dominant_topic] += 1 # add 1 to the most probable topic's counter

    dominant_topics = []

    for tokens in bow_list:
        bow = dictionary.doc2bow(tokens)
        if len(bow) == 0:
            dominant_topics.append(-1)
        else:
            topic_dist = optimal_lda.get_document_topics(bow)
            dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
            dominant_topics.append(dominant_topic)

    topic_df = pd.DataFrame({
        "content": contents,
        "processed": processed,
        "dominant_topic": dominant_topics
    })
    topic_df['dominant_topic'] = topic_df['dominant_topic'].astype(int)

    # topic_df.to_csv("document_topics.csv", index=False)

    # Display the topic counts
    for i, count in enumerate(topic_counts):
        print(f"Topic {i}: {count} posts")
    
    # Display the top10 popular topics
    popular_topics = sorted(enumerate(topic_counts), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 popular topics:")
    for topic_id, count in popular_topics:
        print(f"Topic {topic_id}: {count} posts")

    # Visualization of Training
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, coherence_values, marker='o')
    plt.title('LDA Model Coherence vs Number of Topics')
    plt.xlabel('Number of Topics (k)')
    plt.ylabel('Coherence Score (c_v)')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

    # Visualization of topic distribution
    plt.figure(figsize=(25, 5))
    topic_pairs = sorted(enumerate(topic_counts), key=lambda x: x[1], reverse=True)
    topic_ids, sorted_counts = zip(*topic_pairs)
    bars = plt.bar(range(len(topic_ids)), sorted_counts, tick_label=topic_ids)
    plt.title('Document Distribution Across Topics')
    plt.xlabel('Topic')
    plt.ylabel('Number of Documents')
    plt.xticks(rotation=0)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{int(height)}", ha='center', va='bottom', fontsize=8)
    plt.show()

    # # Visualization with pyLDAvis
    lda_display = pyLDAvis.gensim_models.prepare(optimal_lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, 'lda_visualization.html')

    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    topic_df['sentiment'] = topic_df['content'].apply(lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0)
    topic_df['label'] = topic_df['sentiment'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))
    # overrall tone
    overall_tone = topic_df['label'].value_counts(normalize=True) * 100
    print("Overall Tone Distribution (%):")
    print(overall_tone)
    # tone per topic
    tone_per_topic = topic_df.groupby('dominant_topic')['label'].value_counts(normalize=True).unstack(fill_value=0) * 100
    print("Tone Distribution per Topic (%):")
    print(tone_per_topic)
    # average sentiment per topic
    avg_topic = topic_df.groupby('dominant_topic')['sentiment'].mean()
    print("Average Sentiment per Topic:")
    print(avg_topic)
    # Visualization different colors on sentiment polarity
    plt.figure(figsize=(20, 5))

    def get_color(val):
        if val > 0.05:
            return 'green'
        elif val < -0.05:
            return 'red'
        else:
            return 'blue'
    colors = [get_color(val) for val in avg_topic]
    bars = plt.bar(avg_topic.index.astype(int), avg_topic.values, color=colors)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                f"{height:.2f}", ha='center', va='bottom', fontsize=8)

    plt.title('Average Sentiment per Topic')
    plt.xlabel('Topic ID')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(avg_topic.index.astype(int))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()



if __name__ == "__main__":
        main()





