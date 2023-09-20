import streamlit as st
from tweeterpy import TweeterPy
import datetime
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from textblob import TextBlob
import logging
from collections import Counter
from collections import defaultdict
from gensim import corpora, models
import emoji
import pandas as pd
import plotly.express as px
from pyvis.network import Network
import holoviews as hv
hv.extension('plotly')
import plotly.graph_objects as go
import text2emotion as te
import yfinance as yf
import random
from wordcloud import STOPWORDS
from sklearn.manifold import TSNE
import numpy as np

# Streamlit App Configuration
st.set_page_config(
    page_title="Major Player Analysis Console",
    page_icon="🚀",
    layout="wide"
)

# Add this near the top of your script (after initializing Streamlit, but before any other Streamlit commands)
if 'word_of_interest' not in st.session_state:
    st.session_state.word_of_interest = ""

# Add this below
if 'fetch_info_pressed' not in st.session_state:
    st.session_state.fetch_info_pressed = False    

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("watchdog.observers.inotify_buffer").disabled = True
logger = logging.getLogger('tweeterpy')
for handler in logger.handlers[:]:
    if isinstance(handler, logging.FileHandler):
        logger.removeHandler(handler)


# Initialize TweeterPy
twitter = TweeterPy()

def user_association_graph(username, hashtags, mentions, common_words):
    # Initialize graph
    G = nx.Graph()
    
    # Add the main user to the graph with a fixed size
    G.add_node(username, size=30, color='black')
    
    # Extract frequencies for each category and add them to the graph
    hashtag_counts = Counter(hashtags)
    mention_counts = Counter(mentions)
    word_counts = Counter(common_words)

    # Add hashtags to graph
    for hashtag, count in hashtag_counts.items():
        G.add_node(hashtag, size=10 + count*5, color='purple')
        G.add_edge(username, hashtag, weight=count)

    # Add mentions to graph
    for mention, count in mention_counts.items():
        G.add_node(mention, size=10 + count*5, color='yellow')
        G.add_edge(username, mention, weight=count)
    
    # Add common words to graph
    for word, count in word_counts.items():
        G.add_node(word, size=10 + count*5, color='red')
        G.add_edge(username, word, weight=count)

    return G

def association_graph_plotly(G):
    pos = nx.spring_layout(G, dim=3, seed=42)  # 3D layout

    edge_trace = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='lines',
        line=dict(width=2, color='#888'),
    )

    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        edge_trace['z'] += tuple([z0, z1, None])

    node_trace = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        text=[],
        mode='markers+text',
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=[],
            color=[],
            opacity=0.8,
            sizemode='diameter'
        )
    )

    for node in G.nodes():
        x, y, z = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['z'] += tuple([z])
        node_trace['text'] += tuple([node])
        node_trace['marker']['color'] += tuple([G.nodes[node]['color']])
        node_trace['marker']['size'] += tuple([G.nodes[node]['size']])

    layout = go.Layout(
        title="Association Graph",
        scene=dict(
            xaxis=dict(nticks=4, range=[-1, 1]),
            yaxis=dict(nticks=4, range=[-1, 1]),
            zaxis=dict(nticks=4, range=[-1, 1]),
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    return fig

def mention_relationship_graph(tweets, main_username):
    # Initialize graph
    G = nx.Graph()
    
    # Extract mentions from tweets
    mentions = []
    for tweet in tweets:
        text = tweet['content']['itemContent']['tweet_results']['result']['legacy'].get('full_text', '')
        mentions.extend(re.findall(r'@(\w+)', text))
    
    # Count mentions
    mention_counts = Counter(mentions)

    # Add the main user to the graph
    G.add_node(main_username, size=30, color='blue')
    
    # Add mentions and edges to the graph
    for mention, count in mention_counts.items():
        G.add_node(mention, size=10 + count*5, color='purple' if count > 5 else 'red')
        G.add_edge(main_username, mention, weight=count)
    
    return G
   

# Define a domain-specific lexicon for AMC and meme stocks
domain_specific_lexicon = {
    "amc": 1.5,
    "moass": 2.0,
    "ape": 2.0,
    "moon": 1.5,
    "rocket": 1.5,
    "buy": 1.2,
    "hold": 1.2,
    "short": -1.5,
    "shill": -2.0,
    "crash": -2.0,
    "distort": -1.2,
    "amy": -2.0
}

def compute_domain_specific_sentiment(text, lexicon):
    tokens = tokenize(preprocess_text(text))
    score = sum(lexicon.get(token, 0) for token in tokens)
    return score / (len(tokens) + 1e-7)  # Add a small value to avoid division by zero

def enhanced_sentiment_visualization(enhanced_sentiments):
    # Creating the violin plot
    fig = go.Figure()

    # Violin plot
    fig.add_trace(go.Violin(y=enhanced_sentiments, box_visible=True, line_color='blue',
                             meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,
                             y0="Sentiments"))

    # Histogram
    fig.add_trace(go.Histogram(y=enhanced_sentiments, opacity=0.75))

    # Updating layout for better clarity and adding titles
    fig.update_layout(title="Enhanced Sentiment Analysis Visualization",
                      yaxis_title="Sentiment Value",
                      xaxis_title="Frequency",
                      template="plotly_white",
                      showlegend=False)
    
    # Displaying the figure
    st.plotly_chart(fig)

# Additional Preprocessing
def additional_preprocessing(text):
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Consider boosting certain keywords by repeating them to give them more weight
    boost_keywords = ['amc', 'apes', 'moass', 'short']
    for keyword in boost_keywords:
        if keyword in text:
            text += (' ' + keyword) * 5  # Boost by repeating keyword
    return text    

# Preprocessing and tokenization functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = additional_preprocessing(text)  # Add the additional preprocessing step
    return text

stopwords_set = set(stopwords.words('english'))

def tokenize(text):
    tokens = word_tokenize(text)
    return [token for token in tokens if token not in stopwords_set]    

def extract_emojis(s):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  
        "]+"
    )
    return emoji_pattern.findall(s)

# Extract hashtags and mentions from text
def extract_hashtags_mentions(text):
    hashtags = re.findall(r"#(\w+)", text)
    mentions = re.findall(r"@(\w+)", text)
    return hashtags, mentions  

def extract_cashtags(text):
    return re.findall(r"\$(\w+)", text)      

# Extract bigrams from tokens
def extract_bigrams(tokens):
    bigrams = list(nltk.bigrams(tokens))
    return Counter(bigrams)

def extract_ngrams(tokens, n=2):
    ngrams = list(nltk.ngrams(tokens, n))
    return Counter(ngrams)    

# Extract and count emojis
def extract_most_common_emojis(tweets, top_n=10):
    all_emojis = []
    for tweet in tweets:
        text = tweet['content']['itemContent']['tweet_results']['result']['legacy'].get('full_text', '')
        all_emojis.extend(extract_emojis(text))
    return Counter(all_emojis).most_common(top_n)

# Ego weights definition
ego_weights = {
    # Self-focused
    "i": 1, "me": 1, "my": 1.5, "mine": 1.5, "myself": 2, 
    "i'm": 1.2, "i've": 1.2, "i'd": 1.2, "i'll": 1.2, "me!": 1.3, 
    "yo": 1.3, "yours truly": 1.8, 

    # Ownership or possession
    "my thing": 1.5, "my bad": 1.1, "my way": 1.5, "my turf": 1.6,

    # Group belonging but self-focused
    "we": 0.5, "us": 0.5, "our": 0.5, "ours": 0.5, "ourselves": 1,
    "we're": 0.6, "we've": 0.6, "we'd": 0.6, "we'll": 0.6,

    # Self-aggrandizement
    "legend": 2, "king": 2.5, "queen": 2.5, "boss": 2, "hero": 2,
    "master": 2.3, "god": 2.5, "goddess": 2.5, "star": 2, 
    "number one": 2.2, "best ever": 2.3, "on top": 2, 
    "leading": 2.1, "prime": 2.2, "champion": 2.3, 
    "victor": 2.3, "mvp": 2.4, "genius": 2.5, "prodigy": 2.5, 
    "maestro": 2.3, "expert": 2.1, "ace": 2, "whiz": 2.2, 
    "top dog": 2.3, "big shot": 2.4,

    # Self-importance or superiority
    "better than": 2.2, "superior": 2.3, "unbeatable": 2.4, 
    "peerless": 2.5, "paramount": 2.5, "preeminent": 2.5, 
    "supreme": 2.6, "unrivaled": 2.5, "matchless": 2.5, 
    "incomparable": 2.6,

    # Vanity
    "sexy": 2.1, "hottest": 2.2, "irresistible": 2.3, "stunning": 2.2, 
    "dazzling": 2.2, "ravishing": 2.3, "alluring": 2.2, "magnetic": 2.2, 
    "hypnotic": 2.3, "mesmerizing": 2.3, "captivating": 2.3, "gorgeous": 2.3, 
    "divine": 2.4, "flawless": 2.4, "perfect": 2.5,

    # AMC and Meme Stocks related weights
    "hodl": 2, "to the moon": 2.5, "diamond hands": 2.2, "rocket": 2.3, 
    "moonshot": 2.4, "tendies": 2.1, "squeeze": 2.3, "paper hands": 1.8,
    "apes": 1.7, "not financial advice": 1.5, "buy the dip": 2.3
}

def calculate_ego_score(tokens):
    if not tokens:
        return 0
    total_count = len(tokens)
    ego_count = sum(ego_weights.get(word, 0) for word in tokens)
    return ego_count / total_count

def ego_classification(score):
    if score > 70:
        return "High Ego"
    elif 30 <= score <= 70:
        return "Medium Ego"
    else:
        return "Low Ego"

def generate_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Ego Score"},
        gauge={
            "axis": {"range": [None, 100]},
            "steps": [
                {"range": [0, 30], "color": "green"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"}
            ],
            "bar": {"color": "blue"}
        }
    ))
    st.plotly_chart(fig)  # Display the gauge chart directly within the function

def generate_word_cloud(word_freq):
    if not word_freq:
        st.warning("No words provided for the word cloud!")
        return

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    
    st.pyplot(fig)  # Pass the figure to Streamlit's st.pyplot()

# Initializing ego_word_freq at the beginning
ego_word_freq = defaultdict(int)

def analyze_ego_centricity(tweets):
    all_tokens = []
    ego_scores = []
    
    for tweet in tweets:
        text = tweet['content']['itemContent']['tweet_results']['result']['legacy'].get('full_text', '')
        tokens = tokenize(preprocess_text(text))  # Assuming you have a tokenize and preprocess_text function
        all_tokens.extend(tokens)
        score = calculate_ego_score(tokens)
        ego_scores.append(score)
        
        for token in tokens:
            if token in ego_weights:
                ego_word_freq[token] += 1 

    # Calculate ego-related metrics
    ego_result = ego_classification(np.mean(ego_scores))
    ego_score = np.mean(ego_scores)
    ego_words = dict(Counter(ego_word_freq).most_common())  # Use Counter for word frequency
    

    return ego_result, ego_score, ego_words

def get_ego_word_frequencies(text, ego_words):
    # Tokenize the text
    words = text.split()
    
    # Get frequencies of each word
    word_counts = Counter(words)
    
    # Filter out words that are not ego words
    ego_word_freq = {word: count for word, count in word_counts.items() if word in ego_words}

    return ego_word_freq    

# Function to calculate toxic weights
# Toxic weights definition
toxic_weights = {
    # Aggression
    "attack": 2, "destroy": 2, "hate": 2, "retaliate": 2, "vengeance": 2, 
    "violent": 2.5, "threaten": 2, "kill": 2.5, "harm": 2,

    # Belittlement
    "uneducated": 1.5, "idiot": 2, "stupid": 2, "dumb": 2, "worthless": 2,
    "useless": 1.5, "pathetic": 2, "inferior": 2,

    # Misinformation
    "fake news": 2.5, "hoax": 2, "fraud": 2, "lie": 2, "manipulation": 2,
    "deception": 2, "misleading": 2, "misinformation": 2.5, "propaganda": 2,
    "disinformation": 2.5, "false flag": 2, "conspiracy": 2,

    # Discrimination
    "racist": 2.5, "bigot": 2.5, "sexist": 2.5, "homophobe": 2.5, "xenophobe": 2.5,
    "prejudiced": 2.5,

    # Others
    "leftist": 1, "manipluation": 2, "ignored": 1.5, "victims": 1.5, "silenced": 1.5,
    "maliciously": 1.5, "shame": 2, "defame": 2, "discredit": 2, "lies": 2
} 

def calculate_toxic_score(tokens):
    score = sum([toxic_weights.get(token, 0) for token in tokens])
    return score / len(tokens) if tokens else 0

def get_tweet_emotions(tweet):
    return te.get_emotion(tweet) 

@st.cache_data
def get_amc_price():
    stock = yf.Ticker("AMC")
    return stock.history(period="1d")['Close'][0]  
               

st.title("👜 Baggie Archiver 👜")

if 'fetch_info_pressed' not in st.session_state:
    st.session_state.fetch_info_pressed = False

if 'prev_slider_values' not in st.session_state:
    st.session_state.prev_slider_values = (1, 10)

usernames = st.text_input("Yo fam, drop the @ of your mark (no '@' needed, we chillin'):")
usernames = [username.strip() for username in usernames.split(",")]
num_tweets = st.slider("How many tweets we scrapin' today? Don't be mid af MAX it out for the best experience no cap...", min_value=1, max_value=100, value=5, step=1)
max_edge_width = st.slider("Max Edge Width for the Network Graph - Make it thicc or slim, shady...", min_value=1, max_value=20, value=5, step=1)

if st.button("Fetch the deets! 🚀"):
    st.session_state.fetch_info_pressed = True

if st.session_state.fetch_info_pressed:
    for username in usernames:  # Loop through each username
        if not username:
            st.warning("Bruh, drop a name, don't leave me hanging!")
            continue  # Skip to the next iteration if the username is empty

        # Resetting the ego_word_freq dictionary for each user
        
        ego_word_freq.clear()    

        try:
            user_id_from_tweeterpy = twitter.get_user_id(username)
            tweets_data = twitter.get_user_tweets(user_id_from_tweeterpy, total=num_tweets)
            tweets = tweets_data['data']

            all_tokens = []
            all_emojis = []
            all_hashtags = []
            all_mentions = []
            all_cashtags = []
            all_unigrams = []
            all_bigrams = []
            all_trigrams = []
            total_ego_score = 0
            most_common_tokens = []
            most_common_emojis = []
            most_common_hashtags = []
            most_common_mentions = []
            most_common_cashtags = []
            most_common_bigrams = []
            all_tokens = []
            all_dates = []

            if not tweets:
                st.warning(f"Bro, @{username} is ghosting us. No tweets found!")
            else:
                for tweet in tweets:
                    text = tweet['content']['itemContent']['tweet_results']['result']['legacy'].get('full_text', '')
                    tokens = tokenize(text)
                    all_tokens.extend(tokens)  # Collect tokens from all tweets
                    ego_score = calculate_ego_score(tokens)
                    total_ego_score += ego_score  # accumulate the ego scores

                st.markdown("---")
                with st.expander("Word Frequency Over Time"):
                    st.write("Wanna know how often our mark's been vibin' with a word? This graph's like, 'Yo, here's how much they couldn't shut up about it over time.' 📈")

                    # Collect the dates of the tweets
                    date = datetime.datetime.strptime(tweet['content']['itemContent']['tweet_results']['result']['legacy']['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                    all_dates.append(date)

                # Word Frequency Over Time
                word_of_interest = st.text_input("Got a word that's sus? Track it over time:", value=st.session_state.word_of_interest)
                track_button = st.button("Track it like it's hot 🎵")

                if track_button:
                    if word_of_interest:
                        st.session_state.word_of_interest = word_of_interest

                        # Count occurrences on specific dates
                        occurrences = {date: 0 for date in set(all_dates)}
                        for date, token in zip(all_dates, all_tokens):
                            if token == word_of_interest:
                                occurrences[date] += 1

                    # Convert dictionary to dataframe for visualization
                    df_occurrences = pd.DataFrame(list(occurrences.items()), columns=['date', 'count'])
                    fig = px.line(df_occurrences, x="date", y="count", title=f"Frequency of '{word_of_interest}' Over Time")
                    st.plotly_chart(fig)

                # Inserting custom CSS to make the profile image round
                st.markdown("""
                    <style>
                        .rounded-profile {
                            border-radius: 50%;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(f"## 🕵️‍♂️ OpSec insights for: @{username} 🕵️‍♂️")

                # Call the function to get AMC price and store it in amc_price variable
                amc_price = get_amc_price()         
                
                # Display User Information as a Profile Card
                user_data = tweets[0]['content']['itemContent']['tweet_results']['result']['core']['user_results']['result']['legacy']
                created_at = datetime.datetime.strptime(user_data['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                
                st.markdown("---")
                with st.expander("Baggie Data"):
                    st.write("Yo, peep this stash! It's all the 411 we scooped up on our mark. Hope it ain't too sus. 🕵️‍♂️")

                st.markdown(f"## Shill id #:{user_id_from_tweeterpy} ")

                # Display banner image
                if 'profile_banner_url' in user_data:
                    st.image(user_data['profile_banner_url'], use_column_width=True)

                # Layout for profile image and user details
                col1, col2, col3, col4 = st.columns([1, 3, 3, 1])

                with col1:
                    # Display profile image with rounded corners using custom HTML
                    st.components.v1.html(f"<img src='{user_data.get('profile_image_url_https', 'N/A')}' class='rounded-profile' style='width: 100%; height: auto;'/>", height=110)

                with col2:
                    st.markdown(f"### **[{username}](https://twitter.com/{username})**")
                    st.write(f"**Username**: {username}")
                    st.write(f"**Secret ID**: {user_id_from_tweeterpy}")
                    st.write(f"**Account Created**: {created_at.strftime('%a %b %d %H:%M:%S %Y')}")
                    st.write(f"**Bio**: {user_data['description']}")

                with col3:
                    st.write(f"**Location**: {user_data['location']}")
                    st.write(f"**Pays for X!**: {'Yes' if user_data['verified'] else 'No'}")
                    st.write(f"**Followers**: {user_data['followers_count']:,}")
                    st.write(f"**Following**: {user_data['friends_count']:,}")
                    st.write(f"**Total Favorites**: {user_data['favourites_count']:,}")
                    st.write(f"**Total Tweets**: {user_data['statuses_count']:,}")
                    st.write(f"**Listed In**: {user_data['listed_count']:,}")

                # You can add some stylings, icons, or colors to highlight this data if needed
                with col4:
                    st.write(f"**MOASS Ticket Price**: ${amc_price:.2f}")                    

                st.markdown("---")

                st.write(user_data)


                # Token Extraction and Visualization
                all_tokens = []
                for tweet in tweets:
                    text = tweet['content']['itemContent']['tweet_results']['result']['legacy'].get('full_text', '')
                    processed_text = preprocess_text(text)
                    tokens = tokenize(processed_text)
                    all_tokens.extend(tokens)                   

                most_common_tokens = Counter(all_tokens).most_common(25)

                # Ensure that most_common_tokens are correctly initialized and populated
                if not most_common_tokens:
                    st.error("Error: most_common_tokens is empty!")
                    continue

                # Hashtag, Mention, and Cashtag Analysis
                all_hashtags, all_mentions, all_cashtags = [], [], []
                all_emojis = []
                all_bigrams = []
                for tweet in tweets:
                    text = tweet['content']['itemContent']['tweet_results']['result']['legacy'].get('full_text', '')
                    hashtags, mentions = extract_hashtags_mentions(text)
                    cashtags = extract_cashtags(text)
                    all_hashtags.extend(hashtags)
                    all_mentions.extend(mentions)
                    all_cashtags.extend(cashtags)
                    all_emojis.extend(extract_emojis(text))
                    tokens = tokenize(preprocess_text(text))
                    all_unigrams.extend(extract_ngrams(tokens, 1))
                    all_bigrams.extend(extract_ngrams(tokens, 2))
                    all_trigrams.extend(extract_ngrams(tokens, 3))
                    total_ego_score += calculate_ego_score(tokens)

                all_unique_hashtags = set(all_hashtags)  # Convert to set to get unique hashtags
                all_unique_mentions = set(all_mentions)    
        
                most_common_emojis = extract_most_common_emojis(tweets)
                most_common_hashtags = Counter(all_hashtags).most_common(30)
                most_common_mentions = Counter(all_mentions).most_common(30)
                most_common_cashtags = Counter(all_cashtags).most_common(20)
                most_common_unigrams = Counter(all_unigrams).most_common(25)
                most_common_bigrams = Counter(all_bigrams).most_common(25)
                most_common_trigrams = Counter(all_trigrams).most_common(25)
                
                # Ensure that most_common_tokens are correctly initialized and populated
                if not most_common_tokens:
                    st.error("Error: most_common_tokens is empty!")
                    continue

                # Ego-centricity Analysis
                with st.expander("Ego Alert 🚨"):
                    st.write("Yo, ever met that one dude who's like, 'Me, myself, and I'? This score's gonna spill the tea on how much they're vibin' with themselves. Too much self-love? 🤣")    

                ego_scores = [calculate_ego_score(tokenize(tweet['content']['itemContent']['tweet_results']['result']['legacy'].get('full_text', ''))) for tweet in tweets]
                average_ego_score = sum(ego_scores) / len(tweets) if tweets else 0
                absolute_ego_score = sum(1 for score in ego_scores if score > 0.2) / len(tweets) if tweets else 0
                
                ego_words = dict(Counter(ego_word_freq).most_common())  # Use Counter for word frequency

                # Run the analysis
                ego_result, ego_score, ego_words = analyze_ego_centricity(tweets)

                # Check if there are ego-centric words, then display the word cloud
                if ego_words:
                    generate_word_cloud(ego_words)

                # Display ego words with better formatting
                st.markdown(f"**Ego Words for {username}:**")
                for word, count in ego_words.items():
                    st.markdown(f"- **{word}**: {count}")    
                
                # Display ego score with enhanced formatting
                st.markdown(f"**{username}'s Ego Score:** {ego_score:.2%}")
                st.markdown(f"*Vibe:* {('High-key self-love vibes 💅' if ego_score > 0.3 else 'Pretty chill 🍹')}")
                
                fig_ego = px.bar(x=[username], y=[ego_score], labels={'y':'Ego Score (%)'}, title=f"Ego-centricity Score for {username}")
                st.plotly_chart(fig_ego)

                # Visualization
                fig = generate_gauge_chart(ego_score * 100)
                

                # Toxic Speech Identification
                with st.expander("Toxic Speech Identification Details"):
                    st.write("Ever heard someone spit straight fire but in the bad way? This section breaks down if our mark's been serving some toxic vibes on Twitter. 🤢")

                toxic_scores = [calculate_toxic_score(tokenize(tweet['content']['itemContent']['tweet_results']['result']['legacy'].get('full_text', ''))) for tweet in tweets]
                average_toxic_score = sum(toxic_scores) / len(tweets) if tweets else 0

                st.write(f"Counts of toxic words for {username}: They been naughty or nice? 🎅")
                toxic_word_counts = {word: all_tokens.count(word) for word in toxic_weights.keys()}
                toxic_words_list, toxic_counts = zip(*toxic_word_counts.items())
                fig_toxic = px.bar(x=toxic_words_list, y=toxic_counts, title=f"Counts of Toxic Words for {username}")
                st.plotly_chart(fig_toxic)  

                # Word Cloud
                with st.expander("Word Kloud"):
                    st.write("""
                        Ayy, peep this! It's the lingo our mark's been throwin' around the most on Twitter. Bigger the word, more they're obsessed with it. Wonder if 'I' is huge? 🤣
                    """)

                # Define words to exclude
                exclude_words = ["amp", "then", "there"]  # Replace with words you want to exclude
                stopwords = set(STOPWORDS)
                stopwords.update(exclude_words)
                    
                processed_texts = [preprocess_text(tweet['content']['itemContent']['tweet_results']['result']['legacy'].get('full_text', '')) for tweet in tweets]
                combined_processed_text = ' '.join(processed_texts)

                word_counter = Counter(tokenize(combined_processed_text))
                min_freq, max_freq = st.slider("Select range of word frequencies for word cloud:", min_value=1, max_value=max(word_counter.values()), value=(1,10))

                filtered_words = [word for word, count in word_counter.items() if min_freq <= count <= max_freq]
                filtered_text = ' '.join(filtered_words)

                wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(filtered_text)
    
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt.gcf())

                # Sentiment Analysis
                enhanced_sentiments = []
                for tweet in tweets:
                    text = tweet['content']['itemContent']['tweet_results']['result']['legacy'].get('full_text', '')
                    base_sentiment = TextBlob(text).sentiment.polarity
                    domain_sentiment = compute_domain_specific_sentiment(text, domain_specific_lexicon)
                    combined_sentiment = base_sentiment + domain_sentiment  # Combine the general and domain-specific sentiments
                    enhanced_sentiments.append(combined_sentiment)

                enhanced_sentiment_visualization(enhanced_sentiments)
                
                with st.expander("Enhanced Sentiment Analysis"):
                    st.write("""
                        Check out this spicy violin plot! It's got the vibe check on the tweets, mixin' general feels with some AMC and meme stock sauce. Fat parts? Lots of tweets with that vibe. Skinny parts? Meh, not so much.
                    """)    
                st.write(f"Average Enhanced Sentiment: {sum(enhanced_sentiments)/len(enhanced_sentiments):.2f}")

                # Create a violin plot for enhanced sentiments
                fig_enhanced_sentiment = px.violin(enhanced_sentiments, box=True, points="all", title="Distribution of Enhanced Tweet Sentiments")
                # Add mean sentiment line
                mean_enhanced_sentiment = sum(enhanced_sentiments) / len(enhanced_sentiments)
                fig_enhanced_sentiment.add_shape(
                    type="line", line=dict(dash='dash'),
                    x0=0, x1=1, y0=mean_enhanced_sentiment, y1=mean_enhanced_sentiment
                )
                
                # Annotations to explain sentiment scores
                fig_enhanced_sentiment.add_annotation(text="Positive Sentiment", x=0.6, y=0.8, arrowhead=4, showarrow=True, font=dict(color="green"))
                fig_enhanced_sentiment.add_annotation(text="Neutral Sentiment", x=0.6, y=0.1, arrowhead=4, showarrow=True, font=dict(color="gray"))
                fig_enhanced_sentiment.add_annotation(text="Negative Sentiment", x=0.6, y=-0.8, arrowhead=4, showarrow=True, font=dict(color="red"))

                # Modify x-axis title
                fig_enhanced_sentiment.update_xaxes(title_text="Density")

                # Modify y-axis title
                fig_enhanced_sentiment.update_yaxes(title_text="Sentiment Score")

                # Show the figure
                st.plotly_chart(fig_enhanced_sentiment)

                sentiments_classified = ["positive" if s > 0.1 else "negative" if s < -0.1 else "neutral" for s in enhanced_sentiments]
                sentiment_counts = Counter(sentiments_classified)
                fig_enhanced_sentiment_pie = px.pie(names=sentiment_counts.keys(), values=sentiment_counts.values(), title="Sentiment Distribution")
                st.plotly_chart(fig_enhanced_sentiment_pie)

                # Tracked Words Analysis
                tracked_words = ["gasparino", "adam aron", "apes", "ken griffin", "doug cifu", "cover", "shill", "shills", "moass", "ftd", "squeeze", "darkpool", "short and distort", "off exchange", "flash crash", "naked shorts", "sec", "grifter", "retail investors", "gary gensler", "citadel", "virtu"]
                word_occurrences = {word: 0 for word in tracked_words}
                for tweet in tweets:
                    tweet_text = tweet['content']['itemContent']['tweet_results']['result']['legacy'].get('full_text', '').lower()
                    for word in tracked_words:
                        word_occurrences[word] += tweet_text.count(word)        

                with st.expander("Tracked Word Occurrences"):
                    st.write("""
                        Got some words you're sus about? Let's see how many times our mark's been droppin' them.
                    """)
                fig_tracked_words = px.bar(
                    x=list(word_occurrences.keys()),
                    y=list(word_occurrences.values()),
                    labels={'x': 'Tracked Words', 'y': 'Occurrences'},
                    title="Occurrences of Tracked Words"
                )
                st.plotly_chart(fig_tracked_words)                    

                # Temporal Analysis using Plotly
                tweet_dates = [datetime.datetime.strptime(tweet['content']['itemContent']['tweet_results']['result']['legacy']['created_at'], '%a %b %d %H:%M:%S +0000 %Y') for tweet in tweets]
                with st.expander("🕒 Timey Wimey Stuff"):
                    st.write("""
                        We're about to pull a Doctor Who up in here! Check out how our mark's tweet vibes change through time. It's like time travel, but you're just scrolling.
                    """)    
                fig_dates = px.line(x=tweet_dates, title="Tweet Frequencies Over Time")
                st.plotly_chart(fig_dates)

                # Mention Analysis using Plotly
                with st.expander("🗣️ Who's Getting the Shoutouts?"):
                    st.write("These are the homies our mark's been hollerin' at. Who's the main squeeze? 🤔")

                mentions, counts = zip(*most_common_mentions)
                fig_mentions = px.bar(x=list(mentions), y=list(counts), orientation='h', title="Top 5 Most Mentioned Users")
                st.plotly_chart(fig_mentions)                

                # Hashtag Analysis using Plotly
                with st.expander("Hashtag Analysis"):
                    st.write("""
                        Hashtags are like the secret sauce of Twitter. Let's see which ones our mark's been drowning their tweets in. #Overkill or #OnPoint? 🤔
                    """)
                if most_common_hashtags:
                    hashtags, counts = zip(*most_common_hashtags)
                    df_hashtags = pd.DataFrame({
                        'hashtags': hashtags,
                        'counts': counts
                    }) 
                    fig_hashtags_bubble = px.scatter(df_hashtags, x='hashtags', y='counts', size='counts', hover_data=['hashtags', 'counts'], title="Top 10 Most Common Hashtags")
                    st.plotly_chart(fig_hashtags_bubble)
                else:
                    st.write("No hashtags found in the dataset.")

                # Cashtag Analysis using Plotly
                with st.expander("Cashtags Analysis"):
                    st.write("Dolla dolla bills, y'all! 🤑 See which stocks our mark's blabbering about. Maybe they got some hot stock tips?")
    
                if most_common_cashtags:
                    cashtags, counts = zip(*most_common_cashtags)
                else:
                    cashtags, counts = [], []

                # Create a DataFrame from the cashtags and counts
                df_cashtags = pd.DataFrame({
                    'Cashtags': cashtags,
                    'Counts': counts
                })

                # Now, use column references for x and y in px.bar
                fig_cashtags = px.bar(df_cashtags, x='Counts', y='Cashtags', orientation='h', title="Top 5 Most Common Cashtags")
                st.plotly_chart(fig_cashtags)

                # Extract all bigrams from the tokens collected from all tweets
                bigram_counts = extract_bigrams(all_tokens)

                # Get the top 10 most common bigrams (you can adjust this number as needed)
                most_common_bigrams = bigram_counts.most_common(25)

                # Extract unique words from these top 10 bigrams for the heatmap matrix
                common_bigram_words = list(set(word for bigram in most_common_bigrams for word in bigram[0]))

                # Construct the heatmap matrix
                heatmap_matrix = [[bigram_counts.get((w1, w2), 0) for w2 in common_bigram_words] for w1 in common_bigram_words]

                # N-gram Analysis Heatmap with Plotly
                with st.expander("N-gram Analysis"):
                    st.write("Peek at the dynamic duos! These are the word pairs our mark can't help but use together. PB&J of words, ya know?")
    
                heatmap = go.Figure(data=go.Heatmap(z=heatmap_matrix, x=common_bigram_words, y=common_bigram_words, colorscale='Viridis'))
                heatmap.update_layout(title="Bigram Frequency Heatmap", xaxis_nticks=len(common_bigram_words), yaxis_nticks=len(common_bigram_words))
                st.plotly_chart(heatmap)

                max_count = max([max(row) for row in heatmap_matrix]) if heatmap_matrix else 0

                # Check if max_count is non-zero
                if max_count != 0:
                    heatmap_matrix_normalized = [[val/max_count for val in row] for row in heatmap_matrix]
                else:
                    heatmap_matrix_normalized = heatmap_matrix  # or a suitable default value
                
                threshold = 5
                heatmap_matrix_thresholded = [[val if val > threshold else 0 for val in row] for row in heatmap_matrix]
               
                
                df_unigrams = pd.DataFrame(most_common_unigrams, columns=['Words', 'Count'])
                
                fig_unigrams = px.bar(df_unigrams, x='Words', y='Count', title='Top 10 Unigrams')

                st.plotly_chart(fig_unigrams)

                
                df_trigrams = pd.DataFrame(most_common_trigrams, columns=['Words', 'Count'])

                fig_trigrams = px.bar(df_trigrams, x='Words', y='Count', title='Top 10 Trigrams')
                
                st.plotly_chart(fig_trigrams)

                # Emoji Analysis
                with st.expander("Emoji Analysis"):
                    st.write("Wanna see our mark's emoji game? 🔥 or ❄️? Let's find out!")

                if most_common_emojis:
                    emojis, counts = zip(*most_common_emojis)

                    # Create a scatter plot using plotly
                    fig = go.Figure()

                    for i, emoji in enumerate(emojis):
                        fig.add_trace(go.Scatter(
                            x=[emoji],
                            y=[1],
                            mode="markers+text",
                            marker=dict(
                                size=counts[i]*20,  # Adjust size multiplier as needed
                                sizemode='diameter',
                                color=['rgba(255, 182, 193, 0.9)']
                            ),
                            text=[emoji],
                            textposition="bottom center"
                        ))
                    # Adjust layout to better fit emojis
                    fig.update_layout(
                        showlegend=False,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        hovermode=False
                    )                

                    st.plotly_chart(fig)                    
                else:
                    st.write("No emojis found in the dataset.")    
    

                # Bar Plot for Most Common Words
                words, counts = zip(*most_common_tokens)
                with st.expander("Most Common Words"):
                    st.write("Let's dive into our mark's vocab, shall we? Check out which words they're totally obsessed with. 'LOL' better be there, no cap. 😂")
                fig_common_words = px.bar(
                    x=list(words),
                    y=list(counts),
                    labels={'x': 'Words', 'y': 'Frequency'},
                    title="Top 10 Most Common Words"
                )
                st.plotly_chart(fig_common_words)

                # Extract all unique hashtags and mentions
                all_unique_hashtags = list(set(all_hashtags))
                all_unique_mentions = list(set(all_mentions))

                # Extract bigrams
                bigram_counts = extract_bigrams(all_tokens)
                # Get most common bigrams (or you can adjust the number as per your need)
                most_common_tokens = bigram_counts.most_common(25)
                # Extract common words from most common tokens
                common_words, _ = zip(*most_common_tokens)


                # Association Network Graph using Plotly
                with st.expander("Association Network Graph using Plotly"):
                    st.write("This ain't just a graph, it's a whole web of who's who and what's what in our mark's tweets. Spider-Man would be jealous! 🕷️")
                    
                # Create the Association Graph using NetworkX
                G_association = user_association_graph(username, all_unique_hashtags, all_unique_mentions, common_words)
                    
                # Visualize the Association Graph using Plotly
                fig_association = association_graph_plotly(G_association)
                st.plotly_chart(fig_association)

                with st.expander("Association Network Graph Legend"):
                    st.write("""
                        * **Black Node**: That's our main dude or dudette!
                        * **Purple Node**: Hashtag vibes.
                        * **Yellow Node**: Peeps they're shouting out.
                        * **Red Node**: Words they just can't quit.
                    """)

                # Mention Relationship Network Graph using Plotly
                with st.expander("Mention Relationship Network Graph using Plotly"):
                    st.write("Ever wonder who our mark mentions the most? This graph spills the tea. 👀")

                G = mention_relationship_graph(tweets, username)
            
                # Convert G to a 3D Plotly graph
                pos = nx.spring_layout(G, dim=3, seed=42)  # 3D layout

                edge_trace = go.Scatter3d(
                    x=[],
                    y=[],
                    z=[],
                    mode='lines',
                    line=dict(width=2, color='#888'),
                )

                for edge in G.edges():
                    x0, y0, z0 = pos[edge[0]]
                    x1, y1, z1 = pos[edge[1]]
                    edge_trace['x'] += tuple([x0, x1, None])
                    edge_trace['y'] += tuple([y0, y1, None])
                    edge_trace['z'] += tuple([z0, z1, None])

                node_trace = go.Scatter3d(
                    x=[],
                    y=[],
                    z=[],
                    text=[],
                    mode='markers+text',
                    textposition="top center",
                    hoverinfo='text',
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        size=[],
                        color=[],
                        opacity=0.8,
                        sizemode='diameter'
                    )
                )

                for node in G.nodes():
                    x, y, z = pos[node]
                    node_trace['x'] += tuple([x])
                    node_trace['y'] += tuple([y])
                    node_trace['z'] += tuple([z])
                    node_trace['text'] += tuple([node])
                    node_trace['marker']['color'] += tuple([G.nodes[node]['color']])
                    node_trace['marker']['size'] += tuple([G.nodes[node]['size']])

                layout = go.Layout(
                    title="Mention Relationship Graph",
                    scene=dict(
                        xaxis=dict(nticks=4, range=[-1, 1]),
                        yaxis=dict(nticks=4, range=[-1, 1]),
                        zaxis=dict(nticks=4, range=[-1, 1]),
                    )
                )

                fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
                st.plotly_chart(fig)

                with st.expander("Mention Relationship Graph Legend"):
                    st.write("""
                        * **Blue Node**: That's our main dude or dudette!
                        * **Purple Node**: Frequent Mention vibes.
                        * **Red Node**: Peeps the get no love, get rekt #SFYL.
                    """)    
                
                # Process the tweets for LDA
                processed_tweets = [preprocess_text(tweet['content']['itemContent']['tweet_results']['result']['legacy'].get('full_text', '')) for tweet in tweets]

                # Tokenize the processed tweets
                tokenized_tweets = [tokenize(tweet) for tweet in processed_tweets]

                # Create a dictionary and corpus for LDA using tokenized tweets
                dictionary = corpora.Dictionary(tokenized_tweets)
                corpus = [dictionary.doc2bow(tweet_tokens) for tweet_tokens in tokenized_tweets]


                # Determine number of topics
                if len(tweets) > 50:
                    num_topics = 25
                elif len(set(tokens)) < 5:
                    num_topics = len(set(tokens)) - 1
                else:
                    num_topics = 5

                # Build the LDA model
                lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

                # Get topic distributions for each document
                topic_distributions = [lda_model.get_document_topics(bow) for bow in corpus]
                # Convert to dense format
                dense_topic_distributions = np.zeros((len(topic_distributions), num_topics))
                for i, topic_probs in enumerate(topic_distributions):
                    for topic, prob in topic_probs:
                        dense_topic_distributions[i, topic] = prob

                # Apply t-SNE for dimensionality reduction to 3D
                tsne = TSNE(n_components=3)
                embedding = tsne.fit_transform(dense_topic_distributions)

                # Convert to DataFrame for plotting
                df = pd.DataFrame(embedding, columns=['x', 'y', 'z'])

                # Plotting the 3D scatter plot
                with st.expander("LDA Topic 3D scatter plot Visualization"):
                    st.write("Put on your nerd glasses 🤓! Here's a 3D view of the topics from the tweets.")
                
                # Determine dominant topic for each document
                df['dominant_topic'] = dense_topic_distributions.argmax(axis=1)

                # Get top words for each topic
                def get_topic_top_words(lda_model, topic_id, n=5):
                    return [word for word, _ in lda_model.show_topic(topic_id, topn=n)]

                df['topic_words'] = df['dominant_topic'].apply(lambda x: ", ".join(get_topic_top_words(lda_model, x)))

                # Calculate centroids
                # Drop 'topic_words' column before computing the mean
                df_numeric = df.drop(columns=['topic_words'])
                df_centroids = df_numeric.groupby('dominant_topic').mean().reset_index()

                # Add the topic_words column back to df_centroids
                df_centroids['topic_words'] = df_centroids['dominant_topic'].apply(lambda x: ", ".join(get_topic_top_words(lda_model, x)))

                # Custom Color Mapping
                unique_topics = df['dominant_topic'].unique()
                color_list_length = len(px.colors.qualitative.Plotly)
                color_map = {
                    topic: px.colors.qualitative.Plotly[i % color_list_length]
                    for i, topic in enumerate(unique_topics)
                }

                fig = px.scatter_3d(df, x='x', y='y', z='z',
                                    color='dominant_topic',
                                    labels={'color': 'Dominant Topic'},
                                    title="LDA Topic Visualization in 3D",
                                    hover_name='topic_words',
                                    hover_data={'x': False, 'y': False, 'z': False, 'dominant_topic': True},
                                    color_discrete_map=color_map)  # Use the custom color map here
                
                # Add centroids to the plot
                centroids_fig = px.scatter_3d(df_centroids, x='x', y='y', z='z', 
                                              text='topic_words', 
                                              color='dominant_topic',
                                              color_discrete_map=color_map)  # Use the custom color map here too
                
                for trace in centroids_fig.data:
                    fig.add_trace(trace)

                st.plotly_chart(fig)

                start_date = st.date_input('Start date', datetime.date(2021, 1, 1))
                end_date = st.date_input('End date', datetime.date(2023, 12, 31))
                filtered_tweets = [tweet for tweet in tweets if start_date <= datetime.datetime.strptime(tweet['content']['itemContent']['tweet_results']['result']['legacy']['created_at'], '%a %b %d %H:%M:%S +0000 %Y').date() <= end_date]

                # Display Tweets
                with st.expander("Marks Dataset"):
                    st.write("Brace yoself! Dive into the wild world of our mark's last 100 tweets. It's like reading their diary, but way more public. 🙈")
                for i, tweet in enumerate(tweets):
                    tweet_info = tweet['content']['itemContent']['tweet_results']['result']['legacy']

                    # Set card styling based on index
                    if i % 3 == 0:
                        border_color = "#e63946"
                    elif i % 3 == 1:
                        border_color = "#a8dadc"
                    else:
                        border_color = "#457b9d"

                    # Adjusted card style to mimic a tweet
                    card_style = f"""
                    background-color: white; 
                    border: 2px solid {border_color}; 
                    border-radius: 15px; 
                    padding: 20px; 
                    margin-bottom: 20px; 
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                    """    
                    with st.container():
                        st.markdown(f"<div style='{card_style}'>", unsafe_allow_html=True)
        
                        # Tweet Content
                        st.markdown(f"<h3 style='color: white; font-weight: bold;'>Tweet {i + 1}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: white; font-size: 16px;'>{tweet_info.get('full_text', 'N/A')}</p>", unsafe_allow_html=True)

                        # Baseball Card styled stats
                        tweet_created_at = datetime.datetime.strptime(tweet_info['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                        st.markdown(f"<p style='font-size: 14px; color: white;'>Posted on {tweet_created_at.strftime('%a %b %d %H:%M:%S %Y')}</p>", unsafe_allow_html=True)

                        # Get the tweet data from result directly (instead of legacy)
                        tweet_data = tweet['content']['itemContent']['tweet_results']['result']
        
                        # Columns for displaying stats
                        col1, col2, col3, = st.columns(3)
        
                        with col1:
                            # Extract the source (e.g., Twitter for iPhone) from the HTML content
                            source_text = re.findall(r'>(.*?)<', tweet_info.get('source', 'N/A'))
                            source = source_text[0] if source_text else 'N/A'
                            st.write(f"**Source**: {source}")

                        with col2:
                            st.markdown(f"_Engagement Stats:_")
                            st.write(f"❤️ **Favorites**: {tweet_info.get('favorite_count', 'N/A')}")
                            st.write(f"🔄 **Retweets**: {tweet_info.get('retweet_count', 'N/A')}")
                            views = tweet_data.get('views', {}).get('count', 'N/A')
                            st.write(f"👁️ **Views**: {views}")
            
                        with col3:
                            st.markdown(f"_More Stats:_")
                            st.write(f"💬 **Replies**: {tweet_info.get('reply_count', 'N/A')}")
                            st.write(f"🗨️ **Quotes**: {tweet_info.get('quote_count', 'N/A')}")
                            
                            # Check if 'entities' are present
                        if 'entities' in tweet_info:
                            entities = tweet_info['entities']

                            # Display hashtags
                            hashtags = [f"#{tag['text']}" for tag in entities.get('hashtags', [])]
                            st.markdown("Hashtags: " + " ".join(hashtags))

                            # Display mentions
                            mentions = [f"@{mention['screen_name']}" for mention in entities.get('user_mentions', [])]
                            st.markdown("Mentions: " + " ".join(mentions))

                            # Display URLs
                            if 'urls' in entities:
                                for url in entities['urls']:
                                    st.write(f"[Link]({url.get('expanded_url', 'N/A')})")
        
                        st.markdown("</div>", unsafe_allow_html=True)

                # Visualization of the most common words
                st.subheader("💬 Wordy words that's poppin'!")  # Updated
                df_tokens = pd.DataFrame(most_common_tokens, columns=["Word", "Count"])
                fig_tokens = px.bar(df_tokens, x='Word', y='Count', title="Most Common Words")
                st.plotly_chart(fig_tokens)

                # Visualization of the most common emojis
                st.subheader("😄 Emoji game strong!")  # Updated
                df_emojis = pd.DataFrame(most_common_emojis, columns=["Emoji", "Count"])
                fig_emojis = px.bar(df_emojis, x='Emoji', y='Count', title="Most Common Emojis")
                st.plotly_chart(fig_emojis)

                # Visualization of the most common hashtags
                st.subheader("🔖 Hashtags on point!")  # Updated
                df_hashtags = pd.DataFrame(most_common_hashtags, columns=["Hashtag", "Count"])
                fig_hashtags = px.bar(df_hashtags, x='Hashtag', y='Count', title="Most Common Hashtags")
                st.plotly_chart(fig_hashtags)

                # Visualization of the most common mentions
                st.subheader("🤝 Homies they're shouting out!")  # Updated
                df_mentions = pd.DataFrame(most_common_mentions, columns=["Mention", "Count"])
                fig_mentions = px.bar(df_mentions, x='Mention', y='Count', title="Most Common Mentions")
                st.plotly_chart(fig_mentions)

                # Visualization of the most common cashtags
                st.subheader("💸 Making it rain with these cashtags!")  # Updated
                df_cashtags = pd.DataFrame(most_common_cashtags, columns=["Cashtag", "Count"])
                fig_cashtags = px.bar(df_cashtags, x='Cashtag', y='Count', title="Most Common Cashtags")
                st.plotly_chart(fig_cashtags)

                # Visualization of the most common bigrams
                st.subheader("👯‍♀️ Dynamic duos in their tweets!")  # Updated
                df_bigrams = pd.DataFrame(most_common_bigrams, columns=["Bigram", "Count"])
                fig_bigrams = px.bar(df_bigrams, x='Bigram', y='Count', title="Most Common Bigrams")
                st.plotly_chart(fig_bigrams)        

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            import traceback
            st.error(traceback.format_exc())

    # Clear Session State button
    if st.button("Clear the vibes 🧹"):
        st.session_state.word_of_interest = ""
        st.session_state.fetch_info_pressed = False

    # Footer
    st.markdown("---")
    st.write("🧠 Powered by Rocks - Cuz sometimes it feels like that's all we got upstairs. 🪨")
    st.markdown("Disclaimer: This tool's got the tea ☕ based on public Twitter chit-chat. It's all fun and games until someone makes a bad financial decision. So, don't! 😉")

