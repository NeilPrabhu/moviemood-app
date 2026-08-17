import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from typing import List

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from sklearn.preprocessing import MinMaxScaler
from redis import asyncio as aioredis

import pickle
import numpy as np
import pandas as pd
import json
from scipy.stats import boxcox

logger = logging.getLogger("moviemood")
logging.basicConfig(level=logging.INFO)

class Song(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    type: str
    id: str
    uri: str
    track_href: str
    analysis_url: str
    duration_ms: float
    time_signature: int

'''
class Music(BaseModel):
    music_list: List[Song]
'''

class Music(BaseModel):
    music_list: List[List[str]]
    drop_movies: List[dict] | None = None
    filter_ratings: List[str] | None = None
    filter_genres: List[str] | None = None
    imdb_ratings: float | None = 5.0
    imdb_votes: float | None = 500.0



class Movie(BaseModel):
    omdb_title: str
    genres: str
    omdb_plot: str
    omdb_director:  str
    omdb_actors: str
    imdb_score: float
    omdb_poster: str
    omdb_runtime: str
    rated: str
    rotten_tomatoes_score: float

class Movies(BaseModel):
    movies_list: List[Movie]

@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            redis = aioredis.from_url(redis_url, encoding="utf8", decode_responses=True)
            FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
            logger.info("Cache backend: Redis (%s)", redis_url.split("@")[-1])
        except Exception as e:
            logger.warning("Redis init failed (%s), falling back to in-memory cache", e)
            FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    else:
        FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
        logger.info("Cache backend: in-memory (REDIS_URL not set)")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#model = joblib.load("trainer/music_to_mood_model.pickle")
loaded_rb = pickle.load(open("trainer/rb_scaler.pkl", "rb"))
loaded_model = pickle.load(open("trainer/music_to_mood_model.pickle", "rb"))

loaded_rb_ec = pickle.load(open("trainer/rb_scaler_ec.pkl", "rb"))
loaded_model_ec = pickle.load(open("trainer/music_to_mood_model_ec.pickle", "rb"))

 # Load movie genre scalers
action_genre_rb_scaler = pickle.load(open("trainer/action_genre_rb_scaler.pkl", "rb"))
comedy_genre_rb_scaler = pickle.load(open("trainer/comedy_genre_rb_scaler.pkl", "rb"))
drama_genre_rb_scaler = pickle.load(open("trainer/drama_genre_rb_scaler.pkl", "rb"))
horror_genre_rb_scaler = pickle.load(open("trainer/horror_genre_rb_scaler.pkl", "rb"))
romance_genre_rb_scaler = pickle.load(open("trainer/romance_genre_rb_scaler.pkl", "rb"))
scifi_genre_rb_scaler = pickle.load(open("trainer/scifi_genre_rb_scaler.pkl", "rb"))
                   
# Load movie genre classification models 
loaded_model_action = pickle.load(open('trainer/music_to_action_final.pickle', "rb"))
loaded_model_comedy = pickle.load(open('trainer/music_to_comedy_final.pickle', "rb"))
loaded_model_scifi = pickle.load(open('trainer/music_to_scifi_final.pickle', "rb"))
loaded_model_romance = pickle.load(open('trainer/music_to_romance_final.pickle', "rb"))
loaded_model_horror = pickle.load(open('trainer/music_to_horror_final.pickle', "rb"))
loaded_model_drama = pickle.load(open('trainer/music_to_drama_final.pickle', "rb"))

# Specify cutoff values for each genre (will be called in content layer)
genre_cutoff_dict = {'Action': 0.15, 'Comedy': 0.104, 'Romance': 0.158, 
                               'Horror': 0.156, 'Drama': 0.18, 'Sci-Fi': 0.16}

# Specify features for each genre prediction dataset
action_features = ['tempo', 'instrumentalness', 'acousticness', 'energy', 'valence']
comedy_features = ['loudness', 'tempo', 'danceability', 'instrumentalness', 'acousticness', 'energy']
drama_features = ['loudness', 'danceability', 'acousticness', 'energy']
scifi_features = ['danceability', 'instrumentalness', 'energy', 'valence']
horror_features = ['tempo', 'instrumentalness', 'acousticness', 'energy', 'valence']
romance_features = ['loudness', 'danceability', 'instrumentalness', 'energy', 'valence']

load_movie_data = pd.read_csv('data/cleaned/wiki_movies_mood_label_by_gpt-3.5-turbo_final.csv')
load_movie_data = load_movie_data.drop(columns=['Unnamed: 0'])
load_movie_data['imdb_url'] = "https://www.imdb.com/title/"+load_movie_data['imdb_id'].astype(str)
emotions_mapping = {0: 'sad', 1: 'happy', 2:'energetic', 3:'calm'}

def getIndexOfMovie(df, movie_title, movie_director):
    movie1 = df.loc[(df['omdb_title']==movie_title)]
    movie = df.loc[(df['omdb_title']==movie_title) & (df['omdb_director']==movie_director)]
    return movie.index[0]

def filter_for_genres(df, genres):
    l_of_dfs = []

    for genre in genres:
        l_of_dfs.append(df.loc[df["genres_lowered"].str.contains(genre.lower())])
    
    if len(l_of_dfs) > 1:
        return pd.concat(l_of_dfs).drop_duplicates().reset_index(drop=True)
    elif len(l_of_dfs) == 1:
        return l_of_dfs[0]
    else:
        return df

# Take in music playlist, genre cutoff dictionary, and output genres to remove
def content_layer_null_genres(playlist_df,genre_cutoff_dict):
    # Update columns names (lowercase, underscore, no special chars)
    playlist_df.columns = playlist_df.columns.str.replace(' ','_').str.lower()
    playlist_df.columns = playlist_df.columns.str.replace('(','').str.replace(')','')

    #playlist_df["tempo"] = pd.to_numeric(playlist_df["tempo"])


    # Transform tempo column to be between 0 and 1
    playlist_df['tempo'] = playlist_df['tempo']/240

    # Slice datasets for different models
    X_comedy = playlist_df.loc[:,comedy_features]
    X_action = playlist_df.loc[:,action_features]
    X_horror = playlist_df.loc[:,horror_features]
    X_drama = playlist_df.loc[:,drama_features]
    X_romance = playlist_df.loc[:,romance_features]
    X_scifi = playlist_df.loc[:,scifi_features]

    # Get transformed datasets
    X_action_transformed = action_genre_rb_scaler.transform(X_action)
    X_comedy_transformed = comedy_genre_rb_scaler.transform(X_comedy)
    X_drama_transformed = drama_genre_rb_scaler.transform(X_drama)
    X_horror_transformed = horror_genre_rb_scaler.transform(X_horror)
    X_romance_transformed = romance_genre_rb_scaler.transform(X_romance)
    X_scifi_transformed = scifi_genre_rb_scaler.transform(X_scifi)

    # Predict genre likelihoods and attach back to dataframe
    playlist_df['Action'] = loaded_model_action.predict_proba(X_action_transformed)[:,1]
    playlist_df['Comedy'] = loaded_model_comedy.predict_proba(X_comedy_transformed)[:,1]
    playlist_df['Sci-Fi'] = loaded_model_scifi.predict_proba(X_scifi_transformed)[:,1]
    playlist_df['Romance'] = loaded_model_romance.predict_proba(X_romance_transformed)[:,1]
    playlist_df['Horror'] = loaded_model_horror.predict_proba(X_horror_transformed)[:,1]
    playlist_df['Drama'] = loaded_model_drama.predict_proba(X_drama_transformed)[:,1]

    df_condensed = playlist_df[["Action","Comedy","Romance","Horror","Drama","Sci-Fi"]]

    # Get mean values & list of median values
    genre_stats = df_condensed.describe().loc['mean',:]
    median_vals = [df_condensed[val].median() for val in genre_stats.index]

    # Append to df of means and calculate average of mean/median
    stats_df = pd.DataFrame(genre_stats)
    stats_df['median'] = median_vals
    stats_df = stats_df[['median']].reset_index()
    
    # Determine which genres are below the thresholds and add to list
    null_genres = []
    for i,val in stats_df.iterrows():
        if val['median']<genre_cutoff_dict[val['index']]:
            null_genres.append(val['index'])
    

    return null_genres

@cache()
async def get_cache():
    return 1

@app.get("/")
async def root():
    return {"service": "moviemood-api", "status": "ok"}

@app.get("/hello")
async def hello(name: str = ''):
    if name:
        return {"message": "Hello " + name}
    else:
        raise HTTPException(status_code=400, detail="No value passed")
    
@app.get("/health")
async def health():
    today = datetime.now()
    return {"time":today.isoformat()}

@cache(expire=60)
@app.post("/predict")#, response_model=Movies)
async def predict(music: Music):
    arr = music.music_list
    arr[0] = [x.lower() for x in arr[0]]
    df = pd.DataFrame(data = arr[1:], columns=arr[0])

    df[['danceability','acousticness','energy','instrumentalness', \
        'liveness','valence','loudness','speechiness','tempo']] = df[['danceability','acousticness','energy','instrumentalness', \
                                                                      'liveness','valence','loudness','speechiness','tempo']].apply(pd.to_numeric)


    X = df[['danceability','acousticness','energy','instrumentalness','liveness','valence','loudness','speechiness','tempo']]
    
    X_transformed = loaded_rb.transform(X)
    X_transformed_ec = loaded_rb_ec.transform(X)

    y_pred = loaded_model.predict(X_transformed)
    y_pred_ec = loaded_model_ec.predict(X_transformed_ec)

    df['mood'] = y_pred
    df['mood'] = df['mood'].map(emotions_mapping)
    df[['sad','happy','energetic','calm']] = loaded_model.predict_proba(X_transformed)
    df[['energetic','calm']] = loaded_model_ec.predict_proba(X_transformed_ec)

    if df.shape[0] > 1:
        # Extract var3 and var4 columns
        var3 = df['energetic']
        var4 = df['calm']

        # Apply Box-Cox transformation
        transformed_var3, lambda_var3 = boxcox(var3 + 1)  # Adding 1 to handle zero values
        transformed_var4, lambda_var4 = boxcox(var4 + 1)
        
        # Scale the transformed variables to the range [0, 1]
        scaler = MinMaxScaler()
        scaled_var3 = scaler.fit_transform(transformed_var3.reshape(-1, 1))
        scaled_var4 = scaler.fit_transform(transformed_var4.reshape(-1, 1))

        # Update the DataFrame with scaled variables
        df['energetic'] = scaled_var3
        df['calm'] = scaled_var4
    else:
        pass
    
    columns_for_clustering = ['danceability','acousticness','energy','instrumentalness','liveness','valence','loudness','speechiness','tempo']

    user_mood_vector = df[['happy', 'sad', 'energetic', 'calm']].mean()
    user_mood_reshaped = user_mood_vector.values.reshape(1, -1)

    if df.shape[0] < 5:
        df['cluster_v2'] = 0
        cluster_counts = df['cluster_v2'].value_counts().sort_index()
        cluster_song_counts = cluster_counts.to_list()
    else:
        # Extract the feature matrix
        X = df[columns_for_clustering]

        # Standardize the feature matrix
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialize variables
        best_num_clusters = 1  # Start with a minimum number of clusters
        best_silhouette_score = -1  # Initialize with a low value

        # Loop through different cluster numbers
        for num_clusters in range(1, min(6, len(X) - 1)):  # Try up to 5 clusters or less based on the number of data points
            if num_clusters == 1:
                df['cluster_v2'] = 0
            else:
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                df['cluster_v2'] = kmeans.fit_predict(X_scaled)

                # Calculate silhouette score
                silhouette_avg = silhouette_score(X_scaled, df['cluster_v2'])

                # Check if the current number of clusters improves silhouette score
                if silhouette_avg > best_silhouette_score:
                    best_silhouette_score = silhouette_avg
                    best_num_clusters = num_clusters

        # Apply K-Means clustering with the best number of clusters
        kmeans = KMeans(n_clusters=best_num_clusters, random_state=42)
        df['cluster_v2'] = kmeans.fit_predict(X_scaled)

        # Count the number of songs in each cluster
        cluster_counts = df['cluster_v2'].value_counts().sort_index()
        cluster_song_counts = cluster_counts.to_list()
    
    movie_df = load_movie_data
    movie_df['genres_lowered'] = movie_df['genres'].str.lower()
    movie_df['rated_lowered'] = movie_df['rated'].str.lower()

    if music.drop_movies:
        movies_to_drop = [getIndexOfMovie(movie_df, movie['omdb_title'], movie['omdb_director']) for movie in music.drop_movies]
        movie_df = movie_df.drop(movies_to_drop)

    movie_df = movie_df[movie_df['imdb_score'] >= music.imdb_ratings]
    movie_df = movie_df[movie_df['imdb_votes'] >= music.imdb_votes]

    if music.filter_ratings:
        ratings = [rating.lower() for rating in music.filter_ratings]
        movie_df = movie_df[movie_df['rated_lowered'].isin(ratings)]

    if music.filter_genres:
        movie_df = filter_for_genres(movie_df, music.filter_genres)
    else:
        # Get bad genres and exclude them from movie dataframe
        bad_genres = content_layer_null_genres(df,genre_cutoff_dict)
        movie_df = movie_df[~movie_df['genres'].str.contains('|'.join(bad_genres), case=False, na=False)]

    movie_mood_vectors = movie_df[['happy','sad','energetic','calm']].values

    cluster_movie_counts = {}
    for i in range(len(cluster_song_counts)):
        cluster_movie_counts[i] = int(np.round(5 * (cluster_song_counts[i] / sum(cluster_song_counts))))
    print(cluster_movie_counts)
    cluster_avg_mood = df.groupby('cluster_v2')[['happy', 'sad', 'energetic', 'calm']].mean()
    print(cluster_avg_mood)
    # If the amount of movies recommended is less than 5, add a movie to the cluster with the most songs
    while sum(cluster_movie_counts.values()) < 5:
        max_cluster = max(cluster_movie_counts, key=cluster_movie_counts.get)
        cluster_movie_counts[max_cluster] += 1

    # If the amount of movies recommended is more than 5, remove a movie from the cluster with the most songs
    while sum(cluster_movie_counts.values()) > 5:
        max_cluster = max(cluster_movie_counts, key=cluster_movie_counts.get)
        cluster_movie_counts[max_cluster] -= 1

    result_df = pd.DataFrame(columns=['year', 'href', 'title', 'plot', 'premise', 'synopsis', 'directed_by',
       'screenplay_by', 'based_on', 'starring', 'box_office', 'written_by',
       'budget', 'omdb_response', 'omdb_title', 'genres', 'omdb_plot',
       'release_date', 'omdb_director', 'omdb_writer', 'omdb_actors',
       'imdb_score', 'imdb_votes', 'rotten_tomatoes_score', 'metacritic_score',
       'directed_by_clean', 'screenplay_by_clean', 'written_by_clean',
       'starring_clean', 'based_on_binary', 'based_on_binary_characters',
       'based_on_binary_comic_books', 'box_office_clean', 'budget_clean',
       'chosen_plot_feature', 'writers_to_use', 'actors_to_use',
       'plot_parenthetical', 'num_parentheticals',
       'plot_parenthetical_fuzzy_scores', 'new_plot',
       'all_writers_first_movie', 'any_writers_first_movie', 'roi',
       'roi_binary', 'roi_multi', 'rt_binary', 'rt_multi', 'happy', 'sad',
       'energetic', 'calm', 'error_labeling'])
    spotify_values = {}
    movie_belongs_to_cluster = []
    for key in cluster_movie_counts:
        for i in range(cluster_movie_counts[key]):
            movie_belongs_to_cluster.append(key)
    print("movie_belongs_to_cluster")
    print(movie_belongs_to_cluster)
    df[columns_for_clustering] = df[columns_for_clustering].apply(pd.to_numeric)
    for user_cluster, num_movies in cluster_movie_counts.items():
        # Calculate the average mood vector for each cluster
        cluster_avg_mood = df.groupby('cluster_v2')[['happy', 'sad', 'energetic', 'calm']].mean()
        cluster_avg_spotify = df.groupby('cluster_v2')[columns_for_clustering].mean()
        print(cluster_avg_mood)
        # Reshape the user's cluster average mood vector
        user_mood_reshaped = cluster_avg_mood.loc[user_cluster].values.reshape(1, -1)
        user_spotify_values_reshaped = cluster_avg_spotify.loc[user_cluster].values.reshape(1, -1)[0]
        spotify_values[user_cluster] = {'mood_vector':list(user_mood_reshaped[0])}
        #print(list(user_mood_reshaped[0]))
        for i in range(len(columns_for_clustering)):
            spotify_values[user_cluster][columns_for_clustering[i]] = user_spotify_values_reshaped[i]
        
        # Use cosine similarity to find the top movies similar to the average mood vectors of the clusters
        similarities = cosine_similarity(user_mood_reshaped, movie_mood_vectors)
        sim_scores = list(enumerate(similarities[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get top movie titles based on top similarity scores
        top_movies = [i[0] for i in sim_scores[1:num_movies+1]]
        recommended_movies = movie_df.iloc[top_movies]
        #print(recommended_movies)
        result_df = pd.concat([result_df, recommended_movies], ignore_index=True)  
    print(result_df[['title','happy', 'sad','energetic', 'calm']])
    #top_movies = result_df[:5]
    #top_movie_indices = [i[0] for i in top_movies]
    #recommended_movies = movie_df.iloc[top_movie_indices]
    #pprint(spotify_values)
    #result_df['cluster'] = movie_belongs_to_cluster
    recommended_movies = result_df[['omdb_title','genres','omdb_plot','omdb_director', \
                                            'omdb_actors', 'imdb_score', 'omdb_poster', 'omdb_runtime', 'rated', 'imdb_url', 'rotten_tomatoes_score']]
    recommended_movies_json = recommended_movies.to_json(orient='records')
    keys_to_del = []
    for key in spotify_values:
        if key not in movie_belongs_to_cluster:
            keys_to_del.append(key)

    for k in keys_to_del:
        del spotify_values[k]
    
    return {"movies_list":json.loads(recommended_movies_json), "spotify_information": spotify_values}