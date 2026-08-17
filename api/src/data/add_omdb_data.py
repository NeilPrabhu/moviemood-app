import requests
import pandas as pd
import json
import sys
import os
import datetime

def flatten_extend(matrix):
     flat_list = []
     for row in matrix:
         flat_list.extend(row)
     return flat_list

counter = 0
id_list_completed = []
if os.path.exists('movies_data_completed.csv'):
        movies_completed_id_raw = open("movies_completed.csv").read().splitlines()
        movies_completed_id_list = [int(float(id)) for id in movies_completed_id_raw]
        movies_failed_id_raw = open("movies_not_found_in_tmdb.txt").read().splitlines()
        movies_failed_id_list = [int(float(id.replace("id: ", ""))) for id in movies_failed_id_raw]
        print("length of total ids: " + str(len(movies_completed_id_list)+len(movies_failed_id_list)))
        id_list_completed = movies_completed_id_list + movies_failed_id_list
        id_list_completed.sort()
movies = pd.read_csv("english_movies_gt_2000_dataset_100823.csv", on_bad_lines='skip')
#new_cols = ["omdb_Title", "omdb_Year", "omdb_Rated", "omdb_Released", "omdb_Runtime", "omdb_Genre", "omdb_Director", "omdb_Writer", "omdb_Actors", \
#            "omdb_Plot", "omdb_Language", "omdb_Country", "omdb_Awards", "omdb_Poster", "omdb_Ratings","omdb_Metascore", \
#            "omdb_imdbRating", "omdb_imdbVotes", "omdb_imdbID", "omdb_Type", "omdb_DVD", "omdb_BoxOffice", "omdb_Production", "omdb_Website"]
movies['omdb_Title'] = ''
movies['omdb_Year'] = ''
movies['omdb_Rated'] = ''
movies['omdb_Released'] = ''
movies['omdb_Runtime'] = ''
movies['omdb_Genre'] = ''
movies['omdb_Director'] = ''
movies['omdb_Writer'] = ''
movies['omdb_Actors'] = ''
movies['omdb_Plot'] = ''
movies['omdb_Language'] = ''
movies['omdb_Country'] = ''
movies['omdb_Awards'] = ''
movies['omdb_Poster'] = ''
movies['omdb_Ratings'] = ''
movies['omdb_Metascore'] = ''
movies['omdb_imdbRating'] = ''
movies['omdb_imdbVotes'] = ''
movies['omdb_imdbID'] = ''
movies['omdb_Type'] = ''
movies['omdb_DVD'] = ''
movies['omdb_BoxOffice'] = ''
movies['omdb_Production'] = ''
movies['omdb_Website'] = ''
movies['year'] = movies['release_date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y').year)
#for col in new_cols:
#     movies[col] = ''
if not os.path.exists('movies_data_completed.csv'):
    temp_df = movies.iloc[:0,:].copy()
else:
    temp_df = pd.DataFrame()
temp_df.to_csv('movies_data_completed.csv', index=False, mode='a+')

for i, row in enumerate(movies.itertuples(), 1):
    if row.id in id_list_completed:
        print(str(row.id) + " already in movies_completed")
        continue
    if row.title and row.year:
        print(str(row.id) + " not in movies_completed")
        title = row.title
        year = row.year
        try:
            url = 'https://www.omdbapi.com/?i=tt3896198&apikey=4afdfa9d&t=' + '"' + str(title) + '"&year='+ str(year)
        except Exception as e:
            with open('movies_not_found_in_tmdb.txt', 'a') as f:
                f.write("id: " + str(row.id) +'\n')
        #print(url)
        response = requests.get(url)
        if response.ok:
            try:
                result_json = response.json()
            except Exception as e:
                with open('movies_not_found_in_tmdb.txt', 'a') as f:
                    f.write("id: " + str(row.id) +'\n')
            #print(result_json)
            movies.at[row.Index,'omdb_Title'] = result_json.get('Title', "")
            movies.at[row.Index,'omdb_Year'] = result_json.get('Year', "")
            movies.at[row.Index,'omdb_Rated'] = result_json.get('Rated', "")
            movies.at[row.Index,'omdb_Released'] = result_json.get('Released', "")
            movies.at[row.Index,'omdb_Runtime'] = result_json.get('Runtime', "")
            movies.at[row.Index,'omdb_Genre'] = result_json.get('Genre', "")
            movies.at[row.Index,'omdb_Director'] = result_json.get('Director', "")
            movies.at[row.Index,'omdb_Writer'] = result_json.get('Writer', "")
            movies.at[row.Index,'omdb_Actors'] = result_json.get('Actors', "")
            movies.at[row.Index,'omdb_Plot'] = result_json.get('Plot', "")
            movies.at[row.Index,'omdb_Language'] = result_json.get('Language', "")
            movies.at[row.Index,'omdb_Country'] = result_json.get('Country', "")
            movies.at[row.Index,'omdb_Awards'] = result_json.get('Awards', "")
            movies.at[row.Index,'omdb_Poster'] = result_json.get('Poster', "")
            movies.at[row.Index,'omdb_Ratings'] = result_json.get('Ratings', "")
            movies.at[row.Index,'omdb_Metascore'] = result_json.get('Metascore', "")
            movies.at[row.Index,'omdb_imdbRating'] = result_json.get('imdbRating', "")
            movies.at[row.Index,'omdb_imdbVotes'] = result_json.get('imdbVotes', "")
            movies.at[row.Index,'omdb_imdbID'] = result_json.get('imdbID', "")
            movies.at[row.Index,'omdb_Type'] = result_json.get('Type', "")
            movies.at[row.Index,'omdb_DVD'] = result_json.get('DVD', "")
            movies.at[row.Index,'omdb_BoxOffice'] = result_json.get('BoxOffice', "")
            movies.at[row.Index,'omdb_Production'] = result_json.get('Production', "")
            movies.at[row.Index,'omdb_Website'] = result_json.get('Website', "")
            print("added omdb data for movie " + result_json.get('Title', ""))
            temp_df = temp_df._append(movies.iloc[row.Index], ignore_index = True)                  # add row to dataframe
            temp_df.to_csv('movies_data_completed.csv', index=False, mode='a+', header=False)       # append row to saved csv
            temp_df = movies.iloc[:0,:].copy()                                                      # empty dataframe
            with open('movies_completed.csv', 'a') as f:
                f.write(str(row.id) +'\n')
        else:
            with open('movies_not_found_in_tmdb.txt', 'a') as f:
                f.write("id: " + str(row.id) +'\n')
    else:
        with open('movies_not_found_in_tmdb.txt', 'a') as f:
                f.write(str(row.id)+' is blank \n')

#movies.to_csv('/Users/neilp/Desktop/W210/moviemood/src/data/english_movies_gt_2000_dataset_101923.csv')