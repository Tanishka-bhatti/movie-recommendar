import numpy as np
import pandas as pd
import os
file_path = os.path.join(os.path.dirname(__file__), "movie_dataset.csv")
final_data = pd.read_csv(file_path)
print(final_data.head())
print(final_data.columns)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=6000, stop_words='english')
vectors = cv.fit_transform(final_data['tags']).toarray()
print(vectors)
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(x):
    stem_word = []
    string = ''
    for i in x.split():
        stem_word.append(ps.stem(i))
    return string.join(stem_word)

final_data['tags'] = final_data['tags'].map(lambda x: stem(x))

print(cv.get_feature_names_out())

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
print(similarity)

from difflib import get_close_matches

def recommend(movie):
    if movie not in final_data['movie_title'].values:
        close_matches = get_close_matches(movie, final_data['movie_title'].values)
        if close_matches:
            print(f"Did you mean '{close_matches[0]}'? Using that instead.")
            movie = close_matches[0]
        else:
            print("No similar movie found in the database.")
            return
    movie_index = final_data[final_data['movie_title'] == movie].index[0]
    distances = similarity[movie_index]
    recommend_movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in recommend_movie_list:
        print(final_data.iloc[i[0]].movie_title)


recommend('Spider-Man')

import pickle
pickle.dump(final_data, open('final_data_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

