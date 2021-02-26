root = "/home/lucien/Desktop/rec_systems"
datapath = root + "/data"

import numpy as np 
import pandas as pd 

#importing csv data 
creds = pd.read_csv(datapath + "/credits.csv")
keywords = pd.read_csv(datapath + "/keywords.csv")
links = pd.read_csv(datapath + "/links.csv")
links_small = pd.read_csv(datapath + "/links_small.csv")
metadata = pd.read_csv(datapath + "/movies_metadata.csv")
ratings = pd.read_csv(datapath + "/ratings.csv")
ratings_small = (datapath + "/ratings_small.csv")

#metadata
metadata.head(3)

################################################
# Recommender based on product description 
# In this case, movie textual description
# Features are built based on corpus vocabulary
################################################

#We are using textual data (text description), and therefore we will convert the text into a bow or tfidf matrix 
metadata['overview'].head()

#checking for missing values 
metadata['overview'].isnull().any() #True 
#imputing missing values 
metadata[metadata['overview'].isnull()] = ""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words='english') #had to reduce vocabulary shape

tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#checking shape 
tfidf_matrix.shape #75827 = vocabulary size = features size 

#Features names 
f_names = tfidf.get_feature_names()

#Running into memory issues because of the size of that sparse matrix if we try to compute all matrix.

#Getting cosine similarities testing (only for one row ..)
cs1 = linear_kernel(tfidf_matrix, tfidf_matrix[0]) #since it is tfidf, we can calculate the dot product. 

#Creating a reverse map of indices and movie titles 
indices = pd.Series(metadata.index, index = metadata['title']).drop_duplicates()
#testing it 
indices['Jumanji'] #returns 1

#########################
# Function to get the recommended movies based on movie title 

def get_recommendations(title):
    """
    This function is used to give the movie recomendations for this particular recommender system 
    given the movie title
    """
    #get index 
    index = indices[title]
    #get cosine sim 
    cs = linear_kernel(tfidf_matrix, tfidf_matrix[index])
    cs = list(enumerate(cs)) #Pairwise list (index, cosine_sim)
    #sorting the pairwise list by highest cosine similarity
    sim_scores = sorted(cs, key = lambda x: x[1], reverse = True)
    
    #get best scores 
    best_scores = sim_scores[1:11] #starting at 1 because 0 is just the identical movie
    #best recommendations indices 
    recs = [i[0] for i in best_scores]
    return metadata['title'].iloc[recs]

get_recommendations('The Dark Knight Rises')
get_recommendations('Jumanji')
get_recommendations('Toy Story')






