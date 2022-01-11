from argparse import ArgumentParser
import json
import numpy as np

from compute_scores import pearson_score
from collaborative_filtering import find_similar_users



def build_arg_parser():
    parser = ArgumentParser(description = 'Find recommendations for the given user')
    parser.add_argument('--user', dest = 'user', required = True, help = 'Input user')
    
    return parser

def get_recommendations(dataset, input_user):
    #Get movie recommendations for the input user
    if input_user not in dataset:
        raise TypeError(f'Cannot find {input_user} in the dataset')
        
    overall_scores = {}
    similarity_scores = {}
    
    for user in [x for x in dataset if x != input_user]:
        similarity_score = pearson_score(dataset, input_user, user)
        
        if similarity_score <= 0:
            continue
            
        #Extract a list of movies rated by the current user but not yet rated by the input user
        filtered_list = [x for x in dataset[user] if x not in dataset[input_user] or dataset[input_user][x] == 0]
        
        for item in filtered_list:
            overall_scores.update({item: dataset[user][item] * similarity_score})
            similarity_scores.update({item: similarity_score})
            
    if len(overall_scores) == 0:
        return ['No recommendations Possible']
        
    #Generate movie ranks by normalization
    movie_scores = np.array([[score / similarity_scores[item], item] for item, score in overall_scores.items()])
        
    #Sort in decreasing order
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]
        
    #Extract the movie recommendations
    movie_recommendations = [movie for _, movie in movie_scores]
        
    return movie_recommendations


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user
    
    ratings_file = 'data/ratings.json'
    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())
        
    print(f'Movie recommendations for {user}:')
    movies = get_recommendations(data, user)
    
    for i, movie in enumerate(movies, start = 1):
        print(f'{i}.{movie}')