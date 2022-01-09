from argparse import ArgumentParser
import json
import numpy as np



def build_arg_parser():
    parser = ArgumentParser(description = 'Compute Similarity Score')
    
    parser.add_argument('--user1', dest = 'user1', required = True, help = 'First User')
    parser.add_argument('--user2', dest = 'user2', required = True, help = 'Second User')
    
    parser.add_argument('--score-type', dest = 'score_type', required = True, choices = ['Euclidean', 'Pearson'],
                       help = 'Similarity metric to be used')
    return parser

def euclidean_score(dataset, user1, user2):
    #Compute the Euclidean distance score between user1 and user2
    if user1 not in dataset:
        raise TypeError(f'Cannot find {user1} in the dataset')
    if user2 not in dataset:
        raise TypeError(f'Cannot find {user2} in the dataset')
        
    #Movies rated by both user1 and user2
    common_movies = {}
    
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1
            
    if len(common_movies) == 0:
        return 0
    
    #Compute the squared difference between the ratings
    squared_diff = []
     
    for item in common_movies:
        squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))
        
    #Return the Euclidean score
    return 1 / (1 + np.sqrt(np.sum(squared_diff)))

def pearson_score(dataset, user1, user2):
    #Compute the Pearson correlation score between user1 and user2
    if user1 not in dataset:
        raise TypeError(f'Cannot find {user1} in the dataset')
    if user2 not in dataset:
        raise TypeError(f'Cannot find {user2} in the dataset')
        
    #Movies rated by both user1 and user2
    common_movies = {}
    
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1
    
    num_ratings = len(common_movies)
    if num_ratings == 0:
        return 0
    
    #Calculate the sum of ratings of all the common movies
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])
    
    #Calculate the sum of squares of ratings of all the common movies
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])
    
    #Calculate the sum of products of the ratings of all the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])
    
    #Calculate the Pearson correlation score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings
    
    #If there is no deviation, then the score is 0
    if Sxx * Syy == 0:
        return 0
    else:
        return Sxy / np.sqrt(Sxx * Syy)
    

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    
    user1 = args.user1
    user2 = args.user2
    score_type = args.score_type
    
    ratings_file = 'data/ratings.json'
    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())
        
    if score_type == 'Euclidean':
        print('Euclidean Score:')
        print(euclidean_score(data, user1, user2))
        
    else:
        print('Pearson Score:')
        print(pearson_score(data, user1, user2))