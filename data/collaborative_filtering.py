from argparse import ArgumentParser
import json
import numpy as np

from compute_scores import pearson_score



def build_arg_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--user', dest = 'user', required = True, help = 'Input user')
    return parser

def find_similar_users(dataset, user, num_users):
    #Find users in the dataset that are similar to the input user
    if user not in dataset:
        raise TypeError(f'Cannot find {user} in the dataset')
    
    #Compute Pearson score between input user and all the users in the dataset
    scores = np.array([[x, pearson_score(dataset, user, x)] for x in dataset if x != user])
    
    #Sort the scores in decreasing order
    scores_sorted = np.argsort(scores[:, 1])[::-1]
    
    #Extract the top 'num_users' scores
    top_users = scores_sorted[:num_users]
    
    return scores[top_users]


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user
    
    ratings_file = 'data/ratings.json'
    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())
          
    print(f'Users similar to {user}:\n')
    similar_users = find_similar_users(data, user, 3)
    print('User\t\t\tSimilarity score')
    
    print('-' * 41)
    for name, score in similar_users:
        print(f'{name}\t\t{round(float(score), 2)}')