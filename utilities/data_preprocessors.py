import argparse
import numpy as np
import pandas as pd

def normalize_ratings(ratings: pd.DataFrame):
    # calculate mean ratings of all unique items first
    unique_item_ids = ratings['item_id'].unique()

    # build dictionary that maps unique item id's to their respective means
    items_means = {item_id: ratings.loc[ratings['item_id'] == item_id, 'rating'].mean() for item_id in unique_item_ids}

    # build list of values for mean column of new dataframe
    avg_rating = [items_means[item_id] for item_id in ratings['item_id']]

    # create avg_rating and normed_rating columns 
    temp = ratings.copy()
    temp['avg_rating'] = avg_rating
    temp['normed_rating'] = temp['rating'] - avg_rating

    # return new dataframe
    return temp




def normalize_rating_matrix(Y, R):
    """
    normalizes the ratings of user-item rating matrix Y
    note: the 1e-12 is to avoid dividing by 0 just in case
    that items aren't at all rated by any user and the sum
    of this user-item interaction matrix is not 0 which leads
    to a mathematical error.

    how this works is it takes the mean of all the user ratings
    per item, excluding of course the rating of users who've
    not yet rated the item

    args:
        Y - user-item rating matrix of (n_items x n_users) 
        dimensionality

        R - user-item interaction matrix of (n_items x n_users) 
        dimensionality
    """
    Y_mean = np.sum(Y * R, axis=1) / (np.sum(R, axis=1) + 1e-12).reshape(-1)
    Y_normed = Y - (Y_mean * R)
    return [Y_normed, Y_mean]



def is_strictly_inc_by_k(unique_ids, k):
    """
    checks if array values are increasing by a certain value 'k'
    e.g. we want to check if array [1, 2, 3, 4, 5, 6, 7] is
    strictly increasing by 1, to do this we take difference of
    each adjacent value from 0 to n - 2, which will be |1 - 2|,
    |2 - 3|, |3 - 4|, |4 - 5|, |5 - 6|, |6 - 7|. General formula
    would be |unique_user_ids[i] - unique_user_ids[i + 1]. Note:
    this would not go out of range since we only loop till n - 2
    inclusively.

    if one difference between adjacent numbers is greater than 1
    then label as True, since values like 1 would be labeled as False.
    Then turn each boolean value to ints of 1's or 0's then sum across
    all samples. If sum is greater than 0 then given array did not
    strictly increment by 'k' for all values.
    """

    return bool((np.diff(unique_ids) > k).astype(np.int64).sum())


def build_value_to_index(unique_ids):
    """
    returns a dictionary mapping each unique user id to 
    sicne there may be users where there are e.g. user 1 maps to 0
    but user 2 may not exist so we move on to user 3 which exists
    let's say so map this to (not 2) but 1. No matter if a user id
    may not exist keep incrementing the index by 1 and only by 1
    strictly.

    this is akin to generating a word to index dictionary where each
    unique word based on their freqeuncy will be mapped from indeces 1 to |V|.

    args:
        unique_user_ids - an array/vector/set of all unique user id's from
        perhaps a ratings dataset
    """
    return {id: index for index, id in enumerate(unique_ids)}


def read_item_index_to_entity_id_file():
    file = '../data/' + DATASET + '/item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


def convert_rating():
    """
    converts the rating.dat file to user-item interaction dataset
    where 1 means user has rated an item and 0 means user has not
    rated the item
    """
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]

    # this reads the rating file and replaces the delimiter :: in movies for exaple
    # to 
    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    # subsequent blocks of statemens below will create the ratings_final.txt file
    print('converting rating file ...')
    writer = open('../data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg():
    print('converting kg.txt file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')
    file = open('../data/' + DATASET + '/kg.txt', encoding='utf-8')

    for line in file:
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            continue
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    RATING_FILE_NAME = dict({'movie': 'ratings.dat',
                            'book': 'BX-Book-Ratings.csv',
                            'music': 'user_artists.dat',
                            'news': 'ratings.txt'})
    SEP = dict({'movie': '::', 'book': ';', 'music': '\t', 'news': '\t'})
    THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0, 'news': 0})
    
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.d

    entity_id2index = {}
    relation_id2index = {}
    item_index_old2new = {}

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_kg()

    print('done')
