import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools as it
from concurrent.futures import ThreadPoolExecutor



def has_duplicates(arr: list | np.ndarray) -> bool:
    """
    check if list or numpy array has duplicate values
    """
    if type(arr) is list:
        return len(arr) != len(list(set(arr)))
    elif type(arr) is np.ndarray:
        return arr.size != np.unique(arr).size



def build_results(history, metrics: list=['loss', 'val_loss']):
    """
    builds the dictionary of results based on metric history of both models

    args:
        history - the history object returned by the self.fit() method of the
        tensorflow Model object

        metrics - a list of strings of all the metrics to extract and place in
        the dictionary
    """
    results = {}
    for metric in metrics:
        if metric not in results:
            results[metric] = history.history[metric]

    return results



def normalize_ratings(ratings: pd.DataFrame):
    """
    normalizes the ratings dataframe by subtracting each original
    rating of a user to an item by the mean of all users rating
    to that item

    args: 
        ratings - a
        
    """
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



def get_length__build_value_to_index(ratings: pd.DataFrame, column: str, show_logs=True):
    """
    gets all unique values given a specified column of a dataframe, 
    length of all the unique values in this column, and a dictionary
    to map these unique values to their new indeces

    args:
        ratings - the dataframe to use in order to extract a 
        specified column's unique values

        column - the specified column in which to extract all
        unique values from
    """
    # get user_id unique values and sort them
    unique_ids = ratings[column].unique()
    unique_ids.sort()

    # get number of all unique users/user id's
    n_users = unique_ids.shape[0]

    # build dictionary to map unique id's to new indeces
    vals_to_index = _build_value_to_index(unique_ids)
    sampled = _sample_first_n(vals_to_index, 15)

    if show_logs is True:
        print(f"unique {column}'s: {unique_ids[:15]}")
        print(f"do unique {column}'s have missing {column}'s? {_is_strictly_inc_by_k(unique_ids, 1)}")
        print(f"number of unique {column}: {n_users}")
        print(f"sampled dictionary of all unique {column} mapped to their respective indeces from 0 to |n_{'u' if column == 'user_id' else 'i'} - 1| {sampled}")
    

    return n_users, vals_to_index



def _sample_first_n(sample_dict: dict, first_n: int=15):
    """
    args:
        sample_dict - dictionary containing key-value pairs that will
        be used to sample its first n elements

        first_n - number of first items to sample in sample_dict, e.g.
        a first_n of 15 will take the first 15 key-value pairs of 
        sample_dict
    """
    sampled = dict(it.islice(sample_dict.items(), first_n))
    return sampled



def _is_strictly_inc_by_k(unique_ids: list | pd.Series, k: int):
    """
    checks if array values are increasing by a certain value 'k'
    e.g. we want to check if array [1, 2, 3, 4, 5, 6, 7] is
    strictly increasing by 1, to do this we take difference of
    each adjacent value from 0 to n - 2, which will be |2 - 1|,
    |3 - 2|, |4 - 3|, |5 - 4|, |6 - 5|, |7 - 6|. General formula
    would be |unique_user_ids[i] - unique_user_ids[i + 1]. Note:
    this would not go out of range since we only loop till n - 2
    inclusively.

    Count also negative values e.g. if |1 - 2| then this would be
    negative meaning array does not strictly increment for isntance
    by 1

    if one difference between adjacent numbers is greater than 1
    then label as True, since values like 1 would be labeled as False.
    Then turn each boolean value to ints of 1's or 0's then sum across
    all samples. If sum is greater than 0 then given array did not
    strictly increment by 'k' for all values.
    """

    # keys would be a list of strings ints so convert to 
    converted = list(map(int, unique_ids))
    print(len(converted))
    # print(converted[:15])
    
    # true values are if difference are exactly equal to k e.g.
    # [true, true, false] -> [false, false, true]. If one is true
    # then 1 diff is not equal to k meaning two values did not 
    # strictly increment by k

    # [false, false, true] -> [true, true, false] there are still 
    # 2 "false" values due to initial array before negation

    # [true, true, true] -> [false, false, false] no "false" before 
    # negation values and when all added after negation is exactly zero
    # meaning false, thus array elements increments strictly by k
    negated_bools = ~(np.diff(converted) == k)
    index = np.where(negated_bools == True)

    # returns true if the sum is greater than or less than 0
    # because any value < 0 or > 0 is a non-zero value, and a zero
    # value is simply a false value
    return not bool(negated_bools.astype(np.int64).sum()), index



def _build_value_to_index(unique_ids):
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



def split_data(X: pd.DataFrame, Y: pd.Series):
    """
    splits the given dataframe into training, cross-validation, and testing sets
    """
    
    X_trains, X_, Y_trains, Y_ = train_test_split(X, Y, test_size=0.3, random_state=0)
    X_cross, X_tests, Y_cross, Y_tests = train_test_split(X_, Y_, test_size=0.3, random_state=0)

    # reintegrate Y outputs to X inputs because the combined
    # dataframe will be used as a whole for the normalization
    # function
    train_data = pd.concat([X_trains, Y_trains], axis=1).reset_index(drop=True)
    cross_data = pd.concat([X_cross, Y_cross], axis=1).reset_index(drop=True)
    test_data = pd.concat([X_tests, Y_tests], axis=1).reset_index(drop=True)

    return train_data, cross_data, test_data



def separate_pos_neg_ratings(ratings: pd.DataFrame, threshold: int=4, with_kg: bool=False) -> (dict, dict):
    """
    returns two dictionaries one of them being the negative ratings
    made by each user and the other the positive ratings made by 
    each user

    args:
        ratings - 
        threshold - 
        with_kg - 
    """
    df = ratings.copy()
    
    bools = df['rating'] >= threshold
    # print(bools)

    pos_ratings = df[bools]
    neg_ratings = df[~bools]
    
    temp_pos = pos_ratings.groupby('user_id')['item_id'].agg(set)
    temp_neg = neg_ratings.groupby('user_id')['item_id'].agg(set)

    final_pos_ratings = temp_pos.to_dict()
    final_neg_ratings = temp_neg.to_dict()

    return final_pos_ratings, final_neg_ratings



def refactor_raw_ratings(pos_user_ratings: dict, neg_user_ratings: dict, item_to_index: dict, show_logs=True):
    # bug may be in item_to_index
    def helper(pos_user_rating):
        users_id = []
        users_new_item_set = []
        users_interaction = []

        # extract current user_id, and the corresponding item set
        # they've positively rated, in the concurrent process
        user_id, pos_item_set = pos_user_rating

        item_set = set(item_to_index.values())

        # subtract the new item set indeces to the new positive item set indeces
        # of a user to determine which items have not been interacted by user
        unrated_item_set = item_set - pos_item_set

        # if a negative rating exists for a user extract this negative item set
        # and subtract again the potential unrated items of a user from this 
        # negative item set
        if user_id in neg_user_ratings:
            neg_item_set = neg_user_ratings[user_id]
            unrated_item_set = unrated_item_set - neg_item_set

        for pos_item in pos_item_set:
            users_id.append(user_id)
            users_new_item_set.append(pos_item)
            users_interaction.append(1)

        # check if users unrated item set is equal to or greater than length
        # of positive item set of that user. This will be a constraint we 
        # must add to avoid any future errors when sampling. 
        num_items_to_sample = len(pos_item_set) if len(unrated_item_set) >= len(pos_item_set) else len(unrated_item_set)
        
        # for every positive interaction we sample the same 
        # amount from the unrated item set and use these
        # as our not interacted samples
        for unrated_item in np.random.choice(list(unrated_item_set), size=num_items_to_sample, replace=False):
            users_id.append(user_id)
            users_new_item_set.append(unrated_item)
            users_interaction.append(0)

        if show_logs is True:
            print(user_id)
            print(pos_item_set)
            print(len(pos_item_set), len(unrated_item_set))
        
        return pd.DataFrame({'user_id': users_id, 'item_id': users_new_item_set, 'interaction': users_interaction})

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(helper, pos_user_ratings.items()))
        comb_results = pd.concat(results, axis=0)
        final_result = comb_results.sample(frac=1).reset_index(drop=True, inplace=False)

    return final_result



"""
FOLLOWING FUNCTIONS WILL PROBABLY BE IMPORTATNT SINCE IT WILL BE USED IN PROCESSING THE KNOWLEDGE
GRAPH AS WELL AS THE
"""
def read_item_index_to_entity_id_file(item_index2entity_id_path: str, item_index_old2new: dict, entity_id2index: dict):
    """
    args:
        item_index_old2new - a dictionary that will be subsequently used by the convert_ratings function
        entity_id2index - a dictionary that will be subsequently used by the convert_kg function
    ...
    5	4
    8	5
    10	6
    11	7
    12	8
    13	9
    ...

    this is what is inside the item_id2entity_id.txt
    where numbers on the left are the item_index
    numbers on the right are the satori_id
    """
    file = item_index2entity_id_path
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]

        """THIS LINE POPULATES THE ITEM_INDEX_OLD2NEW EMPTY DICTIONARY DECLARED IN MAIN
        AND THIS WILL CONTAIN THE KEYS OF THE UNIQUE ITEMS EXCLUSIVELY IN THE KNOWLEDGE GRAPH"""
        item_index_old2new[item_index] = i

        """THIS LINE POPULATES THE ENTITY_ID2INDEX EMPTY DICTIONARY DECLARED IN MAIN
        AND THIS WILL CONTAIN THE KEYS OF THE ID OF THE ENTITIES IN THE KNOWLEDGE GRAPH"""
        entity_id2index[satori_id] = i
        i += 1

    return item_index_old2new, entity_id2index


def convert_rating(input_rating_file_path: str, output_rating_file_path: str, threshold: int, sep: str, item_index_old2new: dict):
    # CONVERTS THE RATING.DAT FILE TO USER-ITEM INTERACTION DATASET
    # WHERE 1 MEANS USER HAS RATED AN ITEM HIGHER THAN THE THRESHOLD
    # INDICATING THAT THE USER HAS A POSITIVE INTERACTION WITH THE ITEM
    # IN OTHER WORDS THEY LIKED IT AND ANYTHING BELOW IT WILL MEANS USER
    # HAS NOT LIKED THE ITEM OR HAS NOT RATED THE ITEM
    file = input_rating_file_path
    print('reading rating file ...')

    """
    GETS THE VALUES OF THE ITEM_INDEX_OLD2NEW DICTIONARY WHICH IS JUST THE
    INDECES FROM 0 TO N AND PASSES IT INTO A SET TO MAKE SURE THE ITEM_SET
    VARIABLE ONLY CONTAINS UNIQUE VALUES

    BUT CHECK AS WELL IF ITEM_SET STRICTLY INCREASES BY 1
    """
    item_set = set(item_index_old2new.values())
    print(f"item_set: {list(item_set)[:5]}")
    print(f"item_set length: {len(item_set)}")
    

    """
    DECLARE EMPTY USER_POS_RATINGS AND USER_NEG_RATINGS
    OR THE DICTIONARIES THAT POSITIVE RATINGS AND NEGATIVE RATINGS 
    """
    user_pos_ratings = {}
    user_neg_ratings = {}

    # START READING LINES AFTER THE HEADER OF THE DATASETS THAT'S WHY WE
    # START AT INDEX 1 AND NOT 0 SINCE 0 WOULD BE THE HEADER HOWEVER FOR 
    # THE MOVIE DATASET BECAUSE IT DOES NOT CONTAIN ANY HEADER WE WILL 
    # HAVE TO SKIP ONE ROW OF A USER, AN ITEM, THEIR RATING, AND THE TIMESTAMP
    for line in open(file, encoding='utf-8').readlines()[1:]:
        """
        BECAUSE WE ARE READING THE RATING FILE FOR EACH LINE OF THE PERHAPS
        MOVIE DATASET WE WILL HAVE 4 COLUMNS NAMELY THE USER_ID, ITEM_ID, RATING
        AND THE TIMESTAMP
        """
        row = line.strip().split(sep)

        # REMOVE PREFIX AND SUFFIX QUOTATION MARkS FOR BOOK DATASET
        if 'book' in input_rating_file_path:
            row = list(map(lambda x: x[1: -1], row))

        """
        EXTRACTS ITEM_ID/INDEX FROM THE RATING DATASET FILE
        """
        item_index_old = row[1]

        """AGAIN THIS FUNCTION DEPENDS ON THE NEWLY POPULATED ITEM_INDEX_OLD2NEW DICT
        WHICH STILL CONTAIN THE OLD VALUES OF ITEM_INDEX2ENTITY_ID.TXT FILE AS KEYS
        SO WHEN WE CHECK IF THE ITEM_INDEX_OLD VALUE IS ACTUALLY IN ITEM_INDEX_OLD2NEW
        WE ARE ACTUALLY USING THOSE SAME VALUE FROM THAT FILE ONLY NOW THEY ARE KEY
        VALUES IN THE DICTIONARY, AND WHEN WE CHECK IF A VALUE IS IN THIS DICITONARY WE
        ARE ACTUALLY USING THE KEY LIST AS BASIS IF THIS VALUE OR ITEM_INDEX_OLD IS IN
        THIS LIST 
    
        IF IT IS NOT IN THE ITEM_INDEX_OLD2NEW DICTIONARY THEN WE GO ON TO THE 
        NEXT ITERATION. BECAUSE IF IT IS THEN THE KNOWLEDGE GRAPH DOES NOT CONTAIN
        SUCH AN ITEM AS AN ENTITY
        """
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(row[0])

        rating = float(row[2])

        """
        ASSUMING THE ITEM OCCURS IN THE KNOWLEDGE GRAPH WITH THE LOOP NOT EXITING ITS CURRENT
        ITERATION AND CONTINUING ONTO THE NEXT WE AGAIN MAKE THE CURRENT USER AND THEIR RATED 
        ITEM FACE THE NEXT CONSTRAINT WHICH IS WHETHER SUCH AN ITEM WAS INDEED A POSITIVE ITEM 
        TO THE USER

        THIS CODE BLOCK ORGANIZES THE RATINGS OF USERS INTO POSITIVE AND NEGATIVE RATINGS
        BY USING A THRESHOLD VALUE FOR EACH DATASET E.G. FOR MOVIE RATINGS THRESHOLD FOR
        LOWEST POSITIVE RATING IS 4 AND ANYTHING STRICTLY BELOW THIS WILL BE CONSIDERED
        AS A NEGATIVE RATING, SEE THE THRESHOLD DICTIONARY BELOW

        THIS DEPENDS ON THE THRESHOLD DICTIONARY BELOW
        """
        if rating >= threshold:
            """
            IF THE USER_ID IN THE RATING DATASET DOES NOT ALREADY EXIST YET IN OUR
            USER_POS_RATINGS DICTIONARY CREATE A KEY VALUE PAIR WITH THE USER_ID AS
            THE KEY AND AN EMPTY SET AS THE VALUE

            THEN WE ADD THE ITEM_INDEX THAT OCCURS IN ALSO HAPPENS TO OCCUR IN THE KNOWLEDGE GRAPH
            TO THE SET

            IF USER_ID ALREADY EXISTS THEN WE JUST ADD THE CORRESPONDING ITEM_INDEX OF THIS USER
            TO THE SET
            """
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            """
            THIS IS ONLY FOR NEGATIVE RATINGS
            """
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    # print(f"user_pos_ratings: {user_pos_ratings}")
    # print(f"user_neg_ratings: {user_neg_ratings}")



    """
    SUBSEQUENT BLOCKS OF STATEMENTS BELOW WILL CREATE THE RATINGS_FINAL.TXT FILE

    THESE BLOCKS WILL USE THE NEWLY POPULATED USER_POS_RATINGS DICT AND 
    USER_NEG_RATINGS DICT CONTAINING THE USER_INDECES OF THE RATING DATASET 
    AS THE KEYS AND THE SET OF ITS UNIQUE RATED ITEMS FOR BOTH THE USER_POS_RATINGS 
    AND USER_NEG_RATINGS_DICT
    """
    print('converting rating file ...')
    writer = open(output_rating_file_path, 'w', encoding='utf-8')
    

    """
    THIS IS A DICTIONARY TO BE POPULATED USING THE USER_POS_RATINGS DICTIONARY'S
    """
    user_index_old2new = dict()

    """
    """
    user_cnt = 0

    """
    **ADDED BY ME**

    ADD NOW A VARIABLE THAT WILL SAVE THE NEW POSITIVELY RATED ITEMS
    OF A USER AND ALSO THEIR NEGATIVE ITEMS

    WILL CONTAIN FUTURE VALUES LIKE [user 1, user 2, user 3] FOR THE 'user_index'
    KEY [item 1, item 2, item 3] FOR THE ITEM THAT WAS RATED OR UNRATED BY THESE 
    USERS
    
    MEANING USER 1 RATED ITEM 1 POSITIVELY
    """
    rated_unrated_items = {
        'user_id': [],
        'item_id': [],
        'response': []
    }



    """
    WE LOOP THROUGH ALL KEY-VALUE PAIRS OF THE NEWLY POPULATED AND THE REORGANIZED
    RATINGS DATASET WHICH IS THE USER_POST_RATINGS DICTIONARY
    """
    for user_index_old, pos_item_set in user_pos_ratings.items():
        """
        HERE WE CHECK IF THE 
        """
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            rated_unrated_items['user_id'].append(user_index)
            rated_unrated_items['item_id'].append(item)
            rated_unrated_items['response'].append(1)
            writer.write('%d\t%d\t1\n' % (user_index, item))

        """
        HERE WE WILL FINALLY USE THE ITEM_SET SET DECLARED EARLIER IN
        THE FUNCTION WHICH RECALL CONTAINS ALL THE UNIQUE VALUES OF THE
        ITEM_INDEX_OLD2NEW DICTIONARY VALUES

        UNWATCHED_SET HERE JUST CONTAINS THE COMPLEMENT OF THE ITEMS
        INTERACTED POSITIVELY WITH BY A USER TO ALL THE UNIQUE VALUES
        OF THE ITEM_SET (WHICH IS ALSO IN THE KG)

        E.G. POS_ITEM_SET OF USER 1 IS [0, 5, 4, 6] AND ITEM_SET IS 
        [0, 1, 2, 3, 4, 5, 6, 7] THEREFORE 
        [0, 1, 2, 3, 4, 5, 6, 7] - [0, 5, 4, 6] IS {1, 2, 3, 7}
        """
        unwatched_set = item_set - pos_item_set

        """
        AND HERE WE CHECK IF THE USER_INDEX_OLD ID VALUE IN THE POSITIVE 
        RATING SET DICTIONARY ALSO EXISTS IN THE USER_NEG_RATINGS DICTIONARY
        BECAUSE IF IT IS

        BECAUSE THIS TELLS US THAT A USER THOUGH IT HAS POSITIVE INTERACTED
        ITEMS CAN ALSO HAVE ITEMS THAT IT HAS NEGATIVELY INTERACTED WITH 
        REPRESENTED THROUGH THE UESR_NEG_RATINGS DICTIONARY
        """
        if user_index_old in user_neg_ratings:
            """
            TAKE THE UNWATCHED_SET OF A USER SAY THE ABOVE WHICH IS {1, 2, 3, 7}
            THEN ACCESS THE USER_NEG_RATINGS OF USER_INDEX_OLD (WHICH IS BASICALLY
            AGAIN THE UNIQUE ID OF THE USER) AND FURTHER TRY TO REDUCE THE UNWATCHED
            OR THE UNRATED OR UNINTERACTED ITEMS OF A USER USING THE ITEM SET THEY
            NEGATIVELY RATED.

            ALTERNATIVELY user_neg_ratings[user_index_old] COULD BE SEE AS neg_item_set
            CONTRARY TO THE ABOVE WHICH IS THE pos_item_set set.

            SO FOR INSTNCE IF OUR NEG_ITEM_SET IS [1, 7, 10, 12] THEN OUR UNWATCHED_SET
            [1, 2, 3, 7] - NEG_ITEM_SET [1, 7, 10, 12] WOULD RESULT IN {2, 3}
            """
            unwatched_set = unwatched_set - user_neg_ratings[user_index_old]

        """
        MY QUESTION HERE IS IF LENGTH OF UNWATCHED SET IS LESS THAN POS_ITEM_SET E.G.
        {2, 3} FOR TEH UNWATCHED_SET AND {0, 5, 4, 6} FOR THE POSITIVELY INTERACTED WITH
        ITEMS BY USERS THEN SHOULD WE SAMPLE 4 ITEMS FROM {2, 3} WITHOUT REPLACEMENT THEN
        THIS WOULD RESULT IN AN ERROR E.G.

        numpy.random.choice(list([2, 3]), size=4, replace=False)
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        File "mtrand.pyx", line 965, in numpy.random.mtrand.RandomState.choice
        ValueError: Cannot take a larger sample than population when 'replace=False'

        OR IS IT ALWAYS GUARANTEED THAT THE UNWATCHED_SET HAVE GREATER ELEMENTS THAN THE
        POS_ITEM_SET?
        """
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            rated_unrated_items['user_id'].append(user_index)
            rated_unrated_items['item_id'].append(item)
            rated_unrated_items['response'].append(0)
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))

    return item_set, user_pos_ratings, user_neg_ratings, rated_unrated_items


def convert_kg(input_kg_file_path: str, output_kg_file_path: str, entity_id2index: dict, relation_id2index: dict):
    print('converting kg.txt file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open(output_kg_file_path, 'w', encoding='utf-8')
    file = open(input_kg_file_path, encoding='utf-8')

    for line in file:
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        """
        A TRIPLE IS EXCLUDED TO BE PUT IN ENTITY_ID2INDEX IF IT IS NOT IN THE
        ENTITY_ID2INDEX DICTIONARY WE BUILT EARLIER
        """
        if head_old not in entity_id2index:
            continue
        head = entity_id2index[head_old]

        """
        TAILS IN KG.TXT ARE ALL TOGETHER INCLUDED IF THEY EXIST OR DON'T YET EXIST
        IN THE ENTITY_ID2INDEX
        DICTIONARY WE BUILT EARLIER

        HERE IF A UNIQUE TAIL ID IN THE KG.TXT DOES NOT EXIST THEN IT IS COUNTED
        AS ANOTHER ENTITY
        """
        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        """
        RELATIONS IN KG.TXT ARE ONLY INCLUDED IF THEY EXIST IN THE ENTITY_ID2INDEX
        DICTIONARY WE BUILT EARLIER
        """
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

    """
    PHASE 1: READING ITEM_INDEX2ENTITY_ID.TXT FILE
    """
    # for read_item_index_to_entity_id_file function
    item_index2entity_id_path = f'../data/{DATASET}/item_index2entity_id.txt'
    read_item_index_to_entity_id_file(item_index2entity_id_path=item_index2entity_id_path, item_index_old2new=item_index_old2new, entity_id2index=entity_id2index)
    
    # since item_index_old2new will be modified in 
    # read_item_index_to_entity_id_file test here 
    # whether keys increment by 1 strictly
    print(f"item_index_old2new keys: {item_index_old2new.keys()}")
    print(f"item_index_old2new values: {item_index_old2new.values()}")
    print(f"entity_id2index keys: {entity_id2index.keys()}")
    print(f"entity_id2index values: {entity_id2index.values()}")

    """
    PHASE 2: PASSING RAW RATING FILE TO MULTIPLE CONSTRAINTS TO GET FINAL
    PREPROCESSED RATING FILE DATASET
    """
    # for convert_rating function
    input_rating_file_path = f'../data/{DATASET}/{RATING_FILE_NAME[DATASET]}'
    output_rating_file_path = f'../data/{DATASET}/ratings_final.txt'
    thresh = THRESHOLD[DATASET]
    sep = SEP[DATASET]
    convert_rating(input_rating_file_path=input_rating_file_path, output_rating_file_path=output_rating_file_path, threshold=thresh, sep=sep, item_index_old2new=item_index_old2new)


    """
    PHASE 3:
    """
    convert_kg()

    print('done')
