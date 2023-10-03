# **STILL IN PRODUCTION**
This is the second phase of my undergraduate thesis which will recommend jurisprudence documents to legal practitioners specializing in the labor sector. Based on the paper of Wang, H. et. al. 



# Usage:
1. clone repository with `git clone https://github.com/08Aristodemus24/phil-jurisprudence-recsys.git`
2. navigate to directory with `readme.md` and `requirements.txt` file
3. run command; `conda create -n <name of env e.g. phil-jurisprudence-recsys> python=3.10.11`. Note that 3.10.11 must be the python version otherwise packages to be installed would not be compatible with a different python version
4. once environment is created activate it by running command `conda activate`
5. then run `conda activate phil-jurisprudence-recsys`
6. check if pip is installed by running `conda list -e` and checking list
7. if it is there then move to step 8, if not then install `pip` by typing `conda install pip`
8. if `pip` exists or install is done run `pip install -r requirements.txt` in the directory you are currently in



# Recommender System Building
## Model building:
**Prerequisites to do:** 
1. 

**To do:**
1. <s>see shape of user input in DeepFM model</s>
2. <s>test run</s>
3. <s>label each line of execution in Recommender-System repository particularly in the using deepfm model</s>
4. create data loader for movie ratings dataset *priority*
    a. item_index2entity_id.txt 
5. item_index2entity_id.txt actually goes hand in hand with the knowledge graph dataset ml1m-kg1m and ml1m-kg20k
6. mean adder to the predicted ratings
7. adder of a new user to the user-item rating matrix and user-item interaction matrix
8. being able to update a single rated item-rating by a single user in the user-item rating matrix and the user-item interaction matrix
    a. $Y_{i, j}$ is, 0.5 user turns it to 3.5, $R_{i, j}$ is 1 initially and after update $R_{i, j}$ is still 1
9. being able to update a single unrated item-rating by a single user in the user-item rating matrix and the user-item interaction matrix
    a. $Y_{i, j}$ is, 0 user turns it to 5, $R_{i, j}$ is 0 initially and after update $R_{i, j}$ is now 1
10. confine ratings to only 0 or any number in the set {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5}
    a. predictions like 0.75 should be rounded up to 1, 1.75 to 2, 2.8 to 3, 3.9 to 4, 4.85 to 5
    b. predictions like 0.25 should be rounded down to 0, 0.499999 to 0. Basically anything than below an itnerval of 0.5 must be rounded down
    c. prediction 3.25 -> 3.25 - 3 = 0.25 < 0.5 therefore round 3.25 down to 3.0
    d. prediciton 3.5 -> 3.5 - 3 = 0.5 >= 0.5 therefore round 3.5 to 4.0
11. <s>because initial factorization machine (FM) architecture of collaborative filtering model already exists, using the the Functional API of tensorflow by using the built-in Model class, I need to build a more tailored version of this Model class that implements this FM architecture</s>
12. <s>build DeepFM architecture</s>
13. <s>write MKR architecture</s>

**Questions:**
1. how do I split the data into training, validation, and testing sets?
2. how do I batch train the model?
3. how do I make predictions with the model
4. how do I implement the AUC, Precision@k, Recall@k, and F1-score in this model?

**Insights:**
1. User will always have negative ratings so remove this by subtracting all unique items from negative rating set and positive rating set to get all items not rated by a user. When sampling with replacement, sample size can be greater than population size. And the population mean is a parameter; the sample mean is a statistic e.g. [1, 2, 3] sample 10 can be permitted if replace is true or we sample with replacement or return the value we took out in our "bag"

**Conclusions**
1. 

**Articles:**
1. building a matrix factorization model and normalizing ratings: https://www.kaggle.com/code/colinmorris/matrix-factorization

**Side notes:**

## Preprocessing data:
**Prerequisites to do:**
1. fix file structure of `Recommender-System` repository *priority*
2. translate chinese characters `Recommender-System` repository *priority*
a. data_loader.py
b. kg_load.py
c. evaluation.py
d. decorator.py
e. competition.py

**To do:**
1. <s>create preprocess rating tomorrow to keep only positive interactions as 1 and unrated items as 0.</s>
2. <s>unwatched item set of a user (all unique values) must be equal to or greater than length of positive item set of that user. This will be a constraint we must add to avoid any future errors when sampling. Because if it is the case that unwatched items is less than positive e.g. user rated 10 items positively and 3 items negatively out of all 20 items, unwatched items would be 7, this would be an error since we aer sampling without replacement the same number of positive items which is 10 from an unrated set of only 7 items. But since this is a recommendation system where usually users do not rate most items and the user-item interaction and rating matrix is sparse, such an error could be avoided for the mean time, but we still need to find a way to handle this</s>
3. need to find the end result of entity_id2index after convert_kg function
4. <s>add logs to get_length__build_value_to_index</s>
5. <s>pipeline of separate_pos_neg_ratings() function:</s>
a. pass ratings df, item_id string
b. return values will have var names n_items, old_item_idx2new_item_idx
c. for users the same thing...n_users, old_user_idx2new_user_idx
d. use built lookup dictionaries to renew user id and item id columns using `.apply()` method of dataframe
e. pass new dataframe to `separate_pos_neg_ratings()` func
f. separate pos and neg ratings based on threshold by vectorization
```
>>> import pandas as pd
>>>
>>> ratings = pd.DataFrame({'user_id': [2, 2, 5, 16, 16, 16, 20, 1, 1, 3, 56, 32], 'item_id': [9, 2, 4, 99, 9, 4, 9, 1, 2, 50, 21, 100], 'rating': [4, 5, 4, 5, 4, 5, 5, 5, 5, 4, 5, 5]})
>>>
>>> ratings.groupby('user_id')
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001A9759E7AF0>
>>> ratings.groupby('user_id').agg(set)
            item_id  rating
user_id
1            {1, 2}     {5}
2            {9, 2}  {4, 5}
3              {50}     {4}
5               {4}     {4}
16       {9, 99, 4}  {4, 5}
20              {9}     {5}
32            {100}     {5}
56             {21}     {5}
>>>
```

or

```
>>> ratings.groupby('user_id', as_index=False).agg(set)
   user_id     item_id  rating
0        1      {1, 2}     {5}
1        2      {9, 2}  {4, 5}
2        3        {50}     {4}
3        5         {4}     {4}
4       16  {9, 99, 4}  {4, 5}
5       20         {9}     {5}
6       32       {100}     {5}
7       56        {21}     {5}
```

```
>>> user_pos_ratings.to_dict()
{'user_id': {0: 1, 1: 2, 2: 3, 3: 5, 4: 16, 5: 20, 6: 32, 7: 56}, 'item_id': {0: {1, 2}, 1: {9, 2}, 2: {50}, 3: {4}, 4: {9, 99, 4}, 5: {9}, 6: {100}, 7: {21}}, 'rating': {0: {5}, 1: {4, 5}, 2: {4}, 3: {4}, 4: {4, 5}, 5: {5}, 6: {5}, 7: {5}}}
>>>
```

```
>>> user_pos_ratings_2 = ratings.groupby('user_id').agg(set)
>>> user_pos_ratings_2.to_dict()
{'item_id': {1: {1, 2}, 2: {9, 2}, 3: {50}, 5: {4}, 16: {9, 99, 4}, 20: {9}, 32: {100}, 56: {21}}, 'rating': {1: {5}, 2: {4, 5}, 3: {4}, 5: {4}, 16: {4, 5}, 20: {5}, 32: {5}, 56: {5}}}
```

```
>>> user_pos_ratings_2 = ratings.groupby('user_id')[['item_id']].agg(set)
>>> user_pos_ratings_2.to_dict()
{'item_id': {1: {1, 2}, 2: {9, 2}, 3: {50}, 5: {4}, 16: {9, 99, 4}, 20: {9}, 32: {100}, 56: {21}}}
>>>
```

```
>>> temp = ratings.groupby('user_id')['item_id'].agg(set)
>>> temp
user_id
1         {1, 2}
2         {9, 2}
3           {50}
5            {4}
16    {9, 99, 4}
20           {9}
32         {100}
56          {21}
Name: item_id, dtype: object
>>> temp.to_dict()
{1: {1, 2}, 2: {9, 2}, 3: {50}, 5: {4}, 16: {9, 99, 4}, 20: {9}, 32: {100}, 56: {21}}
```

6. make a function that saves preprocessed adn split data for easier access and its meta data like n_users, n_items for model arguments

**Problems:**
1. <s>A big problem is that upon using refactor_raw_ratings() to get only positive ratings and sample unwatched ratings, item_ids unsually doubles from 3706 items to now 6040 items. So why is that?</s>

**Questions:**

**Insights:**

**Conclusions:**

**Articles:**
1. somehow convert each row that has a unique user and their corresponding rated item to a dictionary with each value as a set in a vectorized way: https://stackoverflow.com/questions/65436865/how-to-convert-dataframe-into-dictionary-of-sets

**Side notes:**

## Tuning Model
**Prerequisites to do:**
1. raise problem in stackvoerflow:
RecSys model performance stalling at 47% AUC and F1-Score. Is the problem due to ratio of users to items in my dataset?

I'm having trouble with making my validation metrics go down for the binary_crossentropy and go up for the F1-score and AUC. I've tried tuning my hyper parameters such as the number of latent features of the model (8), the learning rate (0.000075), the lambda in the regularization term (1.2), the the drop out rate (0.4), and the batch size (16384), which have seemingly maximum values you can give to where you can ensure the prevention of overfitting, but to no avail my validation F1-Score and AUC always stalls at around 47%, 52% at its highest if I increase my epochs to 500. It even got to the point that a higher batch size gave my RAM problems since I only use my mere CPU in this ML task (Because I've no graphics card unfortunately).

Here is my model architecture which uses an embedding layer initially then essentially takes the output of this embedding layer and feeds it into two phases so to speak, one that will flatten the output of the embedding layer and one which concatenates the output of the embedding layer and feeds it into a fully connected network.

So my question Could it be that my architecture is too complex or is my dataset the problem?

2. use github URL of larj-corpus dataset instead of local path in retrieving the rating data

**To do:**
1. try `python train_model.py -d juris-600k --protocol A --model_name DFM --n_features 8 --n_epochs 100 --rec_alpha 0.000075 --rec_lambda 1 --rec_keep_prob 0.8 --batch_size 8192`
2. `python train_model.py -d juris-600k --protocol A --model_name FM --n_epochs 100 --rec_lambda 1 --rec_keep_prob 0.8 --batch_size 8192`
3. in both models try bigger batch of 65536, 32768, 16384
4. in DFM try keep probability of 0.6, alpha of 0.0001, lambda of 1.2
5. commands to use:
* python train_model.py -d ml-1m --protocol A --model_name FM --n_features 32 --n_epochs 100 --rec_alpha 0.0003 --rec_lambda 0.9 --batch_size 8192

* python train_model.py -d juris-300k --protocol A --model_name FM --n_features 32 --n_epochs 100 --rec_alpha 0.0003 --rec_lambda 0.9 --rec_keep_prob 0.1 --batch_size 8192
* python train_model.py -d juris-300k --protocol A --model_name DFM --n_features 32 --layers_dims 32 16 16 16 8 8 4 4 3 1 --n_epochs 100 --rec_alpha 0.0003 --rec_lambda 0.9 --rec_keep_prob 0.1 --batch_size 8192
* python train_model.py -d ml-1m --protocol A --model_name FM --n_features 32 --n_epochs 100 --rec_alpha 0.0003 --rec_lambda 0.9 --batch_size 8192
* python train_model.py -d juris-300k --protocol A --model_name FM --n_features 32 --n_epochs 100 --rec_alpha 0.0003 --rec_lambda 0.9 --rec_keep_prob 0.1 --batch_size 8192
* python train_model.py -d juris-300k --protocol A --model_name DFM --n_features 32 --layers_dims 32 16 16 16 8 8 4 4 3 1 --n_epochs 100 --rec_alpha 0.0003 --rec_lambda 0.9 --rec_keep_prob 0.1 --batch_size 8192

* python train_model.py -d juris-600k --protocol A --model_name FM --n_features 32 --n_epochs 100 --rec_alpha 0.0001 --rec_lambda 0.9 --batch_size 8192
* python train_model.py -d juris-600k --protocol A --model_name DFM --n_features 32 --layers_dims 32 16 16 16 8 8 4 4 3 1 --n_epochs 100 --rec_alpha 0.0001 --rec_lambda 0.9 --rec_keep_prob 0.7 --batch_size 8192

* python train_model.py -d juris-3m --protocol A --model_name FM --n_features 32 --n_epochs 100 --rec_alpha 0.0001 --rec_lambda 0.9 --batch_size 32768
* python train_model.py -d juris-3m --protocol A --model_name DFM --n_features 32 --layers_dims 32 16 16 16 8 8 4 4 3 1 --n_epochs 100 --rec_alpha 0.0001 --rec_lambda 0.9 --rec_keep_prob 0.7 --batch_size 32768



**Problems:**
1. <s>There is something wrong with split data or refactor raw ratings because there seems to be a mismatch in original number of user id's and item_id's. I suspect because user id and item ids are lessened because negative ratings are removed. Nevertheless following models and used dataset produce the ff. results:</s>
* <s>FM with juris_300k is ok</s>
* <s>DFM with juris_300k causes `Allocation of 268435456 exceeds 10% of free system memory.` & `OP_REQUIRES failed at segment_reduction_ops_impl.h:478 : INVALID_ARGUMENT: data.shape = [8192] does not start with segment_ids.shape = [67108864]`. I suspect this has something to do with batch size and the model architecture itself</s>
* <s>FM with juris_600k is not ok to begin with (even if it runs fine albeit with abysmal AUC, F1-Score, and Binary Accuracy results) since there is one user that is missing in the final refactored juris_600k dataset, where instead of 12034 users all in all there are now only 12033 users</s>
* <s>DFM with juris_600k causes `Allocation of 268435456 exceeds 10% of free system memory.` & `OP_REQUIRES failed at segment_reduction_ops_impl.h:478 : INVALID_ARGUMENT: data.shape = [8192] does not start with segment_ids.shape = [67108864]`. Again I suspect this has something to do with batch size and the model architecture itself. **Ok found the problem because if I remove deep neural network architecture model works fine**. Resolved just added flatten layer after concatednation layer because I forgot</s>
2. there seems to be overfitting due to the dataset itself because of the rat
3. somehow the MKR model seems to be working fine giving out the ff. results
```
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:From C:\ProgramData\Anaconda3\envs\phil-jurisprudence-recsys\lib\site-packages\tensorflow\python\util\dispatch.py:1176: calling expand_dims (from tensorflow.python.ops.array_ops) with dim is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
epoch 0    train auc: 0.8884  acc: 0.8032    eval auc: 0.8833  acc: 0.8001    test auc: 0.8836  acc: 0.8003
epoch 1    train auc: 0.8945  acc: 0.8123    eval auc: 0.8862  acc: 0.8051    test auc: 0.8867  acc: 0.8060
epoch 2    train auc: 0.9007  acc: 0.8163    eval auc: 0.8873  acc: 0.8063    test auc: 0.8878  acc: 0.8059
epoch 3    train auc: 0.9102  acc: 0.8260    eval auc: 0.8919  acc: 0.8101    test auc: 0.8919  acc: 0.8110
epoch 4    train auc: 0.9210  acc: 0.8395    eval auc: 0.8987  acc: 0.8210    test auc: 0.8990  acc: 0.8203
epoch 5    train auc: 0.9287  acc: 0.8498    eval auc: 0.9035  acc: 0.8265    test auc: 0.9038  acc: 0.8271
epoch 6    train auc: 0.9331  acc: 0.8558    eval auc: 0.9056  acc: 0.8299    test auc: 0.9054  acc: 0.8298
epoch 7    train auc: 0.9383  acc: 0.8620    eval auc: 0.9076  acc: 0.8328    test auc: 0.9076  acc: 0.8321
epoch 8    train auc: 0.9404  acc: 0.8643    eval auc: 0.9097  acc: 0.8345    test auc: 0.9096  acc: 0.8339
epoch 9    train auc: 0.9427  acc: 0.8675    eval auc: 0.9105  acc: 0.8347    test auc: 0.9103  acc: 0.8345
epoch 10    train auc: 0.9453  acc: 0.8706    eval auc: 0.9112  acc: 0.8372    test auc: 0.9108  acc: 0.8358
epoch 11    train auc: 0.9460  acc: 0.8714    eval auc: 0.9123  acc: 0.8376    test auc: 0.9117  acc: 0.8370
epoch 12    train auc: 0.9474  acc: 0.8731    eval auc: 0.9118  acc: 0.8369    test auc: 0.9117  acc: 0.8370
epoch 13    train auc: 0.9481  acc: 0.8744    eval auc: 0.9134  acc: 0.8390    test auc: 0.9125  acc: 0.8361
epoch 14    train auc: 0.9491  acc: 0.8750    eval auc: 0.9133  acc: 0.8386    test auc: 0.9126  acc: 0.8374
epoch 15    train auc: 0.9483  acc: 0.8740    eval auc: 0.9131  acc: 0.8380    test auc: 0.9122  acc: 0.8367
epoch 16    train auc: 0.9502  acc: 0.8770    eval auc: 0.9134  acc: 0.8389    test auc: 0.9128  acc: 0.8384
epoch 17    train auc: 0.9505  acc: 0.8776    eval auc: 0.9129  acc: 0.8397    test auc: 0.9123  acc: 0.8381
epoch 18    train auc: 0.9509  acc: 0.8777    eval auc: 0.9140  acc: 0.8401    test auc: 0.9134  acc: 0.8392
epoch 19    train auc: 0.9516  acc: 0.8787    eval auc: 0.9136  acc: 0.8398    test auc: 0.9129  acc: 0.8386
```

4. So my problem lies with where I preprocessed the data because in wang's paper they managed to change the order of the user id's themselves such that they were all ordered from the first user (even if it had user id 1000 for instance) which was now set to a new id 0 representing a user with the positive item set. 

This hypothesis has been rejected because even ordering the data doesn't have an effect on performance

4. next hypothesis is, does embeddings have to do with it? When wang preprocessed the movie lens data set did both the train and cross data splits still preserve the number of unique users and unique items

**Questions:**

**Insights:**
1. the higher the auc the more accurate the model is in classifying the 0 class as a 0 class and the 1 class as a 1 class for instance in a binary classification task. The more it is closer to one the more accurate it is the more it is closer to 0 the more it is inaccurate, if it is closer to 0.5 it means the model has no class separation capacity whatsoever
2. My hypothesis is why precision@k, recall@k, accuracy@k, and f1@k is used in binary framed recommender systems is because the positively interacted upon items labeled as 1 and the unrated items of the users that have rated at least 1 positive item labeled 0

For example, the user has watched 6 movies, and in the first recommendation list, 2 of them are relevant. In the second list, 1 of them are relevant, the meaning of the two relevant movies in the former user is the items he/she has had a positive interaction with or will have a positive interaction with

```
______|item 1|item 2|item 3|item 4|item 5|
user 1|  1   |  0   |  0   |  0   |  1   |
------|------|------|------|------|------|
user 2|  0   |  1   |  1   |  1   |  0   |
------|------|------|------|------|------|
user 3|  1   |  1   |  0   |  1   |  0   |
------|------|------|------|------|------|
user 4|  1   |  0   |  1   |  1   |  1   |
------|------|------|------|------|------|
user 5|  0   |  1   |  0   |  1   |  0   |
```

In training say for user 1 we learned to predict properly the interaction between this user and item 1 item 2 and item 4 as our part training set, which are 1, 0, and 0. And we wanted to predict the rest of the items of user 1 which are items 3 and 5 which have interactions 0 and 1 respectively. Should the model hypothetically not overfit then in our cross validation data if we predict 1 correctly as the interaction between item 5 and user 1 then we would have now recommended an item that they may potentially like

3. It could be possible that even if movielens had 6000+ plus and 3000+ items that the reason why our models did not stall was becuase user-item matrix was not sparse. It could be that the reason why our model was stalling was because our user-item rating matrix was too sparse. An experiment that I could execute is to compare whether the movelens dataset is indeed dense in data and whether the juris-300k or juris-600k dataset is sparse.

Should such expectations come to fruition it would mean that my hypothesis of the model performing well on movielens due to it not being sparse and the model not performing well on juris-300k/600k due to it being sparse would be correct and thus lead to the key conclusion that our dataset juris300k and juris-600k are in need of resynthesizing for the final time.

**Articles:**
1. Evaluating recommender systems
* https://neptune.ai/blog/how-to-test-recommender-system
* https://www.shaped.ai/blog/evaluating-recommendation-systems-part-1


## Analyzing embeddings
**Prerequisites to do:**

**To do:**

**Questions:**


**Insights:**

**Articles:**
1. An algorithm similar to (or based on) K-means that do not require the 'k' number of clusters
* https://stats.stackexchange.com/questions/319807/an-algorithm-similar-to-or-based-on-k-means-that-do-not-require-the-k-number

2. Introduction to Embedding, Clustering, and Similarity
* https://towardsdatascience.com/introduction-to-embedding-clustering-and-similarity-11dd80b00061

3. K Means Clustering on High Dimensional Data.
* https://medium.com/swlh/k-means-clustering-on-high-dimensional-data-d2151e1a4240

4. How to transform a 3d arrays into a dataframe in python
* https://stackoverflow.com/questions/35525028/how-to-transform-a-3d-arrays-into-a-dataframe-in-python



# Named Entity Recognition Model building for creating knowledge graph
## Annotation:
**Preqrequisites to do:**
1. use each created and properly separated .txt files for the NER annotator
use https://tecoholic.github.io/ner-annotator/ for annotating organized text files manually
2. annotate manually and save which will result in a .json file with format:
{
    "classes":["CITATION", ... ,"ORGANIZATION"],
    "annotations":[
        ["\" LABOR CIRCULAR ON-LINE No. 61 Series of 1998 TOPIC At a Glance PETITIONS FOR CERTIORARI UNDER RULE 65 OF THE RULES OF COURT\r", {"entities":[[2,31,"CIRCULAR"],[32,46,"SERIES"],[65,89,"PETITION"],[96,103,"RULE"]]}],
        ...
        ["xxx\"\" \"\r",{"entities":[]}],
        ["",{"entities":[]}]
    ]
}
3. create a parser that will take all the annotations arrays of each text file, extract each element and plcae it into one final data file e.g.
[
    ["sentence/line/string 1", {"entities":[(<start index>, <end index>, "<entity type>"), ..., (<start index>, <end index>, "<entity type>")]}],
    ["sentence/line/string 2", {"entities":[(<start index>, <end index>, "<entity type>"), ..., (<start index>, <end index>, "<entity type>")]}],
    ...,
    ["sentence/line/string n", {"entities":[(<start index>, <end index>, "<entity type>"), ..., (<start index>, <end index>, "<entity type>")]}],
]

**To do:**
1. 

**Questions:**
1. 

**Insights:**
1. 

**Conclusions:**
1. 

**Side notes:**
1. 

## Training Model:
**Prerequisites to do:**
1. sample dat for named entity recognition

// TRAIN_DATA = [
//     ("Apple is looking at buying U.K. startup for $1 billion", {"entities": [(0, 5, "ORG"), (29, 32, "GPE"), (36, 46, "MONEY")]}),
//     ("John lives in New York City and works for IBM", {"entities": [(0, 4, "PERSON"), (16, 29, "GPE"), (43, 46, "ORG")]}),
//     ("The Mona Lisa is a painting by Leonardo da Vinci", {"entities": [(4, 14, "WORK_OF_ART"), (25, 42, "PERSON")]}),
//     ("President Biden visited Detroit to talk about job opportunities", {"entities": [(10, 15, "PERSON"), (23, 30, "GPE")]}),
//     ("The Great Barrier Reef is located off the coast of Australia", {"entities": [(4, 23, "LOC"), (36, 45, "GPE")]}),
// ]

// you'd have to create a dataset like the above so...
// [
//     ["\" LABOR CIRCULAR ON-LINE No. 61 Series of 1998 TOPIC At a Glance PETITIONS FOR CERTIORARI UNDER RULE 65 OF THE RULES OF COURT\r", {"entities":[[2,31,"CIRCULAR"],[32,46,"SERIES"],[65,89,"PETITION"],[96,103,"RULE"]]}],
//     ["FROM DECISIONS OF THE NLRC NOW TO BE INITIALLY FILED WITH THE COURT OF APPEALS AND NO LONGER DIRECTLY WITH THE SUPREME COURT\r", {"entities":[[22,26,"ORGANIZATION"],[62,78,"COURT"],[111,124,"COURT"]]}]
// ]

{
    "classes":["CITATION","AMOUNT","COMPANY","CONSTRAINT","COPYRIGHT","COURT","DATE","DEFINITION","DISTANCE","DURATION","GEOENTITY","PERCENT","REGULATION","TRADEMARK","JUDGEMENT","GAZETTE","PROCEEDINGS","ARTICLE","SECTION","CLAUSE","PARAGRAPH","DEFENDANT","PROSECUTOR","APPEAL","APPELANT","PLAINTIFF","INVOLVED ENTITY","ADVOCATE","LEARNED COUNSEL","ROLE","JUDGE","OFFENCE","ACCUSATION","OBJECTION","JURISDICTION","PENALTY","COMPENSATION","EVIDENCE","EVIDENCE DESCRIPTION","ACT","CIRCULAR","SERIES","CASE","GENERAL REGISTRY NUMBER","PETITION","RULE","ORGANIZATION"],
    "annotations":[
        ["\" LABOR CIRCULAR ON-LINE No. 61 Series of 1998 TOPIC At a Glance PETITIONS FOR CERTIORARI UNDER RULE 65 OF THE RULES OF COURT\r", {"entities":[[2,31,"CIRCULAR"],[32,46,"SERIES"],[65,89,"PETITION"],[96,103,"RULE"]]}],
        ["FROM DECISIONS OF THE NLRC NOW TO BE INITIALLY FILED WITH THE COURT OF APPEALS AND NO LONGER DIRECTLY WITH THE SUPREME COURT\r", {"entities":[[22,26,"ORGANIZATION"],[62,78,"COURT"],[111,124,"COURT"]]}],
        ["[en banc]\r",{"entities":[]}],
        ["[New Interpretation of \"\"Appeals\"\" from NLRC Decisions]\r",{"entities":[]}],
        ["Case Title:\r",{"entities":[]}],
        ["ST. MARTIN FUNERAL HOME VS. NATIONAL LABOR RELATIONS COMMISSION, ET AL.\r",{"entities":[[0,23,"PLAINTIFF"],[28,70,"ORGANIZATION"]]}],
        ["[G. R. No. 130866, September 16, 1998]\r",{"entities":[[0,38,"GENERAL REGISTRY NUMBER"]]}],
        ["[en banc]\r",{"entities":[]}],
        ["FACTS & RULING OF THE COURT:\r",{"entities":[]}],
        ["The Supreme Court [en banc] did not rule on the factual issues of the case but instead re-examined, inter alia, Section 9 of Batas Pambansa Bilang 129, as amended by Republic Act No. 7902 [effective March 18, 1995] on the issue of where to elevate on appeal the decisions of the National Labor Relations Commission [NLRC].\r",{"entities":[[0,17,"COURT"],[112,121,"SECTION"],[125,150,"ACT"],[166,187,"ACT"],[199,213,"DATE"],[279,321,"ORGANIZATION"]]}],["The High Court remanded the case to the Court of Appeals consistent with the new ruling enunciated therein that the \"\"appeals\"\" contemplated under the law from the decisions of the National Labor Relations Commission to the Supreme Court should be interpreted to mean \"\"petitions for certiorari under Rule 65\"\" and consequently, should no longer be brought directly to the Supreme Court but initially to the Court of Appeals.\r",{"entities":[[0,14,"COURT"],[40,56,"COURT"],[181,216,"ORGANIZATION"],[224,237,"COURT"],[268,294,"PETITION"],[301,310,"RULE"],[373,386,"COURT"],[408,424,"COURT"]]}],["Before this new en banc ruling, the Supreme Court has consistently held that decisions of the NLRC may be elevated directly to the Supreme Court only by way of a special civil action for certiorari under Rule 65. There was no ruling allowing resort to the Court of Appeals.\r",{"entities":[[36,49,"COURT"],[94,98,"ORGANIZATION"],[131,144,"COURT"],[204,212,"RULE"],[256,272,"COURT"]]}],["In support of this new view, the Supreme Court ratiocinated, insofar as pertinent, as follows: \"\"While we do not wish to intrude into the Congressional sphere on the matter of the wisdom of a law, on this score we add the further observations that there is a growing number of labor cases being elevated to this Court which, not being a trier of fact, has at times been constrained to remand the case to the NLRC for resolution of unclear or ambiguous factual findings; that the Court of Appeals is procedurally equipped for that purpose, aside from the increased number of its competent divisions; and that there is undeniably an imperative need for expeditious action on labor cases as a major aspect of constitutional protection to labor.\r",{"entities":[[33,46,"COURT"],[408,412,"ORGANIZATION"],[479,495,"COURT"]]}],
        ["\"\"Therefore, all references in the amended Section 9 of B. P. No. 129 to supposed appeals from the NLRC to the Supreme Court are interpreted and hereby declared to mean and refer to petitions for certiorari under Rule 65. Consequently, all such petitions should henceforth be initially filed in the Court of Appeals in strict observance of the doctrine on the hierarchy of courts as the appropriate forum for the relief desired.\r",{"entities":[[43,52,"SECTION"],[56,69,"ACT"],[99,103,"ORGANIZATION"],[111,124,"COURT"],[182,206,"PETITION"],[213,221,"RULE"],[299,315,"COURT"]]}],
        ["xxx\"\" \"\r",{"entities":[]}],
        ["",{"entities":[]}]
    ]
}

**To do:**
1. 

**Questions:**
1. 

**Insights:**
1. 

**Conclusions:**
1. 

**Side notes:**
1. 



# Things I Learned:
1. batch size can affect performance of model on validation set
2. hyper-parameters with n_features of 32, n_epochs of 200, rec_alpha of 0.0003, --rec_lambda of 1, --regularization of "L2", and --batch_size of 8192 seem to give good baseline results for the FM (factorization machine) model



# References:
LINK_TO_PAPER, LINK_TO_PAPERS_GITHUB, CITATION
1. https://www.researchgate.net/publication/333072348_Multi-Task_Feature_Learning_for_Knowledge_Graph_Enhanced_Recommendation/stats, https://github.com/hwwang55/MKR, Wang, Hongwei & Zhang, Fuzheng & Zhao, Miao & Li, Wenjie & Xie, Xing & Guo, Minyi. (2019). Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation. WWW '19: The World Wide Web Conference. 2000-2010. 10.1145/3308558.3313411. 

2. https://www.researchgate.net/publication/358851413_DFM-GCN_A_Multi-Task_Learning_Recommendation_Based_on_a_Deep_Graph_Neural_Network, https://github.com/SSSxCCC/Recommender-System, Xiao, Yan & Li, Congdong & Liu, Vincenzo. (2022). DFM-GCN: A Multi-Task Learning Recommendation Based on a Deep Graph Neural Network. Mathematics. 10. 721. 10.3390/math10050721.

3. https://www.kaggle.com/code/colinmorris/embedding-layers?fbclid=IwAR0WuU4rP6M5Mz92jkrEH-sau17G11MA__c1ndMoi7gnfpq4xne38QQbLZs

4. https://www.researchgate.net/publication/332750505_Knowledge_Graph_Convolutional_Networks_for_Recommender_Systems, https://github.com/hwwang55/KGCN, Wang, Hongwei & Zhao, Miao & Xie, Xing & Li, Wenjie & Guo, Minyi. (2019). Knowledge Graph Convolutional Networks for Recommender Systems. 10.1145/3308558.3313417. 