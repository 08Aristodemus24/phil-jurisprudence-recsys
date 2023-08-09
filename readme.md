# **STILL IN PRODUCTION**
this is the second phase of my undergraduate thesis which will recommend jurisprudence documents to legal practitioners specializing in the labor sector. Based on the paper of Wang, H. et. al. 

# Usage:
1. clone repository with `git clone https://github.com/08Aristodemus24/phil-jurisprudence-recsys.git`
2. navigate to directory with `readme.md` and `requirements.txt` file
3. run command; `conda create -n <name of env e.g. phil-jurisprudence-recsys> python=3.10.11`. Note that 3.10.11 must be the python version otherwise packages to be installed would not be compatible with a different python version
4. once environment is created activate it by running command `conda activate`
5. then run `conda activate phil-jurisprudence-recsys`
6. check if pip is installed by running `conda list -e` and checking list
7. if it is there then move to step 8, if not then install `pip` by typing `conda install pip`
8. if `pip` exists or install is done run `pip install -r requirements.txt` in the directory you are currently in

# Model Building:
## Hypotheses/Tests to do
1. <s>see shape of user input in DeepFM model</s>
2. <s>test run</s>
3. <s>label each line of execution in Recommender-System repository particularly in the using deepfm model</s>
4. *priority* create data loader for movie ratings dataset
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
11. because initial factorization machine (FM) architecture of collaborative filtering model already exists, using the the Functional API of tensorflow by using the built-in Model class, I need to build a more tailored version of this Model class that implements this FM architecture
12. I need to build DeepFM architecture, and MKR architecture of these models

## Questions:
1. how do I split the data into training, validation, and testing sets?
2. how do I batch train the model?
3. how do I make predictions with the model
4. how do I implement the AUC, Precision@k, Recall@k, and F1-score in this model?

## Things I Learned:
1. batch size can affect performance of model on validation set
2. hyper-parameters with n_features of 32, n_epochs of 200, rec_alpha of 0.0003, --rec_lambda of 1, --regularization of "L2", and --batch_size of 8192 seem to give good baseline results for the FM (factorization machine) model

## Labor Corpus Juris
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

4. sample dat for named entity recognition

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

5. Things to add for the main collaborative filtering recommender system:
    
## references:
LINK_TO_PAPER, LINK_TO_PAPERS_GITHUB, CITATION
1. https://www.researchgate.net/publication/333072348_Multi-Task_Feature_Learning_for_Knowledge_Graph_Enhanced_Recommendation/stats, https://github.com/hwwang55/MKR, Wang, Hongwei & Zhang, Fuzheng & Zhao, Miao & Li, Wenjie & Xie, Xing & Guo, Minyi. (2019). Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation. WWW '19: The World Wide Web Conference. 2000-2010. 10.1145/3308558.3313411. 

2. https://www.researchgate.net/publication/358851413_DFM-GCN_A_Multi-Task_Learning_Recommendation_Based_on_a_Deep_Graph_Neural_Network, https://github.com/SSSxCCC/Recommender-System, Xiao, Yan & Li, Congdong & Liu, Vincenzo. (2022). DFM-GCN: A Multi-Task Learning Recommendation Based on a Deep Graph Neural Network. Mathematics. 10. 721. 10.3390/math10050721.

3. https://www.kaggle.com/code/colinmorris/embedding-layers?fbclid=IwAR0WuU4rP6M5Mz92jkrEH-sau17G11MA__c1ndMoi7gnfpq4xne38QQbLZs