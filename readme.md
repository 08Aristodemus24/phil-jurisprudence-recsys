# **STILL IN PRODUCTION**
this is the second phase of my undergraduate thesis which will recommend jurisprudence documents to legal practitioners specializing in the labor sector. Based on the paper of Wang, H. et. al. 


# Model Building:
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
    a. mean adder to the predicted ratings
    b. adder of a new user to the user-item rating matrix and user-item interaction matrix

references:
1. https://github.com/hwwang55/MKR, Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation
Hongwei Wang, Fuzheng Zhang, Miao Zhao, Wenjie Li, Xing Xie, and Minyi Guo.
In Proceedings of The 2019 Web Conference (WWW 2019)
2. 