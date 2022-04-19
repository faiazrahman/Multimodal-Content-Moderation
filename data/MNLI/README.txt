Run the `download_data.sh` shell script to download the data from its URL and unzip it.

```
bash download_data.sh
```

This will create a subdirectory `multinli_1.0`, in which the data will then be available as both a json and a text file. We will use `multinli_1.0_train.txt`.

The first line in the file is the header, which has the following column names (in order):
    gold_label
    sentence1_binary_parse
    sentence2_binary_parse
    sentence1_parse
    sentence2_parse
    sentence1
    sentence2
    promptID
    pairID
    genre
    label1
    label2
    label3
    label4
    label5

Thus, we will only use 'sentence1', 'sentence2', and 'gold_label'.

The labels (strings) are as follows:
    neutral
    entailment
    contradiction

For reference, here are the first few lines of `multinli_1.0_train.txt`:

gold_label      sentence1_binary_parse  sentence2_binary_parse  sentence1_parse sentence2_parse sentence1       sentence2       promptID        pairID  genre   label1  label2  label3  label4  label5
neutral ( ( Conceptually ( cream skimming ) ) ( ( has ( ( ( two ( basic dimensions ) ) - ) ( ( product and ) geography ) ) ) . ) )      ( ( ( Product and ) geography ) ( ( are ( what ( make ( cream ( skimming work ) ) ) ) ) . ) )   (ROOT (S (NP (JJ Conceptually) (NN cream) (NN skimming)) (VP (VBZ has) (NP (NP (CD two) (JJ basic) (NNS dimensions)) (: -) (NP (NN product) (CC and) (NN geography)))) (. .)))  (ROOT (S (NP (NN Product) (CC and) (NN geography)) (VP (VBP are) (SBAR (WHNP (WP what)) (S (VP (VBP make) (NP (NP (NN cream)) (VP (VBG skimming) (NP (NN work)))))))) (. .)))   Conceptually cream skimming has two basic dimensions - product and geography.   Product and geography are what make cream skimming work.        31193   31193n  government      neutral
entailment      ( you ( ( know ( during ( ( ( the season ) and ) ( i guess ) ) ) ) ( at ( at ( ( your level ) ( uh ( you ( ( ( lose them ) ( to ( the ( next level ) ) ) ) ( if ( ( if ( they ( decide ( to ( recall ( the ( the ( parent team ) ) ) ) ) ) ) ) ( ( the Braves ) ( decide ( to ( call ( to ( ( recall ( a guy ) ) ( from ( ( triple A ) ( ( ( then ( ( a ( double ( A guy ) ) ) ( ( goes up ) ( to ( replace him ) ) ) ) ) and ) ( ( a ( single ( A guy ) ) ) ( ( goes up ) ( to ( replace him ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )     ( You ( ( ( ( lose ( the things ) ) ( to ( the ( following level ) ) ) ) ( if ( ( the people ) recall ) ) ) . ) )       (ROOT (S (NP (PRP you)) (VP (VBP know) (PP (IN during) (NP (NP (DT the) (NN season)) (CC and) (NP (FW i) (FW guess)))) (PP (IN at) (IN at) (NP (NP (PRP$ your) (NN level)) (SBAR (S (INTJ (UH uh)) (NP (PRP you)) (VP (VBP lose) (NP (PRP them)) (PP (TO to) (NP (DT the) (JJ next) (NN level))) (SBAR (IN if) (S (SBAR (IN if) (S (NP (PRP they)) (VP (VBP decide) (S (VP (TO to) (VP (VB recall) (NP (DT the) (DT the) (NN parent) (NN team)))))))) (NP (DT the) (NNPS Braves)) (VP (VBP decide) (S (VP (TO to) (VP (VB call) (S (VP (TO to) (VP (VB recall) (NP (DT a) (NN guy)) (PP (IN from) (NP (NP (RB triple) (DT A)) (SBAR (S (S (ADVP (RB then)) (NP (DT a) (JJ double) (NNP A) (NN guy)) (VP (VBZ goes) (PRT (RP up)) (S (VP (TO to) (VP (VB replace) (NP (PRP him))))))) (CC and) (S (NP (DT a) (JJ single) (NNP A) (NN guy)) (VP (VBZ goes) (PRT (RP up)) (S (VP (TO to) (VP (VB replace) (NP (PRP him)))))))))))))))))))))))))))) (ROOT (S (NP (PRP You)) (VP (VBP lose) (NP (DT the) (NNS things)) (PP (TO to) (NP (DT the) (JJ following) (NN level))) (SBAR (IN if) (S (NP (DT the) (NNS people)) (VP (VBP recall))))) (. .))) you know during the season and i guess at at your level uh you lose them to the next level if if they decide to recall the the parent team the Braves decide to call to recall a guy from triple A then a double A guy goes up to replace him and a single A guy goes up to replace him You lose the things to the following level if the people recall.
