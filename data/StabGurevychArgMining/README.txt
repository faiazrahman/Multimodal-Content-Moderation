The argument mining dataset from Stab and Gurevych is available at the following GitHub repo:

https://github.com/textmining-project/ArgumentMining-Backend

In case that link is no longer available, we also provide a fork:

https://github.com/faiazrahman/ArgumentMining-Backend

The annotated data for the argumentative unit classification task is located in the bratessays/ folder. Copy that folder into `data/StabGurevychArgMining/`, and rename it to `annotated_data`. (This renaming step is needed; our data preprocessing code assumes that this data is located at `data/StabGurevychArgMining/annotated_data`.)

The annotated data is in the `*.ann` files. Note that this is a tab-separated text file (like a .tsv), where the second column is the label and the fifth column is the argumentative unit (i.e. sentence). Also note that there are many `*.ann` files (one for each original document), and they will be aggregated into a single .tsv file containing all of the relevant data.

Labels
    Claim
    MajorClaim
    Premise

Note that the annotated data has other labels, like "Stance", "supports", and so on. However, we will only be using the above three labels since we are using this data (along with AMPERSAND) to train an argumentative unit classification model.

Additionally, we will assign both "Claim" and "MajorClaim" to the label 1 (for claim), and "Premise" to the label 2 (for premise), matching the AMPERSAND data. (Note that the AMPERSAND data also has the label 0 for non-argumentative units.)
