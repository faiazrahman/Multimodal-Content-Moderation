# Multi-Modal Machine Learning Systems for Social Media Content Moderation with Dialogue Summarization and Argument Graphs

_This project was developed by Faiaz Rahman as his senior thesis submitted in partial fulfillment of the requirements for the degree Bachelor of Science in Computer Science at Yale University, in May 2022, advised by Dr. Dragomir R. Radev._

_Note:_ A PDF of the thesis is included in the root directory of this codebase. Refer to the thesis for an overview of the project, related work, data, model architectures for the multi-modal models, the formalization of the ArgSum algorithm, and an explanation of the multi-modal fusion methods, in addition to an overview of the experiments, evaluation, results, analysis, and discussion.

## Abstract

The proliferation of misinformation and hate speech online has created an era of digital disinformation, public mistrust, and even violence, particularly on social media platforms where users can engage in dialogue with such content. Fake news and hate speech exist not only in text form, but also include any accompanying images and video with the original post. Multi-modal models (i.e., those incorporating multiple modalities of data like text and images) offer a powerful approach in detecting such content. Prior work has both developed hate and misinformation datasets for experimentation, and examined different multi-modal representations in general, particularly for text–image data.

Given that user dialogue (e.g., comment threads, Tweet replies, etc.) can often give more insight into the integrity or hatefulness of a post (e.g., by indicating how extreme of a response was garnered, by introducing viewpoints beyond those of the original author, etc.), we investigate methods for modeling and incorporating the dialogue modality into multi-modal models. Specifically, we (1) develop multi-modal models for content moderation tasks (for the modalities of text, image, and dialogue), (2) improve dialogue modeling within those models by introducing ArgSum, an argument graph–based approach to dialogue summarization, and (3) improve the modeling of cross-modal interactions through the multi-modal fusion methods of uni-modal early fusion and low-rank tensor fusion.

Our experiments find that (1) the incorporation of the dialogue modality in multi-modal models improves performance on fake news detection, (2) modeling argumentative structures in dialogues via ArgSum improves both summarization quality and multi-modal model performance, and (3) low-rank tensor fusion is able to better model cross-modal interactions than early fusion. Additionally, we release a public codebase including all of our PyTorch models, our ArgSum software package, and our experiment configuration files, built with an extensible design for future work on hate and misinformation detection.

## Setup

We recommend using a virtual environment via Conda. We have provided an environment YAML file to rebuild the same virtual environment used in our experiments. We use Python 3.6, PyTorch (version 1.11.0 with CuDNN version 8.2.0), and PyTorch Lightning (version 1.6.0). 

We run our experiments on 1–4 NVIDIA GeForce RTX 3090 GPUs with Driver version 470.103.01 and CUDA version 11.4. We run some additional experiments on two NVIDIA K80 GPUs with driver version 465.19.01 and CUDA version 11.3. The same virtual environment works for both CUDA versions.

```
conda env create --file environment.yaml
conda activate mmcm
```

## Overview of Codebase

- **`environment.yaml`** &nbsp; Conda environment file to recreate our virtual environment from

- **`data_preprocessing.py`** &nbsp; Script to run data preprocessing for Fakeddit
- **`dataloader.py`** &nbsp; Contains the `torch.utils.data.Dataset` for fake news detection
- **`data/`** &nbsp; Contains a subdirectory for each of the five datasets with Bash scripts and/or instructions to download each (and additional subdirectories for the aggregated argumentative unit classification and aggregated relationship type classification data)
- **`data_sampling/`** &nbsp; Contains scripts to generate samples of the Fakeddit dataset with an even class distribution

- **`run_training.py`** &nbsp; Runs training for fake news detection models
- **`run_evaluation.py`** &nbsp; Runs evaluation for fake news detection models
- **`run_experiments.py`** &nbsp; Automates multiple training and evaluation runs via Python’s `subprocess` module (i.e., running the shell commands simultaneously, similar to a regular Bash script except it also generates log files with specific filename conventions based on the experiment configs)
- **`utils.py`** &nbsp; Various utilities for running experiments and evaluation (e.g. getting the latest model checkpoint from the trained assets folder, etc.)

- **`models/`**
  - Contains all multi-modal models and uni-modal baseline models for fake news detection
    - Each file in this subdirectory contains a `pl.LightningModule` (which is the model that is imported during training and evaluation) and a PyTorch `nn.Module` (which implements the model architecture and is used internally by the Lightning model)
 
- **`argsum/`** (formerly **`argument_graphs/`**)
  - `argsum/`
    - Contains the `ArgSum` class, which can be instantiated to generate argument graphs, linearized argument graphs, and dialogue summaries via the ArgSum algorithm
  - `data_structures/`
    - Contains the data structures for working with argument graphs; specifically, the classes `ArgumentGraph`, `ArgumentativeUnitNode`, `RelationshipTypeEdge`, `ArgumentativeUnitType`, and `RelationshipType`
  - `modules/`
    - Contains the modules for various steps of the ArgSum algorithm; specifically, `utterance_segmentation.py`, `graph_construction.py`, and `graph_linearization.py`, which contain the `UtteranceToArgumentativeUnitSegmenter`, `ArgumentGraphConstructor`, and `ArgumentGraphLinearizer` classes, respectively
  - `submodels/`
    - Contains the models used during argument graph construction; specifically, `ArgumentativeUnitClassificationModel` and `RelationshipTypeClassification`, which are both `pl.LightningModules` with an underlying PyTorch `nn.Module`
  - `utils/`
    - Contains various utilities for the tokenizers used by ArgSum’s submodels and for generating batches during inference
  - `data_preprocessing.py` &nbsp; Runs data preprocessing for the AMPERSAND, Stab & Gurevych, and MNLI datasets
  - `dataloader.py` &nbsp; Contains the `torch.utils.data.Datasets` for the ArgSum submodels
  - `run_argument_graph_submodel_training.py` &nbsp; Runs training for the ArgSum submodels
  - `run_argument_graph_submodel_evaluation.py` &nbsp; Runs evaluation for the ArgSum submodels
  - `run_hyperparameter_tuning.py` &nbsp; Runs hyperparameter tuning for the ArgSum submodels
  - `run_hyperparameter_tuning_evaluation.py` &nbsp; Runs evaluation for ArgSum submodel hyperparameter trials

- **`dialogue_summarization/`**
  - Contains data preprocessing for SAMSum, and ROUGE evaluation scripts for ArgSum-BART and baseline BART 

- **`fusion/`**
  - Contains data preprocessing, the `torch.utils.data.Dataset`, the model, the train and evaluation scripts, and an automated experiments script for multi-modal hate speech detec- tion, which is used for fusion method experiments
    - Note that these same fusion methods are also implemented in the multi-modal fake news detection models in the root’s `models/`

- **`configs/`**
  - Contains all experiment configurations as YAML files, which are passed to training and evaluation scripts via the `--config` arg

- **`lightning_logs/`**
  - This is where all trained model assets (i.e., checkpoints and hparams) are stored by PyTorch Lightning

- **`logs/`**
  - This is where the logs generated by `run_experiments.py` (and the other automated run scripts in the repo) are stored

## Data

We employ five datasets, which can be grouped into three categories: (i) data used for the content moderation tasks: Fakeddit (Nakamura et al., 2020) and MMHS150K (Gomez et al., 2019); (ii) data used to train the submodels in the ArgSum algorithm (prior to its incorporation in the multi-modal models): AMPERSAND (Chakrabarty et al., 2019), Stab & Gurevych’s argument mining dataset (abbreviated as SGAM in some data preprocessing files in our codebase; Stab and Gurevych, 2014), and MNLI (Williams et al., 2018); and (iii) data used to evaluate ArgSum’s general performance via summarization metrics and fine-tune Transformer models for dialogue summarization for usage in the content moderation tasks: SAMSum (Gliwa et al., 2019).

We have a data preprocessing pipeline for each of these three data categories; this is described in more detail in Appendix A of the thesis.

### Multi-Modal Fake News Detection: Fakeddit

Fakeddit (Nakamura et al., 2020) is a multi-modal dataset consisting of over 1 million samples from multiple categories of fake news, labeled with 2-way (true, fake), 3-way (true, fake with true text, fake with false text), and 6-way (true, satire, false connection, imposter content, manipulated content, misleading content) classification categories to allow for both binary classification and, more interestingly, fine-grained classification. The dataset was collected from a diverse array of topic categories (i.e., subreddits) from Reddit, and includes a post’s text, image, and comment threads. We used a randomly-sampled subset of their train and test datasets with a balanced class distribution (and selecting only examples which were multi-modal, i.e., with text, image, and comment data), consisting of 10,000 training examples and 1,000 evaluation examples.

The Fakeddit dataset is made [publicly available](https://github.com/entitize/Fakeddit) for usage; our codebase has a slightly modified version of their image downloading script.

### Multi-Modal Hate Speech Detection: MMHS150K

MMHS150K (Gomez et al., 2019) is a multi-modal dataset consisting of 150,000 samples of online hate speech (containing text, image, and OCR; note that OCR refers to the text within an image, which was extracted using Optical Character Recognition), labeled with 6-way classification categories (not hate speech, racist, sexist, homophobic, hate towards religion, and other types of hate). The dataset was collected from Twitter, with the tweets having been posted from September 2018 to February 2019. Tweets were collected in real-time (prior to Twitter’s own hate speech filters and moderation being applied, which the authors report at the time of their data collection was based on user reports and thus not instantaneous) and then filtered (e.g., to remove retweets, tweets containing less than three words, and tweets without images).

The MMHS150K data is made [publicly available](https://www.kaggle.com/datasets/victorcallejasf/multimodal-hate-speech) for usage; our codebase includes a lightweight requirements.txt file for creating a virtual environment, installing the dependencies via the pip package manager, and downloading the dataset via the kaggle command-line tool.

Additionally, we attempted to augment the original MMHS150K with dialogue data (specifically by scraping tweet replies to the original tweets and building a dialogue thread for each), using the Twitter API, but found that the Twitter API does not allow access to the tweet objects of suspended users (which was the case for all the hateful tweets). As a result, we use Fakeddit data for all experiments involving dialogue data, and use MMHS150K for selected text–image experiments.

### Argumentative Unit Classification: AMPERSAND and Stab & Gurevych

For argumentative unit classification (i.e., the task of classifying an argumentative unit as a claim, premise, or non-argumentative unit; described in more detail in 5.2.1), we use aggregate data from two datasets.

The [AMPERSAND data](https://github.com/tuhinjubcse/AMPERSAND-EMNLP2019) and the [Stab & Gurevych data](https://github.com/textmining-project/ArgumentMining-Backend) are both made publicly available. In our codebase, we provide a README for both datasets outlining the relevant data files to download and describing the data format. Our data preprocessing pipeline prepares, filters, and aggregates both datasets into a single dataset which can then be used directly by the PyTorch torch.utils.data.Dataset for the argumentative unit classification model.

### Textual Entailment for Relationship Type Classification: MNLI

For relationship type classification (i.e., the task of classifying the directed edge from one argumentative unit node to another as supporting, contradicting, or neutral, described in more detail in 5.2.2), we use MNLI data. MNLI (Williams et al., 2018) is a dataset consisting of 433,000 sentence pairs annotated with textual entailment information (specifically, with labels of `ENTAILMENT`, `CONTRADICTION`, and `NEUTRAL`).

The MNLI data is made [publicly available](https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip). Our codebase includes a Bash script for downloading and processing it.

## Running Experiments

`TODO`

### Multi-Modal Models and Baselines

### Training ArgSum's Submodels

### Incorporating ArgSum and GraphLin into Multi-Modal Models

### Fusion Experiments

### Experiment Settings
