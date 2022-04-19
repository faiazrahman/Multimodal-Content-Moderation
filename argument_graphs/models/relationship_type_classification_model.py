import os
import logging

import torch
import torch.nn as nn
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_TOKENIZER = "bert-base-uncased"
DEFAULT_MODEL = "bert-base-uncased"
DEFAULT_NUM_LABELS = 3
LEARNING_RATE = 1e-4

losses = []

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

print("CUDA available:", torch.cuda.is_available())

class SequenceClassificationModel(nn.Module):
    """
    nn.Module for sequence classification, used internally by the
    RelationshipTypeClassificationModel pl.LightningModule (which acts as a
    wrapper for training and evaluation)

    Note that RelationshipTypeClassificationModel should be imported and used
    for inference

    Additional Notes
    - The sequence classification model is implemented using Hugging Face
      Transformers' `AutoTokenizer` and `AutoModelForSequenceClassification`
      (https://huggingface.co/docs/transformers/model_doc/auto);
      thus, the specified tokenizer and model must be in Hugging Face's
      models repository
    - Default model is BERT (https://arxiv.org/abs/1810.04805), specifically
      the pretrained `bert-base-uncased`; i.e. BERT with a sequence
      classification head
    - We are doing single-label classification (i.e. each example only has one
      label) with `num_labels = 3` (i.e. there are three possible labels); this
      is specified to `AutoModelForSequenceClassification` via `num_labels`
    """

    def __init__(
        self,
        # tokenizer: str = DEFAULT_TOKENIZER,
        model: str = DEFAULT_MODEL,
        num_labels: int = DEFAULT_NUM_LABELS,
    ):
        super(SequenceClassificationModel, self).__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, model_max_length=512)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=num_labels)

    def forward(self, encoded_text, label=None):
        output = self.model(**encoded_text, labels=label)
        logits, loss = output.logits, output.loss
        pred = torch.argmax(logits, dim=1)
        return (pred, loss)

class RelationshipTypeClassificationModel(pl.LightningModule):
    """
    pl.LightningModule for relationship type classification; wrapper for
    SequenceClassificationModel nn.Module (see above)

    This model should be imported and used for inference
    ```
    # Scripts which use this model should be run from root ./
    # Import this model in those scripts as follows
    from argument_graphs.models import RelationshipTypeClassificationModel
    ```

    Calling the model for inference
    TODO: Add docs

    hparams
        tokenizer: str = "bert-base-uncased"
        model: str = "bert-base-uncased"
        num_labels: int = 3
        learning_rate: float = 1e-4
    """
