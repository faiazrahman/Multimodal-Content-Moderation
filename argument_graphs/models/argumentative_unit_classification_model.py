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
    ArgumentativeUnitClassificationModel pl.LightningModule (which acts as a
    wrapper for training and evaluation)

    Note that ArgumentativeUnitClassificationModel should be imported and used
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

    def forward(self, encoded_text, label):
        output = self.model(**encoded_text, labels=label)
        logits, loss = output.logits, output.loss
        pred = torch.argmax(logits, dim=1)
        return (pred, loss)

class ArgumentativeUnitClassificationModel(pl.LightningModule):
    """
    pl.LightningModule for argumentative unit classification; wrapper for
    SequenceClassificationModel nn.Module (see above)

    This model should be imported and used for inference
    ```
    # Run from root ./
    from argument_graphs.models import ArgumentativeUnitClassificationModel
    ```

    hparams
        tokenizer: str = "bert-base-uncased"
        model: str = "bert-base-uncased"
        num_labels: int = 3
        learning_rate: float = 1e-4
    """

    def __init__(self, hparams=None):
        super(ArgumentativeUnitClassificationModel, self).__init__()
        if hparams:
            # Cannot reassign self.hparams in pl.LightningModule; must use update()
            # https://github.com/PyTorchLightning/pytorch-lightning/discussions/7525
            self.hparams.update(hparams)

        self.model = SequenceClassificationModel(
            # tokenizer=self.hparams.get("tokenizer", DEFAULT_TOKENIZER),
            model=self.hparams.get("model", DEFAULT_MODEL),
            num_labels=self.hparams.get("num_labels", DEFAULT_NUM_LABELS)
        )

        # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
        self.accuracy = torchmetrics.Accuracy()

        # When reloading the model for evaluation and inference, we will need
        # the hyperparameters as they are now
        self.save_hyperparameters()

    # Required for pl.LightningModule
    def forward(self, text, label):
        return self.model(text, label)

    # Required for pl.LightningModule
    def training_step(self, batch, batch_idx):
        global losses
        # pl.Lightning convention: training_step() defines prediction and
        # accompanying loss for training, independent of forward()
        text, label = batch["text"], batch["label"]

        pred, loss = self.model(text, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print(loss.item())
        losses.append(loss.item())
        return loss

    # Optional for pl.LightningModule
    def validation_step(self, batch, batch_idx):
        text, label = batch["text"], batch["label"]
        pred, loss = self.model(text, label)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.accuracy(pred, label)
        self.log("val_acc", self.accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # Optional for pl.LightningModule
    def validation_epoch_end(self, outs):
        print(f"val_acc_epoch: {self.accuracy}")
        self.log("val_acc_epoch", self.accuracy)

    # Optional for pl.LightningModule
    def test_step(self, batch, batch_idx):
        text, label = batch["text"], batch["label"]
        pred, loss = self.model(text, label)
        accuracy = torch.sum(pred == label).item() / (len(label) * 1.0)
        output = {
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy).cuda()
        }
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", torch.tensor(accuracy).cuda(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print(loss.item(), output['test_acc'])
        return output

    # Optional for pl.LightningModule
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["test_acc"] for x in outputs]).mean()
        logs = {
            'test_loss': avg_loss,
            'test_acc': avg_accuracy
        }

        # pl.LightningModule has some issues displaying the results automatically
        # As a workaround, we can store the result logs as an attribute of the
        # class instance and display them manually at the end of testing
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1088
        self.test_results = logs

        return {
            'avg_test_loss': avg_loss,
            'avg_test_acc': avg_accuracy,
            'log': logs,
            'progress_bar': logs
        }

    # Required for pl.LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.get("learning_rate", LEARNING_RATE)
        )
        return optimizer
