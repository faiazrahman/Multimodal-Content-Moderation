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

    def forward(self, encoded_text, label=None):
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
    # Scripts which use this model should be run from root ./
    # Import this model in those scripts as follows
    from argument_graphs.submodels import ArgumentativeUnitClassificationModel
    ```

    Calling the model for inference
    - This model expects as input the encoded text, i.e. the output of
      `transformers.AutoTokenizer()` (which itself is a dict of `input_ids`,
      `token_type_ids`, and `attention_mask` tensors)
    - Thus, when running inference on this model, first tokenize the input
      using `transformers.AutoTokenizer()`, then run it through the model to
      get the prediction
    - Since this is a `pl.LightningModule`, this in turn inherits
      `torch.nn.Module`, which implements `__call__()`, which allows you to
      run inference as follows
    ```
        text = [ "...", "...", "..." ]  # List[str]
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = ArgumentativeUnitClassificationModel()
        encoded_inputs = tokenizer(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        pred = model(encoded_inputs)[0]
    ```
    - Note that the model returns a tuple `(pred, loss)` --- for inference, we
      only need the prediction, so you can unpack it simply via
      `model(encoded_inputs)[0]`
    - Also note that the pred is a tensor of ints (where each int is a label
      0: non-argumentative unit, 1: claim, 2: premise); this tensor can be
      moved onto CPU by casting it to a list via `pred = pred.tolist()`

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
    def forward(self, text, label=None):
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
        # Default to Adam if optimizer is not specified
        if "optimizer" not in self.hparams or self.hparams["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.get("learning_rate", LEARNING_RATE)
            )
        elif self.hparams["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.get("learning_rate", LEARNING_RATE),
                momentum=self.hparams.get("sgd_momentum", 0.9)
            )
        return optimizer
