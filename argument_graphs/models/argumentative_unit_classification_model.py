import os
import logging

import torch
import torch.nn as nn

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
      Transformers' `AutoTokenizer` and `AutoModelForSequenceClassification`;
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
        tokenizer: str = DEFAULT_TOKENIZER,
        model: str = DEFAULT_MODEL,
        num_labels: int = DEFAULT_NUM_LABELS,
    ):
        super(SequenceClassificationModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=num_labels)

    def forward(self, text, label):
        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model(**inputs, labels=label)
        logits, loss = output.logits, output.loss
        pred = logits.argmax().item()
        return (pred, loss)

class ArgumentativeUnitClassificationModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super(ArgumentativeUnitClassificationModel, self).__init__()
        if hparams:
            # Cannot reassign self.hparams in pl.LightningModule; must use update()
            # https://github.com/PyTorchLightning/pytorch-lightning/discussions/7525
            self.hparams.update(hparams)

        self.model = SequenceClassificationModel(
            tokenizer=self.hparams.get("tokenizer", DEFAULT_TOKENIZER),
            model=self.hparams.get("model", DEFAULT_MODEL),
            num_labels=self.hparams.get("num_labels", DEFAULT_NUM_LABELS)
        )

        # When reloading the model for evaluation and inference, we will need
        # the hyperparameters as they are now
        self.save_hyperparameters()

    # Required for pl.LightningModule
    def forward(self, text, label):
        return self.model(text, label)

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
    def training_step_end(self, batch_parts):
        """
        Aggregates results when training using a strategy that splits data
        from each batch across GPUs (e.g. data parallel)
        Note that training_step returns a loss, thus batch_parts returns a list
        of 2 loss values (e.g. if there are 2 GPUs being used)
        """
        return sum(batch_parts) / len(batch_parts)

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
        # optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=0.9)
        return optimizer

if __name__ == "__main__":
    model = SequenceClassificationModel()
    print(model)

    model = ArgumentativeUnitClassificationModel()
    print(model)
