import os
import logging

import torch
import torch.nn as nn
import torchvision
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import transformers
from sentence_transformers import SentenceTransformer

# NOTE: These should be initialized in the calling script
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
DROPOUT_P = 0.1

RESNET_OUT_DIM = 2048

losses = []

logging.basicConfig(level=logging.INFO) # DEBUG, INFO, WARNING, ERROR, CRITICAL

print("CUDA available:", torch.cuda.is_available())

class TextImageResnetModel(nn.Module):

    def __init__(
            self,
            num_classes,
            loss_fn,
            text_module,
            image_module,
            text_feature_dim,
            image_feature_dim,
            fusion_output_size,
            dropout_p,
            hidden_size=512,
            fusion_method="early-concat", # "early-concat" | "low-rank"
        ):
        super(TextImageResnetModel, self).__init__()
        self.text_module = text_module
        self.image_module = image_module
        self.fusion = torch.nn.Linear(in_features=(text_feature_dim + image_feature_dim),
            out_features=fusion_output_size)
        # self.fc = torch.nn.Linear(in_features=fusion_output_size, out_features=num_classes)
        self.fc1 = torch.nn.Linear(in_features=fusion_output_size, out_features=hidden_size)
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, text, image, label):
        text_features = torch.nn.functional.relu(self.text_module(text))
        image_features = torch.nn.functional.relu(self.image_module(image))
        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.dropout(
            torch.nn.functional.relu(self.fusion(combined)))
        # logits = self.fc(fused)
        hidden = torch.nn.functional.relu(self.fc1(fused))
        logits = self.fc2(hidden)

        # nn.CrossEntropyLoss expects raw logits as model output, NOT torch.nn.functional.softmax(logits, dim=1)
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        pred = logits
        loss = self.loss_fn(pred, label)

        return (pred, loss)

class TextImageResnetMMFNDModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super(TextImageResnetMMFNDModel, self).__init__()
        if hparams:
            # Cannot reassign self.hparams in pl.LightningModule; must use update()
            # https://github.com/PyTorchLightning/pytorch-lightning/discussions/7525
            self.hparams.update(hparams)

        self.embedding_dim = self.hparams.get("embedding_dim", 768)
        self.text_feature_dim = self.hparams.get("text_feature_dim", 300)
        self.image_feature_dim = self.hparams.get("image_feature_dim", self.text_feature_dim)

        text_module = torch.nn.Linear(
            in_features=self.embedding_dim, out_features=self.text_feature_dim)

        image_module = torchvision.models.resnet152(pretrained=True)
        # Overwrite last layer to get features (rather than classification)
        image_module.fc = torch.nn.Linear(
            in_features=RESNET_OUT_DIM, out_features=self.image_feature_dim)

        self.model = TextImageResnetModel(
            num_classes=self.hparams.get("num_classes", NUM_CLASSES),
            loss_fn=torch.nn.CrossEntropyLoss(),
            text_module=text_module,
            image_module=image_module,
            text_feature_dim=self.text_feature_dim,
            image_feature_dim=self.image_feature_dim,
            fusion_output_size=self.hparams.get("fusion_output_size", 512),
            dropout_p=self.hparams.get("dropout_p", DROPOUT_P)
        )

        # Metrics for evaluation
        # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
        self.accuracy = torchmetrics.Accuracy()
        # NOTE: You cannot define this as `self.precision`, or else it will
        # raise TypeError: cannot assign 'int' as child module 'precision'
        # (We use `self.precision_metric` instead)
        self.precision_metric = torchmetrics.Precision()
        self.recall = torchmetrics.Recall()
        self.f1_score = torchmetrics.F1Score()
        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            # ConfusionMatrix() takes a required positional argument: num_classes
            self.hparams.get("num_classes", NUM_CLASSES)
        )

        # When reloading the model for evaluation, we will need the
        # hyperparameters as they are now
        self.save_hyperparameters()

    # Required for pl.LightningModule
    def forward(self, text, image, label):
        # pl.Lightning convention: forward() defines prediction for inference
        return self.model(text, image, label)

    # Required for pl.LightningModule
    def training_step(self, batch, batch_idx):
        global losses
        # pl.Lightning convention: training_step() defines prediction and
        # accompanying loss for training, independent of forward()
        text, image, label = batch["text"], batch["image"], batch["label"]

        pred, loss = self.model(text, image, label)
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
        of 2 loss values (since there are 2 GPUs being used)
        """
        return sum(batch_parts) / len(batch_parts)

    # Optional for pl.LightningModule
    def test_step(self, batch, batch_idx):
        text, image, label = batch["text"], batch["image"], batch["label"]
        pred, loss = self.model(text, image, label)
        pred_label = torch.argmax(pred, dim=1)
        accuracy = torch.sum(pred_label == label).item() / (len(label) * 1.0) # TODO deprecate

        self.accuracy(pred_label, label)
        self.precision_metric(pred_label, label)
        self.recall(pred_label, label)
        self.f1_score(pred_label, label)
        self.confusion_matrix(pred_label, label)

        output = {
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy).cuda(), # TODO deprecate
            'test_accuracy': self.accuracy,
            'test_precision': self.precision_metric,
            'test_recall': self.recall,
            'test_f1_score': self.f1_score,
            'test_confusion_matrix': self.confusion_matrix,
        }

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", torch.tensor(accuracy).cuda(), on_step=True, on_epoch=True, prog_bar=True, logger=True) # TODO deprecate
        self.log("test_accuracy", self.accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_precision", self.precision_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_recall", self.recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1_score", self.f1_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print(loss.item(), output['test_acc']) # TODO: deprecate

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

    @classmethod
    def build_image_transform(cls, image_dim=224):
        image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(image_dim, image_dim)),
            torchvision.transforms.ToTensor(),
            # All torchvision models expect the same normalization mean and std
            # https://pytorch.org/docs/stable/torchvision/models.html
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

        return image_transform
