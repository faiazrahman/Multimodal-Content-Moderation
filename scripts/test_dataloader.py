import torch
from torch.utils.data import DataLoader
import torchvision
from sentence_transformers import SentenceTransformer

from dataloader import Modality, MultimodalDataset

def _build_image_transform(image_dim=224):
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

TEXT_EMBEDDER = SentenceTransformer('all-mpnet-base-v2')
IMAGE_TRANSFORM = _build_image_transform()

def test_text_image_dataset():
    print("Testing MultimodalDataset for text-image modality...")

    train_dataset = MultimodalDataset(
        from_preprocessed_dataframe="./data/Fakeddit/train__text_image__dataframe.pkl",
        modality=Modality.TEXT_IMAGE,
        text_embedder=TEXT_EMBEDDER,
        image_transform=IMAGE_TRANSFORM,
        num_classes=6
    )

    assert(len(train_dataset) > 0)
    assert(isinstance(train_dataset[0], dict))

    batch_size = 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0
    )

    for batch in train_loader:
        text, image, label = batch["text"], batch["image"], batch["label"]
        assert(len(list(text[0].shape)) == 1)   # Text is a 1D tensor with 768 values
        assert(len(list(image[0].shape)) == 3)  # Image has 3 channels of 224 by 224 pixels
        assert(len(list(label[0].shape)) == 0)  # Label is a single integer (i.e. dimension of 0)
        break

    print("Testing MultimodalDataset for text-image modality... PASSED")

def test_text_image_dialogue_dataset():
    print("Testing MultimodalDataset for text-image-dialogue modality...")

    train_dataset = MultimodalDataset(
        from_preprocessed_dataframe="./data/Fakeddit/train__text_image_dialogue__dataframe.pkl",
        modality=Modality.TEXT_IMAGE_DIALOGUE,
        text_embedder=TEXT_EMBEDDER,
        image_transform=IMAGE_TRANSFORM,
        num_classes=6
    )

    assert(len(train_dataset) > 0)
    assert(isinstance(train_dataset[0], dict))

    batch_size = 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0
    )

    for batch in train_loader:
        text, image, dialogue, label = batch["text"], batch["image"], batch["dialogue"], batch["label"]
        assert(len(list(text[0].shape)) == 1)   # Text is a 1D tensor with 768 values
        assert(len(list(image[0].shape)) == 3)  # Image has 3 channels of 224 by 224 pixels
        assert(len(list(dialogue[0].shape)) == 1) # Dialogue summary is a 1D tensor with 768 values
        assert(len(list(label[0].shape)) == 0)  # Label is a single integer (i.e. dimension of 0)
        break

    print("Testing MultimodalDataset for text-image-dialogue modality... PASSED")

if __name__ == "__main__":
    test_text_image_dataset()
    test_text_image_dialogue_dataset()
    print("test_dataloader.py: ALL TESTS PASSED")
