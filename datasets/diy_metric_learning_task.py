import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ==================================
# Data processing: Pre-classification and generating image pairs
# ==================================
def get_cifar10_class_indices(dataset):
    """
    Iterate through the dataset and return a dictionary where key is the class label
    and value is the list of indices of all images in that class.
    """
    print("Preprocessing CIFAR10 dataset, classifying image indices by class...")
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    print("Preprocessing complete.")
    return class_indices

class Cifar10PairsDataset(torch.utils.data.Dataset):
    """
    Custom dataset for generating positive and negative sample pairs for Siamese network.
    """
    def __init__(self, original_dataset, class_indices, classes_to_use:set, transform=None):
        self.original_dataset = original_dataset
        self.class_indices = {k: v for k, v in class_indices.items() if k in classes_to_use}
        self.classes_to_use = list(self.class_indices.keys())
        self.transform = transform
        self.pairs = self._create_pairs()

    def _create_pairs(self):
        pairs = []
        # Get all image indices within selected classes
        all_indices_in_scope = [idx for class_label in self.classes_to_use for idx in self.class_indices[class_label]]

        for idx1 in all_indices_in_scope:
            label1 = self.original_dataset.targets[idx1]

            # Create a positive sample pair (label=0.0)
            idx2 = random.choice(self.class_indices[label1])
            while idx1 == idx2: # Ensure not the same image
                idx2 = random.choice(self.class_indices[label1])
            pairs.append((idx1, idx2, 0.0))

            # Create a negative sample pair (label=1.0)
            neg_class_label = random.choice([c for c in self.classes_to_use if c != label1])
            neg_idx2 = random.choice(self.class_indices[neg_class_label])
            pairs.append((idx1, neg_idx2, 1.0))
        
        random.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        idx1, idx2, label = self.pairs[index]
        
        img1 = self.original_dataset.data[idx1]
        img2 = self.original_dataset.data[idx2]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(label, dtype=torch.float32)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        euclidean_distance = F.pairwise_distance(embedding1, embedding2, keepdim=True)

        # Loss for positive sample pairs
        loss_positive = (1 - label) * torch.pow(euclidean_distance, 2)
        # Loss for negative sample pairs
        loss_negative = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        loss_contrastive = torch.mean(loss_positive + loss_negative)

        return loss_contrastive