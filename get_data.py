import numpy as np
import torch
from pathlib import Path

label_tags = ["course", "department", "faculty", "other", "project", "staff", "student"]


def read_data(dir_path):
    dir_path = Path(dir_path)
    texts = []
    labels = []
    for label_path in label_tags:
        for html_file in (dir_path/label_path).glob('**/*.html'):
            texts.append(html_file.read_text())
            labels.append(label_tags.index(label_path))

    return texts, labels


class web_dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



