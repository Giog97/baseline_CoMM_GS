import json
from pathlib import Path
from typing import Union

import PIL
import torch
from torch.utils.data import Dataset


class CC3MLLaVaDataset(Dataset):
    def __init__(self, dataroot, preprocess, return_image=False):
        dataroot = Path(dataroot)
        dataset_path = dataroot / 'LLaVA-CC3M-595K'

        self.preprocess = preprocess
        self.dataset_path = dataset_path
        self.return_image = return_image

        # Get metadata
        with open(dataset_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

    def __getitem__(self, index):
        data = {}
        image_path = self.dataset_path / self.metadata[index]['image']
        image_name = self.metadata[index]['image']
        data['image_name'] = image_name
        if self.return_image:
            image = PIL.Image.open(self.dataset_path / image_path).convert('RGB')
            image = self.preprocess(image)
            data['image'] = image
        text = self.metadata[index]['caption']
        data['text'] = text
        return data

    def __len__(self):
        return len(self.metadata)


if __name__ == '__main__':
    from tqdm import tqdm
    import clip

    _, preprocess = clip.load('ViT-B/32')
    dataset = CC3MLLaVaDataset('/andromeda/datasets', preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=12)
    for data in tqdm(loader):
        pass
