from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import csv

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = []
        self.dataset_path = dataset_path

        with open(dataset_path + '/labels.csv') as csv_file:
          datareader = csv.DictReader(csv_file)
          for row in datareader:
            self.data.append((row['label'], row['file']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.dataset_path + '/' + self.data[idx][1])
        image_to_tensor = transforms.ToTensor()
        imgTensor = image_to_tensor(img)

        label = LABEL_NAMES.index(self.data[idx][0])

        return (imgTensor, label)


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
