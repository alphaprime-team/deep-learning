from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch

def train(args):
    n_epochs = 100
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_data = load_data('data/train')
    valid_data = load_data('data/valid')
  
    model = model_factory[args.model]().to(device)
    loss_fn = ClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    for epoch in range(n_epochs):
      for train_features, train_labels in train_data:
        train_features, train_labels = train_features.to(device), train_labels.to(device)

        output = model(train_features)
        loss = loss_fn(output, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
