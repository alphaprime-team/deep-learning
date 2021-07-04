from .models import CNNClassifier, ClassificationLoss, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb


def train(args):
    from os import path

    n_epochs = 100
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_data = load_data('data/train')
    valid_data = load_data('data/valid')

    model = CNNClassifier()
    loss_fn = ClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    for epoch in range(n_epochs):
      iteration = 0
      for train_features, train_labels in train_data:
        train_features, train_labels = train_features.to(device), train_labels.to(device)

        output = model(train_features)
        loss = loss_fn(output, train_labels)

        if train_logger:
            train_logger.add_scalar('loss', loss.mean(), epoch*20 + iteration)
            train_logger.add_scalar('accuracy', accuracy(output, train_labels), epoch*20 + iteration)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1

      if valid_logger:
        for valid_features, valid_labels = valid_data:
            valid_features, valid_labels = valid_features.to(device), valid_labels.to(device)
            output = model(valid_features)

            valid_logger.add_scalar('accuracy', accuracy(output, valid_labels), (epoch+1)*20-1)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
