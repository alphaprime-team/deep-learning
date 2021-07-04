from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):
    # This is a strongly simplified training loop
    for epoch in range(10):
        torch.manual_seed(epoch)
        accuracies = torch.tensor([])
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            accuracies = torch.cat([accuracies, dummy_train_accuracy])
            train_logger.add_scalar('loss', dummy_train_loss, epoch*20 + iteration)
        train_logger.add_scalar('accuracy', accuracies.mean(), global_step=(epoch+1)*20-1)
        torch.manual_seed(epoch)
        accuracies = torch.tensor([])
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            accuracies = torch.cat([accuracies, dummy_validation_accuracy])
        valid_logger.add_scalar('accuracy', accuracies.mean(), (epoch+1)*20-1)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
