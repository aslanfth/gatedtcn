import torch
import sys
sys.path.append("..")
from torchvision import datasets, transforms
from gatedtcn.models import GatedTemporalConvNet, TemporalConvNet
import os
import torch.optim as optim
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='Sequential MNIST')
parser.add_argument('--architecture', type=str,
                    help="Architecture name (gated or generic).", default=None, metavar='')
parser.add_argument("--batch_size", type=int, help="Batch size.", default=1, metavar='')
parser.add_argument("--gpu", type=bool, help="If GPU will be used.", default=True, metavar='')
parser.add_argument("--perm", type=bool, help="If the sequences are permuted.", default=False, metavar='')
parser.add_argument("--epochs", type=int, help="Number of epochs.", default=1, metavar='')
parser.add_argument("--save", type=int, help="Determine the interval of saving the models.", default=1, metavar='')
parser.add_argument("--print", type=int, help="Determine the interval of printing the models.", default=1, metavar='')

parser.add_argument("--k", type=int, help="Kernel size.", metavar='')
parser.add_argument("--num_layers", type=int, help="Number of layers.", metavar='')
parser.add_argument("--num_filters", type=int, help="Number of filters in each layer.", metavar='')
parser.add_argument("--lr", type=float, help="Learning rate.", metavar='')
parser.add_argument("--clip", type=float, help="Gradient clip.", metavar='')
parser.add_argument("--dropout", type=float, help="Learning rate.", metavar='')

parser.add_argument("--patience", type=int, help="Scheduler patience.", metavar='')
parser.add_argument("--factor", type=float, help="Scheduler factor.", metavar='')
parser.add_argument("--cooldown", type=int, help="Scheduler cooldown.", metavar='')

args = parser.parse_args()


def data_generator(root, batch_size, permute):
    if os.path.exists(os.path.join(root, "MNIST")):
        train_set = datasets.MNIST(root=root, train=True, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
        test_set = datasets.MNIST(root=root, train=False, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
    else:
        train_set = datasets.MNIST(root=root, train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
        test_set = datasets.MNIST(root=root, train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
    train_data = train_set.data.view(-1, 784)
    test_data = test_set.data.view(-1, 784)
    if permute:
        permutations = torch.randperm(784)
        train_data = train_data[:, permutations]
        test_data = test_data[:, permutations]
    train_data = (train_data - 0.1307 * 255) / 0.3081 / 255
    test_data = (test_data - 0.1307 * 255) / 0.3081 / 255

    train_data = train_data.type(torch.FloatTensor).unsqueeze(1)
    test_data = test_data.type(torch.FloatTensor).unsqueeze(1)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_set.targets),
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_set.targets),
                                              batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


gtcn_run = False
tcn_run = False
root = "./datasets"
batch_size = args.batch_size
gpu = args.gpu

num_classes = 10
num_inputs = 1

run_folder = "runs"
model_folder = "models"
setting = "_MNIST"
permute = args.perm
if permute:
    setting += "_perm"
num_epochs = args.epochs
save_every_epoch = args.save
print_every_epoch = args.print

if args.architecture is None:   # Values from the paper is used to compare the architectures.
    gtcn_run = True
    tcn_run = True
    kernel_size = 8
    tcn_channels = [[25] * 8]
    clip = 0.1
    lr = 0.002
    dropout = 0.2
    scheduler_patience = 5
    scheduler_factor = 0.3
    if permute:
        scheduler_cooldown = 5
    else:
        scheduler_cooldown = 10
else:
    if args.architecture == "gated":
        gtcn_run = True
    elif args.architecture == "generic":
        tcn_run = True
    else:
        raise Exception("Invalid architecture.")
    kernel_size = args.k
    tcn_channels = [[args.num_filters] * args.num_layers]
    clip = args.clip
    lr = args.lr
    dropout = args.dropout
    scheduler_patience = args.patience
    scheduler_factor = args.factor
    scheduler_cooldown = args.cooldown

train_loader, test_loader = data_generator(root=root, batch_size=batch_size, permute=permute)
flat_tcn_channels = [item for sublist in tcn_channels for item in sublist]

if gtcn_run:
    if permute and args.architecture is None:
        lr = 0.01
    model_name = "GTCN_"
    model_name += str(kernel_size) + "_" + str(flat_tcn_channels[0]) + "x" + str(len(flat_tcn_channels)) + "_"
    model_name += str(clip) + "_" + str(lr) + "_" + str(scheduler_patience) + "_" + str(scheduler_factor) + "_"
    model_name += str(scheduler_cooldown) + "_" + str(dropout)

    model = GatedTemporalConvNet(num_inputs=num_inputs, num_channels=flat_tcn_channels, kernel_size=kernel_size,
                                 dropout=dropout, runs_folder=os.path.join(run_folder, model_name + setting),
                                 mode="classification", num_classes=num_classes, gpu=gpu)
    if gpu:
        model.cuda()
    num_params = 0
    for p in model.parameters():
        num_params = num_params + p.numel()
    print("Num GTCN params: " + str(num_params))
    model_name += "_" + str(num_params)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, factor=scheduler_factor,
                                                     cooldown=scheduler_cooldown)
    loss_function = F.nll_loss
    model.fit(num_epoch=num_epochs, train_loader=train_loader, optimizer=optimizer, clip=clip,
              loss_function=loss_function, save_every_epoch=save_every_epoch,
              model_path=os.path.join(model_folder, model_name + setting),
              valid_loader=test_loader, scheduler=scheduler, print_every_epoch=print_every_epoch)

if tcn_run:
    flat_tcn_channels = [2 * channel for channel in flat_tcn_channels]
    model_name = "TCN_"
    model_name += str(kernel_size) + "_" + str(flat_tcn_channels[0]) + "x" + str(len(flat_tcn_channels)) + "_"
    model_name += str(clip) + "_" + str(lr) + "_" + str(scheduler_patience) + "_" + str(scheduler_factor) + "_"
    model_name += str(scheduler_cooldown) + "_" + str(dropout)

    model = TemporalConvNet(num_inputs=num_inputs, num_channels=flat_tcn_channels, kernel_size=kernel_size,
                            dropout=dropout, runs_folder=os.path.join(run_folder, model_name + setting),
                            mode="classification", num_classes=num_classes, gpu=gpu)
    if gpu:
        model.cuda()
    num_params = 0
    for p in model.parameters():
        num_params = num_params + p.numel()
    print("Num TCN params: " + str(num_params))
    model_name += "_" + str(num_params)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, factor=scheduler_factor,
                                                     cooldown=scheduler_cooldown)
    loss_function = F.nll_loss
    model.fit(num_epoch=num_epochs, train_loader=train_loader, optimizer=optimizer, clip=clip,
              loss_function=loss_function, save_every_epoch=save_every_epoch,
              model_path=os.path.join(model_folder, model_name + setting),
              valid_loader=test_loader, scheduler=scheduler, print_every_epoch=print_every_epoch)
