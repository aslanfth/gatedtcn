import torch
import sys
sys.path.append("..")
from gatedtcn.models import GatedTemporalConvNet, TemporalConvNet
import os
import torch.optim as optim
import torch.nn.functional as F
import argparse

# The datasets here are generated using the code of https://github.com/dwromero/ckconv

parser = argparse.ArgumentParser(description='Speech Commands')
parser.add_argument('--architecture', type=str,
                    help="Architecture name (gated or generic).", default=None, metavar='')
parser.add_argument("--batch_size", type=int, help="Batch size.", default=1, metavar='')
parser.add_argument("--gpu", type=bool, help="If GPU will be used.", default=True, metavar='')
parser.add_argument("--missing", type=int, help="Missing percent. (0, 30, 50, 70)", default=0, metavar='')
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


def data_generator(root, batch_size, missing_percent=0):
    if os.path.exists(os.path.join(root, "train_X_" + str(missing_percent) + ".pt")):
        train_X = torch.load(os.path.join(root, "train_X_" + str(missing_percent) + ".pt"))
        train_y = torch.load(os.path.join(root, "train_y_" + str(missing_percent) + ".pt"))
        val_X = torch.load(os.path.join(root, "val_X_" + str(missing_percent) + ".pt"))
        val_y = torch.load(os.path.join(root, "val_y_" + str(missing_percent) + ".pt"))
        test_X = torch.load(os.path.join(root, "test_X_" + str(missing_percent) + ".pt"))
        test_y = torch.load(os.path.join(root, "test_y_" + str(missing_percent) + ".pt"))
    else:
        raise Exception("The dataset must be downloaded using datasets/download_speech_data.sh")

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_X, train_y),
                                               batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_X, val_y),
                                             batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_X, test_y),
                                              batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    for batch_idx, (x_, y_) in enumerate(test_loader):
        x_ = x_.cuda()
        y_ = y_.cuda()
        output = model(x_)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y_.data.view_as(pred)).cpu().sum()
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_accuracy


gtcn_run = False
tcn_run = False
# Sequence lengths: 0->16000 30->11200 50->8000 70->4800
root = "datasets/speech"
batch_size = args.batch_size
missing_percent = args.missing
gpu = args.gpu

num_classes = 10
num_inputs = 1 if missing_percent == 0 else 2

run_folder = "runs"
model_folder = "models"
setting = "_speech"
if missing_percent > 0:
    setting += "_" + str(missing_percent)

num_epochs = args.epochs
save_every_epoch = args.save
print_every_epoch = args.print

if args.architecture is None:  # Values from the paper is used to compare the architectures.
    gtcn_run = True
    tcn_run = True
    if missing_percent == 0:
        kernel_size = 8
        tcn_channels = [[25] * 12]
        clip = 0.5
        lr = 0.001
        dropout = 0.2
        scheduler_patience = 3
        scheduler_factor = 0.3
        scheduler_cooldown = 0
    elif missing_percent == 30:
        kernel_size = 6
        tcn_channels = [[25] * 12]
        clip = 0.5
        lr = 0.001
        dropout = 0.2
        scheduler_patience = 3
        scheduler_factor = 0.3
        scheduler_cooldown = 0
    elif missing_percent == 50:
        kernel_size = 8
        tcn_channels = [[25] * 11]
        clip = 0.5
        lr = 0.001
        dropout = 0.2
        scheduler_patience = 3
        scheduler_factor = 0.3
        scheduler_cooldown = 0
    elif missing_percent == 70:
        kernel_size = 5
        tcn_channels = [[25] * 11]
        clip = 0.5
        lr = 0.001
        dropout = 0.2
        scheduler_patience = 3
        scheduler_factor = 0.3
        scheduler_cooldown = 0
    else:
        raise Exception("Invalid missing percent!")

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

train_loader, val_loader, test_loader = data_generator(root, batch_size, missing_percent=missing_percent)
flat_tcn_channels = [item for sublist in tcn_channels for item in sublist]

if gtcn_run:
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
              valid_loader=val_loader, scheduler=scheduler, print_every_epoch=print_every_epoch)
    print("GTCN test accuracy: " + str(evaluate(model=model, test_loader=test_loader)))

if tcn_run:
    if args.architecture is None:
        lr = 0.0001
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
              valid_loader=val_loader, scheduler=scheduler, print_every_epoch=print_every_epoch)
    print("TCN test accuracy: " + str(evaluate(model=model, test_loader=test_loader)))
