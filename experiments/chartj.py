import torch
import sys
sys.path.append("..")
from gatedtcn.models import GatedTemporalConvNet, TemporalConvNet
import os
import torch.optim as optim
import torch.nn.functional as F
import argparse

# The datasets here are generated using the code of https://github.com/dwromero/ckconv

parser = argparse.ArgumentParser(description='Character Trajectories')
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
    if missing_percent > 0:
        train_X_path = os.path.join(root, "train_X_" + str(missing_percent) + ".pt")
        train_y_path = os.path.join(root, "train_y_" + str(missing_percent) + ".pt")
        val_X_path = os.path.join(root, "val_X_" + str(missing_percent) + ".pt")
        val_y_path = os.path.join(root, "val_y_" + str(missing_percent) + ".pt")
        test_X_path = os.path.join(root, "test_X_" + str(missing_percent) + ".pt")
        test_y_path = os.path.join(root, "test_y_" + str(missing_percent) + ".pt")
    else:
        train_X_path = os.path.join(root, "train_X.pt")
        train_y_path = os.path.join(root, "train_y.pt")
        val_X_path = os.path.join(root, "val_X.pt")
        val_y_path = os.path.join(root, "val_y.pt")
        test_X_path = os.path.join(root, "test_X.pt")
        test_y_path = os.path.join(root, "test_y.pt")
    train_X = torch.load(train_X_path)
    train_y = torch.load(train_y_path)
    val_X = torch.load(val_X_path)
    val_y = torch.load(val_y_path)
    test_X = torch.load(test_X_path)
    test_y = torch.load(test_y_path)
    if missing_percent == 0:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_X, train_y),
                                                   batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_X, val_y),
                                                 batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_X, test_y),
                                                  batch_size=batch_size, shuffle=False)
    else:
        interval_lists = []
        train_list = []
        missingness_info = train_X[:, -1, :]
        for i in range(missingness_info.shape[0]):
            interval_list = []
            interval = 1
            for j in range(missingness_info.shape[1]):
                if missingness_info[i, j] == 1:
                    interval_list.append(interval)
                    interval = 1
                else:
                    interval += 1
            interval_list = torch.Tensor(interval_list)
            interval_lists.append(interval_list)
            train_list.append(train_X[i, :-1, missingness_info[i] == 1])
        interval_lists = torch.stack(interval_lists)
        interval_mean = interval_lists.mean()
        interval_std = interval_lists.std()
        interval_lists = (interval_lists - interval_mean) / interval_std
        train_X = torch.stack(train_list)
        train_X = torch.cat((train_X, interval_lists.unsqueeze(1)), dim=1)
        interval_lists = []
        train_list = []
        missingness_info = val_X[:, -1, :]
        for i in range(missingness_info.shape[0]):
            interval_list = []
            interval = 1
            for j in range(missingness_info.shape[1]):
                if missingness_info[i, j] == 1:
                    interval_list.append(interval)
                    interval = 1
                else:
                    interval += 1
            interval_list = torch.Tensor(interval_list)
            interval_lists.append(interval_list)
            train_list.append(val_X[i, :-1, missingness_info[i] == 1])
        interval_lists = torch.stack(interval_lists)
        interval_lists = (interval_lists - interval_mean) / interval_std
        val_X = torch.stack(train_list)
        val_X = torch.cat((val_X, interval_lists.unsqueeze(1)), dim=1)
        interval_lists = []
        train_list = []
        missingness_info = test_X[:, -1, :]
        for i in range(missingness_info.shape[0]):
            interval_list = []
            interval = 1
            for j in range(missingness_info.shape[1]):
                if missingness_info[i, j] == 1:
                    interval_list.append(interval)
                    interval = 1
                else:
                    interval += 1
            interval_list = torch.Tensor(interval_list)
            interval_lists.append(interval_list)
            train_list.append(test_X[i, :-1, missingness_info[i] == 1])
        interval_lists = torch.stack(interval_lists)
        interval_lists = (interval_lists - interval_mean) / interval_std
        test_X = torch.stack(train_list)
        test_X = torch.cat((test_X, interval_lists.unsqueeze(1)), dim=1)
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
# Sequence lengths: 0->182 30->128 50->91 70->55
root = "datasets/chartj"
batch_size = args.batch_size
missing_percent = args.missing
gpu = args.gpu

num_classes = 20
num_inputs = 3 if missing_percent == 0 else 4

run_folder = "runs"
model_folder = "models"
setting = "_CharTJ"
if missing_percent > 0:
    setting += "_" + str(missing_percent)

num_epochs = args.epochs
save_every_epoch = args.save
print_every_epoch = args.print

if args.architecture is None:   # Values from the paper is used to compare the architectures.
    gtcn_run = True
    tcn_run = True
    if missing_percent == 0:
        kernel_size = 6
        tcn_channels = [[20] * 6]
        clip = 0.1
        lr = 0.001
        dropout = 0.4
        scheduler_patience = 3
        scheduler_factor = 0.3
        scheduler_cooldown = 5
    elif missing_percent == 30:
        kernel_size = 8
        tcn_channels = [[20] * 5]
        clip = 0.1
        lr = 0.001
        dropout = 0.4
        scheduler_patience = 3
        scheduler_factor = 0.3
        scheduler_cooldown = 5
    elif missing_percent == 50:
        kernel_size = 6
        tcn_channels = [[20] * 5]
        clip = 0.1
        lr = 0.001
        dropout = 0.4
        scheduler_patience = 3
        scheduler_factor = 0.3
        scheduler_cooldown = 5
    elif missing_percent == 70:
        kernel_size = 8
        tcn_channels = [[20] * 4]
        clip = 0.1
        lr = 0.001
        dropout = 0.4
        scheduler_patience = 3
        scheduler_factor = 0.3
        scheduler_cooldown = 5
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
