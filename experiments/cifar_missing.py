import torch
import sys
sys.path.append("..")
from torchvision import datasets, transforms
from gatedtcn.models import GatedTemporalConvNet, GatedTemporalConvNet_sig, GatedTemporalConvNet_sep, GatedTemporalConvNet_hid
import os
import torch.optim as optim
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='CIFAR Missing')
parser.add_argument("--batch_size", type=int, help="Batch size.", default=1, metavar='')
parser.add_argument("--gpu", type=bool, help="If GPU will be used.", default=True, metavar='')
parser.add_argument("--missing", type=int, help="Number of missing pixels. (0, 256, 512, 768)", default=0, metavar='')
parser.add_argument("--epochs", type=int, help="Number of epochs.", default=1, metavar='')
parser.add_argument("--save", type=int, help="Determine the interval of saving the models.", default=1, metavar='')
parser.add_argument("--print", type=int, help="Determine the interval of printing the models.", default=1, metavar='')

args = parser.parse_args()


def data_generator(root, batch_size, missing_count=0):
    transform_mean = torch.Tensor((0.5, 0.5, 0.5))
    transform_std = torch.Tensor((0.5, 0.5, 0.5))
    if os.path.exists(os.path.join(root, "cifar-10-batches-py")):
        train_set = datasets.CIFAR10(root=root, train=True, download=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(transform_mean, transform_std)
                                     ]))
        test_set = datasets.CIFAR10(root=root, train=False, download=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(transform_mean, transform_std)
                                    ]))
    else:
        train_set = datasets.CIFAR10(root=root, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(transform_mean, transform_std)
                                     ]))
        test_set = datasets.CIFAR10(root=root, train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(transform_mean, transform_std)
                                    ]))
    train_labels = torch.tensor(train_set.targets)
    test_labels = torch.tensor(test_set.targets)
    root = os.path.join(root, "cifar")
    if not os.path.exists(root):
        os.mkdir(root)
    if os.path.exists(os.path.join(root, "train_" + str(missing_count) + ".pt")):
        #  load the precomputed data.
        train_data_masked = torch.load(os.path.join(root, "train_" + str(missing_count) + ".pt"))
        test_data_masked = torch.load(os.path.join(root, "test_" + str(missing_count) + ".pt"))
    else:
        train_data = torch.from_numpy(train_set.data).view(-1, 1024, 3).type(torch.FloatTensor)
        test_data = torch.from_numpy(test_set.data).view(-1, 1024, 3).type(torch.FloatTensor)
        train_data = train_data.permute(0, 2, 1)
        test_data = test_data.permute(0, 2, 1)
        if missing_count == 0:
            train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels),
                                                       batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_labels),
                                                      batch_size=batch_size, shuffle=False)
            return train_loader, test_loader
        train_data_masked = torch.empty(
            size=[train_data.shape[0], train_data.shape[1] - missing_count,
                  train_data.shape[2]]).type(torch.ByteTensor)
        test_data_masked = torch.empty(
            size=[test_data.shape[0], test_data.shape[1] - missing_count,
                  train_data.shape[2]]).type(torch.ByteTensor)
        train_interval = torch.zeros([train_data_masked.shape[0],
                                      train_data.shape[1] - missing_count]).type(torch.FloatTensor)
        test_interval = torch.zeros([test_data_masked.shape[0],
                                     test_data.shape[1] - missing_count]).type(torch.FloatTensor)
        for i in range(train_data.shape[0]):
            interval = torch.ones(1024)
            perm = torch.randperm(1024)
            keep_sorted_train, _ = torch.sort(perm[:-missing_count])
            drop_sorted_train, _ = torch.sort(perm[-missing_count:])
            drop_train = drop_sorted_train.tolist()
            for dropped_idx in drop_train:
                if dropped_idx < 1023:
                    interval[dropped_idx + 1] += interval[dropped_idx]
                interval[dropped_idx] = 0
            train_data_masked[i, :, :] = train_data[i, keep_sorted_train, :]
            interval = interval[interval != 0]
            train_interval[i, :] = interval
        interval_mean = torch.mean(train_interval)
        interval_std = torch.std(train_interval)
        train_interval = (train_interval - interval_mean) / interval_std
        train_data_masked = (train_data_masked - torch.tensor((0.5, 0.5, 0.5)) * 255) / \
                            torch.tensor((0.5, 0.5, 0.5)) / 255
        train_data_masked = torch.cat((train_data_masked, train_interval.unsqueeze(2)), dim=2)
        for i in range(test_data.shape[0]):
            interval = torch.ones(1024)
            perm = torch.randperm(1024)
            keep_sorted_train, _ = torch.sort(perm[:-missing_count])
            drop_sorted_train, _ = torch.sort(perm[-missing_count:])
            drop_train = drop_sorted_train.tolist()
            for dropped_idx in drop_train:
                if dropped_idx < 1023:
                    interval[dropped_idx + 1] += interval[dropped_idx]
                interval[dropped_idx] = 0
            test_data_masked[i, :, :] = test_data[i, keep_sorted_train, :]
            interval = interval[interval != 0]
            test_interval[i, :] = interval
        test_interval = (test_interval - interval_mean) / interval_std
        test_data_masked = (test_data_masked - torch.tensor((0.5, 0.5, 0.5)) * 255) / \
                           torch.tensor((0.5, 0.5, 0.5)) / 255
        test_data_masked = torch.cat((test_data_masked, test_interval.unsqueeze(2)), dim=2)
        torch.save(train_data_masked, os.path.join(root, "train_" + str(missing_count) + ".pt"))
        torch.save(test_data_masked, os.path.join(root, "test_" + str(missing_count) + ".pt"))
    train_data_masked = train_data_masked.permute(0, 2, 1)
    test_data_masked = test_data_masked.permute(0, 2, 1)
    train_loader_missing = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data_masked,
                                                                                      train_labels),
                                                       batch_size=batch_size, shuffle=True)
    test_loader_missing = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data_masked,
                                                                                     test_labels),
                                                      batch_size=batch_size, shuffle=False)
    return train_loader_missing, test_loader_missing


root = "./datasets"
batch_size = args.batch_size
missing_count = args.missing

num_classes = 10
num_inputs = 4 if missing_count > 0 else 3

run_folder = "runs"
model_folder = "models"
setting = "_CIFAR_" + str(missing_count)

num_epochs = args.epochs
save_every_epoch = args.save
print_every_epoch = args.print

if missing_count == 0:
    kernel_size = 8
    tcn_channels = [[25] * 8]
    clip = 0.1
    lr = 0.01
    dropout = 0.2
    scheduler_patience = 3
    scheduler_factor = 0.3
    scheduler_cooldown = 0
elif missing_count == 256:
    kernel_size = 6
    tcn_channels = [[25] * 8]
    clip = 0.1
    lr = 0.01
    dropout = 0.2
    scheduler_patience = 3
    scheduler_factor = 0.3
    scheduler_cooldown = 0
elif missing_count == 50:
    kernel_size = 8
    tcn_channels = [[25] * 7]
    clip = 0.1
    lr = 0.01
    dropout = 0.2
    scheduler_patience = 3
    scheduler_factor = 0.3
    scheduler_cooldown = 0
elif missing_count == 70:
    kernel_size = 8
    tcn_channels = [[25] * 6]
    clip = 0.1
    lr = 0.01
    dropout = 0.2
    scheduler_patience = 3
    scheduler_factor = 0.3
    scheduler_cooldown = 0
else:
    raise Exception("Invalid missing count!")

train_missing, test_missing = data_generator(root, batch_size, missing_count=missing_count)
flat_tcn_channels = [item for sublist in tcn_channels for item in sublist]

model_name = "GTCN_"
model_name += str(kernel_size) + "_" + str(flat_tcn_channels[0]) + "x" + str(len(flat_tcn_channels)) + "_"
model_name += str(clip) + "_" + str(lr) + "_" + str(scheduler_patience) + "_" + str(scheduler_factor) + "_"
model_name += str(scheduler_cooldown) + "_" + str(dropout)

model = GatedTemporalConvNet(num_inputs=num_inputs, num_channels=flat_tcn_channels, kernel_size=kernel_size,
                             dropout=dropout, runs_folder=os.path.join(run_folder, model_name + setting),
                             mode="classification", num_classes=num_classes)
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
model.fit(num_epoch=num_epochs, train_loader=train_missing, optimizer=optimizer, clip=clip,
          loss_function=loss_function, save_every_epoch=save_every_epoch,
          model_path=os.path.join(model_folder, model_name + setting),
          valid_loader=test_missing, scheduler=scheduler, print_every_epoch=print_every_epoch)

model_name = "GTCN_sig_"
model_name += str(kernel_size) + "_" + str(flat_tcn_channels[0]) + "x" + str(len(flat_tcn_channels)) + "_"
model_name += str(clip) + "_" + str(lr) + "_" + str(scheduler_patience) + "_" + str(scheduler_factor) + "_"
model_name += str(scheduler_cooldown) + "_" + str(dropout)

model = GatedTemporalConvNet_sig(num_inputs=num_inputs, num_channels=flat_tcn_channels, kernel_size=kernel_size,
                                 dropout=dropout, runs_folder=os.path.join(run_folder, model_name + setting),
                                 mode="classification", num_classes=num_classes)
model.cuda()
num_params = 0
for p in model.parameters():
    num_params = num_params + p.numel()
print("Num GTCN_sig params: " + str(num_params))
model_name += "_" + str(num_params)

optimizer = optim.Adam(params=model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, factor=scheduler_factor,
                                                 cooldown=scheduler_cooldown)
loss_function = F.nll_loss
model.fit(num_epoch=num_epochs, train_loader=train_missing, optimizer=optimizer, clip=clip,
          loss_function=loss_function, save_every_epoch=save_every_epoch,
          model_path=os.path.join(model_folder, model_name + setting),
          valid_loader=test_missing, scheduler=scheduler, print_every_epoch=print_every_epoch)

model_name = "GTCN_sep_"
model_name += str(kernel_size) + "_" + str(flat_tcn_channels[0]) + "x" + str(len(flat_tcn_channels)) + "_"
model_name += str(clip) + "_" + str(lr) + "_" + str(scheduler_patience) + "_" + str(scheduler_factor) + "_"
model_name += str(scheduler_cooldown) + "_" + str(dropout)

model = GatedTemporalConvNet_sep(num_inputs=num_inputs, num_channels=flat_tcn_channels, kernel_size=kernel_size,
                                 dropout=dropout, runs_folder=os.path.join(run_folder, model_name + setting),
                                 mode="classification", num_classes=num_classes)
model.cuda()
num_params = 0
for p in model.parameters():
    num_params = num_params + p.numel()
print("Num GTCN_sep params: " + str(num_params))
model_name += "_" + str(num_params)

optimizer = optim.Adam(params=model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, factor=scheduler_factor,
                                                 cooldown=scheduler_cooldown)
loss_function = F.nll_loss
model.fit(num_epoch=num_epochs, train_loader=train_missing, optimizer=optimizer, clip=clip,
          loss_function=loss_function, save_every_epoch=save_every_epoch,
          model_path=os.path.join(model_folder, model_name + setting),
          valid_loader=test_missing, scheduler=scheduler, print_every_epoch=print_every_epoch)

model_name = "GTCN_hid_"
model_name += str(kernel_size) + "_" + str(flat_tcn_channels[0]) + "x" + str(len(flat_tcn_channels)) + "_"
model_name += str(clip) + "_" + str(lr) + "_" + str(scheduler_patience) + "_" + str(scheduler_factor) + "_"
model_name += str(scheduler_cooldown) + "_" + str(dropout)

model = GatedTemporalConvNet_hid(num_inputs=num_inputs, num_channels=flat_tcn_channels, kernel_size=kernel_size,
                                 dropout=dropout, runs_folder=os.path.join(run_folder, model_name + setting),
                                 mode="classification", num_classes=num_classes)
model.cuda()
num_params = 0
for p in model.parameters():
    num_params = num_params + p.numel()
print("Num GTCN_hid params: " + str(num_params))
model_name += "_" + str(num_params)

optimizer = optim.Adam(params=model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, factor=scheduler_factor,
                                                 cooldown=scheduler_cooldown)
loss_function = F.nll_loss
model.fit(num_epoch=num_epochs, train_loader=train_missing, optimizer=optimizer, clip=clip,
          loss_function=loss_function, save_every_epoch=save_every_epoch,
          model_path=os.path.join(model_folder, model_name + setting),
          valid_loader=test_missing, scheduler=scheduler, print_every_epoch=print_every_epoch)
