from torch.nn.utils import weight_norm
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch.autograd import Variable

# The code for implementing the generic TCN model is taken from  https://github.com/locuslab/TCN

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class GenericTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2,
                 runs_folder=None, mode=None, num_classes=1, gpu=True):
        super(GenericTemporalConvNet, self).__init__()
        self.mode = mode
        if runs_folder is not None:
            self.tb_writer = SummaryWriter(runs_folder, purge_step=0)
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_inputs = num_inputs
        self.gpu = gpu

        if mode == "classification":
            self.last_layer = nn.Linear(in_features=self.num_channels[-1], out_features=num_classes)
            self.activation = nn.LogSoftmax(dim=1)
            if self.gpu:
                self.last_layer.cuda()
                self.activation.cuda()

    def forward(self, x):
        if self.mode == "classification":
            output = self.network(x)
            fc_output = self.last_layer(output[:, :, -1])
            return self.activation(fc_output)
        else:
            return self.network(x)

    def fit(self, num_epoch, train_loader, optimizer, clip, loss_function, save_every_epoch=20, model_path=None,
            valid_loader=None, scheduler=None, print_every_epoch=20):
        best_val_loss = float('inf')
        old_lr = [group['lr'] for group in optimizer.param_groups][0]
        receptive_field = self.kernel_size * (2 ** (len(self.num_channels) - 1))
        print("Starting LR: " + str(old_lr))
        for epoch in range(num_epoch):
            self.train()
            total_loss = 0
            for batch_idx, (x_, y_) in enumerate(train_loader):
                x_ = Variable(x_)
                y_ = Variable(y_)
                if self.gpu:
                    x_ = x_.cuda()
                    y_ = y_.cuda()
                optimizer.zero_grad()
                output = self(x_)
                loss = loss_function(output, y_)
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
                optimizer.step()
                total_loss += loss.item()
            cur_loss = total_loss / (batch_idx + 1)
            if (epoch + 1) % print_every_epoch == 0:
                print("Epoch: " + str(epoch))
                print('Loss: {:.6f}'.format(cur_loss))
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("training/loss", cur_loss, epoch)
            to_save = ((epoch + 1) % save_every_epoch == 0)
            if to_save and model_path is not None:
                torch.save(self.state_dict(), model_path + "_" + str(epoch))
            if valid_loader is not None:
                self.eval()
                test_loss = 0
                correct = 0
                counter = 0
                for batch_idx, (x_, y_) in enumerate(valid_loader):
                    x_ = Variable(x_)
                    y_ = Variable(y_)
                    if self.gpu:
                        x_ = x_.cuda()
                        y_ = y_.cuda()
                    output = self(x_)
                    loss = loss_function(output, y_)
                    if self.mode == "classification":
                        pred = output.data.max(1, keepdim=True)[1]
                        correct += pred.eq(y_.data.view_as(pred)).cpu().sum()
                    test_loss += loss.item()
                if test_loss / (batch_idx + 1) < best_val_loss and model_path is not None:
                    torch.save(self.state_dict(), model_path + "_best")
                    best_val_loss = test_loss / (batch_idx + 1)
                    print("Best model is saved, epoch: " + str(epoch))
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("validation/loss", test_loss / (batch_idx + 1), epoch)
                    if self.mode == "classification":
                        self.tb_writer.add_scalar("validation/accuracy", 100. * correct / len(valid_loader.dataset),
                                                  epoch)
            if scheduler is not None and valid_loader is not None:
                scheduler.step(test_loss / (batch_idx + 1))
                lr = [group['lr'] for group in optimizer.param_groups][0]
                if lr != old_lr:
                    print("LR is modified, new LR: " + str(lr) + ", step: " + str(epoch))
                    old_lr = lr
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.kernel_size = kernel_size

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalBlockLast(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlockLast, self).__init__()
        self.kernel_size = kernel_size

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.dropout1,
                                 self.conv2, self.chomp2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalConvNet(GenericTemporalConvNet):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2,
                 runs_folder=None, mode=None, num_classes=1, gpu=True):
        super().__init__(num_inputs, num_channels, kernel_size=kernel_size,
                         runs_folder=runs_folder, mode=mode, num_classes=num_classes, gpu=gpu)
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            if i == num_levels - 1:
                layers += [TemporalBlockLast(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                             padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            else:
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)


class GatedTemporalBlock(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.1,
    ):
        super(GatedTemporalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                2 * n_inputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                2 * n_outputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        hidden, gate = self.conv1(x).chunk(2, dim=1)
        hidden = hidden * torch.sigmoid(gate)
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp1(hidden)
        hidden = self.dropout1(hidden)
        hidden, gate = self.conv2(hidden).chunk(2, dim=1)
        hidden = hidden * torch.sigmoid(gate)
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp2(hidden)
        hidden = self.dropout2(hidden)
        res = x if self.downsample is None else self.downsample(x)
        return hidden + res


class GatedTemporalBlockLast(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.1,
    ):
        super(GatedTemporalBlockLast, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                2 * n_inputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                2 * n_outputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        hidden, gate = self.conv1(x).chunk(2, dim=1)
        hidden = hidden * torch.sigmoid(gate)
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp1(hidden)
        hidden = self.dropout1(hidden)
        hidden, gate = self.conv2(hidden).chunk(2, dim=1)
        hidden = hidden * torch.sigmoid(gate)
        hidden = self.chomp2(hidden)
        hidden = self.dropout2(hidden)
        res = x.chunk(2, dim=1)[0] if self.downsample is None else self.downsample(x.chunk(2, dim=1)[0])
        return hidden + res


class GatedTemporalBlockFirst(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.1,
    ):
        super(GatedTemporalBlockFirst, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                2 * n_outputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        hidden, gate = self.conv1(x).chunk(2, dim=1)
        hidden = hidden * torch.sigmoid(gate)
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp1(hidden)
        hidden = self.dropout1(hidden)
        hidden, gate = self.conv2(hidden).chunk(2, dim=1)
        hidden = hidden * torch.sigmoid(gate)
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp2(hidden)
        hidden = self.dropout2(hidden)
        res = x if self.downsample is None else self.downsample(x)
        return hidden + torch.cat((res, res), dim=1)


class GatedTemporalConvNet(GenericTemporalConvNet):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2,
                 runs_folder=None, mode=None, num_classes=1, gpu=True):
        super().__init__(num_inputs, num_channels, kernel_size=kernel_size,
                         runs_folder=runs_folder, mode=mode, num_classes=num_classes, gpu=gpu)
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            if i == num_levels - 1:
                layers += [
                    GatedTemporalBlockLast(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                           padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            elif i == 0:
                layers += [
                    GatedTemporalBlockFirst(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                            padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            else:
                layers += [GatedTemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                              padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)


# For ablation studies:

class GatedTemporalBlock_hid(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.1,
    ):
        super(GatedTemporalBlock_hid, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        hidden, gate = self.conv1(x).chunk(2, dim=1)
        hidden = hidden * torch.sigmoid(gate)
        hidden = self.chomp1(hidden)
        hidden = self.dropout1(hidden)
        hidden, gate = self.conv2(hidden).chunk(2, dim=1)
        hidden = hidden * torch.sigmoid(gate)
        hidden = self.chomp2(hidden)
        hidden = self.dropout2(hidden)
        res = x if self.downsample is None else self.downsample(x)
        return hidden + res


class GatedTemporalBlockLast_hid(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.1,
    ):
        super(GatedTemporalBlockLast_hid, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        hidden, gate = self.conv1(x).chunk(2, dim=1)
        hidden = hidden * torch.sigmoid(gate)
        hidden = self.chomp1(hidden)
        hidden = self.dropout1(hidden)
        hidden, gate = self.conv2(hidden).chunk(2, dim=1)
        hidden = hidden * torch.sigmoid(gate)
        hidden = self.chomp2(hidden)
        hidden = self.dropout2(hidden)
        res = x if self.downsample is None else self.downsample(x)
        return hidden + res


class GatedTemporalConvNet_hid(GenericTemporalConvNet):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2,
                 runs_folder=None, mode=None, num_classes=1, gpu=True):
        super().__init__(num_inputs, num_channels, kernel_size=kernel_size,
                         runs_folder=runs_folder, mode=mode, num_classes=num_classes, gpu=gpu)
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            if i == num_levels - 1:
                layers += [
                    GatedTemporalBlockLast_hid(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                               padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            else:
                layers += [
                    GatedTemporalBlock_hid(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                           padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)


class GatedTemporalBlock_sep(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.1,
    ):
        super(GatedTemporalBlock_sep, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.convgate1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.convgate2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        self.convgate1.weight.data.normal_(0, 0.01)
        self.convgate2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        hidden = self.conv1(x.chunk(2, dim=1)[0])
        gate = self.convgate1(x.chunk(2, dim=1)[1])
        hidden = hidden * torch.sigmoid(gate)
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp1(hidden)
        hidden = self.dropout1(hidden)
        gate = self.convgate2(hidden.chunk(2, dim=1)[1])
        hidden = self.conv2(hidden.chunk(2, dim=1)[0])
        hidden = hidden * torch.sigmoid(gate)
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp2(hidden)
        hidden = self.dropout2(hidden)
        res = x if self.downsample is None else self.downsample(x)
        return hidden + res


class GatedTemporalBlockLast_sep(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.1,
    ):
        super(GatedTemporalBlockLast_sep, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.convgate1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.convgate2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        self.convgate1.weight.data.normal_(0, 0.01)
        self.convgate2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        hidden = self.conv1(x.chunk(2, dim=1)[0])
        gate = self.convgate1(x.chunk(2, dim=1)[1])
        hidden = hidden * torch.sigmoid(gate)
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp1(hidden)
        hidden = self.dropout1(hidden)
        gate = self.convgate2(hidden.chunk(2, dim=1)[1])
        hidden = self.conv2(hidden.chunk(2, dim=1)[0])
        hidden = hidden * torch.sigmoid(gate)
        hidden = self.chomp2(hidden)
        hidden = self.dropout2(hidden)
        res = x.chunk(2, dim=1)[0] if self.downsample is None else self.downsample(x.chunk(2, dim=1)[0])
        return hidden + res


class GatedTemporalBlockFirst_sep(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.1,
    ):
        super(GatedTemporalBlockFirst_sep, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.convgate1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.convgate2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        self.convgate1.weight.data.normal_(0, 0.01)
        self.convgate2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        hidden = self.conv1(x)
        gate = self.convgate1(x)
        hidden = hidden * torch.sigmoid(gate)
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp1(hidden)
        hidden = self.dropout1(hidden)
        gate = self.convgate2(hidden.chunk(2, dim=1)[1])
        hidden = self.conv2(hidden.chunk(2, dim=1)[0])
        hidden = hidden * torch.sigmoid(gate)
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp2(hidden)
        hidden = self.dropout2(hidden)
        res = x if self.downsample is None else self.downsample(x)
        return hidden + torch.cat((res, res), dim=1)


class GatedTemporalConvNet_sep(GenericTemporalConvNet):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2,
                 runs_folder=None, mode=None, num_classes=1, gpu=True):
        super().__init__(num_inputs, num_channels, kernel_size=kernel_size,
                         runs_folder=runs_folder, mode=mode, num_classes=num_classes, gpu=gpu)
        if runs_folder is not None:
            self.tb_writer = SummaryWriter(runs_folder, purge_step=0)
        self.mode = mode
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_inputs = num_inputs
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            if i == num_levels - 1:
                layers += [
                    GatedTemporalBlockLast_sep(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                               padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            elif i == 0:
                layers += [
                    GatedTemporalBlockFirst_sep(in_channels, out_channels, kernel_size, stride=1,
                                                dilation=dilation_size,
                                                padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            else:
                layers += [
                    GatedTemporalBlock_sep(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                           padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)


class GatedTemporalBlock_sig(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.1,
    ):
        super(GatedTemporalBlock_sig, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                2 * n_inputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                2 * n_outputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        hidden, gate = self.conv1(x).chunk(2, dim=1)
        gate = torch.sigmoid(gate)
        hidden = hidden * gate
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp1(hidden)
        hidden = self.dropout1(hidden)
        hidden, gate = self.conv2(hidden).chunk(2, dim=1)
        gate = torch.sigmoid(gate)
        hidden = hidden * gate
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp2(hidden)
        hidden = self.dropout2(hidden)
        res = x if self.downsample is None else self.downsample(x)
        return hidden + res


class GatedTemporalBlockLast_sig(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.1,
    ):
        super(GatedTemporalBlockLast_sig, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                2 * n_inputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                2 * n_outputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        hidden, gate = self.conv1(x).chunk(2, dim=1)
        gate = torch.sigmoid(gate)
        hidden = hidden * gate
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp1(hidden)
        hidden = self.dropout1(hidden)
        hidden, gate = self.conv2(hidden).chunk(2, dim=1)
        gate = torch.sigmoid(gate)
        hidden = hidden * gate
        hidden = self.chomp2(hidden)
        hidden = self.dropout2(hidden)
        res = x.chunk(2, dim=1)[0] if self.downsample is None else self.downsample(x.chunk(2, dim=1)[0])
        return hidden + res


class GatedTemporalBlockFirst_sig(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.1,
    ):
        super(GatedTemporalBlockFirst_sig, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                2 * n_outputs,
                2 * n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        hidden, gate = self.conv1(x).chunk(2, dim=1)
        gate = torch.sigmoid(gate)
        hidden = hidden * gate
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp1(hidden)
        hidden = self.dropout1(hidden)
        hidden, gate = self.conv2(hidden).chunk(2, dim=1)
        gate = torch.sigmoid(gate)
        hidden = hidden * gate
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp2(hidden)
        hidden = self.dropout2(hidden)
        res = x if self.downsample is None else self.downsample(x)
        return hidden + torch.cat((res, res), dim=1)


class GatedTemporalConvNet_sig(GenericTemporalConvNet):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2,
                 runs_folder=None, mode=None, num_classes=1, gpu=True):
        super().__init__(num_inputs, num_channels, kernel_size=kernel_size,
                         runs_folder=runs_folder, mode=mode, num_classes=num_classes, gpu=gpu)
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            if i == num_levels - 1:
                layers += [
                    GatedTemporalBlockLast_sig(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                               padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            elif i == 0:
                layers += [
                    GatedTemporalBlockFirst_sig(in_channels, out_channels, kernel_size, stride=1,
                                                dilation=dilation_size,
                                                padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            else:
                layers += [
                    GatedTemporalBlock_sig(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                           padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
