# Gated Temporal Convolutional Network

This repository contains the experiments done in the work [Handling Irregularly Sampled Sequences with Gated
Temporal Convolutional Networks](https://arxiv.org) by Fatih Aslan and Suleyman Serdar Kozat.

Experiments are done in PyTorch. If you find this repository helpful, please cite our work:

```
Aslan, F., Kozat, S.S. Handling irregularly sampled signals with gated temporal convolutional networks. SIViP (2022). https://doi.org/10.1007/s11760-022-02292-2
```

PyTorch, TorchVision, and TensorBoard must be installed to run the experiments.

To reproduce the results as in the paper, one needs to specify only the following and not modify the other arguments:
```
--batch_size (int) Mini batch size. (default: 1)
--gpu (boolean) Set True to use PyTorch with CUDA. (default: True)
--epochs (int) Number of epochs. (default: 1)
--save (int) Specify the interval to save the model parameters. (default: 1)
--print (int) Specify the interval to print the model parameters. (default: 1)
```

In the experiments for irregular sampling and missing value cases, one needs to specify the `--missing` parameter as
 well, which is set to `0` by default. Prior running the Character Trajectories and the Speech Commands experiments
 , the datasets must be downloaded by running:
```
cd datasets
download-chartj-data.sh
download-speech-data.sh
```
 
To use the GTCN or TCN with different hyperparameters, one needs to specify the architecture and the
 following hyperparameters:
 
```
--architecture (str) gated for the GTCN, generic for the TCN. (default: None)
--k (int) Kernel size.
--num_layers (int) Number of layers.
--num_filters (int) Number of filters in each layer.
--lr (float) Learning rate.
--clip (float) Gradient clip. Set as -1 to disable.
--dropout (float) Learning rate.
--patience (int) Scheduler patience. Set higher than --epochs to disable the scheduler.
--factor (float) Scheduler factor.
--cooldown (int) Scheduler cooldown.
```
The optimizer used is _Adam_ and the scheduler is _ReduceLROnPlateau_. 
Hyperparameters of the ablation study are set as in the paper and can't be modified using arguments.
For help, run `python experiments/[experiment_name].py -h`

Example code for reproducing the sequential CIFAR10 results using cpu is:
```
python experiments/cifar.py --batch_size 128 --epochs 100 --gpu False
```

TensorBoard can be used to check the results by running:
```
tensorboard --logdir runs
```
