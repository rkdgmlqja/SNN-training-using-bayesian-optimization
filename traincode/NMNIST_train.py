import os
import torch
import torch.nn as nn
from ray.tune.search.ax import AxSearch
from ray.tune.search.bayesopt import BayesOptSearch
import tonic.transforms as transforms
from ray import tune

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF

from snntorch import utils
import tonic
from torch.utils.data import DataLoader, random_split
from ray.air import session
from ray.air.checkpoint import Checkpoint

from ray.tune.schedulers import AsyncHyperBandScheduler
#define CSNN##########################################################################################
class FirstBlock (torch.nn.Module):

    def __init__(self, beta, slope, chn):
        super(FirstBlock, self).__init__()
        self.keep_prob = 0.5
        self.layerconv1 = torch.nn.Sequential(
            snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=slope), init_hidden=True),
            nn.Conv2d(2, chn, 7, padding='same'),
            nn.AvgPool2d(3, divisor_override=1)
            )

    def forward(self, data):

        out = self.layerconv1(data)
        return out


class Subblock(torch.nn.Module):

    def __init__(self, beta, slope, inchan, chan):
        super(Subblock, self).__init__()
        self.keep_prob = 0.5
        self.conv1 = nn.Conv2d(inchan, chan, 3, padding='same')
        self.conv2 = nn.Conv2d(chan, chan, 3, padding='same')
        self.syn1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=slope), init_hidden=True)
        self.syn2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=slope), init_hidden=True)
        self.drop = nn.Dropout(p=0.5)


    def forward(self, data):
        out = self.conv1(data)
        skip = out
        out = self.syn1(out)
        out = self.conv2(out)
        out = out+skip
        out = self.syn2(out)
        out = self.drop(out)
        return out

class Block(torch.nn.Module):
    def __init__(self, beta, slope, inchan, chan):
        super(Block, self).__init__()
        self.keep_prob = 0.5
        self.subblock1 = Subblock(beta, slope, inchan, chan)
        self.subblock2 = Subblock(beta, slope, chan, chan)
    def forward(self, data):
        out = self.subblock1(data)
        out = self.subblock2(out)
        return out

class FullBlock(torch.nn.Module):

    def __init__(self,beta, slope):
        super(FullBlock, self).__init__()
        self.keep_prob = 0.5
        self.firstblock = FirstBlock(beta, slope, 16)
        self.Block1 = Block(beta, slope, 16, 16)
        self.Block2 = Block(beta, slope, 16, 32)
        self.Block3 = Block(beta, slope, 32, 64)
        self.Block4 = Block(beta, slope, 64, 128)
        self.Pool = nn.AvgPool2d(3, divisor_override=1)
        self.fc = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(2016, 512),
            nn.Linear(512, 10),
            snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=slope), init_hidden=True))

    def forward(self, data):
        spk_rec = []

        # resets hidden states for all LIF neurons in net
        utils.reset(self.firstblock)
        utils.reset(self.Block1)
        utils.reset(self.Block2)
        utils.reset(self.Block3)
        utils.reset(self.Block4)
        utils.reset(self.fc)


        for step in range(data.size(1)):
            input_torch = data[:, step, :, :, :]
            input_torch = input_torch.cuda()
            out = self.firstblock(input_torch)
            out = self.Block1(out)
            out = self.Block2(out)
            skip2 = out
            out = self.Block3(out)
            skip3 = out
            out = self.Block4(out)
            out = torch.cat((out, skip2, skip3), 1)
            out = self.Pool(out)
            out = self.fc(out)
            spk_rec.append(out)


        return torch.stack(spk_rec)


######################################################################################################


########################################################################################
def train_NMNIST(config):

    net = FullBlock(config["beta"], config["slope"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        #if torch.cuda.device_count() > 1:
        #    net = nn.DataParallel(net)
        net.to(device)

    criterion = SF.mse_count_loss(correct_rate=config["correct_rate"], incorrect_rate=config["incorrect_rate"])
    optimizer = torch.optim.RAdam(net.parameters(), lr=config["lr"], betas=(0.9, 0.999))

    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Define event representation and filter
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, time_window=int(config["time_window"]))

    trainset = tonic.datasets.NMNIST(save_to='/home/hubo1024/PycharmProjects/snn_paper/data/NMNIST',
                                     transform=frame_transform, train=True)


    #trainset, testset = load_data(data_dir)
    dataset_size = len(trainset)
    train_size = int(dataset_size * 0.9)
    validation_size = dataset_size - train_size
    trainset, valset = random_split(trainset, [train_size, validation_size])
    trainloader = DataLoader(trainset, batch_size=int(config["batch_size"]), collate_fn=tonic.collation.PadTensors(), shuffle=True)
    valloader = DataLoader(valset, batch_size=int(config["batch_size"]), collate_fn=tonic.collation.PadTensors(), shuffle=True)


    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.cuda()
            targets = targets.cuda()

            net.train()
            spk_rec = net(data)
            # print(spk_rec.shape)
            loss_val = criterion(spk_rec, targets)
            running_loss += loss_val.item()
            # Gradient calculation + weight update
            optimizer.zero_grad()

            loss_val.backward()
            optimizer.step()
            # print(spk_rec.shape)
            # Store loss history for future plotting
            #loss_hist.append(loss_val.item())
            # print statistics
            #running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        val_acc=0

        for i, (data, targets) in enumerate(iter(valloader)):
            with torch.no_grad():
                data = data.cuda()
                targets = targets.cuda()

                outputs = net(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_steps += 1
                val_acc += SF.accuracy_rate(outputs, targets)

        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (net.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        session.report({"loss": (val_loss / val_steps), "accuracy":(val_acc/val_steps)}, checkpoint=checkpoint)
    print("Finished Training")



###################
def test_best_model(best_result):
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, time_window=int(best_result.config["time_window"]))

    testset = tonic.datasets.NMNIST(save_to='/home/hubo1024/PycharmProjects/snn_paper/data/NMNIST',
                                     transform=frame_transform, train=False)

    testloader = DataLoader(testset, batch_size=50, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    t_acc_sum = 0
    total = 0
    best_trained_model = FullBlock(best_result.config["beta"], best_result.config["slope"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)
    for i, (data, targets) in enumerate(iter(testloader)):
        with torch.no_grad():
            data = data.cuda()
            targets = targets.cuda()

            outputs = best_trained_model(data)
            t_acc = SF.accuracy_rate(outputs, targets)
            t_acc_sum += t_acc
            total += 1
    print("Best trial test set accuracy: {}".format(t_acc_sum / total))



##############


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    space = {
        "beta": (0, 1),
        "slope": (0, 100),
        "correct_rate": (0.5, 1),
        "incorrect_rate": (0, 0.5),
        "time_window": (100, 10000),
        "batch_size": (16, 64),
        "lr": (0.00001, 0.01)
    }

    bayesopt = BayesOptSearch(space, metric="accuracy", mode="min")
    scheduler = AsyncHyperBandScheduler(max_t=max_num_epochs)
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_NMNIST),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            search_alg=bayesopt,
            scheduler=scheduler,
            num_samples=num_samples,
        )
    )
    #tuner = tune.Tuner.restore("/home/hubo1024/ray_results/train_NMNIST_2022-11-14_13-48-48")
    results = tuner.fit()
    best_result = results.get_best_result("accuracy", "max")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["accuracy"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result)








if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=40, max_num_epochs=1, gpus_per_trial=1)
