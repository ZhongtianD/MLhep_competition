import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping






class Print(nn.Module):
    """Debugging only"""

    def forward(self, x):
        print(x.size())
        return x


class Clamp(nn.Module):
    """Clamp energy output"""

    def forward(self, x):
        x = torch.clamp(x, min=0, max=30)
        return x


class SimpleConv(pl.LightningModule):
    def __init__(self, mode: ["classification", "regression"] = "classification"):
        super().__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=3, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.layer1 = nn.Sequential(self.conv1,self.bn1,self.relu, 
                                    #self.maxpool, 
                    self.conv2,self.bn2,self.relu, 
                                    #self.maxpool, 
                    self.conv3,self.bn3,
                    #self.conv4,self.bn4,self.relu, self.maxpool,
                    #ResidualBlock(128, 128, kernel_size=3, stride=1),
                    self.avgpool,
                    nn.Flatten(),
                )


        self.drop_out = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 2)  # for classification
        self.fc3 = nn.Linear(512, 1)  # for regression


        self.stem = nn.Sequential(
            self.layer1, self.drop_out, self.fc1,
            )
        if self.mode == "classification":
            self.classification = nn.Sequential(self.stem,self.drop_out, self.fc2)
        else:
            self.regression = nn.Sequential(self.stem,self.relu, self.drop_out, self.fc3)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        # --------------------------
        x_target, class_target, reg_target, _ = batch
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.train_acc(torch.sigmoid(class_pred), class_target)
            self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
            self.log("classification_loss", class_loss)

            return class_loss

        else:
            reg_pred = self.regression(x_target.float())
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))

            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) / reg_target.float().view(-1, 1))
            self.log("regression_loss", reg_loss)
            return reg_loss

    def training_epoch_end(self, outs):
        # log epoch metric
        if self.mode == "classification":
            self.log("train_acc_epoch", self.train_acc.compute())
        else:
            pass

    def validation_step(self, batch, batch_idx):
        x_target, class_target, reg_target, _ = batch
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.valid_acc(torch.sigmoid(class_pred), class_target)
            self.log("valid_acc", self.valid_acc.compute())
            self.log("classification_loss", class_loss)
            return class_loss

        else:
            reg_pred = self.regression(x_target.float())
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))

            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) / reg_target.float().view(-1, 1))
            self.log("val_regression_loss", reg_loss)
            return reg_loss

    #def test_step(self, batch, batch_idx):
        # --------------------------
        #x_target, class_target, _, reg_target = batch
#        if self.mode == "classification":
#            class_pred = self.classification(x_target.float())
#            class_loss = F.binary_cross_entropy_with_logits(
#                class_pred, class_target.float()
#            )
#            self.test_acc(torch.sigmoid(class_pred), class_target)
#            self.log("test_acc", self.train_acc, on_step=True, on_epoch=False)
#            self.log("classification_loss", class_loss)
#            return class_loss
#
#        else:
#            reg_pred = self.regression(x_target.float())
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
#            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))
#
            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) /reg_target.float().view(-1, 1))
#            self.log("regression_loss", reg_loss)
#            return reg_loss

            #return exp_predicted, class_target

    # --------------------------

    def test_epoch_end(self, test_step_outputs):
        print(self.test_acc.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        if self.mode == "classification":
            class_pred = self.classification(x.float())
            return {"class": torch.sigmoid(class_pred)}
        else:
            reg_pred = self.regression(x.float())
            return {"energy": reg_pred}
        

        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = (kernel_size-1)//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = (kernel_size-1)//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.25)
        #initialization = (lambda w: torch.nn.init.kaiming_normal_(w, nonlinearity='relu'))
        initialization = None
        if initialization is not None:
            initialization(self.conv1.weight)
            initialization(self.conv2.weight)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        #out = self.dropout(out)
        out = self.conv2(out)
        #out = self.relu(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        #out = self.dropout(out)
        return out


class Resnet2(pl.LightningModule):
    def __init__(self, mode: ["classification", "regression"] = "classification"):
        super().__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=3, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.layer1 = nn.Sequential(self.conv1,self.bn1,self.relu,
                                    #self.maxpool, 
                    ResidualBlock(16, 16, kernel_size=7, stride=1),
                    self.conv2,self.bn2,self.relu, 
                    #                self.maxpool, 
                    ResidualBlock(32, 32, kernel_size=5, stride=1),
                    self.conv3,self.bn3,#self.relu,self.maxpool,
                    #ResidualBlock(64, 64, kernel_size=3, stride=1),
                    #self.conv4,self.bn4,self.relu, self.maxpool,
                    #ResidualBlock(128, 128, kernel_size=3, stride=1),
                    self.avgpool,
                    nn.Flatten(),
                )

        self.fc1 = nn.Linear(256, 512)
        #self.fc2 = nn.Linear(512, 2)  # for classification
        self.fc3 = nn.Linear(512, 1)  # for regression


        self.stem = nn.Sequential(
            self.layer1, self.drop_out, self.fc1
            )
        if self.mode == "classification":
            self.classification = nn.Sequential(self.stem, self.relu, self.drop_out, self.fc2)
        else:
            self.regression = nn.Sequential(self.stem, self.relu, self.drop_out, self.fc3)
            


        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        # --------------------------
        x_target, class_target, reg_target, _ = batch
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.train_acc(torch.sigmoid(class_pred), class_target)
            self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
            self.log("classification_loss", class_loss)

            return class_loss

        else:
            reg_pred = self.regression(x_target.float())
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))

            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) / reg_target.float().view(-1, 1))
            self.log("regression_loss", reg_loss)
            return reg_loss

    def training_epoch_end(self, outs):
        # log epoch metric
        if self.mode == "classification":
            self.log("train_acc_epoch", self.train_acc.compute())
        else:
            pass

    def validation_step(self, batch, batch_idx):
        x_target, class_target, reg_target, _ = batch
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.valid_acc(torch.sigmoid(class_pred), class_target)
            self.log("valid_acc", self.valid_acc.compute())
            self.log("classification_loss", class_loss)
            return class_loss

        else:
            reg_pred = self.regression(x_target.float())
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))

            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) / reg_target.float().view(-1, 1))
            self.log("val_regression_loss", reg_loss)
            return reg_loss

    #def test_step(self, batch, batch_idx):
        # --------------------------
        #x_target, class_target, _, reg_target = batch
#        if self.mode == "classification":
#            class_pred = self.classification(x_target.float())
#            class_loss = F.binary_cross_entropy_with_logits(
#                class_pred, class_target.float()
#            )
#            self.test_acc(torch.sigmoid(class_pred), class_target)
#            self.log("test_acc", self.train_acc, on_step=True, on_epoch=False)
#            self.log("classification_loss", class_loss)
#            return class_loss
#
#        else:
#            reg_pred = self.regression(x_target.float())
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
#            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))
#
            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) /reg_target.float().view(-1, 1))
#            self.log("regression_loss", reg_loss)
#            return reg_loss

            #return exp_predicted, class_target

    # --------------------------

    def test_epoch_end(self, test_step_outputs):
        print(self.test_acc.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer

    def forward(self, x):
        if self.mode == "classification":
            class_pred = self.classification(x.float())
            return {"class": torch.sigmoid(class_pred)}
        else:
            reg_pred = self.regression(x.float())
            return {"energy": reg_pred}
