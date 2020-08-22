import torch
import torch.nn as nn
import torch.nn.functional as F


class SudokuSolver(nn.Module):
    def __init__(self, numberOfKernels=32):
        super(SudokuSolver, self).__init__()
        self.featureSize = numberOfKernels*9*4
        self.cellsConv = torch.nn.Conv2d(1, numberOfKernels, 3, stride=3)      
        self.rowsConv = torch.nn.Conv2d(1, numberOfKernels, [1,9])      
        self.columnsConv = torch.nn.Conv2d(1, numberOfKernels, [9,1])
      
        self.dummyFeature = nn.Linear(81, numberOfKernels*9)
        self.bn0 = nn.BatchNorm1d(self.featureSize)
        self.fc1 = nn.Linear(self.featureSize, self.featureSize)
        self.bn1 = nn.BatchNorm1d(self.featureSize)
        self.fc2 = nn.Linear(self.featureSize, self.featureSize)
        self.bn2 = nn.BatchNorm1d(self.featureSize)
        self.classifier = nn.Linear(self.featureSize,81*9)

    def forward(self, x):
        x = x.float()
        x0 = self.dummyFeature(x)
        x = x.view(-1,1,9,9)
        x1 = self.cellsConv(x).view(x.size(0),-1)
        x2 = self.rowsConv(x).view(x.size(0),-1)
        x3 = self.columnsConv(x).view(x.size(0),-1)
        featuresLayer = torch.cat((x0,torch.cat((torch.cat((x1,x2),dim=1),x3),dim=1)),dim=1)
        featuresLayer = F.relu(self.bn0(featuresLayer))
        layer1 = F.relu(self.bn1(self.fc1(featuresLayer)))
        layer2 = F.relu(self.bn2(self.fc2(layer1)))
        labelScores = self.classifier(layer2).view(-1,81,9)
        return labelScores

