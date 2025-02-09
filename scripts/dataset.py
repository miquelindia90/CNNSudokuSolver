import torch.utils.data as data
import torch
import torch.nn.functional as F
import pandas as pd
import random
import copy

def prepareTensor(listOfSudokus):
    tensor = torch.zeros((len(listOfSudokus),81))
    for idx, sudokuString in enumerate(listOfSudokus):
        for position, numberString in enumerate(sudokuString):
            tensor[idx, position] = int(numberString)

    return tensor.long()

def prepareSudokuTensors(dataFrame, train_split=0.9):
    
    dataSize = dataFrame.shape[0]
    quizzes = prepareTensor(dataFrame.quizzes.to_list())
    solutions = prepareTensor(dataFrame.solutions.to_list())

    randomPermutation = torch.randperm(dataSize)
    trainPermutation = randomPermutation[:int(train_split * dataSize)]
    testPermutation = randomPermutation[int(train_split * dataSize):]

    return quizzes[trainPermutation], solutions[trainPermutation],\
        quizzes[testPermutation], solutions[testPermutation]

class Dataset(data.Dataset):

    def __init__(self, sudokusQuizzes, sudokusSolutions, dropSampling=False):
        'Initialization'
        self.sudokusQuizzes = sudokusQuizzes
        self.sudokusSolutions = sudokusSolutions
        self.num_samples = len(sudokusQuizzes)
        self.dropSampling = dropSampling

    def __len__(self):
        return self.num_samples

    def __getInput(self, index):
        likelihood = round(random.uniform(0, 1),1)
        if likelihood>0.3:
            mask = torch.where(torch.randn(self.sudokusSolutions[index,:].size()) > round(random.uniform(-1, 1),1))[0]
            sudokuTensor = copy.deepcopy(self.sudokusSolutions[index,:])
            sudokuTensor[mask]=0        
            return sudokuTensor
        else:
            return self.sudokusQuizzes[index,:]

    def __getitem__(self, index):
        'Generates one sample of data'
        inputSudoku = self.__getInput(index) if self.dropSampling else self.sudokusQuizzes[index,:]
        outputSudoku = self.sudokusSolutions[index,:]
        return inputSudoku, outputSudoku

def loadDataset(dataSet, subsample=1000000, dropSampling=False):
    dataset = pd.read_csv(dataSet, sep=',')
    dataSubset = dataset.sample(subsample)
    trainInputs, trainSolutions, testInputs, testSolutions = prepareSudokuTensors(dataSubset)
    return Dataset(trainInputs, trainSolutions, dropSampling=dropSampling), Dataset(testInputs, testSolutions)
