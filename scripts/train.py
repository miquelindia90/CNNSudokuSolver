import os 
import argparse
import itertools
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from dataset import *
from model import *

class Trainner:
    def __init__(self, parameters):
        self.parameters = parameters
        self.__loadDataLoaders()
        self.__prepareModel()
        self.__prepareOutputDirectories()
        self.__loadOptimizer()
        self.__loadCriterion()
        print('Training Set Up Ready')

    def __loadDataLoaders(self):

        print('Preparing Data')
        trainSet, testSet = loadDataset(self.parameters.CSVDataPath, subsample=self.parameters.dataSamples)
        self.trainDataLoader = data.DataLoader(trainSet, batch_size=parameters.batchSize, shuffle=True)
        self.validDataLoader = data.DataLoader(testSet, batch_size=parameters.batchSize)

    def __prepareModel(self):

        print('Preparing Models')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SudokuNet(numberOfKernels = self.parameters.numberOfKernels, recurrentIterations = self.parameters.recurrentIterations)
        self.model.to(self.device)

    def __prepareOutputDirectories(self):
        if not os.path.exists(self.parameters.outputSamplesDirectory):
            os.makedirs(self.parameters.outputSamplesDirectory)

    def __loadCriterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def __loadOptimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.parameters.learningRate)

    def __logInformation(self, pred, tensorOutput, mask, loss):
        self.trainLoss += loss.item()
        self.trainAccuracy += self.__checkBatchAccuracy(pred.view(-1,9)[mask], tensorOutput.view(-1)[mask])
        self.batchCount+=1
        if self.batchCount % self.parameters.printEvery == 0:
            print('Epoch: {}, Batch: {}, Train Loss: {}, Train Accuracy: {}%'.format(self.epoch, self.batchIndex+1, self.trainLoss/self.batchCount, self.trainAccuracy*100/self.batchCount))
            self.__initLoggingVariables()

    def __initLoggingVariables(self):
        self.trainLoss=0
        self.trainAccuracy=0
        self.batchCount=0
    
    def __getIOTensorFromBatch(self, batch):
        tensorInput = batch[0].squeeze()
        tensorOutput = batch[1].squeeze()-1
        return tensorInput.to(self.device), tensorOutput.to(self.device)

    def __prepareIOTrainTensors(self, batch):
        tensorInput, tensorOutput = self.__getIOTensorFromBatch(batch)
        return tensorInput, tensorOutput.view(-1)
    
    def __sudoku_ok(self, line):
        return (len(line) == 9 and sum(line) == sum(set(line)))

    def __checkSudoku(self, grid):

        bad_rows = [row for row in grid if not self.__sudoku_ok(row)]
        grid = list(zip(*grid))
        bad_cols = [col for col in grid if not self.__sudoku_ok(col)]
        squares = []
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                square = list(itertools.chain(row[j:j+3] for row in grid[i:i+3]))
                squareNumbers = list()
                for column in square:
                    for elem in column:
                        squareNumbers.append(elem)
                squares.append(squareNumbers)
                
        bad_squares = [square for square in squares if not self.__sudoku_ok(square)]
        return not (bad_rows or bad_cols or bad_squares)
    
    @staticmethod
    def __toListFormat(solvedSudokus):
        solvedSudokus = torch.argmax(solvedSudokus, dim=-1) + 1
        return solvedSudokus.view(-1,9,9).tolist()

    def __checkBatchAccuracy(self, pred, outputTensor):
        pred = torch.argmax(pred, dim=-1) + 1        
        correctElements = (torch.sum(pred.cpu() == outputTensor.cpu())).item()
        return float(correctElements)/(pred.size(0))
    
    @staticmethod
    def __toStr(listOfIntegers):

        strList = [str(integer) for integer in listOfIntegers]
        return strList

    def __writeLastBatch(self, unsolvedSudokus, solvedSudokus, targetSudokus):
        solvedSudokus = self.__toListFormat(solvedSudokus)
        unsolvedSudokus = unsolvedSudokus.view(-1,9,9).tolist()
        targetSudokus = targetSudokus.view(-1,9,9).tolist()
        with open('{}/epoch{}.res'.format(self.parameters.outputSamplesDirectory, self.epoch),'w') as outputFile:
            for unsolvedSudoku, solvedSudoku, targetSudoku in zip(unsolvedSudokus, solvedSudokus, targetSudokus):
                outputFile.write('Input\tOutput\tTarget\n')
                for row1, row2, row3 in zip(unsolvedSudoku, solvedSudoku, targetSudoku):
                    outputFile.write('{} | {} | {}\n'.format(' '.join(self.__toStr(row1)), ' '.join(self.__toStr(row2)),' '.join(self.__toStr(row3))))
                outputFile.write('\n')

    def __evaluate(self):
        self.model.eval()
        validationAccuracy = 0.
        with torch.no_grad():
            for batchIndex, batch in enumerate(self.validDataLoader):
                unsolvedSudokus, _ = self.__getIOTensorFromBatch(batch)
                _, solvedSudokus = self.model(unsolvedSudokus)
                mask = torch.where(unsolvedSudokus.view(-1)==0)
                validationAccuracy += self.__checkBatchAccuracy(solvedSudokus.view(-1,9)[mask], batch[1].view(-1)[mask])

        self.__writeLastBatch(unsolvedSudokus, solvedSudokus, batch[1].squeeze())
        print('ValidationAccuracy: {}%'.format(validationAccuracy*100/(batchIndex + 1)))
        self.model.train()

    def __saveModel(self):
        
        modelName='{}/model{}.pt'.format(self.parameters.outputSamplesDirectory, self.epoch)
        torch.save(self.model.state_dict(), modelName)

    def train(self):

        for self.epoch in range(self.parameters.maxEpochs):
            self.__initLoggingVariables()
            for self.batchIndex, batch in enumerate(self.trainDataLoader):
                self.optimizer.zero_grad()
                tensorInput, tensorOutput = self.__prepareIOTrainTensors(batch)
                pred1, pred2 = self.model(tensorInput)
                mask = torch.where(tensorInput.view(-1)==0)
                loss = self.criterion(pred1.view(-1,9)[mask], tensorOutput[mask]) + self.criterion(pred2.view(-1,9)[mask], tensorOutput[mask])
                loss.backward()
                self.optimizer.step()
                self.__logInformation(pred2, batch[1], mask, loss)

            self.__evaluate()
            self.__saveModel()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a DL based SudokuSolver')
    parser.add_argument('--batchSize', type=int, default=128) 
    parser.add_argument('--maxEpochs', type=int, default=10000) 
    parser.add_argument('--CSVDataPath', type=str, default='sudoku.csv')
    parser.add_argument('--outputSamplesDirectory', type=str, default='./out2')
    parser.add_argument('--dataSamples', type=int, default=1000000, help='Number Of Samples Used from the CSV') 
    parser.add_argument('--dropSamplingStrategy', action='store_true')

    parser.add_argument('--numberOfKernels', type=int, default=32)
    parser.add_argument('--recurrentIterations', type=int, default=10)
    parser.add_argument('--learningRate', type=float, default=0.0001)
    
    parser.add_argument('--printEvery', type=int, default=1000, help='Trainning Logging Frequency in # of batches') 
    parameters = parser.parse_args() 
    trainner = Trainner(parameters)
    trainner.train()
