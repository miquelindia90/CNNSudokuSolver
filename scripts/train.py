import os 
import argparse
import itertools
import random
import logging
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from dataset import *
from model import *

class Trainner:
    def __init__(self, parameters):
        self.parameters = parameters
        self.__initSeeds(parameters.seed)
        self.__setLogger(parameters.logFile)
        self.__loadDataLoaders()
        self.__prepareModel()
        self.__prepareOutputDirectories()
        self.__loadOptimizer()
        self.__loadCriterion()
        logging.info('Training Set Up Ready')

    def __initSeeds(self, seed):

        torch.manual_seed(seed)
        random.seed(seed)

    def __setLogger(self, logPath):

        if len(logPath.split('/'))>2:
            self.__createDirectory('/'.join(logPath.split('/')[:-1]))
        logging.basicConfig(level=logging.INFO, filename=logPath, filemode='w', format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.info('Logger Configured.')

    def __loadDataLoaders(self):

        logging.info('Preparing Data')
        trainSet, testSet = loadDataset(self.parameters.CSVDataPath, subsample=self.parameters.dataSamples)
        self.trainDataLoader = data.DataLoader(trainSet, batch_size=parameters.batchSize, shuffle=True)
        self.validDataLoader = data.DataLoader(testSet, batch_size=parameters.batchSize)

    def __prepareModel(self):

        logging.info('Preparing Models')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SudokuNet(numberOfKernels = self.parameters.numberOfKernels, recurrentIterations = self.parameters.recurrentIterations)
        self.model.to(self.device)

    @staticmethod
    def __createDirectory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def __prepareOutputDirectories(self):
        self.__createDirectory(self.parameters.outputSamplesDirectory)

    def __loadCriterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def __loadOptimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.parameters.learningRate)

    def __logInformation(self, pred1,  pred2, tensorOutput, mask, loss):
        self.trainLoss += loss.item()
        self.preLabelsTrainAccuracy += self.__checkBatchAccuracy(pred1.view(-1,9)[mask], tensorOutput.view(-1)[mask])
        self.trainAccuracy += self.__checkBatchAccuracy(pred2.view(-1,9)[mask], tensorOutput.view(-1)[mask])
        self.batchCount+=1
        if self.batchCount % self.parameters.printEvery == 0:
            logging.info('Epoch: {}, Batch: {}, Train Loss: {}, PreLabels Train Accuracy: {}% Train Accuracy: {}%'.format(self.epoch, self.batchIndex+1, self.trainLoss/self.batchCount, self.preLabelsTrainAccuracy*100/self.batchCount, self.trainAccuracy*100/self.batchCount))
            self.__initLoggingVariables()

    def __initLoggingVariables(self):
        self.trainLoss=0
        self.preLabelsTrainAccuracy=0
        self.trainAccuracy=0
        self.batchCount=0
    
    def __getIOTensorFromBatch(self, batch):
        tensorInput = batch[0].squeeze()
        tensorOutput = batch[1].squeeze()-1
        return tensorInput.to(self.device), tensorOutput.to(self.device)

    def __prepareIOTrainTensors(self, batch):
        tensorInput, tensorOutput = self.__getIOTensorFromBatch(batch)
        return tensorInput, tensorOutput.view(-1)
    
    @staticmethod
    def __toListFormat(solvedSudokus, unsolvedSudokus):
        solvedSudokus = torch.argmax(solvedSudokus, dim=-1) + 1
        mask = torch.where(unsolvedSudokus>0)
        solvedSudokus[mask] = unsolvedSudokus[mask]
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
        solvedSudokus = self.__toListFormat(solvedSudokus, unsolvedSudokus)
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
        preValidationAccuracy = 0.
        validationAccuracy = 0.
        with torch.no_grad():
            for batchIndex, batch in enumerate(self.validDataLoader):
                unsolvedSudokus, _ = self.__getIOTensorFromBatch(batch)
                preSolvedSudokus, solvedSudokus = self.model(unsolvedSudokus)
                mask = torch.where(unsolvedSudokus.view(-1)==0)
                preValidationAccuracy += self.__checkBatchAccuracy(preSolvedSudokus.view(-1,9)[mask], batch[1].view(-1)[mask])
                validationAccuracy += self.__checkBatchAccuracy(solvedSudokus.view(-1,9)[mask], batch[1].view(-1)[mask])

        self.__writeLastBatch(unsolvedSudokus, solvedSudokus, batch[1].squeeze())
        logging.info('Pre-Validation Accuracy: {}%, Validation Accuracy: {}%'.format(preValidationAccuracy*100/(batchIndex+1) ,validationAccuracy*100/(batchIndex+1)))
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
                self.__logInformation(pred1, pred2, batch[1], mask, loss)

            self.__evaluate()
            self.__saveModel()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a DL based SudokuSolver')
    parser.add_argument('--batchSize', type=int, default=128) 
    parser.add_argument('--maxEpochs', type=int, default=2000) 
    parser.add_argument('--CSVDataPath', type=str, default='data/sudoku.csv')
    parser.add_argument('--outputSamplesDirectory', type=str, default='./out2')
    parser.add_argument('--logFile', type=str, default='./out2/train.log')
    parser.add_argument('--dataSamples', type=int, default=1000000, help='Number Of Samples Used from the CSV') 
    parser.add_argument('--dropSamplingStrategy', action='store_true')

    parser.add_argument('--numberOfKernels', type=int, default=32)
    parser.add_argument('--recurrentIterations', type=int, default=10)
    parser.add_argument('--learningRate', type=float, default=0.0001)
    
    parser.add_argument('--seed', type=int, default=1234) 
    parser.add_argument('--printEvery', type=int, default=1000, help='Trainning Logging Frequency in # of batches') 
    parameters = parser.parse_args() 
    trainner = Trainner(parameters)
    trainner.train()
