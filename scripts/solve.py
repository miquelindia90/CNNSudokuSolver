import os
import sys
import copy
import torch
import itertools

from model import *
from utilities import *

class SudokuSolver:
    def __init__(self, modelPath, recurrentIterations=10):
        self.sudokuChecker = SudokuChecker()
        self.plotter = SudokuPlotter()
        self.__loadModel(modelPath, recurrentIterations)

    def __loadModel(self, modelPath, recurrentIterations):

        self.model = SudokuNet(recurrentIterations=recurrentIterations)
        self.model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
        self.model.to(torch.device('cpu'))
        self.model.eval()

    @staticmethod
    def __prepareSudokuTensor(sudoku):
        sudokuTensor = torch.zeros((1,81))
        for rowIndex, row in enumerate(sudoku):
            for columnIndex, value in enumerate(row):
                sudokuTensor[0, rowIndex*9 + columnIndex] = int(value)
        return sudokuTensor.long()

    def __solveSudokuInOneStep(self, sudoku):
        sudokuTensor = self.__prepareSudokuTensor(sudoku)
        _, outputSudokuTensor = self.model(sudokuTensor)
        solutionTensor = torch.argmax(outputSudokuTensor,-1)
        solutionTensor += 1
        mask = torch.where(sudokuTensor==0)
        sudokuTensor[mask]+=solutionTensor[mask]
        self.plotter.plotMatrix(sudokuTensor)
        return sudokuTensor

    def __solveSudokuStepByStep(self, sudoku):
        sudokuTensor = self.__prepareSudokuTensor(sudoku)
        step=1
        while torch.sum(sudokuTensor==0)>0:
            _, outputSudokuTensor = self.model(sudokuTensor)
            decisionProbabilities, solutionTensor = torch.max(outputSudokuTensor,2) 
            _, mask = torch.where(sudokuTensor>0)
            decisionProbabilities[0,mask] = float('-inf')
            mostLikelyCell = torch.argmax(decisionProbabilities)
            sudokuTensor[0,mostLikelyCell] = solutionTensor[0,mostLikelyCell]+1
            print('Step: {}'.format(step))
            print('Filled Row {}, Column {}'.format(mostLikelyCell//9+1,mostLikelyCell%9+1))
            self.plotter.plotMatrix(sudokuTensor)
            step+=1
        
        return sudokuTensor

    def __getSudokuSolution(self, sudoku, fast):

        with torch.no_grad():
            if fast:
                solvedSudoku = self.__solveSudokuInOneStep(sudoku)
            else:
                solvedSudoku = self.__solveSudokuStepByStep(sudoku)
        return solvedSudoku


    def __evaluateSudoku(self, sudokuTensor, fast):

        speed = 'Fast' if fast else 'Slow'
        sudokuMatrix = copy.copy(sudokuTensor).view(9,9).tolist()
        if self.sudokuChecker.checkSudoku(sudokuMatrix):
            print('{} Sudoku Is Correct\n'.format(speed))
            return True
        else:
            print('{} Sudoku Is UnCorrect\n'.format(speed))
            return False

    def solveSudoku(self, sudoku, fast=False):
        solvedSudokuTensor = self.__getSudokuSolution(sudoku, fast)
        self.__evaluateSudoku(solvedSudokuTensor, fast)
    
    def solveSudokus(self, sudokus, fast=False):
        accuracyCount = 0.
        for sudoku in sudokus:
            solvedSudokuTensor = self.__getSudokuSolution(sudoku, fast)
            if self.__evaluateSudoku(solvedSudokuTensor, fast):
                accuracyCount += 1
        print('Sudokus Solved: {}%'.format(accuracyCount*100/len(sudokus)))
         

def readSudokusFromTest(sudokuPath):

    listOfInputSudokus = list()
    with open(sudokuPath) as inputFile:
        sudokus = inputFile.read().split('\n\n')
        for sudoku in sudokus:
            sudokuAsList = list()
            lines = sudoku.split('\n')    
            for line in lines:
                sudokuAsList.append(''.join(line.strip().split()))
            listOfInputSudokus.append(sudokuAsList)

    return listOfInputSudokus

if __name__ == '__main__':

    sudokuSolver = SudokuSolver(sys.argv[1])
    sudokus = readSudokusFromTextFile(sys.argv[2])
    sudokuSolver.solveSudokus(sudokus, fast=True)
    
