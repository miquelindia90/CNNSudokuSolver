import os
import sys
import copy
import torch
import itertools


class SudokuChecker:
    def __init__(self):
        pass
   
    @staticmethod 
    def __checkSudokuConstraint(line):
        return (len(line) == 9 and sum(line) == sum(set(line)))

    def checkSudoku(self, grid):

        bad_rows = [row for row in grid if not self.__checkSudokuConstraint(row)]
        grid = list(zip(*grid))
        bad_cols = [col for col in grid if not (col)]
        squares = []
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                square = list(itertools.chain(row[j:j+3] for row in grid[i:i+3]))
                squareNumbers = list()
                for column in square:
                    for elem in column:
                        squareNumbers.append(elem)
                squares.append(squareNumbers)

        bad_squares = [square for square in squares if not self.__checkSudokuConstraint(square)]
        return not (bad_rows or bad_cols or bad_squares)

class SudokuPlotter:
    def __init__(self):
        pass
    
    @staticmethod    
    def __toStr(intList):
    
        strList = [str(element) for element in intList]
        return strList
    
    def plotMatrix(self, sudokuTensor):
        sudokuMatrix = copy.copy(sudokuTensor).view(9,9).tolist()
        for row in sudokuMatrix:
            print('{}'.format(' '.join(self.__toStr(row))))
        print('')

class SudokuSolver:
    def __init__(self, modelPath):
        self.sudokuChecker = SudokuChecker()
        self.plotter = SudokuPlotter()
        self.__loadModel(modelPath)

    def __loadModel(self, modelPath):
        self.model = torch.load(modelPath, map_location=torch.device('cpu'))
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
        outputSudokuTensor = self.model(sudokuTensor)
        solutionTensor = torch.argmax(outputSudokuTensor,2)
        solutionTensor += 1  
        self.plotter.plotMatrix(solutionTensor)
        return solutionTensor

    def __solveSudokuStepByStep(self, sudoku):
        sudokuTensor = self.__prepareSudokuTensor(sudoku)
        step=1
        while torch.sum(sudokuTensor==0)>0:
            outputSudokuTensor = self.model(sudokuTensor)
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
        else:
            print('{} Sudoku Is UnCorrect\n'.format(speed))

    def solveSudoku(self, sudoku, fast=False):
        solvedSudokuTensor = self.__getSudokuSolution(sudoku, fast)
        self.__evaluateSudoku(solvedSudokuTensor, fast)
         

def readSudokuFromTest(sudokuPath):
    sudokuAsList = list()
    with open(sudokuPath) as inputFile:
        for line in inputFile:
            sudokuAsList.append(''.join(line.strip().split()))
    return sudokuAsList

if __name__ == '__main__':

    sudokuSolver = SudokuSolver(sys.argv[1])
    sudoku = readSudokuFromTest(sys.argv[2])
    sudokuSolver.solveSudoku(sudoku)
    
