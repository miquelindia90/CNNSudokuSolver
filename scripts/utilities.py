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

def convertSudokuTextToListFormat(sudoku):

    sudokuAsList = list()
    lines = sudoku.split('\n')
    for line in lines:
        sudokuAsList.append(''.join(line.strip().split()))
    return sudokuAsList

def readSudokusFromTextFile(sudokuPath):

    listOfInputSudokus = list()
    with open(sudokuPath) as inputFile:
        sudokus = inputFile.read().split('\n\n')
        for sudoku in sudokus:
            sudokuAsList = convertSudokuTextToListFormat(sudoku)
            listOfInputSudokus.append(sudokuAsList)

    return listOfInputSudokus
