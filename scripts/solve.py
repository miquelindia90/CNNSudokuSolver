import os
import sys
import copy
import torch
import itertools

def sudoku_ok(line):
    return (len(line) == 9 and sum(line) == sum(set(line)))

def checkSudoku(grid):

    bad_rows = [row for row in grid if not sudoku_ok(row)]
    grid = list(zip(*grid))
    bad_cols = [col for col in grid if not sudoku_ok(col)]
    squares = []
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            square = list(itertools.chain(row[j:j+3] for row in grid[i:i+3]))
            squareNumbers = list()
            for column in square:
                for elem in column:
                    squareNumbers.append(elem)
            squares.append(squareNumbers)

    bad_squares = [square for square in squares if not sudoku_ok(square)]
    return not (bad_rows or bad_cols or bad_squares)

def checkIfSudokuIsCorrect(sudokuTensor):

    sudokuMatrix = copy.copy(sudokuTensor).view(9,9).tolist()
    return checkSudoku(sudokuMatrix)

def toStr(intList):

    strList = [str(element) for element in intList]
    return strList

def plotMatrix(sudokuTensor):
    sudokuMatrix = copy.copy(sudokuTensor).view(9,9).tolist()
    for row in sudokuMatrix:
        print('{}'.format(' '.join(toStr(row))))
    print('')

def prepareTensor(sudoku):
    tensor = torch.zeros((1,81))
    for rowIndex, row in enumerate(sudoku):
        for columnIndex, value in enumerate(row):
            tensor[0, rowIndex*9 + columnIndex] = int(value)

    return tensor.long()

def solveSudokuStepByStep(model, sudokuTensor):

    step=1
    while torch.sum(sudokuTensor==0)>0:
        outputSudokuTensor = model(sudokuTensor)
        decisionProbabilities, solutionTensor = torch.max(outputSudokuTensor,2) 
        _, mask = torch.where(sudokuTensor>0)
        decisionProbabilities[0,mask] = float('-inf')
        mostLikelyCell = torch.argmax(decisionProbabilities)
        sudokuTensor[0,mostLikelyCell] = solutionTensor[0,mostLikelyCell]+1
        print('Step: {}'.format(step))
        print('Filled Row {}, Column {}'.format(mostLikelyCell//9+1,mostLikelyCell%9+1))
        plotMatrix(sudokuTensor)
        step+=1
    
    return sudokuTensor

def solveSudokuFast(model, sudokuTensor):

    outputSudokuTensor = model(sudokuTensor)
    solutionTensor = torch.argmax(outputSudokuTensor,2)
    solutionTensor += 1  
    plotMatrix(solutionTensor)
    return solutionTensor 

if __name__ == '__main__':

    model = torch.load(sys.argv[1], map_location=torch.device('cpu'))
    model.eval()
    #sudoku = ['000402063', '005810097', '431006500', '010904000', '847000006', '500180002', '089571000', '063000000', '000000279']
    sudoku = ['800370200', '305840760', '970065043', '436092507', '008607400', '057403016', '680034105', '700520690', '042906008']
    #864371259 325849761 971265843 436192587 198657432 257483916 689734125 713528694 542916378

    print('Proposed Sudoku')
    plotMatrix(prepareTensor(sudoku))
    
    #Slow Method
    sudokuSolvedTensor = solveSudokuStepByStep(model, prepareTensor(sudoku)) 
    if checkIfSudokuIsCorrect(sudokuSolvedTensor):
        print('Slow Sudoku Is Correct\n')
    else:
        print('Slow Sudoku Is UnCorrect\n')

    print('--------------------------------\n')
    
    ##Fast Method

    #sudokuSolvedTensor = solveSudokuFast(model, prepareTensor(sudoku)) 
    #if checkIfSudokuIsCorrect(sudokuSolvedTensor):
    #    print('Fast Sudoku Is Correct')
    #else:
    #    print('Fast Sudoku Is UnCorrect')

    
