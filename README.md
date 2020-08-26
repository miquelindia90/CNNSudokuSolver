# CNNSudokuSolver
CNN based algorithm to solve a sudoku instead of using the standard backtracking algorithm.

This is a project I just did for fun in some free times when I could not focused in my PHD :).
Initially this repo was thought to develop a Sudoku solver with DNNs from zero and learn about
about the issues and challenges it has. Maybe in future this is more well explained in a blog.

If you are looking for an accurate and fast method to solve sudokus you are not in the right place.
This repository is more focused on testing and tryng to understand what is able to learn a DNN from a
dataset in a supervised way. 

# Requirements

I've been developing these scripts with python 3.6 and pytorch. You can find the dependencies in the
requirements.txt file. I recommend to just use:

```bash
pip install -r requirements.txt
```

# Dataset

If you want to try to train some models so as to reproduce the results you need a dataset with tuples
of unsolved/solved sudokus. Here we have used the [1 million Sudoku games](https://www.kaggle.com/bryanpark/sudoku) corpus from Kaggle. The code
is adapted to use their sudoku.csv file. You can download it from the following [link](https://www.kaggle.com/bryanpark/sudoku).

# Training

To launch a training, you only need to run the following script: 

```batch
python scripts/train.py --CSVDataPath <path_to_csv/sudoku.scv> --dropSamplingStrategy
```

The code automatically detects if there is a GPU available and stars the training.
If not the training is run with CPUs. We recommend to use GPUs for the training for the speed boost.
The system will store the model and a validation few samples for each epoch in the directory path stablished in --outputSamplesDirectory.
 
# Inference 

To use the code for solve a sudoku you need to use the script `solve.py`.

This code reads a quizz sudoku from a file, prints the results and evaluates if is correct.

Example:
Consider the following sudoku, which corresponds to the sample in `data/sudoku1.txt`:

<pre>
1  0  0  0  6  0  0  0  2
0  8  0  0  3  9  0  0  0
3  0  0  4  1  2  0  0  7
0  2  0  0  4  0  7  0  1
0  1  6  0  5  0  9  8  0
9  0  4  0  0  7  0  0  6
0  9  7  5  0  0  6  0  3
0  0  1  0  0  8  5  0  0
2  5  0  0  9  0  0  0  0</pre>


If you run the following script:

```batch
python scripts/solve.py model/modelV1.pt data/sudoku1.txt
```

You will get the result of the sudoku and a message with the result:

<pre>
1  7  9  8  6  5  3  4  2
4  8  2  7  3  9  1  6  5
3  6  5  4  1  2  8  9  7
5  2  8  9  4  6  7  3  1
7  1  6  2  5  3  9  8  4
9  3  4  1  8  7  2  5  6
8  9  7  5  2  4  6  1  3
6  4  1  3  7  8  5  2  9
2  5  3  6  9  1  4  7  8

Fast Sudoku Is Correct

Sudokus Solved: 100.0%</pre>

This script can be run to evaluate more than one sudoku. It is only needed tp separate the sudokus in the test file with break lines. You can 
launch the following example to see how it works:

```batch
python scripts/solve.py model/modelV1.pt data/sudokusTest.txt
```


