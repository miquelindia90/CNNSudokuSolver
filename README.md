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

# Trainning

To launch a training, you only need to run the following script: 

```batch
python scripts/train.py --CSVDataPath <path_to_csv/sudoku.scv> --dropSamplingStrategy
```

The code automatically detects if there is a GPU available and stars the training.
If not the training is run with CPUs. We recommend to use GPUs for the training for the speed boost.
The system will store the model and a few samples each epoch in the directory path stablished --outputSamplesDirectory.
 
# Inference 

Under Construction
