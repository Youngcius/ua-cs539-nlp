# Assignment-3

# Objectives

The learning objectives of this assignment are to:
1. practice performing basic arithmetic on word vectors (e.g. cosine, etc). 
2. demonstrate building and using both sparse and dense word vectors
3. demonstrate loading word vectors from a file (GloVe embeddings)
4. practice calculating centroid vectors and tf.idf vectors
5. apply this knowledge to a multiple-choice question answering task.

# Setup your environment

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.6 or higher)](https://www.python.org/downloads/)
* [pytest](https://docs.pytest.org)

**PLEASE NOTE that you are not allowed to use any external libraries (e.g. no numpy/scipy/etc).** You must implement everything by yourself. 

# Check out a new branch

Before you start editing any code, you will need to create a new branch in your
GitHub repository to hold your work.

1. Go to the repository that GitHub Classroom created for you,
`https://github.com/ling-439-539-spring22/Assignment-3-<your-username>`, where
`<your-username>` is your GitHub username, and
[create a branch through the GitHub interface](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/).
You may name the branch anything you like.
2. Clone the repository to your local machine and checkout the branch you
just created:
   ```
   git clone -b <branch> https://github.com/ling-439-539-spring22/Assignment-3-<your-username>.git
   ```
   where `<branch>` is whatever you named your branch.
   
# Write your code
In `main.py` file you are provided with two classes: *VecDense* and *VecSparseTFIDF*, plus two functions: *loadVectors*, and *doQuestionAnsweringCentroidDense*. 
You should read the documentation strings (docstrings) in each of methods in
that file, and implement the methods as described.
Write your code below the docstring of each method; **do not delete the
docstrings**.


# Test your code

Tests have been provided for you in the `test_vectors.py` file.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_vectors.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
collected 13 items

test_vectors.py .............                                            [100%]

============================== 13 passed in 0.08s ==============================
```


# Submit your code

As you are working on the code, you should regularly `git commit` to save your
current changes locally and `git push` to push all saved changes to the remote
repository on GitHub.


# Submit your code

As you are working on the code, you should regularly `git commit` to save your
current changes locally and `git push` to push all saved changes to the remote
repository on GitHub.

To submit your assignment,
[create a pull request on GitHub](https://help.github.com/articles/creating-a-pull-request/#creating-the-pull-request).
where the "base" branch is "master", and the "compare" branch is the branch you
created at the beginning of this assignment.
Then go to the "Files changed" tab, and make sure that you have only changed
the `main.py` file and that all your changes look as you would expect them
to.
**Do not merge the pull request.**

Your instructor will grade the code of this pull request, and provide you
feedback in the form of comments on the pull request.

# Grading

Assignments will be graded primarily on their ability to pass the tests that have been provided to you.  Assignments that faithfully pass tests will be given a grade corresponding to the proportion of tests they pass.  For example, assignments that pass half the tests will be given a grade of **50%**.  Assignments that pass all **13** tests will be given **100%**. 
