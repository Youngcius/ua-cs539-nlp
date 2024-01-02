# Assignment-1

# Objectives

The learning objectives of this assignment are to:
1. practice spacy processes such as tokenization, parsing dependencies... etc.
2. become familiar with reading library documentation.
3. be able to read & understand [typing hints](https://docs.python.org/3/library/typing.html) provided in functions or methods.
4. learn how to asnwer assignments as described in docstrings.



# Setup your environment

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.8 or higher)](https://www.python.org/downloads/).
* [Conda virtural environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). If you want to have a virtural env. 
* [Spacy](https://spacy.io/usage). *you might need to install spacy models separately if needed*.
*  [pytest](https://docs.pytest.org/)


# Check out a new branch

Before you start editing any code, you will need to create a **new branch** in your
GitHub repository to hold your work. 
1. . Go to the repository that GitHub Classroom created for you, `https://github.com/ling-439-539-spring22/assignment-1-439-539-<your-username>`, where
`<your-username>` is your GitHub username, and
[create a branch through the GitHub interface](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/).
You may name the branch anything you like.
2. Clone the repository to your local machine and checkout the branch you
just created:
   ```
   git clone -b <branch> https://github.com/ling-439-539-spring22/assignment-1-439-539-<your-username>.git
   ```
  where `<branch>` is whatever you named your branch.


# Write your code

You will implement 8 functions in *main.py*. 
You should read the documentation strings (docstrings) in each of functions in
that file, and implement the functions as described. Write your code below the docstring of each function; **do not delete the
docstrings**.


# Testing your code.

**Tests have been provided for you in the `test_main.py` file.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_main.py``.
Initially, you will see output like:**


============================= test session starts ==============================

...
collected 8 items                                                              

test_main.py FFFFx   [100%]

============================= 8 passed in 1.59 seconds =========================


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
feedback in the form of comments on the pull request if needed.

# Grading

Assignments will be graded primarily on their ability to pass the tests that
have been provided to you.

