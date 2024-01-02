# Assignment-2-bootsrapping

## Objectives

The learning objective of this assignment is to:

- practice implementing bootstrap resampling techinques. 

# Setup your environment

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.6 or higher)](https://www.python.org/downloads/)
* [numpy (version 1.11 or higher)](http://www.numpy.org/)
* [Generate pseudo-random numbers](https://docs.python.org/3/library/random.html)

# Check out a new branch

Before you start editing any code, you will need to create a new branch in your
GitHub repository to hold your work.


. Go to the repository that GitHub Classroom created for you,
`https://github.com/ling-439-539-spring22/assignment2-2-bootstrap-resampling-<your-username>`, where
`<your-username>` is your GitHub username, and
[create a branch through the GitHub interface](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/).
You may name the branch anything you like.
2. Clone the repository to your local machine and checkout the branch you
just created:
   ```
   git clone -b <branch> https://github.com/ling-439-539-spring22/assignment2-2-bootstrap-resampling-<your-username>.git
   ```
   where `<branch>` is whatever you named your branch.



# Write your code

You will implement 5 methods in `BootstrapResampling.py`. You should read the docstring in each method, and implement the methods as described. Write your code below the docstring of each function; **do not delete the docstrings**.

# Test your code

Tests have been provided for you in the `test_bootstrap.py` file.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_bootstrap.py``.

- You must pass all tests. 

# Submit your code

As you are working on the code, you should regularly `git commit` to save your
current changes locally and `git push` to push all saved changes to the remote
repository on GitHub.

To submit your assignment,
[create a pull request on GitHub](https://help.github.com/articles/creating-a-pull-request/#creating-the-pull-request).
where the "base" branch is "master", and the "compare" branch is the branch you
created at the beginning of this assignment.
Then go to the "Files changed" tab, and make sure that you have only changed
the `BootstrapResampling.py` file and that all your changes look as you would expect them
to.
**Do not merge the pull request.**

Your instructor will grade the code of this pull request, and provide you
feedback in the form of comments on the pull request.

