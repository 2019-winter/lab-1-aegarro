---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Alexandra Garro
**PUT YOUR FULL NAME(S) HERE**


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


1. I would like a further explanation on the difference between markdown files and notebook files and why the 2 are needed.
2. When working as a team, does it work the same as Git where everyone needs to commit and pull.
3. In the Notebook Basics site, it is mainly about the notebook server. I was wondering if this step is needed if you use the web application as I just went to the terminal and pulled from the Github repo.


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.


## Exercise 1

```python
import numpy as np
```

```python
# YOUR SOLUTION HERE
a = np.full((6,4),2)
a
```

## Exercise 2

```python
# YOUR SOLUTION HERE
b = np.full((6,4),1)
np.fill_diagonal(b,3)
b
```

## Exercise 3

```python
# YOUR SOLUTION HERE
# a * b
# np.dot(a,b)
```

The solution np.dot(a,b) does NOT work since the dot product for arrays expects the same length where the columns of the first need to be the same as the rows of the second array and vise versa.


## Exercise 4

```python
# YOUR SOLUTION HERE
c = np.dot(a.transpose(),b)
d = np.dot(a, b.transpose())

print(c)
print(d)
```

transpose() flips the rows to columns and the columns to rows. Running a.transpose() and b.transpose() both changes a and b into 6x4 instead of a 4x6 array. Running np.dot(a.transpose(),b) and using the rules of a matrix multiplication, the output is a 4x6 array and running np.dot(a, b.transpose()) results in a 6x4 array. 


## Exercise 5

```python
# YOUR SOLUTION HERE
def myFunction():
    print("This is my function")
    
myFunction()
```

## Exercise 6

```python
# YOUR SOLUTION HERE
def randArraySum():
    ra = np.random.rand(3,2,2)
    print(ra)
    print("Sum =",ra.sum())
    print("Mean =",ra.mean())
randArraySum()
```

## Exercise 7


Define a function that counts number of ones

```python
exampleA = np.array([1,2,3,4,1])
```

```python
# YOUR SOLUTION HERE

def countOnes(givenArray):
    count = 0
    for num in givenArray:
        if num == 1:
            count += 1
    return count
countOnes(exampleA)
```

Count number of ones using np.where

```python
def countOnes2(arr):
    outputArray = np.where(arr==1, arr, 0)
    return outputArray.sum()

assert countOnes(np.array([1,2])) == 1
assert countOnes(np.array([1,2,1])) == 2
assert countOnes(np.array([3,2,15])) == 0
```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.

```python
import pandas as pd
```

## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
# YOUR SOLUTION HERE
a = np.full(shape=(6,4), fill_value=2)
a = pd.DataFrame(a)
a
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python
# YOUR SOLUTION HERE
b = np.full((6,4),1)
np.fill_diagonal(b,3)
b = pd.DataFrame(b)
b
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.

```python
# YOUR SOLUTION HERE
a * b
# np.dot(a,b)
```

Same explanation as before -> see explanation for Exercise 3


## Exercise 11
Repeat exercise A.7 using a dataframe.


Define a function that counts number of ones

```python
exampleA = np.array([1,2,3,4,1])
```

```python
# YOUR SOLUTION HERE
def countOnes(givenArray):
    count = 0
    for num in givenArray:
        if num == 1:
            count += 1
    return count
countOnes(exampleA)
```

```python
def countOnes2(df):
    return df.where(df==1,0).sum().sum()

assert countOnes2(b) == 20
assert countOnes2(a) == 0

test = pd.DataFrame([1,2,3,4])
assert countOnes2(test) == 1

test2 = pd.DataFrame([1,2,1,4])
assert countOnes2(test2) == 2
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
## YOUR SOLUTION HERE
titanic_df["name"]
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
titanic_df.set_index('sex',inplace=True)
```

```python
## YOUR SOLUTION HERE
len(titanic_df.loc["female"])
```

## Exercise 14
How do you reset the index?

```python
## YOUR SOLUTION HERE
```

```python
titanic_df.reset_index(inplace=True)
```

```python
titanic_df.head()
```

```python

```
