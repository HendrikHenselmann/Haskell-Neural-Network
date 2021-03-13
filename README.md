# Neural Network written in Haskell
 This is an implementation of a Neural Network written in Haskell. I wrote it only to deepen my understandings of Neural Networks. It is not meant to be used for real world application.

## Introduction

## Running the Main.hs
```
runhaskell -i{path}/src Main.hs
```
Where **{path}** is the path to the local folder of this repository. (Yep, there is no space between -i and the path.)

## Running a test file
```
runhaskell -i{path}/src {file}
```

Where **{path}** is the path to the local folder of this repository and **{file}** is the path to the test file you want to run. (Yep, there is no space between -i and the path.)

## Running all tests

```
for f in ./tests/*.hs; do runhaskell -i./src $f; done
```

Run the command above in the root directory of the project to execute all tests.

## Important notes!
* The are no learning episodes. Backpropagation takes a random observation (or batch of observations) at every backpropagation iteration. It's not going through the whole dataset. I think that this is okay if the number of learning steps is high enough.
