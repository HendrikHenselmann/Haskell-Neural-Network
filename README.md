# Neural Network written in Haskell
 This is an implementation of a Neural Network written in Haskell. I wrote it solely to deepen my understanding of Neural Networks. It is not meant to be used for real world applications.

## Setup
You need the Haskell tool **stack** to setup, build and execute this project.
Execute the following steps:
<ol>
<li> Download this project folder from GitHub.</li>
<li> Open the terminal and navigate into the downloaded project directory.</li>
<li> Use the following commands to setup and build the project using <strong>stack</strong>:
 
         stack setup
         stack build
 </li>
</ol>

## Running the Main file (Main.hs)
Use the following command to execute the Main file in the "./app/" subdirectory, after you have completed the setup steps above.
```
stack exec Haskell-Neural-Network-exe
```

## Running the tests
Use the following command to execute all the unit tests in the "./test/" subdirectory, after you have completed the setup steps above.
```
stack test
```

<br/><br/>

## Important notes / Clean Code violations / gathered experience
* At every learning iteration a random observation (or batch of observations) is selected. There are no learning episodes, meaning that there is no mechanism that ensures that the whole dataset is considered at learning. I think that this is okay if the number of learning steps is high enough.
* I've not implemented Exceptions. In the case of failure, e.g. unexpected / inconsistent input, failure indicating output / output datastructures will be propagated through. For example the emptyMatrix, emptyDNN, emptyPipe are structures indicating failure. This is certainly not good practice, since debugging gets very difficult.
