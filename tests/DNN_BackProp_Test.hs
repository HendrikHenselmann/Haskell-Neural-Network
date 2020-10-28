-- Copyright [2020] <Hendrik Henselmann>

import DNN
import Layer
import Matrix
import LossFuncs
import ActivationFuncs
import WeightInitializations

import Test.HUnit

-- random seed
seed :: Int
seed = 128

-- small constant
epsilon :: Double
epsilon = 0.01

------------------------------------------------------------------------------------
-- Declaration of some test Networks

testDNN_1 :: DNN
testDNN_1 = initDNN 1 [(DenseLayer, 1, identity)] simpleWeightInit seed

------------------------------------------------------------------------------------
-- Test cases

train_tests = TestCase (do
    -- First test      simple learning of input weights = 1.0 and bias weight = 0.0
    let inputMat = initMatrix 5 1 [-1.0 .. 3.0]
    let desiredOutput = initMatrix 5 1  [-1.0 .. 3.0]
    let desiredInputWeightsAfterTraining = initMatrix 1 1 [1.0]
    let desiredBiasWeightsAfterTraining = initMatrix 1 1 [0.0]

    let updatedDnn = snd (train seed 3000 1 0.01 squaredError testDNN_1 inputMat desiredOutput)

    let trainedInputWeights = inputWeights $ last $ layers updatedDnn
    let trainedBiasWeights = biasWeights $ last $ layers updatedDnn

    let correctInputWeights = matsAreEqualWithTolerance epsilon trainedInputWeights desiredInputWeightsAfterTraining
    let correctBiasWeights = matsAreEqualWithTolerance epsilon trainedBiasWeights desiredBiasWeightsAfterTraining

    assertBool "first case" (correctInputWeights && correctBiasWeights)

    )

------------------------------------------------------------------------------------
-- Name tests and group them together

tests = TestList [TestLabel "train" train_tests]

------------------------------------------------------------------------------------
-- Execute tests
main :: IO Counts
main = runTestTT tests
