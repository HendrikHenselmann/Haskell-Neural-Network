-- Copyright [2020] <Hendrik Henselmann>
module BackPropagation_Tests (tests) where

import DNN
import Layer
import Matrix
import Pipeline
import LossFuncs
import ActivationFuncs
import WeightInitializations

import Test.HUnit

-- random seed
seed :: Int
seed = 128

-- small constant
epsilon :: Double
epsilon = 0.1

------------------------------------------------------------------------------------
-- Declaration of some test Networks

simpleDNN :: DNN
simpleDNN = initDNN 1 [(DenseLayer, 1, identity)] simpleWeightInit seed

additionDNN :: DNN
additionDNN = initDNN 2 [(DenseLayer, 1, leakyRelu 0.2)] kaiminWeightInit seed

additionPipe :: Pipeline
additionPipe = initPipeline Nothing additionDNN Nothing

------------------------------------------------------------------------------------
-- Test cases

noLearning_test = TestCase (do
    -- simple learning of input weight = 1.0 and bias weight = 0.0
    let inputMat = initMatrix 5 1 [-1.0 .. 3.0]
    let desiredOutput = initMatrix 5 1  [-1.0 .. 3.0]
    let desiredInputWeightsAfterTraining = initMatrix 1 1 [1.0]
    let desiredBiasWeightsAfterTraining = initMatrix 1 1 [0.0]

    let updatedDNN = snd $ train seed 3000 1 0.01 squaredError simpleDNN inputMat desiredOutput

    let trainedInputWeights = inputWeights $ last $ layers updatedDNN
    let trainedBiasWeights = biasWeights $ last $ layers updatedDNN

    let correctInputWeights = matsAreEqualWithTolerance epsilon trainedInputWeights desiredInputWeightsAfterTraining
    let correctBiasWeights = matsAreEqualWithTolerance epsilon trainedBiasWeights desiredBiasWeightsAfterTraining

    assertBool "Incorrect weights and bias!" (correctInputWeights && correctBiasWeights)
    )

learnAddition_test = TestCase (do
    -- simple learning of input weights = 1.0 and bias weight = 0.0
    let inputMat = initMatrix 6 2 [0.0, 0.0,
                                   1.0, 0.0,
                                   0.0, 1.0,
                                   1.0, 1.0,
                                   3.0, 10.0,
                                   40.0, 20.0]
    let desiredOutput = initMatrix 6 1 [0.0,
                                        1.0,
                                        1.0,
                                        2.0,
                                        13.0,
                                        60.0]
    let desiredInputWeightsAfterTraining = initMatrix 2 1 [1.0, 1.0]
    let desiredBiasWeightsAfterTraining = initMatrix 1 1 [0.0]

    let updatedPipe = snd $ trainPipe seed 100000 3 0.0026 squaredError additionPipe inputMat desiredOutput
    let updatedDNN = getDNN updatedPipe

    let trainedInputWeights = inputWeights $ last $ layers updatedDNN
    let trainedBiasWeights = biasWeights $ last $ layers updatedDNN

    let correctInputWeights = matsAreEqualWithTolerance epsilon trainedInputWeights desiredInputWeightsAfterTraining
    let correctBiasWeights = matsAreEqualWithTolerance epsilon trainedBiasWeights desiredBiasWeightsAfterTraining

    assertBool "Incorrect weights and bias!" (correctInputWeights && correctBiasWeights)
    )

------------------------------------------------------------------------------------
-- Name tests and group them together

tests = [TestLabel "trivial learn nothing task" noLearning_test, TestLabel "Learning addition task" learnAddition_test]

------------------------------------------------------------------------------------
