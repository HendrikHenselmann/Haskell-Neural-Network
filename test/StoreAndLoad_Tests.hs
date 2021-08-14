-- Copyright [2020] <Hendrik Henselmann>
module StoreAndLoad_Tests (tests) where

import DNN
import Layer
import Matrix
import Scaler
import Pipeline
import LossFuncs
import ActivationFuncs
import WeightInitializations

import Test.HUnit
import Data.Maybe
import System.Directory

------------------------------------------------------------------------------------
-- random seed
seed :: Int
seed = 128

-- very small constant
epsilon :: Double
epsilon = 0.0001

------------------------------------------------------------------------------------
-- test functions

-- comparing layer types
checkLayerTypes :: [Layer] -> [Layer] -> Bool
checkLayerTypes [] [] = True
checkLayerTypes _ [] = False
checkLayerTypes [] _ = False
checkLayerTypes (layer1:layers1) (layer2:layers2) = currentAreSame && restAreSame
    where
        currentAreSame = sameLayerType (layerType layer1) (layerType layer2)
        restAreSame = checkLayerTypes layers1 layers2

-- comparing layer sizes
checkLayerSizes :: [Layer] -> [Layer] -> Bool
checkLayerSizes [] [] = True
checkLayerSizes _ [] = False
checkLayerSizes [] _ = False
checkLayerSizes (layer1:layers1) (layer2:layers2) = currentAreSame && restAreSame
    where
        currentAreSame = (size layer1) == (size layer2)
        restAreSame = checkLayerSizes layers1 layers2

-- comparing layer activations
checkLayerActiations :: [Layer] -> [Layer] -> Bool
checkLayerActiations [] [] = True
checkLayerActiations _ [] = False
checkLayerActiations [] _ = False
checkLayerActiations (layer1:layers1) (layer2:layers2) = currentAreSame && restAreSame
    where
        firstActFnc = actFnc layer1
        secondActFnc = actFnc layer2 
        firstActFncName = name firstActFnc
        secondActFncName = name secondActFnc
        currentAreSame = (firstActFncName == secondActFncName) && (paramsAreSame (params firstActFnc) (params secondActFnc))
        restAreSame = checkLayerActiations layers1 layers2

        paramsAreSame :: Maybe [Double] -> Maybe [Double] -> Bool
        paramsAreSame Nothing Nothing = True
        paramsAreSame params1 params2
            | (isNothing params1) = False
            | (isNothing params2) = False
            | otherwise = doublesAreSame (fromJust params1) (fromJust params2)
            where
                doublesAreSame :: [Double] -> [Double] -> Bool
                doublesAreSame [] [] = True
                doublesAreSame [] _ = False
                doublesAreSame _ [] = False
                doublesAreSame (p1:params1) (p2:params2) = (sameWithTolerance p1 p2) && (doublesAreSame params1 params2)
                    where
                        sameWithTolerance :: Double -> Double -> Bool
                        sameWithTolerance param1 param2 = ((param1 + epsilon) >= param2) && ((param1 - epsilon) <= param2)

-- compare layers input weights
checkLayerInputWeights :: [Layer] -> [Layer] -> Bool
checkLayerInputWeights [] [] = True
checkLayerInputWeights [] _ = False
checkLayerInputWeights _ [] = False
checkLayerInputWeights (layer1:layers1) (layer2:layers2) = currentAreSame && restAreSame
    where
        currentAreSame = matsAreEqualWithTolerance epsilon (inputWeights layer1) (inputWeights layer2)
        restAreSame = checkLayerInputWeights layers1 layers2

-- compare layers bias weights
checkLayerBiasWeights :: [Layer] -> [Layer] -> Bool
checkLayerBiasWeights [] [] = True
checkLayerBiasWeights [] _ = False
checkLayerBiasWeights _ [] = False
checkLayerBiasWeights (layer1:layers1) (layer2:layers2) = currentAreSame && restAreSame
    where
        currentAreSame = matsAreEqualWithTolerance epsilon (biasWeights layer1) (biasWeights layer2)
        restAreSame = checkLayerInputWeights layers1 layers2

------------------------------------------------------------------------------------
-- Test cases

store_and_load_dnn_test = TestCase (do
    -- init network and copy weight matrices
    let initialDnn = initDNN 2 [(DenseLayer, 2, leakyRelu 0.2), (DenseLayer, 3, sigmoid)] kaiminWeightInit 0
    let initialLastLayersInputWeights = inputWeights $ last $ layers initialDnn
    let initialLastLayersBiasWeights = biasWeights $ last $ layers initialDnn

    -- store network
    storeDNN initialDnn "./storeDNN_loadDNN_test.data" True

    -- load network
    loadedDnn <- loadDNN "./storeDNN_loadDNN_test.data"

    -- check network properties
    assertBool "layer type" (checkLayerTypes (layers initialDnn) (layers loadedDnn))
    assertBool "layer size" (checkLayerSizes (layers initialDnn) (layers loadedDnn))
    assertBool "layer activation" (checkLayerActiations (layers initialDnn) (layers loadedDnn))

    -- compare weight mats
    assertBool "input weights" (checkLayerInputWeights (layers initialDnn) (layers loadedDnn))
    assertBool "bias weights" (checkLayerBiasWeights (layers initialDnn) (layers loadedDnn))

    -- compare output of initial and loaded dnn at given input
    let inputMat = initMatrix 5 2 [0.0, 0.0,
                                   1.0, 0.0,
                                   0.0, -5.0,
                                   20.43, -89.02,
                                   -3.898, -45.394804]
    let initialOutput = createOutput initialDnn inputMat
    let loadedDnnOutput = createOutput loadedDnn inputMat
    assertBool "output" (matsAreEqualWithTolerance epsilon initialOutput loadedDnnOutput) 

    -- delete test file of stored dnn
    removeFile "./storeDNN_loadDNN_test.data"
    )

store_and_load_pipe_test = TestCase (do
    -- init network and pipe and copy weight matrices
    let initialDnn = initDNN 2 [(DenseLayer, 2, leakyRelu 0.2), (DenseLayer, 3, sigmoid)] kaiminWeightInit 0
    let initialLastLayersInputWeights = inputWeights $ last $ layers initialDnn
    let initialLastLayersBiasWeights = biasWeights $ last $ layers initialDnn
    let initialPipe = initPipeline (Just standardization) initialDnn (Just softmax)

    -- store Pipeline
    storePipe initialPipe "./storePipe_loadPipe_test.data"

    -- load Pipeline
    loadedPipe <- loadPipe "./storePipe_loadPipe_test.data"
    let loadedDnn = getDNN loadedPipe

    -- check input and output scalers
    assertBool "input scaler" ((inScalerName $ fromJust $ getInScaler loadedPipe) == "standardization")
    assertBool "output scaler" ((outScalerName $ fromJust $ getOutScaler loadedPipe) == "softmax")

    -- check network properties
    assertBool "layer type" (checkLayerTypes (layers initialDnn) (layers loadedDnn))
    assertBool "layer size" (checkLayerSizes (layers initialDnn) (layers loadedDnn))
    assertBool "layer activation" (checkLayerActiations (layers initialDnn) (layers loadedDnn))

    -- compare weight mats
    assertBool "input weights" (checkLayerInputWeights (layers initialDnn) (layers loadedDnn))
    assertBool "bias weights" (checkLayerBiasWeights (layers initialDnn) (layers loadedDnn))

    -- compare output of initial and loaded dnn at given input
    let inputMat = initMatrix 5 2 [0.0, 0.0,
                                   1.0, 0.0,
                                   0.0, -5.0,
                                   20.43, -89.02,
                                   -3.898, -45.394804]
    let initialOutput = createOutput initialDnn inputMat
    let loadedDnnOutput = createOutput loadedDnn inputMat
    assertBool "output" (matsAreEqualWithTolerance epsilon initialOutput loadedDnnOutput) 

    -- delete test file of stored dnn
    removeFile "./storePipe_loadPipe_test.data"
    )

------------------------------------------------------------------------------------
-- Name tests and group them together

tests = [TestLabel "store and load dnn test" store_and_load_dnn_test, TestLabel "store and load pipe test" store_and_load_pipe_test]

------------------------------------------------------------------------------------
