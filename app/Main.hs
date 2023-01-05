-- Copyright [2020] <Hendrik Henselmann>

import DNN
import Layer
import Scaler
import Matrix
import Pipeline
import ReadData
import LossFuncs
import OneHotEncoding
import ActivationFuncs
import PerformanceMetrics
import WeightInitializations

import Text.Printf

------------------------------------------------------------------------------------
printLoss :: [Double] -> IO ()
printLoss [] = return ()
printLoss (loss:lossList) = do
    print loss
    printLoss lossList

------------------------------------------------------------------------------------
-- important learning (hyper) parameters
seed :: Int
seed = 128

trainingSteps :: Int
trainingSteps = 3000

learningRate :: Double
learningRate = 0.01

lossFunction :: LossFunc
lossFunction = squaredError

performanceMetric :: Matrix -> Matrix -> Matrix
performanceMetric = meanAbsoluteError

------------------------------------------------------------------------------------
-- self created test data with binary labels
-- output should be addition of input

numFeatures = 2

featuresArray = [0.0, 0.0,
                 1.0, 0.0,
                 0.0, 1.0,
                 1.0, 1.0,
                 3.0, 2.0,
                 0.0, 4.0,
                 2.0, 2.0,
                 4.5, 2.5,
                 2.3, 5.6]

trueFunc :: Double -> Double -> Double
trueFunc x y = 2*x + 0.5*y + 10

features = initMatrix (div (length featuresArray) numFeatures) numFeatures featuresArray
labels = applyToMatRowWise trueFunc features

------------------------------------------------------------------------------------

-- Deep Learning Classification Model
main :: IO ()
main = do
    -- declare model as pipeline
    let dnn = initDNN 2 [(DenseLayer, 1, leakyRelu 0.3)] kaiminWeightInit seed
    let pipe = initPipeline Nothing dnn Nothing

    -- get output of untrained network
    let pipeOutput1 = createPipeOutput pipe features
    let untrainedMSE = performanceMetric labels pipeOutput1

    -- print MSE before training
    printf "\n\nUntrained MSE: %.4f\n" $ head $ array untrainedMSE

    -- train the Neural network
    let res = trainPipe seed trainingSteps learningRate lossFunction pipe features labels
    let lossHistory = fst res
    let updatedPipe = snd res

    -- print loss history
    -- printLoss $ array lossHistory

    -- get output of trained network
    let pipeOutput2 = createPipeOutput updatedPipe features
    let trainedMSE = performanceMetric labels pipeOutput2

    -- print MSE after training
    printf "trained MSE: %.4f\n" $ head $ array trainedMSE
    
    -- print trained weights
    let updatedDNN = getDNN updatedPipe
    let trainedWeights = inputWeights $ last $ layers updatedDNN
    let trainedBias = biasWeights $ last $ layers updatedDNN

    printf "\nInput weights:\n"
    printMat trainedWeights
    printf "Bias:\n"
    printMat trainedBias

    let x = 1.0
    let y = 0.0
    let newObservation = initMatrix 1 2 [x, y]
    let newOutput = createPipeOutput updatedPipe newObservation

    printf "After training: output of input [%.0f, %.0f]: %.2f (desired is %.2f)\n" x y (trueFunc x y) $ head $ array newOutput
------------------------------------------------------------------------------------
