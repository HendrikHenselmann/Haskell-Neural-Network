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
trainingSteps = 1

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
                 3.0, 10.0,
                 40.0, 20.0,
                 10.0, 20.0,
                 20.0, 20.0,
                 150.0, 50.0,
                 32.0, 70.0]

features = initMatrix (div (length featuresArray) numFeatures) numFeatures featuresArray
labels = applyToMatRowWise (\ x y -> 2.0*x + (0.5*y) + 10.0) features

------------------------------------------------------------------------------------

-- Deep Learning Classification Model
main :: IO ()
main = do
    -- declare model as pipeline
    let dnn = initDNN 2 [(DenseLayer, 1, leakyRelu 0.2)] kaiminWeightInit seed
    let pipe = initPipeline Nothing dnn Nothing

    -- one hot encoding labels
    -- let oneHotEncodedLabels = oneHotEncoding labels 1

    -- get output of untrained network
    let pipeOutput1 = createPipeOutput pipe features
    let untrainedMAE = performanceMetric labels pipeOutput1

    -- print MAE before training
    printf "untrained MAE: %.4f\n" $ head $ array untrainedMAE

    -- train the Neural network
    let res = trainPipe seed trainingSteps learningRate lossFunction pipe features labels
    let lossHistory = fst res
    let updatedPipe = snd res

    -- print loss history
    -- printLoss $ array lossHistory

    -- get output of trained network
    let pipeOutput2 = createPipeOutput updatedPipe features
    let trainedMAE = performanceMetric labels pipeOutput2

    -- print MAE after training
    printf "trained MAE: %.4f\n" $ head $ array trainedMAE
    
    -- print trained weights
    let updatedDNN = getDNN updatedPipe
    let trainedWeights = inputWeights $ last $ layers updatedDNN
    let trainedBias = biasWeights $ last $ layers updatedDNN

    printf "Input weights:\n"
    printMat trainedWeights
    printf "Bias:\n"
    printMat trainedBias

    let newObservation = initMatrix 1 2 [3.0, 7.0]
    let newOutput = createPipeOutput updatedPipe newObservation

    printf "output of input observation [3, 7] : %.4f\n" $ head $ array newOutput
------------------------------------------------------------------------------------
