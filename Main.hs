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
-- important learning (hyper) parameters
seed :: Int
seed = 7

batchSize :: Int
batchSize = 2

trainingSteps :: Int
trainingSteps = 10000

learningRate :: Double
learningRate = 0.0026

lossFunction :: LossFunc
lossFunction = crossEntropy

------------------------------------------------------------------------------------
-- self created test data with binary labels
-- label is 1 iff 3rd input is greater than 0; 0 otherwise

features = initMatrix 10 3 [0.0, 0.0, 0.0,
                            1.0, -3.0, 0.0,
                            2.0, 4.0, 1.0,
                            -3.0, 2.0, 4.0,
                            2.24, 5.54, -7.6,
                            3.4, -8.3, -2.0,
                            -3.0, -9.0, 6.0,
                            3.0, 2.0, 1.0,
                            7.0, -9.0, -1.0,
                            4.0, -8.0, 2.0]

labels = initMatrix 10 1 [0.0,
                          0.0, 
                          1.0,
                          1.0,
                          0.0,
                          0.0,
                          1.0,
                          1.0,
                          0.0,
                          1.0]

------------------------------------------------------------------------------------

-- Deep Learning Classification Model
main :: IO ()
main = do
    -- declare model as pipeline
    let dnn = initDNN 3 [(DenseLayer, 10, relu), (DenseLayer, 2, relu)] kaiminWeightInit seed
    let pipe = initPipeline Nothing dnn (Just softmax)

    -- one hot encoding labels
    let oneHotEncodedLabels = oneHotEncoding labels 1

    -- get output of untrained network
    let pipeOutput1 = createPipeOutput pipe features
    let untrainedAcc = accuracy labels pipeOutput1

    -- print accuracy before training
    printf "untrained accuracy: %.4f\n" untrainedAcc

    -- train the Neural network
    let res = trainPipe seed trainingSteps batchSize learningRate lossFunction pipe features oneHotEncodedLabels
    let lossHistory = fst res
    let updatedPipe = snd res

    -- get output of trained network
    let pipeOutput2 = createPipeOutput updatedPipe features
    let trainedAcc = accuracy labels pipeOutput2

    -- print accuracy after training
    printf "trained accuracy: %.4f\n" trainedAcc

------------------------------------------------------------------------------------
