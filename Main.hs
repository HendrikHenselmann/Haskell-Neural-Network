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
seed = 128

batchSize :: Int
batchSize = 1

trainingSteps :: Int
trainingSteps = 10000

learningRate :: Double
learningRate = 0.0026

lossFunction :: LossFunc
lossFunction = squaredError

------------------------------------------------------------------------------------
-- self created test data with binary labels
-- output should be addition of input

features = initMatrix 4 2 [0.0, 0.0,
                            1.0, 0.0,
                            0.0, 1.0,
                            1.0, 1.0]

labels = initMatrix 4 1 [0.0,
                          1.0,
                          1.0,
                          2.0]

------------------------------------------------------------------------------------

-- Deep Learning Classification Model
main :: IO ()
main = do
    -- declare model as pipeline
    let dnn = initDNN 2 [(DenseLayer, 1, relu)] kaiminWeightInit seed
    let pipe = initPipeline Nothing dnn Nothing

    -- one hot encoding labels
    -- let oneHotEncodedLabels = oneHotEncoding labels 1

    -- get output of untrained network
    let pipeOutput1 = createPipeOutput pipe features
    let untrainedAcc = accuracy labels pipeOutput1

    -- print accuracy before training
    printf "untrained accuracy: %.4f\n" untrainedAcc

    -- train the Neural network
    let res = trainPipe seed trainingSteps batchSize learningRate lossFunction pipe features labels
    let lossHistory = fst res
    let updatedPipe = snd res

    -- get output of trained network
    let pipeOutput2 = createPipeOutput updatedPipe features
    let trainedAcc = accuracy labels pipeOutput2

    -- print accuracy after training
    printf "trained accuracy: %.4f\n" trainedAcc

------------------------------------------------------------------------------------
