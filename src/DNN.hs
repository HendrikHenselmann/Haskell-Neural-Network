-- Copyright [2020] <Hendrik Henselmann>
module DNN (DNN(..),
            emptyDNN,
            initDNN,
            evaluate,
            getResult,
            createOutput,
            train,
            storeDNN,
            loadDNN,
            printDNNInfo,
            trainAux1_,
            dnnFromString_,
            parseNextInt_) where

import Layer
import Matrix
import LossFuncs
import ActivationFuncs
import RandomMatrixBatch
import WeightInitializations

import Text.Printf
import System.Random
import Control.Monad

------------------------------------------------------------------------------------
-- Deep Neural Network
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
-- Neural Network consisting of Layers

newtype DNN = DNN {
    layers :: [Layer]
}

------------------------------------------------------------------------------------
-- Initialization

-- Initializing empty DNN
emptyDNN :: DNN
emptyDNN = DNN []

-- Initialize DNN according to given input layer size (inSize), hidden layer sizes, types and activation functions
initDNN :: Int -> [(LayerType, Int, ActivationFunc)] -> WeightInit -> Int -> DNN
initDNN inSize layerSpecifications weightInit seed
    | (inSize == 0) && null layerSpecifications = emptyDNN
    | otherwise = aux ((InputLayer, inSize, identity):layerSpecifications) emptyDNN
    where
        aux :: [(LayerType, Int, ActivationFunc)] -> DNN -> DNN
        aux [] dnn = dnn
        aux ((InputLayer, inSize, _):xs) dnn = aux xs (addInputLayer dnn inSize)
        aux ((DenseLayer, inSize, actFnc):xs) dnn = aux xs (addDenseLayer dnn inSize actFnc weightInit seed)

-- adding DenseLayer of m_i hidden nodes to DNN
addDenseLayer :: DNN -> Int -> ActivationFunc -> WeightInit -> Int -> DNN
addDenseLayer dnn m_i actFnc weightInit seed
    | null (layers dnn) = emptyDNN
    | otherwise = DNN (layers dnn ++ [newLayer])
    where
        previousLayer = last $ layers dnn
        prevLayerSize = size previousLayer
        inputWeightsMatrix = inputWeightInit weightInit seed prevLayerSize m_i
        biasWeightMatrix = biasWeightInit weightInit seed m_i
        newLayer = Layer DenseLayer m_i actFnc inputWeightsMatrix biasWeightMatrix emptyMatrix emptyMatrix  -- emptyMatrix are placeholders

-- adding Input Layer of m_i hidden nodes to DNN
addInputLayer :: DNN -> Int -> DNN
addInputLayer dnn m_i = DNN [Layer InputLayer m_i identity emptyMatrix emptyMatrix emptyMatrix emptyMatrix]  -- identity and emptyMatrix have no meaning here

------------------------------------------------------------------------------------
-- Evaluation

-- Evaluate DNN output for a given input (feddforward through every layer)
-- returning the updatedDNN with actual outputs as last layers outputs
evaluate :: DNN -> Matrix -> DNN
evaluate dnn input
    | null (layers dnn) = emptyDNN
    | null (array input) = emptyDNN
    | otherwise = DNN (evalAux (layers dnn) input)

evalAux :: [Layer] -> Matrix -> [Layer]
evalAux [] _ = []
evalAux (layer:layers) input = updatedLayer : evalAux layers newInput
    where
        updatedLayer = feedForward layer input
        newInput = output updatedLayer

-- get the result of previous evaluation (the output of last layer)
getResult :: DNN -> Matrix
getResult dnn = output (last (layers dnn))

-- function that returns output Matrix for given DNN and input
-- Note: This doesn't change the given DNNs input / output !!!
createOutput :: DNN -> Matrix -> Matrix
createOutput dnn input
    | null (layers dnn) = emptyMatrix
    | otherwise = output
    where
        updatedDnn = evaluate dnn input
        output = getResult updatedDnn

------------------------------------------------------------------------------------
-- Train the DNN
-- Training a DNN using given inputs, desired output and parameters
-- returning training loss vector and trained DNN

-- this function is basically just for error handling and (re-)structuring input and output
train :: Int -> Int -> Double -> LossFunc -> DNN -> Matrix -> Matrix -> (Matrix, DNN)
train seed trainingSteps learningRate lossFunc dnn inputMat desiredOutput
    | null (layers dnn) = (emptyMatrix, emptyDNN)
    | matsAreEqual inputMat emptyMatrix = (emptyMatrix, emptyDNN)
    | matsAreEqual desiredOutput emptyMatrix = (emptyMatrix, emptyDNN)
    | trainingSteps == 0 = (emptyMatrix, dnn)
    | otherwise = (lossMat, updatedDnn)
    where
        -- performing the training steps
        lossAndDnn = trainAux1_ seed trainingSteps learningRate lossFunc dnn inputMat desiredOutput []
        -- perform a last prediction
        updatedDnn = snd lossAndDnn
        finalPrediction = createOutput updatedDnn inputMat
        -- calculate final loss
        finalLoss = function lossFunc desiredOutput finalPrediction
        reversedLossHistory = head (array finalLoss):fst lossAndDnn
        lossHistory = reverse reversedLossHistory
        lossMat = Matrix 1 (trainingSteps+1) lossHistory

-- the outer loop ("training steps"): randomly choosing an input/output sample, calculating error and updating Network accordingly (Backpropagation)
trainAux1_ :: Int -> Int -> Double -> LossFunc -> DNN -> Matrix -> Matrix -> [Double] -> ([Double], DNN)
trainAux1_ seed trainingSteps learningRate lossFunc dnn inputMat desiredOutput reversedLossHistory
    | trainingSteps == 0 = (reversedLossHistory, dnn)
    | otherwise = trainAux1_ nextSeed (trainingSteps-1) learningRate lossFunc updatedDnn inputMat desiredOutput updatedReversedLossHistory
    where
        -- generate new seed so that a new sample is chosen in every iteration
        nextSeed = fst $ uniform $ mkStdGen seed

        -- the following procedure: choose random input/output pair -> feedforward -> calc loss and loss derivative -> backprop
        -- choose random input/output pair
        randomInputDesiredOutput = chooseRandomRowOf2Matrices nextSeed inputMat desiredOutput
        randomInputRow = fst randomInputDesiredOutput
        randomDesiredOutputRow = snd randomInputDesiredOutput
        -- feedforward
        dnnAfterFeedForward = evaluate dnn randomInputRow
        dnnPrediction = getResult dnnAfterFeedForward
        -- calc loss
        loss = function lossFunc randomDesiredOutputRow dnnPrediction
        -- calc loss derivative
        dLoss_dPrevOutput = derivative lossFunc randomDesiredOutputRow dnnPrediction
        -- backprop
        reversedLayers = reverse $ layers dnnAfterFeedForward
        updatedReversedLayers = trainAux2_ learningRate reversedLayers dLoss_dPrevOutput
        updatedDnn = DNN (reverse updatedReversedLayers)
        updatedReversedLossHistory = head (array loss):reversedLossHistory

-- the inner loop: backpropagating error through the whole network / all layers
trainAux2_ :: Double -> [Layer] -> Matrix -> [Layer]
trainAux2_ _ [] _ = []
trainAux2_ learningRate (layer:previousLayer:layers) dLoss_dOutput = updatedLayer:updatedRest
    where
        rawInput = output previousLayer
        backpropResult = backpropSGD learningRate layer rawInput dLoss_dOutput
        updatedLayer = snd backpropResult
        dLoss_dPreOutput = fst backpropResult
        updatedRest = trainAux2_ learningRate (previousLayer:layers) dLoss_dPreOutput

trainAux2_ _ (inputLayer:emptyList) _ = [inputLayer]

------------------------------------------------------------------------------------
-- Storing the dnn parameters in a file
storeDNN :: DNN -> FilePath -> Bool -> IO ()
storeDNN dnn fileName overwriteFile = do
    if overwriteFile
        then writeFile fileName ""
        else appendFile fileName ""
    storeAux (layers dnn) fileName

storeAux :: [Layer] -> FilePath -> IO ()
storeAux [] fileName = return ()
storeAux (layer:layers) fileName = do

    -- print seperators
    appendFile fileName "------------------------------------------------------------------------------------\n"

    -- layer information: encoded LayerType, layer size and encoded activation function with optional parameters
    storeLayerInfo fileName layer

    -- store info of input and bias weight matrices
    storeWeightInfo fileName layer

    -- store information of the remaining
    storeAux layers fileName

-- append layer info to file
storeLayerInfo :: FilePath -> Layer -> IO ()
storeLayerInfo fileName layer = do
    -- store layer info
    appendFile fileName (encodeLayerType $ layerType layer)
    appendFile fileName " "
    appendFile fileName (show $ size layer)
    appendFile fileName " "
    appendFile fileName (encodeActivation $ actFnc layer)
    appendFile fileName " "

-- append weight info to file
storeWeightInfo :: FilePath -> Layer -> IO ()
storeWeightInfo fileName layer = do
    -- store input weights
    let inputWeightsMat = inputWeights layer
    if matsAreEqual inputWeightsMat emptyMatrix
        then appendFile fileName "\n0 "
        else storeMat fileName inputWeightsMat
    -- store bias weights
    let biasWeightMat = biasWeights layer
    if matsAreEqual biasWeightMat emptyMatrix
        then appendFile fileName "\n0 \n"
        else do
            storeMat fileName biasWeightMat
            appendFile fileName "\n"

-- append Matrix info to file
storeMat :: FilePath -> Matrix -> IO ()
storeMat fileName mat = do
    -- print n m in first line
    appendFile fileName "\n"
    appendFile fileName (show $ n mat)
    appendFile fileName " "
    appendFile fileName (show $ m mat)
    appendFile fileName "\n"
    -- append values of matrix array in following n lines
    unless (matsAreEqual mat emptyMatrix) (storeArray fileName (m mat) 0 (array mat))

-- convert Matrix array to string
storeArray :: FilePath -> Int -> Int -> [Double] -> IO ()
storeArray fileName _ _ [] = return ()
storeArray fileName cols j (x:xs)
    | cols == j = do
        appendFile fileName "\n"
        storeArray fileName cols 0 (x:xs)
    | j == 0 =  do
        appendFile fileName (show x)
        storeArray fileName cols (j+1) xs
    | otherwise = do
        appendFile fileName " "
        appendFile fileName (show x)
        storeArray fileName cols (j+1) xs

------------------------------------------------------------------------------------
-- Loading the dnn out of a file
-- Note: It has to be stored with the store function above!!!
loadDNN :: FilePath -> IO DNN
loadDNN fileName = do
    dnnString <- readFile fileName
    return $ dnnFromString_ dnnString

-- wrapping DNN structure around layer array
dnnFromString_ :: String -> DNN
dnnFromString_ dnnString = DNN (parseNextLayer dnnString)

-- parsing layer after layer
parseNextLayer :: String -> [Layer]
parseNextLayer [] = []
parseNextLayer dnnString = layer : parseNextLayer remainingString
    where
        -- skipping seperators
        afterSkippedSeperators = skipSeperators dnnString

        -- parsing layer info (layer type, layer size, activation function)
        -- parsing layer type
        afterTypeParsing = parseLayerType afterSkippedSeperators
        parsedLayerType = fst afterTypeParsing
        -- parsing layer size
        afterSizeParsing = parseLayerSize (snd afterTypeParsing)
        layerSize = fst afterSizeParsing
        -- parsing activation function
        afterActivationParsing = parseLayerActivation (snd afterSizeParsing)
        activation = fst afterActivationParsing

        -- parsing weights
        -- parsing input weights
        afterParsedInputWeights = parseWeightMatrix (snd afterActivationParsing)
        inputWeightMat = fst afterParsedInputWeights
        -- parsing bias weights
        afterParsedBiasWeights = parseWeightMatrix (snd afterParsedInputWeights)
        biasWeightMat = fst afterParsedBiasWeights
        remainingString = snd afterParsedBiasWeights

        -- assemble the parsed layer
        layer = Layer parsedLayerType layerSize activation inputWeightMat biasWeightMat emptyMatrix emptyMatrix

-- skipping seperators
skipSeperators :: String -> String
skipSeperators [] = []
skipSeperators (c:str)
    | c == '-' = skipSeperators str
    | c == '\n' = str
    | otherwise = c:str  -- should not happen!

-- skipping newLine char if thats the next char
skipNewline :: String -> String
skipNewline [] = []
skipNewline (c:str)
    | c == '\n' = str
    | otherwise = c:str  -- should only happen at EOF

-- parsing layer type
parseLayerType :: String -> (LayerType, String)
parseLayerType [] = (InputLayer, [])  -- should not happen!
parseLayerType str = (parsedLayerType, remainingString)
    where
        res = parseNextInt_ str ' '
        parsedLayerType = decodeLayerType $ fst res
        remainingString = snd res

-- parsing layer size
parseLayerSize :: String -> (Int, String)
parseLayerSize str = (parsedLayerSize, remainingString)
    where
        res = parseNextInt_ str ' '
        parsedLayerSize = fst res
        remainingString = snd res

-- parsing activation function
parseLayerActivation :: String -> (ActivationFunc, String)
parseLayerActivation str = (parsedActivation, remainingString)
    where
        -- parsing activation function encoding
        res = parseNextInt_ str ' '
        parsedActivationNum = fst res
        afterParsedActivationNum = snd res
        -- parsing params according to the function
        numParams = getNumParams parsedActivationNum
        afterParsedParams
            | numParams > 0 = parseNextCoupleDouble afterParsedActivationNum numParams
            | otherwise = ([], skipNewline afterParsedActivationNum)
        parsedActivation
            | numParams > 0 = decodeActivation parsedActivationNum (Just $ fst afterParsedParams)
            | otherwise = decodeActivation parsedActivationNum Nothing
        remainingString = snd afterParsedParams

-- parsing matrix (input weights and bias weights)
parseWeightMatrix :: String -> (Matrix, String)
parseWeightMatrix str = (mat, remainingString)
    where
        -- parsing number of rows and cols
        afterParsedRows = parseNextInt_ str ' '
        rows = fst afterParsedRows
        afterParsedCols = parseNextInt_ (snd afterParsedRows) '\n'
        cols = fst afterParsedCols

        -- parsing values (Doubles) of matrix array
        afterParsedMatrixArray = parseWeightMatrixArray (snd afterParsedCols) rows cols 0 0 []
        mat
            | rows == 0 = emptyMatrix
            | otherwise = Matrix rows cols (fst afterParsedMatrixArray)

        remainingString
            | rows == 0 = skipNewline (snd afterParsedRows)
            | otherwise = snd afterParsedMatrixArray

-- parsing matrix array: (rows * cols) Doubles
parseWeightMatrixArray :: String -> Int -> Int -> Int -> Int -> [Double] -> ([Double], String)
parseWeightMatrixArray str rows cols i j acc
    | i == rows = (reverse acc, str)
    | j == (cols-1) =
        let res = parseNextDouble str '\n'
            lastDouble = fst res
            remainingString = snd res
            in parseWeightMatrixArray remainingString rows cols (i+1) 0 (lastDouble:acc)
    | otherwise = parseWeightMatrixArray remainingString rows cols i (j+1) (nextDouble:acc)
        where
            res = parseNextDouble str ' '
            nextDouble = fst res
            remainingString = snd res

-- parsing next Double till specified stop char occures
parseNextDouble :: String -> Char -> (Double, String)
parseNextDouble [] _ = (0.0, [])  -- sould not happen!
parseNextDouble str stopChar = (parsedDouble, remainingString)
    where
        res = parseNextDoubleAux str stopChar []
        parsedDouble = fst res
        remainingString = snd res

parseNextDoubleAux :: String -> Char -> String -> (Double, String)
parseNextDoubleAux [] _ buffer = ( read $ reverse buffer :: Double, [] )
parseNextDoubleAux (c:str) stopChar buffer
    | c == stopChar = ( read $ reverse buffer :: Double, str )
    | otherwise = parseNextDoubleAux str stopChar (c:buffer)

-- parsing next x Doubles seperated by space (' ') and the next '\n' if exists
parseNextCoupleDouble :: String -> Int -> ([Double], String)
parseNextCoupleDouble str x = parseNextCoupleDoubleAux str x []

parseNextCoupleDoubleAux :: String -> Int -> [Double] -> ([Double], String)
parseNextCoupleDoubleAux [] _ _ = ([], [])  -- sould not happen!
parseNextCoupleDoubleAux str x acc
    | x == 0 = (reverse acc, str)
    | otherwise = parseNextCoupleDoubleAux remainingString (x-1) (nextDouble:acc)
        where
            res = parseNextDouble str ' '
            nextDouble = fst res
            remainingString = snd res

-- parsing next Int till specified stop char occures
parseNextInt_ :: String -> Char -> (Int, String)
parseNextInt_ [] _ = (0, [])  -- sould not happen!
parseNextInt_ str stopChar = (parsedInt, remainingString)
    where
        res = parseNextIntAux str stopChar []
        parsedInt = fst res
        remainingString = snd res

parseNextIntAux :: String -> Char -> String -> (Int, String)
parseNextIntAux [] _ buffer = ( read $ reverse buffer :: Int, [] )
parseNextIntAux (c:str) stopChar buffer
    | c == stopChar = (read $ reverse buffer :: Int, str )
    | otherwise = parseNextIntAux str stopChar (c:buffer)

------------------------------------------------------------------------------------
-- Implementing print function for DNN
printDNNInfo :: DNN -> IO ()
printDNNInfo dnn
    | null (layers dnn) = printf "\n"
    | otherwise = do
        printf "\nDeep Neural Network\t-\tInformation\n"
        printf "================================================\n"
        printf "Input Layer size: %d\n" (size (head (layers dnn)))
        printf "================================================\n"
        printf "Hidden Layers: (type, size, activation)\n\n"
        aux (tail (layers dnn)) 1
        printf "================================================\n\n"
        where
            aux :: [Layer] -> Int -> IO ()
            aux [] _ = printf "\n"
            aux (layer:layers) i = do
                printf "\tLayer %d: " i
                printLayerInfo layer
                aux layers (i+1)

------------------------------------------------------------------------------------
