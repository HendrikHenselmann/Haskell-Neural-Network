-- Copyright [2020] <Hendrik Henselmann>
module Pipeline (Pipeline,
                 getInScaler,
                 getDNN,
                 getOutScaler,
                 initPipeline,
                 emptyPipe,
                 createPipeOutput,
                 trainPipe,
                 storePipe,
                 loadPipe,
                 printPipe) where

import DNN
import Layer
import Matrix
import Scaler
import LossFuncs

import Data.Maybe
import Data.Either

import Text.Printf

------------------------------------------------------------------------------------
-- Pipeline:
--            wrapping DNN with Input and Output Scaler

------------------------------------------------------------------------------------
------------------------------------------------------------------------------------

type Pipeline = (Maybe InputScaler, DNN, Maybe OutputScaler)

getInScaler :: Pipeline -> Maybe InputScaler
getInScaler (inScaler, _, _) = inScaler

getDNN :: Pipeline -> DNN
getDNN (_, dnn, _) = dnn

getOutScaler :: Pipeline -> Maybe OutputScaler
getOutScaler (_, _, outScaler) = outScaler

------------------------------------------------------------------------------------
-- Initialization of a Pipeline
initPipeline :: Maybe InputScaler -> DNN -> Maybe OutputScaler -> Pipeline
initPipeline inputScaler dnn outputScaler
    | (null (layers dnn)) = emptyPipe
    | otherwise = (inputScaler, dnn, outputScaler)

emptyPipe :: Pipeline
emptyPipe = (Nothing, emptyDNN, Nothing)

------------------------------------------------------------------------------------
-- Evaluation

-- Function that returns output Matrix for given Pipeline and input
-- Note: This doesn't change the given DNNs input / output !!!
createPipeOutput :: Pipeline -> Matrix -> Matrix
createPipeOutput pipe input
    | ( null $ layers $ getDNN pipe ) = emptyMatrix
    | otherwise = maybeScaledOutput
    where
        inScaler = getInScaler pipe
        outScaler = getOutScaler pipe
        dnn = getDNN pipe

        maybeScaledInput
            | (isNothing inScaler) = input
            | otherwise = (scaleIn $ fromJust inScaler) input

        updatedDnn = evaluate dnn maybeScaledInput
        output = getResult updatedDnn

        maybeScaledOutput
            | (isNothing outScaler) = output
            | otherwise = (scaleOut $ fromJust outScaler) output

------------------------------------------------------------------------------------
-- Training
-- Training a DNN in a Pipeline using given inputs, desired output and parameters
-- returning training loss vector and Pipeline with trained DNN

-- this function is basically just for error handling and (re-)structuring input and output
trainPipe :: Int -> Int -> Int -> Double -> LossFunc -> Pipeline -> Matrix -> Matrix -> (Matrix, Pipeline)
trainPipe seed trainingSteps batchSize learningRate lossFunc pipe inputMat desiredOutput
    | (null (layers dnn)) = (emptyMatrix, emptyPipe)
    | (batchSize > (n inputMat)) = (emptyMatrix, emptyPipe)
    | (matsAreEqual inputMat emptyMatrix) = (emptyMatrix, emptyPipe)
    | (matsAreEqual desiredOutput emptyMatrix) = (emptyMatrix, emptyPipe)
    | (trainingSteps == 0) = (emptyMatrix, pipe)
    | otherwise = (lossMat, updatedPipe)
    where
        -- extract objects of Pipeline
        dnn = getDNN pipe
        inScaler = getInScaler pipe
        outScaler = getOutScaler pipe

        -- scale input
        maybeScaledInputMat
            | (isNothing inScaler) = inputMat
            | otherwise = (scaleIn $ fromJust inScaler) inputMat

        -- change loss function according to output scaler
        -- has to be updated for every possible (scaler x loss) combination
        maybeLossFuncAfterScaling :: LossFunc
        maybeLossFuncAfterScaling
            | (isNothing outScaler) = lossFunc
            | (((outScalerName $ fromJust outScaler) == "softmax") && ((lossName lossFunc) == "crossEntropy")) = crossEntropyAfterSoftmax
            | otherwise = lossFunc

        -- performing the training steps
        lossAndDnn = trainAux1_ seed trainingSteps batchSize learningRate maybeLossFuncAfterScaling dnn maybeScaledInputMat desiredOutput []

        -- perform a last prediction
        updatedDnn = snd lossAndDnn
        finalPrediction = createOutput updatedDnn maybeScaledInputMat

        -- calculate final loss
        finalLoss = (function maybeLossFuncAfterScaling) desiredOutput finalPrediction
        meanFinalLoss = (sum (array finalLoss)) / (fromIntegral (n maybeScaledInputMat))
        reversedLossHistory = meanFinalLoss:(fst lossAndDnn)
        lossHistory = reverse reversedLossHistory
        lossMat = Matrix 1 (trainingSteps+1) lossHistory

        -- create updated pipe from updated DNN
        updatedPipe = (inScaler, updatedDnn, outScaler)

------------------------------------------------------------------------------------
-- Storing a Pipeline in a file (with the dnn parameters)
storePipe :: Pipeline -> FilePath -> IO ()
storePipe pipe fileName = do
    -- store information about the input scaler in first line
    writeFile fileName (encodeInputScaler $ getInScaler pipe)
    appendFile fileName "\n"

    -- store information about the output scaler in second line
    appendFile fileName (encodeOutputScaler $ getOutScaler pipe)
    appendFile fileName "\n"

    -- store DNN parameters in the following lines
    storeDNN (getDNN pipe) fileName False

------------------------------------------------------------------------------------
-- Loading a Pipeline out of a file
-- Note: It has to be stored with the store function above!!!
loadPipe :: FilePath -> IO Pipeline
loadPipe fileName = do
    pipeString <- readFile fileName
    return $ pipeFromString pipeString

pipeFromString :: String -> Pipeline
pipeFromString pipeString = loadedPipe
    where
        -- parse input scaler
        afterParsedInScaler = parseNextInt_ pipeString '\n'
        parsedInScaler = decodeInputScaler $ fst afterParsedInScaler

        -- parse output scaler
        afterParsedOutScaler = parseNextInt_ (snd afterParsedInScaler) '\n'
        parsedOutScaler = decodeOutputScaler $ fst afterParsedOutScaler

        -- parse DNN
        parsedDnn = dnnFromString_ (snd afterParsedOutScaler)

        -- combine the parsed parts
        loadedPipe = (parsedInScaler, parsedDnn, parsedOutScaler) 

------------------------------------------------------------------------------------
-- print Pipeline info
printPipe :: Pipeline -> IO ()
printPipe pipe
    | (null dnnLayers) = print "\n"
    | otherwise = do
        printf "\nDeep Neural Network\t-\tInformation\n"
        printf "================================================\n"
        printf "Input Layer size: %d\n" (size (head dnnLayers))
        printf "Input scaler: %s\n" inScaler
        printf "================================================\n"
        printf "Hidden Layers: (type, size, activation)\n\n"
        aux (tail dnnLayers) 1
        printf "================================================\n"
        printf "Output Layer size: %d\n" (size (last dnnLayers))
        printf "Output scaler: %s\n" outScaler
        printf "================================================\n\n"
        where
            dnn = getDNN pipe
            dnnLayers = layers dnn

            inScaler
                | (isNothing (getInScaler pipe)) = "Nothing"
                | otherwise = inScalerName $ fromJust $ getInScaler pipe

            outScaler
                | (isNothing (getOutScaler pipe)) = "Nothing"
                | otherwise = outScalerName $ fromJust $ getOutScaler pipe

            aux :: [Layer] -> Int -> IO ()
            aux [] _ = printf "\n"
            aux (layer:layers) i = do
            printf "\tLayer %d: " i
            printLayerInfo layer
            aux layers (i+1)

------------------------------------------------------------------------------------
