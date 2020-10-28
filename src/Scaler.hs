-- Copyright [2020] <Hendrik Henselmann>
module Scaler (InputScaler (..),
               OutputScaler (..),
               encodeInputScaler,
               decodeInputScaler,
               encodeOutputScaler,
               decodeOutputScaler,
               emptyInputScaler,
               emptyOutputScaler,
               normalization,
               standardization,
               softmax
               ) where

import Matrix
import OneHotEncoding

import Data.Maybe
import Text.Printf

------------------------------------------------------------------------------------
-- Input and Output Scalers (no reverse functions needed in Deep Learning)
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
-- Input Scaler
data InputScaler = InputScaler {
    inScalerName :: String,
    scaleIn :: Matrix -> Matrix
}

-- Output Scaler
data OutputScaler = OutputScaler {
    outScalerName :: String,
    scaleOut :: Matrix -> Matrix,
    -- derive: for backprop
    derive :: Matrix -> Matrix -> Matrix,
    -- scaleAndPredict for classification: returning the predicted label instead of probabilities
    scaleAndPredict :: Matrix -> Matrix 
}

------------------------------------------------------------------------------------
-- encoding and decoding of Input Scaler (for load and store of Pipeline)

encodeInputScaler :: Maybe InputScaler -> String
encodeInputScaler Nothing = "0"
encodeInputScaler inScaler
    | (scalerName == "normalization") = "1"
    | (scalerName == "standardization") = "2"
    | otherwise = "0"
    where
        scalerName = inScalerName $ fromJust inScaler

decodeInputScaler :: Int -> Maybe InputScaler
decodeInputScaler 0 = Nothing
decodeInputScaler scalerId
    | (scalerId == 1) = Just normalization
    | (scalerId == 2) = Just standardization
    | otherwise = Nothing
------------------------------------------------------------------------------------
-- encoding and decoding of Output Scaler (for load and store of Pipeline)

encodeOutputScaler :: Maybe OutputScaler -> String
encodeOutputScaler Nothing = "0"
encodeOutputScaler outScaler
    | (scalerName == "softmax") = "1"
    | otherwise = "0"
    where
        scalerName = outScalerName $ fromJust outScaler

decodeOutputScaler :: Int -> Maybe OutputScaler
decodeOutputScaler 0 = Nothing
decodeOutputScaler scalerId
    | (scalerId == 1) = Just softmax
    | otherwise = Nothing

------------------------------------------------------------------------------------
-- specify empty input and output scaler
emptyInputScaler :: InputScaler
emptyInputScaler = InputScaler "! Empty_Input_Scaler !" (\_ -> emptyMatrix)

emptyOutputScaler:: OutputScaler
emptyOutputScaler = OutputScaler "! Empty_Output_Scaler !" (\_ -> emptyMatrix) (\_ _ -> emptyMatrix) (\_ -> emptyMatrix)

------------------------------------------------------------------------------------
-- Normalizing Input Scaler / MinMax Scaler -> input in range [0, 1]
normalization = InputScaler "normalization" normalizationScale

normalizationScale :: Matrix -> Matrix
normalizationScale inputMat
    | (matsAreEqual inputMat emptyMatrix) = emptyMatrix
    | otherwise = applyToTwoMatsElementWise (/) xMinusMin maxMinusMinRepeated
    where
        maxArray = applyToMatColWise greaterArg inputMat
        minArray = applyToMatColWise smallerArg inputMat
        xMinusMin = (matAdd inputMat (applyToMatElementWise (*(-1)) minArray))
        maxMinusMin = applyToTwoMatsElementWise (-) maxArray minArray
        maxMinusMinRepeated = Matrix (n inputMat) (m inputMat) (repeatDoubleList (n inputMat) (array maxMinusMin))

greaterArg :: Double -> Double -> Double
greaterArg x y
    | (x > y) = x
    | otherwise = y

smallerArg :: Double -> Double -> Double
smallerArg x y
    | (x < y) = x
    | otherwise = y

------------------------------------------------------------------------------------
-- Standardizing Input Scaler -> input mean at 0 and std deviation of 1
standardization :: InputScaler
standardization = InputScaler "standardization" standardizationScale

standardizationScale :: Matrix -> Matrix
standardizationScale inputMat
    | (matsAreEqual inputMat emptyMatrix) = emptyMatrix
    | otherwise = applyToTwoMatsElementWise (/) (matAdd inputMat minusMeanArray) repeatedStdDeviation
    where
        sumArray = applyToMatColWise (+) inputMat
        meanArray = applyToMatElementWise (/ (fromIntegral (n inputMat))) sumArray
        minusMeanArray = applyToMatElementWise (*(-1.0)) meanArray
        xMinusMean = matAdd inputMat minusMeanArray
        squaredXMinusMean = applyToMatElementWise (**2.0) xMinusMean
        sumSquaredXMinusMean = applyToMatColWise (+) squaredXMinusMean
        stdDeviation = applyToMatElementWise (\x -> sqrt(x / (fromIntegral (n inputMat)))) sumSquaredXMinusMean
        repeatedStdDeviation = Matrix (n inputMat) (m inputMat) (repeatDoubleList (n inputMat) (array stdDeviation))

------------------------------------------------------------------------------------
-- repeating a list of doubles i times
repeatDoubleList :: Int -> [Double] -> [Double]
repeatDoubleList i original = aux i original
    where
        aux :: Int -> [Double] -> [Double]
        aux 0 _ = []
        aux i [] = aux (i-1) original
        aux i (x:xs) = x:(aux i xs)

------------------------------------------------------------------------------------
-- Softmax Output Scaler
softmax :: OutputScaler
softmax = OutputScaler "softmax" softmaxFunc softmaxDeriv applySoftmaxAndExtractLabel

softmaxFunc :: Matrix -> Matrix
softmaxFunc rawInputs = applyToTwoMatsElementWise (/) elementWiseExp repeatedRowWiseSum
    where
        elementWiseExp = applyToMatElementWise (\x -> exp(x)) rawInputs
        rowWiseSum = applyToMatRowWise (+) elementWiseExp
        repeatedRowWiseSum = Matrix (n rawInputs) (m rawInputs) (repeatDoubleListElements (m rawInputs) (array rowWiseSum))

softmaxDeriv :: Matrix -> Matrix -> Matrix
softmaxDeriv actual predictionsBeforeSoftmax = softmaxDerivation
    where
        -- scale predictions
        softmaxScaledPredictions = softmaxFunc predictionsBeforeSoftmax
        -- get prediction where actual is 1.0
        predictionWhereActualIsOne = softmaxDeriv_Aux (array actual) (array softmaxScaledPredictions)
        -- calculate derivative
        softmaxDerivationArray = softmaxDeriv_Aux2 (array actual) (array softmaxScaledPredictions) predictionWhereActualIsOne (m actual) 0
        softmaxDerivation = Matrix (n actual) (m actual) softmaxDerivationArray

softmaxDeriv_Aux :: [Double] -> [Double] -> [Double]
softmaxDeriv_Aux [] _ = []
softmaxDeriv_Aux _ [] = []
softmaxDeriv_Aux (a:actuals) (p:predicteds)
    | (a == 1.0) = p : rest
    | otherwise = rest
    where
        rest = softmaxDeriv_Aux actuals predicteds

softmaxDeriv_Aux2 :: [Double] -> [Double] -> [Double] -> Int -> Int -> [Double]
softmaxDeriv_Aux2 [] _ _ _ _ = []
softmaxDeriv_Aux2 _ [] _ _ _ = []
softmaxDeriv_Aux2 _ _ [] _ _ = []
softmaxDeriv_Aux2 (a:actuals) (p:predictions) (pWhereActualIsOne:predictionsWhereActualIsOne) m j
    | ((a == 1.0) && ((j+1) < m)) = (p*(1-p)) : (softmaxDeriv_Aux2 actuals predictions (pWhereActualIsOne:predictionsWhereActualIsOne) m (j+1))
    | ((j+1) < m) = (-p * pWhereActualIsOne) : (softmaxDeriv_Aux2 actuals predictions (pWhereActualIsOne:predictionsWhereActualIsOne) m (j+1))
    | (a == 1.0) = (p*(1-p)) : (softmaxDeriv_Aux2 actuals predictions predictionsWhereActualIsOne m 0)
    | otherwise = (-p * pWhereActualIsOne) : (softmaxDeriv_Aux2 actuals predictions predictionsWhereActualIsOne m 0)

applySoftmaxAndExtractLabel :: Matrix -> Matrix 
applySoftmaxAndExtractLabel rawInputs = predictionToLabel $ softmaxFunc rawInputs

------------------------------------------------------------------------------------
-- repeating elements in list (of doubles) i times
repeatDoubleListElements :: Int -> [Double] -> [Double]
repeatDoubleListElements m original = aux m original
    where
        aux :: Int -> [Double] -> [Double]
        aux _ [] = []
        aux 0 (x:xs) = aux m xs
        aux j (x:xs) = x:(aux (j-1) (x:xs))

------------------------------------------------------------------------------------
