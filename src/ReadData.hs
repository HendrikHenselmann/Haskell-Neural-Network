-- Copyright [2020] <Hendrik Henselmann>
module ReadData (readTest,
                 readMNIST_testSet50,
                 readMNIST_testSet300,
                 readMNIST_testSet,
                 readMNIST_trainSet) where

import Matrix

import Data.Char
import System.IO( isEOF )

------------------------------------------------------------------------------------
-- Module to extract data out of csv files
------------------------------------------------------------------------------------
-- Important note:
-- This implementation of a Neural Network is actually not efficient enough to
-- be able to learn the categorization of images ( too many pixels / data )
------------------------------------------------------------------------------------

------------------------------------------------------------------------------------
-- cut first line (headings)
cutFirstLine :: String -> String
cutFirstLine [] = []
cutFirstLine (c:str)
    | (c == '\n') = str
    | otherwise = cutFirstLine str

------------------------------------------------------------------------------------
-- Read and Parse MNIST CSV File consisting of Doubles or Ints
-- !!! Values have to be: label, feature1, feature2, .. !!!
readAndParseCSV_MNIST :: Int -> Int -> String -> IO (Matrix, Matrix)
readAndParseCSV_MNIST rows featureCols fileName = do
    fileContent <- readFile fileName
    return $ parseContent_MNIST rows featureCols fileContent

parseContent_MNIST :: Int -> Int -> String -> (Matrix, Matrix)
parseContent_MNIST rows featureCols rawStr = (Matrix rows featureCols featureArray, Matrix rows 1 labelArray)
    where
        rawStrWithoutFirstCol = cutFirstLine rawStr
        res = parseContentAux_MNIST rawStrWithoutFirstCol ([], []) True ""
        labelArray = reverse (snd res)
        featureArray = reverse (fst res)

parseContentAux_MNIST :: String -> ([Double], [Double]) -> Bool -> String -> ([Double], [Double])
parseContentAux_MNIST [] res labelCol buffer
    | (buffer == []) = res
    | otherwise = updatedRes
    where
        features = fst res
        labels = snd res
        updatedFeatures = (read (reverse buffer) :: Double) : features
        updatedRes = (updatedFeatures, labels)
parseContentAux_MNIST (c:str) (featureArray, labelArray) labelCol buffer
    | ((isSpace c) && (c /= '\n')) = parseContentAux_MNIST str (featureArray, labelArray) labelCol buffer
    | (c == '\n') = parseContentAux_MNIST str ((read (reverse buffer) :: Double) : featureArray, labelArray) True ""
    | ((c == ',') && (labelCol)) = parseContentAux_MNIST str (featureArray, (read (reverse buffer) :: Double) : labelArray) False ""
    | (c == ',') = parseContentAux_MNIST str ((read (reverse buffer) :: Double) : featureArray, labelArray) False ""
    | otherwise = parseContentAux_MNIST str (featureArray, labelArray) labelCol (c:buffer)

------------------------------------------------------------------------------------
-- test function
readTest :: IO (Matrix, Matrix)
readTest = do
    let dataSize = 4
    let numFeatures = 3
    dataArrays <- readAndParseCSV_MNIST dataSize numFeatures "./data/Test/test.csv"
    return dataArrays

------------------------------------------------------------------------------------
-- Parsing the MNIST train and test sets

readMNIST_testSet50 :: IO (Matrix, Matrix)
readMNIST_testSet50 = do
    let dataSize = 50
    let numFeatures = 784
    dataArrays <- readAndParseCSV_MNIST dataSize numFeatures "./data/MNIST/mnist_test50.csv"
    return dataArrays

readMNIST_testSet300 :: IO (Matrix, Matrix)
readMNIST_testSet300 = do
    let dataSize = 300
    let numFeatures = 784
    dataArrays <- readAndParseCSV_MNIST dataSize numFeatures "./data/MNIST/mnist_test300.csv"
    return dataArrays

readMNIST_testSet :: IO (Matrix, Matrix)
readMNIST_testSet = do
    let dataSize = 10000
    let numFeatures = 784
    dataArrays <- readAndParseCSV_MNIST dataSize numFeatures "./data/MNIST/mnist_test.csv"
    return dataArrays

readMNIST_trainSet :: IO (Matrix, Matrix)
readMNIST_trainSet = do
    let dataSize = 50000
    let numFeatures = 28*28
    dataArrays <- readAndParseCSV_MNIST dataSize numFeatures "./data/MNIST/mnist_train.csv"
    return dataArrays

------------------------------------------------------------------------------------
