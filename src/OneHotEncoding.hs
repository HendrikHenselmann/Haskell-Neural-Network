-- Copyright [2020] <Hendrik Henselmann>
module OneHotEncoding (oneHotEncoding, predictionToLabel) where

import Matrix

------------------------------------------------------------------------------------
-- One Hot Encoding and kind of an inverse function
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
-- Note to "inverse" function ( predictionToLabel ) : 
--          function to turn a prediction into the label which is most likely

------------------------------------------------------------------------------------

-- one hot encoding (n x 1) Matrix to (n x (max+1)) Matrix
-- result has only one 1.0 in each row and the other values are 0.0's
oneHotEncoding :: Matrix -> Int -> Matrix
oneHotEncoding inputMat max
    | (matsAreEqual inputMat emptyMatrix) = emptyMatrix
    | otherwise = Matrix (n inputMat) (max+1) oneHotEncodedArray
    where
        oneHotEncodedArray = oneHotEncoding_Aux (array inputMat) max 0 

oneHotEncoding_Aux :: [Double] -> Int -> Int -> [Double]
oneHotEncoding_Aux [] _ _ = []
oneHotEncoding_Aux (x:xs) max j
    | (j == (max+1)) = oneHotEncoding_Aux xs max 0
    | ((round x) == j) = (1.0) : oneHotEncoding_Aux (x:xs) max (j+1)
    | otherwise = (0.0) : oneHotEncoding_Aux (x:xs) max (j+1)

------------------------------------------------------------------------------------

-- calculate the predicted label (column with greatest value in row)
-- transforming (n x m) Matrix to (n x 1) Matrix
predictionToLabel :: Matrix -> Matrix
predictionToLabel inputMat
    | (matsAreEqual inputMat emptyMatrix) = emptyMatrix
    | otherwise = Matrix (n inputMat) 1 labelArray
    where
        labelArray = predictionToLabel_Aux (array inputMat) (m inputMat) 0 0 0

predictionToLabel_Aux :: [Double] -> Int -> Int -> Int -> Double -> [Double]
predictionToLabel_Aux [] _ _ greatestCol _ = [fromIntegral greatestCol]
predictionToLabel_Aux (x:xs) m j greatestCol greatestVal
    | (j == m) = (fromIntegral greatestCol) : (predictionToLabel_Aux xs m 1 0 x)
    | otherwise = predictionToLabel_Aux xs m (j+1) updatedGreatestCol updatedGreatestVal
        where
            -- extract the greater column and value in same row
            greaterColVal
                | (greatestVal < x) = (j, x)
                | otherwise = (greatestCol, greatestVal)
            updatedGreatestCol = fst greaterColVal
            updatedGreatestVal = snd greaterColVal

------------------------------------------------------------------------------------
