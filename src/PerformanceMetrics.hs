-- Copyright [2020] <Hendrik Henselmann>
module PerformanceMetrics (accuracy) where

import Matrix
import OneHotEncoding

------------------------------------------------------------------------------------
-- Performance Metrics   for Classification and Regression

-- Note for Classification:
--      actual labels have to be (n x 1) matrix (vector) of labels ( not one hot encoded !! )
--      predictions have to be (n x maxVal) matrix of probabilities

------------------------------------------------------------------------------------
-- Accuracy
accuracy :: Matrix -> Matrix -> Double
accuracy actual predictedProbas
    | (matsAreEqual actual emptyMatrix) = -1.0
    | ( (n actual) /= (n predictedProbas) ) = -1.0
    | ( (m actual) /= (m predictedLabels) ) = -1.0
    | otherwise = percentageRightPredictions
    where
        predictedLabels = predictionToLabel predictedProbas
        numRightPredictions = rightPredictions (array actual) (array predictedLabels) 0
        percentageRightPredictions = (fromIntegral numRightPredictions) / (fromIntegral (n actual))

rightPredictions :: [Double] -> [Double] -> Int -> Int
rightPredictions [] [] acc = acc
rightPredictions [] _ _ = -1  -- should not happen!
rightPredictions _ [] _ = -1  -- should not happen!
rightPredictions (a:as) (p:ps) acc = rightPredictions as ps updatedAcc
    where
        updatedAcc
            | (a == p) = acc + 1
            | otherwise = acc

-- ------------------------------------------------------------------------------------
-- -- Precision
-- precision :: Matrix -> Matrix -> Double
-- precision actual prediction = 

-- ------------------------------------------------------------------------------------
-- -- Recall
-- recall :: Matrix -> Matrix -> Double
-- recall actual prediction =

-- ------------------------------------------------------------------------------------
-- Confusion Matrix

-- ------------------------------------------------------------------------------------
-- -- R2Score
-- r2Score :: Matrix -> Matrix -> Double
-- r2Score actual prediction = 

-- ------------------------------------------------------------------------------------
-- AOC

-- ------------------------------------------------------------------------------------
-- F1

-- ------------------------------------------------------------------------------------
-- Mean absolute error (MAE)

-- ------------------------------------------------------------------------------------
