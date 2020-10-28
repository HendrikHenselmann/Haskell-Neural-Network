-- Copyright [2020] <Hendrik Henselmann>
module LossFuncs (LossFunc(..), squaredError, crossEntropy, crossEntropyAfterSoftmax) where

import Matrix
import Scaler

------------------------------------------------------------------------------------
-- Loss Function object
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------

data LossFunc = LossFunc {
    lossName :: String,
    function :: Matrix -> Matrix -> Matrix,
    derivative :: Matrix -> Matrix -> Matrix  -- derivative of loss wrt. last layer output
}

------------------------------------------------------------------------------------
-- Squared Error on arrays

squaredError :: LossFunc
squaredError = LossFunc "squaredLoss" squaredErrorFnc squaredErrorDerivative

squaredErrorFnc :: Matrix -> Matrix -> Matrix
squaredErrorFnc actual predicted
    | (matsAreEqual actual emptyMatrix) = emptyMatrix
    | (matsAreEqual predicted emptyMatrix) = emptyMatrix
    | otherwise = squaredErr
    where
        elementWiseSquareError = applyToTwoMatsElementWise (\ a p ->((a-p)**2)) actual predicted
        rowWiseAddition = applyToMatRowWise (+) elementWiseSquareError
        squaredErr = applyToMatElementWise (0.5 *) rowWiseAddition

squaredErrorDerivative :: Matrix -> Matrix -> Matrix
squaredErrorDerivative actual predicted
    | (matsAreEqual actual emptyMatrix) = emptyMatrix
    | (matsAreEqual predicted emptyMatrix) = emptyMatrix
    | otherwise = squaredErrDeriv
    where
        squaredErrDeriv = applyToTwoMatsElementWise (-) predicted actual

------------------------------------------------------------------------------------
-- Cross Entropy

crossEntropy :: LossFunc
crossEntropy = LossFunc "crossEntropy" crossEntropyFnc crossEntropyDerivative

crossEntropyFnc :: Matrix -> Matrix -> Matrix
crossEntropyFnc actual predicted
    | (matsAreEqual actual emptyMatrix) = emptyMatrix
    | (matsAreEqual predicted emptyMatrix) = emptyMatrix
    | otherwise = crossEntropyErr
    where
        logPredicted = applyToMatElementWise log predicted
        minusActualTimesLogPredicted = applyToTwoMatsElementWise (\ a p -> ((-a)*p)) actual predicted
        crossEntropyErr = applyToMatRowWise (+) minusActualTimesLogPredicted

crossEntropyDerivative :: Matrix -> Matrix -> Matrix
crossEntropyDerivative actual predicted
    | (matsAreEqual actual emptyMatrix) = emptyMatrix
    | (matsAreEqual predicted emptyMatrix) = emptyMatrix
    | otherwise = crossEntropyDeriv
    where
        crossEntropyDeriv = applyToTwoMatsElementWise (\ a p -> ( (-1.0) * (a/p) )) actual predicted

------------------------------------------------------------------------------------
-- Combination of cross entropy and softmax scaler

crossEntropyAfterSoftmax :: LossFunc
crossEntropyAfterSoftmax = LossFunc "crossEntropy after softmax" crossEntropyAfterSoftmaxFunc crossEntropyAfterSoftmaxDeriv

crossEntropyAfterSoftmaxFunc :: Matrix -> Matrix -> Matrix
crossEntropyAfterSoftmaxFunc actual predictedBeforeSoftmax
    | (matsAreEqual actual emptyMatrix) = emptyMatrix
    | (matsAreEqual predictedBeforeSoftmax emptyMatrix) = emptyMatrix
    | otherwise = res
    where
        -- get predictions (before softmax scaling) where actual is 1.0
        predictedWhereActualIsOne = Matrix (n actual) 1 (aux (array actual) (array predictedBeforeSoftmax))

        aux [] _ = []
        aux _ [] = []
        aux (a:actuals) (p:predicteds)
            | (a == 1.0) = p : rest
            | otherwise = rest
            where
                rest = aux actuals predicteds

        exps = applyToMatElementWise exp predictedBeforeSoftmax
        sumOfExps = applyToMatRowWise (+) exps
        logSumOfExps = applyToMatElementWise log sumOfExps
        res = applyToTwoMatsElementWise (\ x y -> (y - x) ) predictedWhereActualIsOne logSumOfExps

crossEntropyAfterSoftmaxDeriv :: Matrix -> Matrix -> Matrix
crossEntropyAfterSoftmaxDeriv actual predictedBeforeSoftmax
    | (matsAreEqual actual emptyMatrix) = emptyMatrix
    | (matsAreEqual predictedBeforeSoftmax emptyMatrix) = emptyMatrix
    | otherwise = applyToTwoMatsElementWise (\ a p -> p - a) actual ((scaleOut softmax) predictedBeforeSoftmax)

------------------------------------------------------------------------------------
