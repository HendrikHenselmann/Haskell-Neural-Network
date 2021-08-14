-- Copyright [2020] <Hendrik Henselmann>
module Layer (LayerType(..),
              sameLayerType,
              encodeLayerType,
              decodeLayerType,
              Layer(..),
              feedForward,
              backpropMBGD,
              printLayerInfo) where

import Matrix
import LossFuncs
import ActivationFuncs

import Data.Maybe
import Text.Printf
import System.Random

------------------------------------------------------------------------------------
-- Layers for Deep Neural Network
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------

data LayerType = InputLayer | DenseLayer

------------------------------------------------------------------------------------
-- defining layer type equality as function
sameLayerType :: LayerType -> LayerType -> Bool
sameLayerType InputLayer InputLayer = True
sameLayerType DenseLayer DenseLayer = True
sameLayerType _ _ = False

------------------------------------------------------------------------------------

-- Dense Layer
data Layer = Layer {              -- shape example for DenseLayer:
    layerType :: LayerType,
    size :: Int,                  -- m_i
    actFnc :: ActivationFunc,
    inputWeights :: Matrix,       -- (m_i-1 x m_i)
    biasWeights :: Matrix,        -- (1 x m_i)
    input :: Matrix,              -- sum of weighted inputs + bias  | (n x m_i)
    output :: Matrix              -- activation function applied on input  | (n x m_i)
}
------------------------------------------------------------------------------------
-- encoding and decoding of LayerType (for load and store of dnn / pipe)
-- not very efficient, but thats ok since they will not be used very often

encodeLayerType :: LayerType -> String
encodeLayerType InputLayer = "1"
encodeLayerType DenseLayer = "2"

decodeLayerType :: Int -> LayerType
decodeLayerType layerId
    | (layerId == 2) = DenseLayer
    | otherwise = InputLayer

------------------------------------------------------------------------------------
-- Feedforward

-- main feedforward function -> calling feedforward for special layer types
feedForward :: Layer -> Matrix -> Layer
feedForward layer rawInput = select (layerType layer)
    where
        select DenseLayer = feedForwardDense layer rawInput
        select InputLayer = Layer InputLayer (size layer) identity emptyMatrix emptyMatrix emptyMatrix rawInput

-- feedforward input values through DenseLayer, returning the updated DenseLayer
feedForwardDense :: Layer -> Matrix -> Layer
feedForwardDense layer rawInput = Layer DenseLayer m_i activation weights bias input output
    where
        m_i = size layer
        activation = actFnc layer
        weights = inputWeights layer
        bias = biasWeights layer
        input = matAdd (matMul rawInput weights) bias
        output = applyToMatElementWise (fnc activation) input

------------------------------------------------------------------------------------
-- Backpropagation of single layer

-- general backprop function (using mini batch gradient descent) -> calling backprop for different layer types
backpropMBGD :: Double -> Layer -> Matrix -> Matrix -> (Matrix, Layer)
backpropMBGD learningRate layer rawInput dLoss_dOutput = select (layerType layer)
    where
        select InputLayer = (emptyMatrix, layer)
        select DenseLayer = backpropDenseLayerMBGD learningRate layer rawInput dLoss_dOutput

-- Backpropagation of a single DenseLayer using mini batch gradient descent
-- returns derived loss wrt. current output and the updated Layer (updated weights)
-- rawInput: is the output of previouse layer, which is basically
--           the derivative of current layers input wrt. current layers weights
backpropDenseLayerMBGD :: Double -> Layer -> Matrix -> Matrix -> (Matrix, Layer)
backpropDenseLayerMBGD learningRate layer rawInput dLoss_dOutput = (dLoss_dPrevOutput, updatedLayer)
    where
        batchSize = n rawInput

        layerSize = size layer
        activation = actFnc layer
        inputWeightsMat = inputWeights layer      -- (m_i-1 x m_i)
        biasWeightsMat = biasWeights layer        -- (1 x m_i)
        inputMat = input layer                    -- (n x m_i)
        outputMat = output layer                  -- (n x m_i)

        -- (n x mi): elementWise on (n x mi)
        dOutput_dInput = applyToMatElementWise (fncDerive activation) inputMat
        -- (n x mi): elementWise (n x mi) (n x mi)
        dLoss_dInput = matMulElementWise dLoss_dOutput dOutput_dInput
        -- (n x mi-1): matMul (n x mi) (mi x mi-1)
        dLoss_dPrevOutput = matMul dLoss_dInput (transpose inputWeightsMat)
        -- (mi-1 x mi): matMul (mi-1 x n) (n x mi)
        dLoss_dInputWeights = matMul (transpose rawInput) dLoss_dInput
        -- (1 x mi): matMul (1 x n) (n x mi)
        dLoss_dBiasWeights = matMul (initOnes 1 batchSize) dLoss_dInput

        updatedInputWeights = matAdd inputWeightsMat (matScalarMult (-learningRate / (fromIntegral batchSize)) dLoss_dInputWeights)
        updatedBiasWeights = matAdd biasWeightsMat (matScalarMult (-learningRate / (fromIntegral batchSize)) dLoss_dBiasWeights)

        updatedLayer = Layer DenseLayer batchSize activation updatedInputWeights updatedBiasWeights inputMat outputMat

------------------------------------------------------------------------------------
-- Implementing print functions for different layers
printLayerInfo :: Layer -> IO ()
printLayerInfo layer = select (layerType layer)
    where
        select InputLayer = printInputLayerInfo layer
        select DenseLayer = printDenseLayerInfo layer

-- Implementing print function for Input layer
printInputLayerInfo :: Layer -> IO ()
printInputLayerInfo layer = printf "( Input, %d, - )\n" (size layer)

-- Implementing print function for DenseLayer layer
printDenseLayerInfo :: Layer -> IO ()
printDenseLayerInfo layer = printf "( Dense, %d, %s%s )\n" (size layer) (name (actFnc layer)) (convertActivationParamsToString $ params (actFnc layer))

convertActivationParamsToString :: Maybe [Double] -> String
convertActivationParamsToString params
    | (isNothing params) = ""
    | otherwise = aux (fromJust params)
    where
        aux :: [Double] -> String
        aux [] = ""
        aux (param:params) = " " ++ (show param) ++ (aux params)

------------------------------------------------------------------------------------
