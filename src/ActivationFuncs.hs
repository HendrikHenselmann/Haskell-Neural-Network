-- Copyright [2020] <Hendrik Henselmann>
module ActivationFuncs (ActivationFunc(..),
                        encodeActivation,
                        getNumParams,
                        decodeActivation,
                        identity,
                        sigmoid,
                        relu,
                        leakyRelu,
                        tanH,
                        expLinUnit) where

import Data.Maybe

------------------------------------------------------------------------------------
-- Activation Function object
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------

data ActivationFunc = ActivationFunc {
    name :: String,
    fnc :: Double -> Double,
    fncDerive :: Double -> Double,
    params :: Maybe [Double]
}

------------------------------------------------------------------------------------
-- encoding and decoding of ActivationFunc (for load and store of DNN / Pipeline)
-- not very efficient, but thats ok since they will not be used very often

encodeActivation :: ActivationFunc -> String
encodeActivation actFunc
    | (actName == "identity") = "1"
    | (actName == "sigmoid") = "2"
    | (actName == "ReLU") = "3"
    | (actName == "leaky ReLU") = "4" ++ (encodeActivationParams (fromJust $ params actFunc))
    | (actName == "tanH") = "5"
    | (actName == "Exponential Linear Unit") = "6 " ++ (encodeActivationParams (fromJust $ params actFunc))
    | otherwise = "0"
    where
        actName = name actFunc

encodeActivationParams :: [Double] -> String
encodeActivationParams [] = []
encodeActivationParams (x:xs) = " " ++ (show x) ++ (encodeActivationParams xs)

getNumParams :: Int -> Int
getNumParams actFuncId
    | (actFuncId == 4) = 1
    | (actFuncId == 6) = 1
    | otherwise = 0

decodeActivation :: Int -> Maybe [Double] -> ActivationFunc
decodeActivation actFuncId params
    | (isNothing params) = decodeActivationWithoutParams actFuncId
    | otherwise = decodeActivationWithParams actFuncId (fromJust params)

decodeActivationWithoutParams :: Int -> ActivationFunc
decodeActivationWithoutParams actFuncId
    | (actFuncId == 2) = sigmoid
    | (actFuncId == 3) = relu
    | (actFuncId == 5) = tanH
    | otherwise = identity

decodeActivationWithParams :: Int -> [Double] -> ActivationFunc
decodeActivationWithParams actFuncId params
    | (actFuncId == 4) = leakyRelu (params!!0)
    | (actFuncId == 6) = expLinUnit (params!!0)
    | otherwise = identity

------------------------------------------------------------------------------------
-- Linear Function / Placeholder
identity :: ActivationFunc
identity = ActivationFunc "identity" identityFunc identityDeriv Nothing

identityFunc :: Double -> Double
identityFunc rawInput = rawInput

identityDeriv :: Double -> Double
identityDeriv x = 1.0

------------------------------------------------------------------------------------
-- Sigmoid Function

sigmoid :: ActivationFunc
sigmoid = ActivationFunc "sigmoid" sigmoidFunc sigmoidDeriv Nothing

sigmoidFunc :: Double -> Double
sigmoidFunc x = 1.0 / (1.0 + exp(-x))

sigmoidDeriv :: Double -> Double
sigmoidDeriv x = s * (1.0 - s)
    where
        s = sigmoidFunc x

------------------------------------------------------------------------------------
-- ReLU (Rectified Linear Unit)

relu :: ActivationFunc
relu = ActivationFunc "ReLU" reluFunc reluDeriv Nothing

reluFunc :: Double -> Double
reluFunc x
    | (x > 0.0) = x
    | otherwise = 0.0

reluDeriv :: Double -> Double
reluDeriv x
    | (x > 0.0) = 1.0
    | otherwise = 0.0

------------------------------------------------------------------------------------
-- Leaky ReLU

leakyRelu :: Double -> ActivationFunc
leakyRelu alpha = ActivationFunc "leaky ReLU" (leakyReluFunc alpha) (leakyReluDeriv alpha) (Just $ [alpha])

leakyReluFunc :: Double -> Double -> Double
leakyReluFunc alpha x
    | (x > 0.0) = x
    | otherwise = alpha * x

leakyReluDeriv :: Double -> Double -> Double
leakyReluDeriv alpha x
    | (x > 0.0) = 1.0
    | otherwise = alpha

------------------------------------------------------------------------------------
-- tanh Function

tanH :: ActivationFunc
tanH = ActivationFunc "tanH" tanHFunc tanHDeriv Nothing

tanHFunc :: Double -> Double
tanHFunc x = (2.0 / (1.0 + exp(-2.0*x))) - 1.0

tanHDeriv :: Double -> Double
tanHDeriv x = 1.0 - ((tanHFunc x) ** 2)

--------------------------------------------------------------------------------------
-- exponential linear Unit Function

expLinUnit :: Double -> ActivationFunc
expLinUnit alpha = ActivationFunc "Exponential Linear Unit" (expLinUnitFunc alpha) (expLinUnitDeriv alpha) (Just $ [alpha])

expLinUnitFunc :: Double -> Double -> Double
expLinUnitFunc alpha x
    | (x < 0.0) = alpha * (exp(x) - 1.0)
    | otherwise = x

expLinUnitDeriv :: Double -> Double -> Double
expLinUnitDeriv alpha x
    | (x < 0.0) = (expLinUnitFunc alpha x) + alpha
    | otherwise = 1.0

------------------------------------------------------------------------------------
