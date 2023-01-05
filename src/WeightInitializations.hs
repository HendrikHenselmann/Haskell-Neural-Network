-- Copyright [2020] <Hendrik Henselmann>
module WeightInitializations (WeightInit(..),
                              simpleWeightInit,
                              xavierWeightInit,
                              kaiminWeightInit) where

import Matrix

import System.Random

------------------------------------------------------------------------------------
-- Weight Initialization Functions
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------

-- first ints are random seeds
data WeightInit = WeightInit {
    inputWeightInit :: Int -> Int -> Int -> Matrix,
    biasWeightInit :: Int -> Int -> Matrix
}

------------------------------------------------------------------------------------
-- simple one: all weights are 1.0  -> Just for testing !
simpleWeightInit :: WeightInit
simpleWeightInit = WeightInit (const initOnes) (const (initOnes 1))

------------------------------------------------------------------------------------
-- Xavier weight initialization
xavierWeightInit :: WeightInit
xavierWeightInit = WeightInit xavierInputWeightInit (const (initOnes 1))

xavierInputWeightInit :: Int -> Int -> Int -> Matrix
xavierInputWeightInit seed n m = inputWeights
    where
        bound :: Double
        bound = sqrt $ 6 / fromIntegral (n + m)
        pureGen = mkStdGen seed
        randoms = uniformRs (-bound, bound) pureGen
        inputWeights = initMatrix n m (take (n*m) randoms)

------------------------------------------------------------------------------------
-- Kaiming weight initialization
-- espacially useful if assymmetric activation functions (eg. ReLUs) are used
-- link: https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
kaiminWeightInit :: WeightInit
kaiminWeightInit = WeightInit kaiminInputWeightInit (const (initOnes 1))

kaiminInputWeightInit :: Int -> Int -> Int -> Matrix
kaiminInputWeightInit seed n m = inputWeightsScaled
    where
        standardNormalDistribution = stdNormDistr seed
        inputWeightsRaw = initMatrix n m (take (n*m) standardNormalDistribution)
        inputWeightsScaled = applyToMatElementWise (* sqrt (2.0 / fromIntegral n)) inputWeightsRaw

-- Transform uniform distribution generator into standard normal distribution generator
-- using the Inverse transform sampling - Method, more precise the Box-Muller transform
-- Yielding an infinite array of standard normal distributed doubles
stdNormDistr :: Int -> [Double]
stdNormDistr seed = boxMullerTransform uniformDistributedDoubles
    where
        pureGen = mkStdGen seed
        uniformDistributedDoubles = uniformRs (0.0 :: Double, 1.0 :: Double) pureGen

boxMullerTransform :: [Double] -> [Double]
boxMullerTransform [] = []
boxMullerTransform (u1:u2:us) = (sqrt(-2 * log u1) * cos(2 * pi * u2)) : boxMullerTransform us

-- implementing uniformRs equivalent to randomRs of System.Random (randomRs is deprecated)
-- Function that returns an infinite list of uniform distributed random values
uniformRs :: (RandomGen g, UniformRange a) => (a, a) -> g -> [a]
uniformRs range gen = val : rest
    where
        result = uniformR range gen
        val = fst result
        rest = uniformRs range (snd result)

------------------------------------------------------------------------------------

-- Write values of stdNormDistr to file to be able to check if distribution fits
writeStdNormDistrToFile :: IO ()
writeStdNormDistrToFile = do
    let stdDistr = take 100000 (stdNormDistr 0)
    let stdDistrPrintableFormat = distrToPlottableForm stdDistr
    aux stdDistrPrintableFormat
    where
        aux [] = return ()
        aux ((x, y):xys) = do
            appendFile "./data/stdNormDistr.data" (show x ++ "\t" ++ show y ++ "\n")
            aux xys

-- calculate probability distribution in plottable format (x, y)
distrToPlottableForm :: [Double] -> [(Double, Double)]
distrToPlottableForm raw = aux raw 0 [(x/20.0, 0) | x <- [-100 .. 100]]
    where
        aux :: [Double] -> Int -> [(Double, Int)] -> [(Double, Double)]
        aux [] total res = map (\ (x, y) -> (x, fromIntegral y / fromIntegral total)) res
        aux (x:xs) total res = aux xs (total+1) (distrToPlottableFormAux res x)

distrToPlottableFormAux :: [(Double, Int)] -> Double -> [(Double, Int)]
distrToPlottableFormAux [] _ = []
distrToPlottableFormAux ((x, y):xys) z
    | z <= x = (x, y+1) : xys
    | otherwise = (x, y) : distrToPlottableFormAux xys z

------------------------------------------------------------------------------------
