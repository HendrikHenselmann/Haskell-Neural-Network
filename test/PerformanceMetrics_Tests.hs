-- Copyright [2020] <Hendrik Henselmann>
module PerformanceMetrics_Tests (tests) where

import Matrix
import PerformanceMetrics

import TestAuxiliaries

import Test.HUnit

-- small constant
epsilon :: Double
epsilon = 0.00001

------------------------------------------------------------------------------------
-- Declaration of some test Matrices for Classification
actualMat1 = initMatrix 6 1 [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]

-- all right
predictionMat1 = initMatrix 6 2 [0.0, 1.0,
                                 0.0, -1.0,
                                 0.5, 0.6,
                                 0.25, 0.75,
                                 0.1, 0.0,
                                 5.0, 4.0]

-- 2 wrong
predictionMat2 = initMatrix 6 2 [0.0, 1.0,
                                 0.0, 1.0,  -- wrong
                                 0.5, 0.6,
                                 0.25, 0.75,
                                 0.0, 0.1,  -- wrong
                                 5.0, 4.0]

-- 5 wrong
predictionMat3 = initMatrix 6 2 [1.0, 0.0,  -- wrong
                                 0.0, 1.0,  -- wrong
                                 0.6, 0.5,  -- wrong
                                 0.25, 0.75,
                                 0.0, 0.1,  -- wrong
                                 5.0, 6.0]  -- wrong

------------------------------------------------------------------------------------
-- Declaration of some test Matrices for Regression
actualRegressionMat = initMatrix 6 1 [1.0, 2.0, -4.0, 10.0, -50.0, 0.0]

-- all right
predictionRegressionMat1 = initMatrix 6 1 [1.0, 2.0, -4.0, 10.0, -50.0, 0.0]
desiredMSEOutputMat1 = initZeroes 1 1

-- first off by 1
predictionRegressionMat2 = initMatrix 6 1 [2.0, 2.0, -4.0, 10.0, -50.0, 0.0]
desiredMSEOutputMat2 = initMatrix 1 1 [1.0 / 6.0]

-- first off by 2
predictionRegressionMat3 = initMatrix 6 1 [3.0, 2.0, -4.0, 10.0, -50.0, 0.0]
desiredMSEOutputMat3 = initMatrix 1 1 [2.0 / 6.0]

-- negative one (third) is off by 3
predictionRegressionMat4 = initMatrix 6 1 [1.0, 2.0, -7.0, 10.0, -50.0, 0.0]
desiredMSEOutputMat4 = initMatrix 1 1 [3.0 / 6.0]

-- multiple ones (first, third and last) are off by 3+6+20 = 29
predictionRegressionMat5 = initMatrix 6 1 [-2.0, 2.0, 2.0, 10.0, -50.0, -20.0]
desiredMSEOutputMat5 = initMatrix 1 1 [29.0 / 6.0]

------------------------------------------------------------------------------------
-- Declaration of some test Matrices for multivariate Regression
actualMultivariateRegressionMat = initMatrix 3 2 [1.0, 2.0, -4.0, 10.0, -50.0, 0.0]

-- all right
predictionMultivariateRegressionMat1 = initMatrix 3 2 [1.0, 2.0, -4.0, 10.0, -50.0, 0.0]
desiredMultivariateOutputMat1 = initZeroes 1 2

-- first off by 1
predictionMultivariateRegressionMat2 = initMatrix 3 2 [2.0, 2.0, -4.0, 10.0, -50.0, 0.0]
desiredMultivariateOutputMat2 = initMatrix 1 2 [1.0 / 3.0, 0.0]

-- first off by 2
predictionMultivariateRegressionMat3 = initMatrix 3 2 [3.0, 2.0, -4.0, 10.0, -50.0, 0.0]
desiredMultivariateOutputMat3 = initMatrix 1 2 [2.0 / 3.0, 0.0]

-- negative one (third) is off by 3
predictionMultivariateRegressionMat4 = initMatrix 3 2 [1.0, 2.0, -7.0, 10.0, -50.0, 0.0]
desiredMultivariateOutputMat4 = initMatrix 1 2 [3.0 / 3.0, 0.0]

-- multiple ones (first, third and last) are off by 3+6 = 9 and 20
predictionMultivariateRegressionMat5 = initMatrix 3 2 [-2.0, 2.0, 2.0, 10.0, -50.0, -20.0]
desiredMultivariateOutputMat5 = initMatrix 1 2 [9.0 / 3.0, 20.0 / 3.0]

------------------------------------------------------------------------------------
-- Test cases

accuracy_tests = TestCase (do
    -- First test
    assertBool "first case" ((accuracy actualMat1 emptyMatrix) == -1.0)
    -- Second test
    assertBool "second case" ((accuracy emptyMatrix predictionMat1) == -1.0)
    -- Third test
    assertBool "third case" (equalityWithTolerance epsilon (accuracy actualMat1 predictionMat1) 1.0)
    -- Fourth test
    assertBool "fourth case" ( equalityWithTolerance epsilon (accuracy actualMat1 predictionMat2) (4.0 / 6.0) )
    -- Six test
    assertBool "six case" ( equalityWithTolerance epsilon (accuracy actualMat1 predictionMat3) (1.0 / 6.0) )
    )

meanAbsoluteError_tests = TestCase (do
    -- First test
    assertBool "first case" ( matsAreEqual (meanAbsoluteError actualRegressionMat emptyMatrix) emptyMatrix )
    -- Second test
    assertBool "second case" ( matsAreEqual (meanAbsoluteError emptyMatrix predictionRegressionMat1) emptyMatrix )
    -- Third test
    assertBool "third case" ( matsAreEqualWithTolerance epsilon (meanAbsoluteError actualRegressionMat predictionRegressionMat1) desiredMSEOutputMat1 )
    -- Fourth test
    assertBool "fourth case" ( matsAreEqualWithTolerance epsilon (meanAbsoluteError actualRegressionMat predictionRegressionMat2) desiredMSEOutputMat2 )
    -- Six test
    assertBool "six case" ( matsAreEqualWithTolerance epsilon (meanAbsoluteError actualRegressionMat predictionRegressionMat3) desiredMSEOutputMat3 )
    -- Seventh test
    assertBool "seventh case" ( matsAreEqualWithTolerance epsilon (meanAbsoluteError actualRegressionMat predictionRegressionMat4) desiredMSEOutputMat4 )
    -- Eighth test
    assertBool "eigth case" ( matsAreEqualWithTolerance epsilon (meanAbsoluteError actualRegressionMat predictionRegressionMat5) desiredMSEOutputMat5 )

    -- Multivariate tests
    -- Nineth test
    assertBool "nineth case" ( matsAreEqual (meanAbsoluteError actualMultivariateRegressionMat emptyMatrix) emptyMatrix )
    -- Tenth test
    assertBool "tenth case" ( matsAreEqual (meanAbsoluteError emptyMatrix predictionMultivariateRegressionMat1) emptyMatrix )
    -- Eleventh test
    assertBool "eleventh case" ( matsAreEqualWithTolerance epsilon (meanAbsoluteError actualMultivariateRegressionMat predictionMultivariateRegressionMat1) desiredMultivariateOutputMat1 )
    -- Twelveth test
    assertBool "twelveth case" ( matsAreEqualWithTolerance epsilon (meanAbsoluteError actualMultivariateRegressionMat predictionMultivariateRegressionMat2) desiredMultivariateOutputMat2 )
    -- Thirteenth test
    assertBool "thirteenth case" ( matsAreEqualWithTolerance epsilon (meanAbsoluteError actualMultivariateRegressionMat predictionMultivariateRegressionMat3) desiredMultivariateOutputMat3 )
    -- Fourteenth test
    assertBool "fourteenth case" ( matsAreEqualWithTolerance epsilon (meanAbsoluteError actualMultivariateRegressionMat predictionMultivariateRegressionMat4) desiredMultivariateOutputMat4 )
    -- Fifteenth test
    assertBool "fifteenth case" ( matsAreEqualWithTolerance epsilon (meanAbsoluteError actualMultivariateRegressionMat predictionMultivariateRegressionMat5) desiredMultivariateOutputMat5 )
    )

------------------------------------------------------------------------------------
-- Name tests and group them together

tests = [TestLabel "accuracy" accuracy_tests, TestLabel "mean absolute squared error" meanAbsoluteError_tests]

------------------------------------------------------------------------------------
