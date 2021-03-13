-- Copyright [2020] <Hendrik Henselmann>

import Matrix
import PerformanceMetrics

import TestAuxiliaries

import Test.HUnit

-- small constant
epsilon :: Double
epsilon = 0.00001

------------------------------------------------------------------------------------
-- Declaration of some test Matrices
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

------------------------------------------------------------------------------------
-- Name tests and group them together

tests = TestList [TestLabel "accuracy" accuracy_tests]

------------------------------------------------------------------------------------
-- Execute tests
main :: IO Counts
main = runTestTT tests
