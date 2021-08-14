-- Copyright [2021] <Hendrik Henselmann>
module WeightInitialization_Tests (tests) where

import Matrix
import WeightInitializations

import TestAuxiliaries

import Test.HUnit

------------------------------------------------------------------------------------
-- Declaration of some weight initialization tests

-- random seed (not used in simple weight initialization for these tests)
seed :: Int
seed = 128

-- small constant
epsilon :: Double
epsilon = 0.00001

------------------------------------------------------------------------------------
-- Test cases

simpleWeightInit_tests = TestCase (do
    let weightMatrix = (inputWeightInit simpleWeightInit) seed 3 4
    let biasMatrix = (biasWeightInit simpleWeightInit) seed 4
    -- dimensionality tests
    assertBool "First weight dimension does not fit!" ((n weightMatrix) == 3)
    assertBool "Second weight dimension does not fit!" ((m weightMatrix) == 4)
    assertBool "First bias dimension does not fit!" ((n biasMatrix) == 1)
    assertBool "Second bias dimension does not fit!" ((m biasMatrix) == 4)
    -- matrix size test
    assertBool "Actual weight matrix size does not fit!" ((length $ array weightMatrix) == 12)
    assertBool "Actual bias matrix size does not fit!" ((length $ array biasMatrix) == 4)
    -- bias all 1
    assertBool "A bias is not initialized with 1!" (matsAreEqualWithTolerance epsilon (initOnes 1 4) biasMatrix)
    -- weights all 1
    assertBool "A weight is not initialized with 1!" (matsAreEqualWithTolerance epsilon (initOnes 3 4) weightMatrix)
    )

xavierWeightInit_tests = TestCase (do
    let weightMatrix = (inputWeightInit xavierWeightInit) seed 3 4
    let biasMatrix = (biasWeightInit xavierWeightInit) seed 4
    -- dimensionality tests
    assertBool "First weight dimension does not fit!" ((n weightMatrix) == 3)
    assertBool "Second weight dimension does not fit!" ((m weightMatrix) == 4)
    assertBool "First bias dimension does not fit!" ((n biasMatrix) == 1)
    assertBool "Second bias dimension does not fit!" ((m biasMatrix) == 4)
    -- matrix size test
    assertBool "Actual weight matrix size does not fit!" ((length $ array weightMatrix) == 12)
    assertBool "Actual bias matrix size does not fit!" ((length $ array biasMatrix) == 4)
    -- bias all 1
    assertBool "A bias is not initialized with 1!" (matsAreEqualWithTolerance epsilon (initOnes 1 4) biasMatrix)
    )

kaiminWeightInit_tests = TestCase (do
    let weightMatrix = (inputWeightInit kaiminWeightInit) seed 3 4
    let biasMatrix = (biasWeightInit kaiminWeightInit) seed 4
    -- dimensionality tests
    assertBool "First weight dimension does not fit!" ((n weightMatrix) == 3)
    assertBool "Second weight dimension does not fit!" ((m weightMatrix) == 4)
    assertBool "First bias dimension does not fit!" ((n biasMatrix) == 1)
    assertBool "Second bias dimension does not fit!" ((m biasMatrix) == 4)
    -- matrix size test
    assertBool "Actual weight matrix size does not fit!" ((length $ array weightMatrix) == 12)
    assertBool "Actual bias matrix size does not fit!" ((length $ array biasMatrix) == 4)
    -- bias all 1
    assertBool "A bias is not initialized with 1!" (matsAreEqualWithTolerance epsilon (initOnes 1 4) biasMatrix)
    )

------------------------------------------------------------------------------------
-- Name tests and group them together

tests = [TestLabel "simpleWeightInit" simpleWeightInit_tests, TestLabel "xavierWeightInit" xavierWeightInit_tests, TestLabel "kaiminWeightInit" kaiminWeightInit_tests]

------------------------------------------------------------------------------------
