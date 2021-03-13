-- Copyright [2020] <Hendrik Henselmann>

import Matrix
import LossFuncs

import Test.HUnit

-- small constant
epsilon :: Double
epsilon = 0.001

------------------------------------------------------------------------------------
-- Declaration of some test Matrices

testMat1 :: Matrix
testMat1 = initMatrix 2 2 [1.0, 0.0, 0.0, 1.0]

testMat2 :: Matrix
testMat2 = initMatrix 2 2 [1.0, 2.0, 3.0, 4.0]

testMat3 :: Matrix
testMat3 = initMatrix 2 2 [1.0, 0.0, 1.0, 0.0]

------------------------------------------------------------------------------------
-- Test cases

crossEntropyAfterSoftmax_tests = TestCase (do
    -- First test
    assertBool "first case" (matsAreEqual ((function crossEntropyAfterSoftmax) emptyMatrix emptyMatrix) emptyMatrix)
    -- Second test
    assertBool "second case" (matsAreEqual ((derivative crossEntropyAfterSoftmax) emptyMatrix emptyMatrix) emptyMatrix)
    -- Third test
    let desiredOutput3 = initMatrix 2 1 [log(exp(1.0)+exp(2.0))-1.0, log(exp(3.0)+exp(4.0)) - 4.0]
    assertBool "third case" (matsAreEqualWithTolerance epsilon ((function crossEntropyAfterSoftmax) testMat1 testMat2) desiredOutput3)
    )

------------------------------------------------------------------------------------
-- Name tests and group them together

tests = TestList [TestLabel "crossEntropy After Softmax" crossEntropyAfterSoftmax_tests]

------------------------------------------------------------------------------------
-- Execute tests
main :: IO Counts
main = runTestTT tests
