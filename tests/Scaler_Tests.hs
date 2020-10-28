-- Copyright [2020] <Hendrik Henselmann>

import Scaler
import Matrix

import Test.HUnit

-- small constant
epsilon :: Double
epsilon = 0.00001

------------------------------------------------------------------------------------
-- Declaration of some test Matrices

testMat1 :: Matrix
testMat1 = initZeroes 1 1

testMat2 :: Matrix
testMat2 = initOnes 1 1

testMat3 :: Matrix
testMat3 = initMatrix 2 2 [1.0 .. 4.0]

testMat4 :: Matrix
testMat4 = initMatrix 2 2 [1.0, 0.0, 0.0, 1.0]

testMat5 :: Matrix
testMat5 = initMatrix 3 4 [1.0 .. 12.0]

testMat6 :: Matrix
testMat6 = initMatrix 1 2 [1.0, 2.0]

testMat7 :: Matrix
testMat7 = initMatrix 8 2 [1.0 .. 16.0]

------------------------------------------------------------------------------------
-- Test cases

normalization_tests = TestCase (do
    -- First test
    assertBool "first case" (matsAreEqual ((scaleIn normalization) emptyMatrix) emptyMatrix)
    -- Second test
    let inputMat2 = initMatrix 3 3 [1.0, 4.0, 7.0,
                                    2.0, 5.0, 8.0,
                                    3.0, 6.0, 9.0]
    let desiredOutput2 = initMatrix 3 3 [0.0, 0.0, 0.0,
                                         0.5, 0.5, 0.5,
                                         1.0, 1.0, 1.0]
    assertBool "second case" (matsAreEqualWithTolerance epsilon ((scaleIn normalization) inputMat2) desiredOutput2)
    -- Third test
    let inputMat3 = initMatrix 3 3 [1.0, 1.0, 1.0,
                                    2.0, 2.0, 2.0,
                                    3.0, 4.0, 5.0]
    let desiredOutput3 = initMatrix 3 3 [0.0, 0.0, 0.0,
                                         0.5, (1.0 / 3.0), 0.25,
                                         1.0, 1.0, 1.0]
    assertBool "second case" (matsAreEqualWithTolerance epsilon ((scaleIn normalization) inputMat3) desiredOutput3)
    )

------------------------------------------------------------------------------------
-- Name tests and group them together

tests = TestList [TestLabel "normalization" normalization_tests]

------------------------------------------------------------------------------------
-- Execute tests
main :: IO Counts
main = runTestTT tests
