-- Copyright [2020] <Hendrik Henselmann>
module OneHotEncoding_Tests (tests) where

import Matrix
import OneHotEncoding

import Test.HUnit

------------------------------------------------------------------------------------
-- Declaration of some test Matrices

testMat1 :: Matrix
testMat1 = initMatrix 2 1 [0.0, 1.0]

testMat2 :: Matrix
testMat2 = initMatrix 2 2 [1.0, 0.0, 0.0, 1.0]

testMat3 :: Matrix
testMat3 = initMatrix 5 1 [1.0, 0.0, 2.0, 4.0, 3.0]

testMat4 :: Matrix
testMat4 = initMatrix 5 5 [0.0, 1.0, 0.0, 0.0, 0.0,
                          1.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 1.0,
                          0.0, 0.0, 0.0, 1.0, 0.0]

------------------------------------------------------------------------------------
-- Test cases

oneHotEncoding_tests = TestCase (do
    -- first test
    assertBool "second case" (matsAreEqual (oneHotEncoding emptyMatrix 0) emptyMatrix)
    -- second test
    assertBool "second case" (matsAreEqual (oneHotEncoding testMat1 1) testMat2)
    -- third test
    assertBool "third case" (matsAreEqual (oneHotEncoding testMat3 4) testMat4)
    )

predictionToLabel_tests = TestCase (do
    -- first test
    assertBool "second case" (matsAreEqual (predictionToLabel emptyMatrix) emptyMatrix)
    -- second test
    assertBool "second case" (matsAreEqual (predictionToLabel testMat2) testMat1)
    -- third test
    assertBool "third case" (matsAreEqual (predictionToLabel testMat4) testMat3)
    )

------------------------------------------------------------------------------------
-- Name tests and group them together

tests = [TestLabel "oneHotEncoding" oneHotEncoding_tests, TestLabel "prediction to label" predictionToLabel_tests]

------------------------------------------------------------------------------------
