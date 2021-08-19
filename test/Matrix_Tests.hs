-- Copyright [2020] <Hendrik Henselmann>
module Matrix_Tests (tests) where

import Matrix
import RandomMatrixBatch

import Test.HUnit

-- small constant
epsilon :: Double
epsilon = 0.00001

------------------------------------------------------------------------------------
-- Naive Testing if all Numbers in array are unique
-- Note: O(n**2)
-- Naive implementation, because the function should test the more complex version [with O(n*log2 n)]
allUniqueNaive :: [Int] -> Bool
allUniqueNaive [] = True
allUniqueNaive (x:xs) = (notIn x xs) && (allUniqueNaive xs)
    where
        notIn :: Int -> [Int] -> Bool
        notIn _ [] = True
        notIn x (y:ys)
            | (x == y) = False
            | otherwise = notIn x ys

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
general_tests = TestCase (do
    -- First test case
    assertBool "first case" (((n testMat5) == 3) && ((m testMat5) == 4))
    -- Second test case
    assertBool "second case" (not (matsAreEqual testMat1 testMat2))
    -- Third test case
    assertBool "third case" (matsAreEqual testMat3 testMat3)
    -- Fourth test case
    let mat4 = initMatrix 2 2 [1.0 .. 4.0]
    assertBool "fourth case" (matsAreEqual testMat3 mat4)
    -- Fifth test case
    assertBool "fifth case" (not (matsAreEqual testMat4 testMat5))
    -- Sixth test case
    assertBool "sixth case" (matsAreEqual (reshape testMat1 1 2) emptyMatrix)
    -- Seventh test case
    let mat7 = initMatrix 16 1 [1.0 .. 16.0]
    assertBool "seventh case" (matsAreEqual (reshape testMat7 16 1) mat7)
    -- Eighth test case
    let mat8 = initMatrix 4 3 [1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0]
    assertBool "eighth case" (matsAreEqual (transpose testMat5) mat8)
    -- Nineth test case
    assertBool "nineth case" ((matSum testMat3) == 10.0)
    -- Tenth test case
    assertBool "tenth case" (matsAreEqualWithTolerance epsilon testMat3 testMat3)
    -- Eleventh test case
    assertBool "eleventh case" (matsAreEqualWithTolerance epsilon testMat3 mat4)
    -- Twelvth test case
    assertBool "twelvth case" (not (matsAreEqualWithTolerance epsilon testMat4 testMat5))
    )

matAdd_tests = TestCase (do
    -- First test case
    assertBool "first case" (matsAreEqual (matAdd testMat1 testMat2) testMat2)
    -- Second test case
    assertBool "second case" (matsAreEqual (matAdd testMat1 testMat3) emptyMatrix)
    -- Third test case
    let desiredOutput3 = initMatrix 2 2 [2.0, 2.0, 3.0, 5.0]
    assertBool "third case" (matsAreEqual (matAdd testMat3 testMat4) desiredOutput3)
    -- Fourth test case
    assertBool "fourth case" (matsAreEqual (matAdd testMat4 testMat5) emptyMatrix)
    -- Fifth test case
    let desiredOutput5 = initMatrix 2 2 [2.0, 4.0, 4.0, 6.0]
    assertBool "fifth case" (matsAreEqual (matAdd testMat3 testMat6) desiredOutput5)
    -- Sixth test case
    assertBool "sixth case" (matsAreEqual (matAdd testMat6 testMat3) desiredOutput5)
    )

matMul_tests = TestCase (do
    -- First test case
    assertBool "first case" (matsAreEqual (matMul testMat1 testMat2) testMat1)
    -- Second test case
    assertBool "second case" (matsAreEqual (matMulElementWise testMat1 testMat2) testMat1)
    -- Third test case
    assertBool "third case" (matsAreEqual (matMul testMat3 testMat4) testMat3)
    -- Fourth test case
    assertBool "fourth case" (matsAreEqual (matMul testMat3 testMat6) emptyMatrix)
    -- Fifth test case
    let desiredOutput5 = initMatrix 1 2 [7.0, 10.0]
    assertBool "fifth case" (matsAreEqual (matMul testMat6 testMat3) desiredOutput5)
    -- Sixth test case
    let desiredOutput6 = initMatrix 2 2 [1.0, 0.0, 0.0, 4.0]
    assertBool "sixth case" (matsAreEqual (matMulElementWise testMat3 testMat4) desiredOutput6)
    -- Seventh test case
    let desiredOutput7 = initMatrix 8 2 [0.5*x | x <- [1.0 .. 16.0]]
    assertBool "seventh case" (matsAreEqual (matScalarMult 0.5 testMat7) desiredOutput7)
    )

applyToMat_tests = TestCase (do
    -- First test case
    let desiredOutput1 = initMatrix 2 1 [3.0, 7.0]
    assertBool "first case" (matsAreEqual (applyToMatRowWise (+) testMat3) desiredOutput1)
    -- Second test case
    let desiredOutput2 = initMatrix 2 2 [3.0, 6.0, 9.0, 12.0]
    assertBool "second case" (matsAreEqual (applyToMatElementWise (*3) testMat3) desiredOutput2)
    -- Third test case
    let desiredOutput3 = initMatrix 3 1 [10.0, 26.0, 42.0]
    assertBool "third case" (matsAreEqual (applyToMatRowWise (+) testMat5) desiredOutput3)
    -- Fourth test case
    let desiredOutput4 = initMatrix 3 4 [2.0 .. 13.0]
    assertBool "fourth case" (matsAreEqual (applyToMatElementWise (+1) testMat5) desiredOutput4)
    -- Fifth test case
    assertBool "fifth case" (matsAreEqual (applyToTwoMatsElementWise (+) testMat1 testMat3) emptyMatrix)
    -- Sixth test case
    let desiredOutput6 = initMatrix 2 2 [2.0, 2.0, 3.0, 5.0]
    assertBool "sixth case" (matsAreEqual (applyToTwoMatsElementWise (+) testMat3 testMat4) desiredOutput6)
    -- Seventh test case
    let desiredOutput7 = initMatrix 2 2 [2.0, 0.0, 0.0, 8.0]
    assertBool "seventh case" (matsAreEqual (applyToTwoMatsElementWise (\x y -> 2*x*y) testMat3 testMat4) desiredOutput7)
    -- Eighth test case
    assertBool "eighth case" (matsAreEqual (applyToMatColWise (+) emptyMatrix) emptyMatrix)
    -- Nineth test case
    let desiredOutput9 = initMatrix 1 2 [1.0, 1.0]
    assertBool "Nineth case" (matsAreEqual (applyToMatColWise (+) testMat4) desiredOutput9)
    -- Tenth test case
    let desiredOutput10 = initMatrix 1 4 [15.0, 18.0, 21.0, 24.0]
    assertBool "tenth case" (matsAreEqual (applyToMatColWise (+) testMat5) desiredOutput10)
    )

chooseBatch_tests = TestCase (do
    -- First test case
    let desiredOutput1 = initMatrix 1 2 [1.0, 2.0]
    assertBool "first case" (matsAreEqual (Matrix 1 2 (cut_ (array testMat3) (m testMat3) [0])) desiredOutput1)
    -- Second test case
    let desiredOutput2 = initMatrix 1 2 [3.0, 4.0]
    assertBool "first case" (matsAreEqual (Matrix 1 2 (cut_ (array testMat3) (m testMat3) [1])) desiredOutput2)
    -- Third test case
    assertBool "third case" (matsAreEqual (chooseBatch 0 testMat7 0) emptyMatrix)
    -- Fourth test case
    assertBool "third case" (matsAreEqual (chooseBatch 0 testMat7 9) emptyMatrix)
    -- Fifth test case
    let batch5 = chooseBatch 0 testMat7 2
    assertBool "fifth case" (((n batch5) == 2) && ((m batch5) == 2))
    -- Sixth test case
    let batch6 = chooseBatch 0 testMat7 7
    assertBool "sixth case" (((n batch6) == 7) && ((m batch6) == 2))
    )

cut1row_tests = TestCase (do
    -- First test case
    assertBool "first case" (matsAreEqual (cut1row emptyMatrix 0) emptyMatrix)
    -- Second test case
    assertBool "first case" (matsAreEqual (cut1row testMat3 2) emptyMatrix)
    -- Third test case
    let desiredOutput3 = initMatrix 1 2 [1.0, 2.0]
    assertBool "third case" (matsAreEqual (cut1row testMat3 0) desiredOutput3)
    -- Fourth test case
    let desiredOutput4 = initMatrix 1 2 [3.0, 4.0]
    assertBool "fourth case" (matsAreEqual (cut1row testMat3 1) desiredOutput4)
    -- Fifth test case
    let desiredOutput5 = initMatrix 1 2 [15.0, 16.0]
    assertBool "fifth case" (matsAreEqual (cut1row testMat7 7) desiredOutput5)
    )

chooseBatchOf2Matrices_tests = TestCase (do
    -- First test case: batch size == 0
    let batches = chooseBatchOf2Matrices 0 testMat1 testMat2 0
    let batch1_1 = fst batches
    let batch1_2 = snd batches
    assertBool "first case" ((matsAreEqual batch1_1 emptyMatrix) && (matsAreEqual batch1_2 emptyMatrix))
    -- Second test case: batch size == n of both mats -> should return the input mats
    let batches = chooseBatchOf2Matrices 0 testMat3 testMat4 2
    let batch2_1 = fst batches
    let batch2_2 = snd batches
    assertBool "second case" ((matsAreEqual batch2_1 testMat3) && (matsAreEqual batch2_2 testMat4))
    -- Third test case: n of mats differ
    let batches = chooseBatchOf2Matrices 0 testMat4 testMat5 1
    let batch3_1 = fst batches
    let batch3_2 = snd batches
    assertBool "third case" ((matsAreEqual batch3_1 emptyMatrix) && (matsAreEqual batch3_2 emptyMatrix))
    -- Fourth test case
    let batches = chooseBatchOf2Matrices 0 testMat3 testMat4 1
    let batch3_1 = fst batches
    let batch3_2 = snd batches
    assertBool "third case" (((n batch3_1) == 1) && ((m batch3_1) == 2) && ((n batch3_2) == 1) && ((m batch3_2) == 2))
    )

randomUnique_tests = TestCase (do
    -- First test case
    assertBool "first case" (null (randomUnique 0 10 0))
    -- Second test case
    assertBool "second case" (allUniqueNaive (randomUnique 0 100 100))
    -- Third test case
    assertBool "third case" (allUniqueNaive (randomUnique 128 100 100))
    -- Fourth test case
    assertBool "fourth case" (allUniqueNaive (randomUnique 16 100 100))
    )

------------------------------------------------------------------------------------
-- Name tests and group them together
tests = [TestLabel "general" general_tests,
        TestLabel "matAdd" matAdd_tests,
        TestLabel "matMul" matMul_tests,
        TestLabel "applyToMat" applyToMat_tests,
        TestLabel "chooseBatch" chooseBatch_tests,
        TestLabel "cut1row" cut1row_tests,
        TestLabel "chooseBatchOf2Matrices" chooseBatchOf2Matrices_tests,
        TestLabel "randomUnique" randomUnique_tests]

------------------------------------------------------------------------------------
