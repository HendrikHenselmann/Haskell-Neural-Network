-- Copyright [2020] <Hendrik Henselmann>

import DNN
import Layer
import Matrix
import LossFuncs
import ActivationFuncs
import WeightInitializations

import Test.HUnit

------------------------------------------------------------------------------------
-- Declaration of some test Networks

-- random seed (not used in simple weight initialization for these tests)
seed :: Int
seed = 128

testDNN_1 :: DNN
testDNN_1 = initDNN 1 [(DenseLayer, 1, relu)] simpleWeightInit seed

testDNN_2 :: DNN
testDNN_2 = initDNN 1 [(DenseLayer, 1, sigmoid)] simpleWeightInit seed

testDNN_3 :: DNN
testDNN_3 = initDNN 2 [(DenseLayer, 1, relu)] simpleWeightInit seed

testDNN_4 :: DNN
testDNN_4 = initDNN 2 [(DenseLayer, 2, relu), (DenseLayer, 1, relu)] simpleWeightInit seed

------------------------------------------------------------------------------------
-- Test cases

dnn1_tests = TestCase (do
    -- First test
    let input = initMatrix 1 1 [0.0]
    let desiredOutput = applyToMatElementWise (+1) input
    assertBool ("Input: " ++ (show (array input))) (matsAreEqual (createOutput testDNN_1 input) desiredOutput)
    -- Second test
    let input2 = initMatrix 1 1 [1.0]
    let desiredOutput2 = applyToMatElementWise (+1) input2
    assertBool ("Input: " ++ (show (array input2))) (matsAreEqual (createOutput testDNN_1 input2) desiredOutput2)
    -- Third test
    let input3 = initMatrix 1 1 [-2.0]
    let desiredOutput3 = initMatrix 1 1 [0.0]
    assertBool ("Input: " ++ (show (array input3))) (matsAreEqual (createOutput testDNN_1 input3) desiredOutput3)
    )

dnn2_tests = TestCase (do
    -- First test case
    let input = initMatrix 1 1 [0.0]
    let desiredOutput = initMatrix 1 1 [((fnc sigmoid) 1.0)]
    assertBool ("Input: " ++ (show (array input))) (matsAreEqual (createOutput testDNN_2 input) desiredOutput)
    -- Second test case
    let input2 = initMatrix 1 1 [1.0]
    let desiredOutput2 = initMatrix 1 1 [((fnc sigmoid) 2.0)]
    assertBool ("Input: " ++ (show (array input2))) (matsAreEqual (createOutput testDNN_2 input2) desiredOutput2)
    -- Third test case
    let input3 = initMatrix 1 1 [-1.0]
    let desiredOutput3 = initMatrix 1 1 [0.5]
    assertBool ("Input: " ++ (show (array input3))) (matsAreEqual (createOutput testDNN_2 input3) desiredOutput3)
    -- Fourth test case
    let input4 = initMatrix 1 1 [-2.0]
    let desiredOutput4 = initMatrix 1 1 [((fnc sigmoid) (-1.0))]
    assertBool ("Input: " ++ (show (array input4))) (matsAreEqual (createOutput testDNN_2 input4) desiredOutput4)
    )

dnn3_tests = TestCase (do
    -- First test case
    let input = initMatrix 1 2 [0.0, 0.0]
    let desiredOutput = initMatrix 1 1 [1.0]
    assertBool ("Input: " ++ (show (array input))) (matsAreEqual (createOutput testDNN_3 input) desiredOutput)
    -- Second test case
    let input2 = initMatrix 1 2 [2.0, 3.0]
    let desiredOutput2 = initMatrix 1 1 [6.0]
    assertBool ("Input: " ++ (show (array input2))) (matsAreEqual (createOutput testDNN_3 input2) desiredOutput2)
    -- Third test case
    let input3 = initMatrix 1 2 [6.0, 8.0]
    let desiredOutput3 = initMatrix 1 1 [15.0]
    assertBool ("Input: " ++ (show (array input3))) (matsAreEqual (createOutput testDNN_3 input3) desiredOutput3)
    -- Fourth test case
    let input4 = initMatrix 1 2 [-1.0, -1.0]
    let desiredOutput4 = initMatrix 1 1 [0.0]
    assertBool ("Input: " ++ (show (array input4))) (matsAreEqual (createOutput testDNN_3 input4) desiredOutput4)
    -- Fifth test case
    let input4 = initMatrix 1 2 [-3.0, -4.0]
    let desiredOutput4 = initMatrix 1 1 [0.0]
    assertBool ("Input: " ++ (show (array input4))) (matsAreEqual (createOutput testDNN_3 input4) desiredOutput4)
    )

dnn4_tests = TestCase (do
    -- First test case
    let input = initMatrix 1 2 [0.0, 0.0]
    let desiredOutput = initMatrix 1 1 [3.0]
    assertBool ("Input: " ++ (show (array input))) (matsAreEqual (createOutput testDNN_4 input) desiredOutput)
    -- Second test case
    let input2 = initMatrix 1 2 [2.0, 4.0]
    let desiredOutput2 = initMatrix 1 1 [15.0]
    assertBool ("Input: " ++ (show (array input2))) (matsAreEqual (createOutput testDNN_4 input2) desiredOutput2)
    -- Third test case
    let input3 = initMatrix 1 2 [-1.0, 0.0]
    let desiredOutput3 = initMatrix 1 1 [1.0]
    assertBool ("Input: " ++ (show (array input3))) (matsAreEqual (createOutput testDNN_4 input3) desiredOutput3)
    -- Fourth test case
    let input4 = initMatrix 1 2 [-4.0, -8.0]
    let desiredOutput4 = initMatrix 1 1 [1.0]
    assertBool ("Input: " ++ (show (array input4))) (matsAreEqual (createOutput testDNN_4 input4) desiredOutput4)
    -- Fifth test case
    let input4 = initMatrix 1 2 [12.0, 15.0]
    let desiredOutput4 = initMatrix 1 1 [57.0]
    assertBool ("Input: " ++ (show (array input4))) (matsAreEqual (createOutput testDNN_4 input4) desiredOutput4)
    )

------------------------------------------------------------------------------------
-- Name tests and group them together

tests = TestList [TestLabel "DNN 1" dnn1_tests, TestLabel "DNN 2" dnn2_tests, TestLabel "DNN 3" dnn3_tests, TestLabel "DNN 4" dnn4_tests]

------------------------------------------------------------------------------------
-- Execute tests
main :: IO Counts
main = runTestTT tests
