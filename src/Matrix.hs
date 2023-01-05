-- Copyright [2020] <Hendrik Henselmann>
module Matrix (Matrix(..),
               initMatrix,
               emptyMatrix,
               initZeroes,
               initOnes,
               matsAreEqual,
               matsAreEqualWithTolerance,
               matSum,
               matMul,
               matMulElementWise,
               matScalarMult,
               matAdd,
               applyToMatRowWise,
               applyToMatColWise,
               applyToMatElementWise,
               applyToTwoMatsElementWise,
               cut1row,
               reshape,
               transpose,
               printMat) where

import Text.Printf

------------------------------------------------------------------------------------
-- 2D Matrix of Doubles
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
-- Implementing a 2D Matrix of Doubles
data Matrix = Matrix {
    n :: Int,  -- number of rows
    m :: Int,  -- number of columns
    array :: [Double]  -- 1D array of values
} deriving (Show)

-- Initialize new Matrix with given values
-- Important: This is the way a Matrix should be initialized!!!
--            Because it checks for consistency.
initMatrix :: Int -> Int -> [Double] -> Matrix
initMatrix n m arr
    | length arr == n*m = Matrix n m arr
    | otherwise = emptyMatrix

-- Initialize Matrix of Zeroes
initZeroes :: Int -> Int -> Matrix
initZeroes n m = Matrix n m (replicate (n*m) 0.0)

-- Initialize Matrix of Ones
initOnes :: Int -> Int -> Matrix
initOnes n m = Matrix n m (replicate (n*m) 1.0)

-- Function that maps 2D coordinates to 1D coordinates
index :: Matrix -> Int -> Int -> Int
index mat i j
    | i >= n mat = -1
    | j >= m mat = -1
    | otherwise = m mat * i + j

-- Reshaping only changes the dimension sizes, eg. (2 x 2) to (1 x 4) or to (4 x 1)
-- Note: It doesn't change the underlying array of values!
-- Useful: reshape of (1 x n) to (n x 1) and the other way 'round is basically a very cheap transposing
reshape :: Matrix -> Int -> Int -> Matrix
reshape mat new_n new_m
    | n mat * m mat /= new_n * new_m = emptyMatrix
    | otherwise = Matrix new_n new_m (array mat)

-- calculating the transposed Matrix
transpose :: Matrix -> Matrix
transpose mat = Matrix n_out m_out (aux n_out m_out 0 0 mat)
    where
        n_out = m mat
        m_out = n mat
        aux :: Int -> Int -> Int -> Int -> Matrix -> [Double]
        aux n_out m_out i j mat
            | i == n_out = []
            | j == m_out = aux n_out m_out (i+1) 0 mat
            | otherwise = get mat j i : aux n_out m_out i (j+1) mat

-- Function to get value at position (i, j) -> row i and col j
get :: Matrix -> Int -> Int -> Double
get mat i j = arr !! ind
    where
        arr = array mat
        ind = index mat i j

-- Declaring the empty matrix
emptyMatrix :: Matrix
emptyMatrix = Matrix 0 0 []

-- Defining matrix equality -> same shape and values
matsAreEqual :: Matrix -> Matrix -> Bool
matsAreEqual mat1 mat2
    | n mat1 /= n mat2 = False
    | m mat1 /= m mat2 = False
    | otherwise = elementWiseEqual (array mat1) (array mat2)

elementWiseEqual :: [Double] -> [Double] -> Bool
elementWiseEqual [] [] = True
elementWiseEqual _ [] = False
elementWiseEqual [] _ = False
elementWiseEqual (x:xs) (y:ys) = (x==y) && elementWiseEqual xs ys

-- Defining matrix equality -> same shape and values with error tolerance of a given (! small !) constant (epsilon)
matsAreEqualWithTolerance :: Double -> Matrix -> Matrix -> Bool
matsAreEqualWithTolerance eps mat1 mat2
    | n mat1 /= n mat2 = False
    | m mat1 /= m mat2 = False
    | otherwise = elementWiseEqualWithTolerance eps (array mat1) (array mat2)

elementWiseEqualWithTolerance :: Double -> [Double] -> [Double] -> Bool
elementWiseEqualWithTolerance _ [] [] = True
elementWiseEqualWithTolerance _ _ [] = False
elementWiseEqualWithTolerance _ [] _ = False
elementWiseEqualWithTolerance eps (x:xs) (y:ys) = (x >= (y-eps)) && (x <= (y+eps)) && elementWiseEqualWithTolerance eps xs ys

------------------------------------------------------------------------------------
-- Implementing sum on matrix
matSum :: Matrix -> Double
matSum mat = sum (array mat)

------------------------------------------------------------------------------------
-- Implementing matrix multiplication

-- Regular matrix multiplication of 2D Matrices
matMul :: Matrix -> Matrix -> Matrix
matMul mat1 mat2
    | m mat1 /= n mat2 = emptyMatrix
    | otherwise = Matrix (n mat1) (m mat2) (matMulAux1_ mat1 mat2 0 0)

matMulAux1_ :: Matrix -> Matrix -> Int -> Int -> [Double]
matMulAux1_ mat1 mat2 i j
    | i == n_out = []
    | j == m_out = matMulAux1_ mat1 mat2 (i+1) 0
    | otherwise = matMulAux2_ mat1 mat2 i j 0 0 : matMulAux1_ mat1 mat2 i (j+1)
    where
        n_out = n mat1
        m_out = m mat2

matMulAux2_ :: Matrix -> Matrix -> Int -> Int -> Int -> Double -> Double
matMulAux2_ mat1 mat2 i j k res
    | k == k_max = res
    | otherwise = matMulAux2_ mat1 mat2 i j (k+1) updatedResult
    where
        k_max = m mat1
        val1 = get mat1 i k
        val2 = get mat2 k j
        updatedResult = res + val1 * val2

-- Element-wise multiplication of two matrices (also called Hadamard product)
matMulElementWise :: Matrix -> Matrix -> Matrix
matMulElementWise mat1 mat2
    | (n mat1 /= n mat2) || (m mat1 /= m mat2) = emptyMatrix
    | otherwise = Matrix (n mat1) (m mat1) (aux (array mat1) (array mat2))
    where
        aux [] [] = []
        aux (x:xs) (y:ys) = (x*y) : aux xs ys

-- multiplication of given scalar with every element of given matrix
matScalarMult :: Double -> Matrix -> Matrix
matScalarMult scalar mat = Matrix (n mat) (m mat) (map (scalar*) (array mat))

------------------------------------------------------------------------------------
-- Implementing matrix addition

-- Matrix addition, element wise or matrix with vector row-wise
matAdd :: Matrix -> Matrix -> Matrix
matAdd mat1 mat2
    | (n mat1 == n mat2) && (m mat1 == m mat2) = matAddElementWise_ mat1 mat2
    | (n mat2 == 1) && (m mat1 == m mat2) = matAddRowWise_ mat1 mat2
    | (n mat1 == 1) && (m mat1 == m mat2) = matAddRowWise_ mat2 mat1
    | otherwise = emptyMatrix

matAddElementWise_ :: Matrix -> Matrix -> Matrix
matAddElementWise_ mat1 mat2 = Matrix (n mat1) (m mat1) (aux (array mat1) (array mat2))
    where
        aux [] [] = []
        aux (x:xs) (y:ys) = (x+y) : aux xs ys

-- Mat2 is the vector
matAddRowWise_ :: Matrix -> Matrix -> Matrix
matAddRowWise_ mat1 mat2 = Matrix (n mat1) (m mat1) (aux (array mat1) (array mat2) (array mat2))
    where
        aux [] _ _ = []
        aux xs arr2 [] = aux xs arr2 arr2
        aux (x:xs) arr2 (y:ys) = (y+x) : aux xs arr2 ys

------------------------------------------------------------------------------------
-- Apply functions element-wise or row-wise

-- Row-wise application of Double -> Double -> Double function (to the whole row)
applyToMatRowWise :: (Double -> Double -> Double) -> Matrix -> Matrix
applyToMatRowWise fnc mat
    | matsAreEqual mat emptyMatrix = emptyMatrix
    | otherwise = Matrix (n mat) 1 (aux fnc rest (m mat) 1 first)
    where
        arr = array mat
        first = head arr
        rest = tail arr
        aux fnc [] _ _ acc = [acc]
        aux fnc (x:xs) m j acc
            | m == j = acc : aux fnc xs m 1 x
            | otherwise = aux fnc xs m (j+1) (fnc acc x)

-- Column-wise application of Double -> Double -> Double function (to the whole column)
applyToMatColWise :: (Double -> Double -> Double) -> Matrix -> Matrix
applyToMatColWise fnc mat
    | matsAreEqual mat emptyMatrix = emptyMatrix
    | otherwise = Matrix 1 (m mat) (aux mat 0)
    where
        aux :: Matrix -> Int -> [Double]
        aux mat j
            | j >= m mat = []
            | otherwise = applyToMatColWiseAux fnc mat j : aux mat (j+1)

-- folding column j
applyToMatColWiseAux :: (Double -> Double -> Double) -> Matrix -> Int -> Double
applyToMatColWiseAux fnc mat j = aux fnc mat 1 j first
    where
        first = get mat 0 j
        aux :: (Double -> Double -> Double) -> Matrix -> Int -> Int -> Double -> Double
        aux fnc mat i j acc
            | i >= n mat = acc
            | otherwise = aux fnc mat (i+1) j (fnc acc currentElement)
            where
                currentElement = get mat i j

-- Element-wise application of Double -> Double function
applyToMatElementWise :: (Double -> Double) -> Matrix -> Matrix
applyToMatElementWise fnc mat = Matrix (n mat) (m mat) (map fnc (array mat))

-- Element-wise application of Double -> Double -> Double function
-- it takes two matrices and combines them by element-wise application of the function
applyToTwoMatsElementWise :: (Double -> Double -> Double) -> Matrix -> Matrix -> Matrix
applyToTwoMatsElementWise fnc mat1 mat2
    | n mat1 /= n mat2 = emptyMatrix
    | m mat1 /= m mat2 = emptyMatrix
    | otherwise = Matrix (n mat1) (m mat1) (aux (array mat1) (array mat2))
    where
        aux :: [Double] -> [Double] -> [Double]
        aux [] _ = []
        aux _ [] = []
        aux (x:xs) (y:ys) = fnc x y : aux xs ys

------------------------------------------------------------------------------------
-- Implementing row extraction
-- return matrix specified by input matrix and given row numbers
cut1row :: Matrix -> Int -> Matrix
cut1row mat ind
    | matsAreEqual mat emptyMatrix = emptyMatrix
    | ind >= n mat = emptyMatrix
    | otherwise = Matrix 1 (m mat) (aux (array mat) (m mat) 0 0 ind)
    where
        aux [] _ _ _ _ = []
        aux (x:xs) m i j ind
            | (i == ind) && (j /= m) = x : aux xs m i (j+1) ind  -- chosen row
            | i == ind = []  -- chosen row finished
            | j == m = aux (x:xs) m (i+1) 0 ind  -- not chosen row finished
            | otherwise = aux xs m i (j+1) ind  -- not chosen row

------------------------------------------------------------------------------------
-- Visualizing Matrix object
printMat :: Matrix -> IO ()
printMat mat = do
    printf "\n(%d x %d) Matrix:\n\n" (n mat) (m mat)
    printMatArray (array mat) (m mat) 0
    printf "\n\n"

printMatArray :: [Double] -> Int -> Int -> IO ()
printMatArray [] _ _ = return ()
printMatArray (x:xs) m j
    | j == m = do
        printf "\n"
        printMatArray (x:xs) m 0
    | otherwise = do
        printf " %0.3f" x
        printMatArray xs m (j+1)

------------------------------------------------------------------------------------
