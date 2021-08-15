-- Copyright [2020] <Hendrik Henselmann>
module RandomMatrixBatch (chooseBatch, chooseBatchOf2Matrices, chooseRandomRowOf2Matrices, randomUnique, cut_) where

import Matrix

import System.Random

------------------------------------------------------------------------------------
-- Randomly choose a batch of given batch size out of a matrix
-- bad if batch size is almost the size of possible chosen rows (n)
-- but fairly good if batchSize >> (n mat)
chooseBatch :: Int -> Matrix -> Int -> Matrix
chooseBatch seed mat batchSize
    | (batchSize == 0) = emptyMatrix
    | (null (array mat)) = emptyMatrix
    | (batchSize > (n mat)) = emptyMatrix
    | (batchSize == (n mat)) = mat
    | otherwise = Matrix batchSize (m mat) randomBatch
    where
        randomIndices = randomUnique seed ((n mat)-1) batchSize
        randomBatch = cut_ (array mat) (m mat) randomIndices

-- Randomly choose a batch of given batch size out of two matrices
-- bad if batch size is almost the size of possible chosen rows (n)
-- but fairly good if batchSize << (n mat)
chooseBatchOf2Matrices :: Int -> Matrix -> Matrix -> Int -> (Matrix, Matrix)
chooseBatchOf2Matrices seed mat1 mat2 batchSize
    | (n mat1 /= n mat2) = (emptyMatrix, emptyMatrix)
    | (batchSize == 0) = (emptyMatrix, emptyMatrix)
    | (batchSize > (n mat1)) = (emptyMatrix, emptyMatrix)
    | (batchSize == (n mat1)) = (mat1, mat2)
    | otherwise = (Matrix batchSize (m mat1) randomBatch1, Matrix batchSize (m mat2) randomBatch2)
    where
        randomIndices = randomUnique seed ((n mat1)-1) batchSize
        randomBatch1 = cut_ (array mat1) (m mat1) randomIndices
        randomBatch2 = cut_ (array mat2) (m mat2) randomIndices

-- Randomly choose a row out of two matrices
chooseRandomRowOf2Matrices :: Int -> Matrix -> Matrix -> (Matrix, Matrix)
chooseRandomRowOf2Matrices seed mat1 mat2 = chooseBatchOf2Matrices seed mat1 mat2 1

-- return matrix array specified by input matrix array and given array of chosen row numbers
-- Important Note: row indices in array have to be unique and sorted in ascending order!
cut_ :: [Double] -> Int -> [Int] -> [Double]
cut_ _ _ [] = []
cut_ matArray m indices = aux matArray m 0 0 indices
    where
        aux _ _ _ _ [] = []
        aux [] _ _ _ _ = []
        aux (x:xs) m i j (ind:inds)
            | ((i == ind) && (j /= m)) = x:(aux xs m i (j+1) (ind:inds))  -- chosen row
            | (i == ind) = aux (x:xs) m (i+1) 0 inds  -- chosen row finished
            | (j == m) = aux (x:xs) m (i+1) 0 (ind:inds)  -- not chosen row finished
            | otherwise = aux xs m i (j+1) (ind:inds)  -- not chosen row

-- return array of x unique Ints between 0 and max (including)
randomUnique :: Int -> Int -> Int -> [Int]
randomUnique seed max x = aux max x Leaf pureGen
    where
        pureGen = mkStdGen seed
        aux max i tree gen
            | (i == 0) = getSortedList tree
            | notUpdated = aux max i tree updatedGen
            | otherwise = aux max (i-1) updatedTree updatedGen
            where
                res = uniformR (0 :: Int, max :: Int) gen
                uniqueRandomInt = fst res
                updatedGen = snd res
                updatedTree = checkAndInsert tree uniqueRandomInt
                notUpdated = (updatedTree == Found)

------------------------------------------------------------------------------------

-- Tree structure to realize chooseBatch in O(n + m*logn)
-- Found for checkAndInsert
data SearchTree a = Leaf | Found | Node (SearchTree a) a (SearchTree a)
    deriving (Show)

instance (Eq (SearchTree a)) where
    (==) Found Found = True
    (==) Leaf Leaf = True
    (==) _ _ = False

-- insert Int into SearchTree
insert :: SearchTree Int -> Int -> SearchTree Int
insert Found _ = Found
insert Leaf insertVal = Node Leaf insertVal Leaf
insert (Node left val right) insertVal
    | (val == insertVal) = Node left val right
    | (insertVal < val) = Node (insert left insertVal) val right
    | otherwise = Node left val (insert right insertVal)

-- check if Int in SearchTree
elementOf :: SearchTree Int -> Int -> Bool
elementOf Found _ = False
elementOf Leaf _ = False
elementOf (Node left val right) searchVal
    | (val == searchVal) = True
    | (searchVal < val) = elementOf left searchVal
    | otherwise = elementOf right searchVal

-- check if Int is in SearchTree: return Found if yes, otherwise the updated Tree
checkAndInsert :: SearchTree Int -> Int -> SearchTree Int
checkAndInsert Found _ = Found
checkAndInsert Leaf searchVal = Node Leaf searchVal Leaf
checkAndInsert (Node left val right) searchVal
    | (val == searchVal) = Found
    | (searchVal < val) = returnedTreeLeft
    | otherwise = eturnedTreeRight
    where
        newLeft = checkAndInsert left searchVal
        newRight = checkAndInsert right searchVal
        returnedTreeLeft
            | (newLeft == Found) = Found
            | otherwise = Node newLeft val right
        eturnedTreeRight
            | (newRight == Found) = Found
            | otherwise = Node left val newRight

-- transform tree to list of sorted values
getSortedList :: SearchTree Int -> [Int]
getSortedList Found = []
getSortedList Leaf = []
getSortedList (Node left val right) = (getSortedList left) ++ [val] ++ (getSortedList right)

------------------------------------------------------------------------------------
