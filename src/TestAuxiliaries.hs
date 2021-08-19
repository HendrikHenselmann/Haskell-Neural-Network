-- Copyright [2021] <Hendrik Henselmann>
module TestAuxiliaries (equalityWithTolerance) where

------------------------------------------------------------------------------------
-- Auxiliary functions for testing
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
equalityWithTolerance :: Double -> Double -> Double -> Bool
equalityWithTolerance epsilon val1 val2 = (val1 - epsilon <= val2) && (val1 + epsilon >= val2)

------------------------------------------------------------------------------------
