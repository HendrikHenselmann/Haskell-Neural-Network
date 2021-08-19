import LossFuncs_Tests (tests)
import Feedforward_Tests (tests)
import PerformanceMetrics_Tests (tests)
import BackPropagation_Tests (tests)
import Scaler_Tests (tests)
import StoreAndLoad_Tests (tests)
import Matrix_Tests (tests)
import OneHotEncoding_Tests (tests)
import WeightInitialization_Tests (tests)

import System.Exit
import Test.HUnit

main :: IO ()
main = do
    results <- runTestTT $ test (LossFuncs_Tests.tests
                    ++ Feedforward_Tests.tests
                    ++ PerformanceMetrics_Tests.tests
                    ++ BackPropagation_Tests.tests
                    ++ Scaler_Tests.tests
                    ++ StoreAndLoad_Tests.tests
                    ++ Matrix_Tests.tests
                    ++ OneHotEncoding_Tests.tests
                    ++ WeightInitialization_Tests.tests)
    if errors results + failures results == 0 then
        putStrLn "Tests passed."
    else
        die "Tests failed."
