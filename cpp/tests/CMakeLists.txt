include(GoogleTest)

add_executable(test_mask test_mask.cpp)
target_link_libraries(test_mask PRIVATE omr_core GTest::gtest_main)
gtest_discover_tests(test_mask)

add_executable(test_primitives test_primitives.cpp)
target_link_libraries(test_primitives PRIVATE omr_core GTest::gtest_main)
gtest_discover_tests(test_primitives)

add_executable(test_splitter_helpers test_splitter_helpers.cpp)
target_link_libraries(test_splitter_helpers PRIVATE omr_core GTest::gtest_main)
gtest_discover_tests(test_splitter_helpers)

add_executable(test_metrics test_metrics.cpp)
target_link_libraries(test_metrics PRIVATE omr_core GTest::gtest_main)
gtest_discover_tests(test_metrics)

add_executable(test_evaluation test_evaluation.cpp)
target_link_libraries(test_evaluation PRIVATE omr_core GTest::gtest_main)
gtest_discover_tests(test_evaluation)

add_executable(test_preprocessor test_preprocessor.cpp)
target_link_libraries(test_preprocessor PRIVATE omr_core GTest::gtest_main)
gtest_discover_tests(test_preprocessor)

add_executable(test_deepscores_dataset test_deepscores_dataset.cpp)
target_link_libraries(test_deepscores_dataset PRIVATE omr_core GTest::gtest_main)
gtest_discover_tests(test_deepscores_dataset)

add_executable(test_deepscores_io test_deepscores_io.cpp)
target_link_libraries(test_deepscores_io PRIVATE omr_core GTest::gtest_main)
gtest_discover_tests(test_deepscores_io)

add_executable(test_assembly test_assembly.cpp)
target_link_libraries(test_assembly PRIVATE omr_core GTest::gtest_main)
gtest_discover_tests(test_assembly)

add_executable(test_semantics test_semantics.cpp)
target_link_libraries(test_semantics PRIVATE omr_core GTest::gtest_main)
gtest_discover_tests(test_semantics)
