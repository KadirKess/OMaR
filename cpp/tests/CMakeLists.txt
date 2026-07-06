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
