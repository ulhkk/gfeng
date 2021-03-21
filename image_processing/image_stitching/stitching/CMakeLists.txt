add_executable(multi_stitching multi_stitching.cpp)
add_executable(apply_homo apply_homo.cpp)
target_link_libraries(apply_homo ${OpenCV_LIBS})
target_link_libraries(multi_stitching stitching ${OpenCV_LIBS})