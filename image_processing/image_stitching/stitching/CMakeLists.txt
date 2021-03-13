add_executable(multi_stitching multi_stitching.cpp)
target_link_libraries(multi_stitching stitching ${OpenCV_LIBS})