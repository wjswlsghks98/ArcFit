cmake_minimum_required(VERSION 3.10)
set(SOURCES test.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${Boost_INCLUDE_DIR})

add_executable(arcfit_test ${SOURCES})
set_property(TARGET arcfit_test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(arcfit_test 
                      Arcfit 
                      Eigen3::Eigen
                      Matplot++::matplot
                      ${Boost_LIBRARIES})