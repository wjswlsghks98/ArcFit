cmake_minimum_required(VERSION 3.10)
project(Arcfit)

find_package(Eigen3 REQUIRED)
find_package(Matplot++ REQUIRED)
find_package(Boost REQUIRED)

add_subdirectory(test)

set(SOURCES 
    src/SingleArcFit.cpp
    src/MultipleArcFit.cpp
    src/Models.cpp
    src/Utils.cpp
)



# include_directories(${CMAKE_SOURCE_DIR}/external/eigen)
# include_directories(${CMAKE_SOURCE_DIR}/external/ceres-solver)
add_library(Arcfit ${SOURCES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set_property(TARGET Arcfit PROPERTY POSITION_INDEPENDENT_CODE ON)

include_directories(Arcfit PUBLIC 
                    ${PROJECT_SOURCE_DIR}/include
                    ${Boost_INCLUDE_DIR})

target_link_libraries(Arcfit PUBLIC
                      Eigen3::Eigen 
                      Matplot++::matplot
                      ${Boost_LIBRARIES}
                      )
set_target_properties(Arcfit PROPERTIES
                      CXX_STANDARD 17)