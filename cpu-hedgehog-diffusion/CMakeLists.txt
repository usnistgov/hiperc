cmake_minimum_required(VERSION 3.9)
project(diffusion_Hedgehog)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-report")

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)

add_definitions(-DPROFILE)

find_package(Threads REQUIRED)
find_package(Hedgehog REQUIRED)
find_package(PNG REQUIRED)

set(SRC
        data/GridPtrData.cpp
        data/GridPtrData.h
        tasks/DiffOpTask.cpp
        tasks/DiffOpTask.h
        utils/type.h
        utils/output.cpp
        utils/output.h
        utils/boundaries.h
        utils/mesh.cpp
        utils/mesh.h
        utils/numerics.cpp
        utils/numerics.h)

add_executable(diffusion_Hedgehog main.cpp ${SRC})

target_include_directories(diffusion_Hedgehog PUBLIC ${Hedgehog_INCLUDE_DIR})
target_link_libraries(diffusion_Hedgehog ${Hedgehog_LIBRARIES})
target_link_libraries(diffusion_Hedgehog ${PNG_LIBRARIES})
