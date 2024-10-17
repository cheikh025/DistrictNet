cmake_minimum_required(VERSION 3.13.0 FATAL_ERROR)

project(GeoScenario LANGUAGES CXX)

# Find the JlCxx package
find_package(JlCxx REQUIRED)

add_library(GenerateScenario SHARED
    Block.cpp
    Params.cpp
    Point.cpp
    Polygon.cpp
    SCmain.cpp
)

# Include directories
target_include_directories(GenerateScenario PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link against the JlCxx library
target_link_libraries(GenerateScenario PUBLIC JlCxx::cxxwrap_julia)

# Configure RPATH so the shared library can find its dependencies
set_target_properties(GenerateScenario PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;$<TARGET_FILE_DIR:JlCxx::cxxwrap_julia>"
)