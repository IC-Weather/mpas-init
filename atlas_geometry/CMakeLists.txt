cmake_minimum_required(VERSION 3.20)
project(mpas_atlas_candidates LANGUAGES CXX)

find_package(atlas 0.46 REQUIRED CONFIG)
add_executable(mpas-atlas-candidates main.cc)
target_compile_features(mpas-atlas-candidates PRIVATE cxx_std_17)
target_link_libraries(mpas-atlas-candidates PRIVATE atlas)
target_compile_definitions(mpas-atlas-candidates PRIVATE
    MPAS_ATLAS_VERSION="${atlas_VERSION}"
    MPAS_ATLAS_REVISION="${atlas_GIT_SHA1}")
if(APPLE)
    set_property(TARGET mpas-atlas-candidates PROPERTY INSTALL_RPATH "@loader_path/../lib")
else()
    set_property(TARGET mpas-atlas-candidates PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
endif()
set_property(TARGET mpas-atlas-candidates PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS mpas-atlas-candidates RUNTIME DESTINATION bin)
