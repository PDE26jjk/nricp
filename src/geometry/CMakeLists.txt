add_library(nricp_geometry STATIC)

target_sources(nricp_geometry PRIVATE
    closest_point.cpp
    closest_point.h
)

target_include_directories(nricp_geometry PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(nricp_geometry PUBLIC nricp_core)

if(MSVC)
    # Makes the IDE resolve the project's own includes for these files.
    set_target_properties(nricp_geometry PROPERTIES
        VS_GLOBAL_IncludePath "${PROJECT_SOURCE_DIR}/src")
endif()
