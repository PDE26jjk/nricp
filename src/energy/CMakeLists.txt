add_library(nricp_energy STATIC)

target_sources(nricp_energy PRIVATE
    energy.cpp
    energy.h
    robust.cpp
    robust.h
)

target_include_directories(nricp_energy PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(nricp_energy PUBLIC nricp_core)

if(MSVC)
    set_target_properties(nricp_energy PROPERTIES
        VS_GLOBAL_IncludePath "${PROJECT_SOURCE_DIR}/src")
endif()
