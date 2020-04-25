#Generate the local Makefile
if(CREATE_LOCAL_MAKEFILES)
  file(RELATIVE_PATH ZIRAN_RELATIVE_PATH ${ZIRAN_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
  configure_file(${ZIRAN_SOURCE_DIR}/Scripts/Makefile.in ${CMAKE_CURRENT_SOURCE_DIR}/Makefile @ONLY)
endif()

if(OPENVDB_FOUND)

  add_ziran_executable(multigrid main.cpp)
  target_include_directories(multigrid PUBLIC ./)
  target_compile_options(multigrid PRIVATE ${ZIRAN_CXXFLAGS_LTO})
  target_link_libraries(multigrid ziran mpmdev tbbmalloc tbbmalloc_proxy arpack ${ZIRAN_CXXFLAGS_LTO})
  
endif()

