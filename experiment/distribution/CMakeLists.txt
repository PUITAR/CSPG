set(ITEM distribution)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin/${ITEM})

foreach(_target
  gcd_get_gt
  gcd_hcnng
  gcd_hnsw
  gcd_nsg
  gcd_vamana

  gud_get_gt
  gud_hcnng
  gud_hnsw
  gud_nsg
  gud_vamana
)
  add_executable(${ITEM}_${_target} "${_target}.cpp" ${index_source})
  target_compile_options(${ITEM}_${_target} PUBLIC 
    -Ofast 
    -march=native 
    -mtune=native 
    -lrt
  )
  target_link_libraries(${ITEM}_${_target} PUBLIC
    OpenMP::OpenMP_CXX
  )
endforeach()