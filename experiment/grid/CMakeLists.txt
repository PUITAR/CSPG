set(ITEM grid)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin/${ITEM})

foreach(_target

  sift1m_vamana 
  sift1m_hnsw 
  sift1m_hcnng
  sift1m_nsg

  deep1m_vamana 
  deep1m_hnsw
  deep1m_hcnng
  deep1m_nsg

  gist1m_hnsw
  gist1m_vamana
  gist1m_hcnng
  gist1m_nsg

  
  sift1m_vamana_cmp 
  sift1m_hnsw_cmp 
  sift1m_hcnng_cmp
  sift1m_nsg_cmp

  deep1m_vamana_cmp 
  deep1m_hnsw_cmp
  deep1m_hcnng_cmp
  deep1m_nsg_cmp

  gist1m_hnsw_cmp
  gist1m_vamana_cmp
  gist1m_hcnng_cmp
  gist1m_nsg_cmp
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