set(ITEM scale)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin/${ITEM})

foreach(_target
  sample_dataset_construction

  sift0.1m_hnsw    
  sift0.2m_hnsw    
  sift0.5m_hnsw    
  sift2m_hnsw
  sift5m_hnsw

  sift0.1m_vamana
  sift0.2m_vamana
  sift0.5m_vamana
  sift2m_vamana
  sift5m_vamana

  sift0.1m_hcnng            
  sift0.2m_hcnng
  sift0.5m_hcnng
  sift2m_hcnng   
  sift5m_hcnng

  sift0.1m_nsg
  sift0.2m_nsg
  sift0.5m_nsg
  sift2m_nsg
  sift5m_nsg
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