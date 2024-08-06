set(ITEM hard)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin/${ITEM})

foreach(_target
  get_text2image1m_gt
  get_turing1m_gt
  get_audio5w_gt

  sift100m_vamana 
  sift100m_hnsw 
  sift100m_hcnng
  sift100m_nsg

  text2image1m_vamana 
  text2image1m_hnsw 
  text2image1m_hcnng
  text2image1m_nsg

  turing1m_vamana 
  turing1m_hnsw 
  turing1m_hcnng
  turing1m_nsg

  audio5w_hcnng_cspg
  audio5w_hnsw_cspg
  audio5w_nsg_cspg
  audio5w_vamana_cspg
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