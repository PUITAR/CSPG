import subprocess
from tqdm import tqdm
import time
import sys

programs = []

modes = ['overall', 'partition', 'redundancy', 'stages', 'scale', 'waste', 'hard', 'grid', 'distribution']

if len(sys.argv) != 2:
  print(f'Error: please specify one mode of {modes} by passing only one parameter')
  exit(1)

mode = sys.argv[1]

if mode not in modes:
  print(f'Error: no experiment item named {mode}, please select one mode of {modes}')
  exit(1)

programs_list = {
  'overall': [
    # 'deep1m_hcnng', 'deep1m_hnsw', 'deep1m_vamana', 'gist1m_hcnng', 'gist1m_hnsw', 'gist1m_vamana', 'sift1m_hcnng', 'sift1m_hnsw', 'sift1m_vamana', 'sift10m_hcnng', 'sift10m_hnsw', 'sift10m_vamana',
    'deep1m_nsg', 'gist1m_nsg', 'sift1m_nsg', 'sift10m_nsg'
  ], 
  'partition': [
    # 'deep1m_hcnng', 'deep1m_hnsw', 'deep1m_vamana', 'gist1m_hcnng', 'gist1m_hnsw', 'gist1m_vamana', 'sift1m_hcnng', 'sift1m_hnsw', 'sift1m_vamana',
    'deep1m_nsg', 'gist1m_nsg', 'sift1m_nsg'
  ],
  'redundancy': [
    # 'deep1m_hcnng', 'deep1m_hnsw', 'deep1m_vamana', 'gist1m_hcnng', 'gist1m_hnsw', 'gist1m_vamana', 'sift1m_hcnng', 'sift1m_hnsw', 'sift1m_vamana',
    'deep1m_nsg', 'gist1m_nsg', 'sift1m_nsg'
  ],
  'scale': [
    # 'sift0.1m_hnsw', 'sift0.2m_hnsw', 'sift0.5m_hnsw', 'sift2m_hnsw', 'sift5m_hnsw', 'sift0.1m_vamana', 'sift0.2m_vamana', 'sift0.5m_vamana', 'sift2m_vamana', 'sift5m_vamana', 'sift0.1m_hcnng', 'sift0.2m_hcnng', 'sift0.5m_hcnng', 'sift2m_hcnng', 'sift5m_hcnng',
    'sift0.1m_nsg', 'sift0.2m_nsg', 'sift0.5m_nsg', 'sift2m_nsg', 'sift5m_nsg'
  ], 
  'stages': [
    # 'deep1m_hcnng', 'deep1m_hnsw', 'deep1m_vamana', 'gist1m_hcnng', 'gist1m_hnsw', 'gist1m_vamana', 'sift1m_hcnng', 'sift1m_hnsw', 'sift1m_vamana',
    'deep1m_nsg', 'gist1m_nsg', 'sift1m_nsg'
  ],
  'waste': [
    # 'sift0.1m_hnsw', 'sift0.2m_hnsw', 'sift0.5m_hnsw', 'sift2m_hnsw', 'sift5m_hnsw', 'sift0.1m_vamana', 'sift0.2m_vamana', 'sift0.5m_vamana', 'sift2m_vamana', 'sift5m_vamana', 'sift0.1m_hcnng', 'sift0.2m_hcnng', 'sift0.5m_hcnng', 'sift2m_hcnng', 'sift5m_hcnng',
    'sift0.1m_nsg', 'sift0.2m_nsg', 'sift0.5m_nsg', 'sift2m_nsg', 'sift5m_nsg'
  ], 
  'hard': [
    # 'sift100m_hcnng', 'sift100m_vamana', 'sift100m_nsg', 'sift100m_hnsw', 
    # 'kosarak_hcnng', 'kosarak_hnsw', 'kosarak_vamana', 'kosarak_nsg',
    # 'turing1m_hcnng', 'turing1m_hnsw', 'turing1m_vamana', 'turing1m_nsg',
    # 'text2image1m_hcnng', 'text2image1m_hnsw', 'text2image1m_vamana', 'text2image1m_nsg',
    'audio5w_hcnng_cspg', 'audio5w_hnsw_cspg', 'audio5w_vamana_cspg', 'audio5w_nsg_cspg'
  ],
  'grid': [
    # 'sift1m_hcnng', 'sift1m_hnsw', 'sift1m_vamana', 'sift1m_nsg',
    # 'deep1m_hcnng', 'deep1m_hnsw', 'deep1m_vamana', 'deep1m_nsg',
    # 'gist1m_hcnng', 'gist1m_hnsw', 'gist1m_vamana', 'gist1m_nsg',
    'sift1m_hcnng_cmp', 'sift1m_hnsw_cmp', 'sift1m_vamana_cmp', 'sift1m_nsg_cmp',
    'deep1m_hcnng_cmp', 'deep1m_hnsw_cmp', 'deep1m_vamana_cmp', 'deep1m_nsg_cmp',
    'gist1m_hcnng_cmp', 'gist1m_hnsw_cmp', 'gist1m_vamana_cmp', 'gist1m_nsg_cmp',
  ], 
  'distribution': [
    # 'gcd_hnsw', 'gcd_hcnng', 'gcd_vamana', 'gcd_nsg',
    'gud_hnsw', 'gud_hcnng', 'gud_vamana', 'gud_nsg'
  ]
}

for p in programs_list[mode]:
  programs.append(p.strip())
  
subprocess.run(f'mkdir -p output/{mode}', shell=True)

print(f'Running experiment for \033[92m{mode}\033[0m')
print(f'Executing programs: \033[92m{programs}\033[0m')

for program in tqdm(programs):
    subprocess.run(f'bin/{mode}/{mode}_{program}', shell=True)
    time.sleep(1)
