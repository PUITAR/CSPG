#pragma once

#include <vector>

double GetWasteFactor(const std::vector<std::vector<float>> & lengths) {
  size_t nl = lengths.size();
  if (nl == 0) return 0;
  double w = 0;
  for (size_t i = 0; i < nl; i++) {
    if (lengths[i].empty()) continue;
    float bound = lengths[i][0];
    size_t waste = 0;
    for (float l: lengths[i]) {
      if (l > bound) waste++;
      else bound = l;
    }
    w += double(waste) / lengths[i].size();
  } 
  return w / nl;
}