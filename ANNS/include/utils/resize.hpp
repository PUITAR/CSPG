#ifndef INCLUDE_RESIZE_HPP
#define INCLUDE_RESIZE_HPP

#include <vector>
#include <stdint.h>

namespace utils
{

  template <typename vdim_t>
  std::vector<vdim_t> Flatten(const std::vector<std::vector<vdim_t>> &nested)
  {
    size_t d0 = nested.size();
    size_t d1 = nested[0].size();
    std::vector<vdim_t> flattened(d0 * d1);
    for (size_t i = 0; i < d0; i++)
    {
      std::copy(nested[i].begin(), nested[i].end(), flattened.begin() + i * d1);
    }
    return flattened;
  }

  template <typename vdim_t>
  std::vector<vdim_t> Flatten(const std::vector<std::vector<std::vector<vdim_t>>> &nested)
  {
    size_t d0 = nested.size();
    size_t d1 = nested[0].size();
    size_t d2 = nested[0][0].size();
    std::vector<vdim_t> flattened(d0 * d1 * d2);
    for (size_t i = 0; i < d0; i++)
    {
      for (size_t j = 0; j < d1; j++)
      {
        std::copy(nested[i][j].begin(), nested[i][j].end(), flattened.begin() + i * d1 * d2 + j * d2);
      }
    }
    return flattened;
  }

  template <typename vdim_t>
  std::vector<std::vector<vdim_t>> Nest(const std::vector<vdim_t> &flattened, size_t d0, size_t d1)
  {
    std::vector<std::vector<vdim_t>> nested(d0);

    for (size_t i = 0; i < d0; i++)
    {
      nested[i].resize(d1);
      std::copy(flattened.begin() + i * d1, flattened.begin() + (i + 1) * d1, nested[i].begin());
    }

    return nested;
  }

  template <typename vdim_t>
  std::vector<std::vector<std::vector<vdim_t>>> Nest(const std::vector<vdim_t> &flattened, size_t d0, size_t d1, size_t d2)
  {
    assert(flattened.size() == d0 * d1 * d2);
    std::vector<std::vector<std::vector<vdim_t>>> nested(d0, std::vector<std::vector<vdim_t>>(d1, std::vector<vdim_t>(d2)));
    for (size_t i = 0; i < d0; i++)
    {
      for (size_t j = 0; j < d1; j++)
      {
        std::copy(
            flattened.begin() + i * d1 * d2 + j * d2,
            flattened.begin() + i * d1 * d2 + (j + 1) * d2,
            nested[i][j].begin());
      }
    }
    return nested;
  }

};

#endif