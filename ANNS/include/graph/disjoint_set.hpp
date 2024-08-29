#pragma once

#include <vector>
#include <sys/types.h>

namespace anns
{

class DisjointSet 
{

public:
    DisjointSet(size_t size)
    {
        parent.resize(size);
        for (size_t i = 0; i < size; ++i)
            parent[i] = i;
    }

    id_t Find(id_t x)
    {
        if (parent[x] != x)
            parent[x] = Find(parent[x]);
        return parent[x];
    }

    void UnionSet(id_t x, id_t y)
    {
        parent[Find(x)] = Find(y);
    }

private:
    std::vector<id_t> parent;

};

} // namespace index
