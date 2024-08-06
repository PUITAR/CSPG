#ifndef INCLUDE_DISJOINT_SET_HPP
#define INCLUDE_DISJOINT_SET_HPP

#include <vector>
#include <sys/types.h>

namespace puiann
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


#endif // !INCLUDE_DISJOINT_SET_HPP