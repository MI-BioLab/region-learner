#include "rtabmap/core/graph_clustering/edge.hpp"

namespace graph_clustering
{
    Edge::Edge(unsigned long id_from, unsigned long id_to) : id_from_(id_from), id_to_(id_to) {}

    std::string Edge::prettyPrint()
    {
        return "\nEDGE\nid_from: " + std::to_string(this->id_from_) + "\nid_to: " + std::to_string(this->id_to_) + "\n";
    }
}