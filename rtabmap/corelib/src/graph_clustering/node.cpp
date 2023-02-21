#include "rtabmap/core/graph_clustering/node.hpp"

namespace graph_clustering
{
   
    Node::Node(unsigned long id, const Position &position) : id_(id), position_(position), is_assigned_(false)
    {}

    bool Node::operator<(const Node &node)
    {
        if (this->position_.getX() < node.position_.getX())
            return true;
        if (this->position_.getX() > node.position_.getX())
            return false;
        if (this->position_.getY() < node.position_.getY())
            return true;
        return false;
    }

    std::string Node::prettyPrint()
    {
        return "\nNODE:\nnode_id: " + std::to_string(this->id_) + "\n" + this->position_.prettyPrint() + "region: " + (this->is_assigned_ ? std::to_string(this->region_) : "not assigned") + "\n";
    }
}
