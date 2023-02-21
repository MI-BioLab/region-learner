#pragma once

#ifndef NODE_HPP
#define NODE_HPP

#include <iostream>
#include <memory>
#include <vector>

#include "rtabmap/core/graph_clustering/position.hpp"

namespace graph_clustering
{
    /**
     * Class that represents a node of the graph.
    */
    class Node
    {
    public:
        typedef std::shared_ptr<Node> SharedPtr;
        typedef std::shared_ptr<const Node> ConstSharedPtr;

        typedef std::unique_ptr<Node> UniquePtr;
        typedef std::unique_ptr<const Node> ConstUniquePtr;

        typedef std::weak_ptr<Node> WeakPtr;
        typedef std::weak_ptr<const Node> ConstWeakPtr;

        Node(unsigned long id,
             const Position &position);

        inline unsigned long getId() const { return this->id_; }
        inline float getX() const { return this->position_.getX(); }
        inline float getY() const { return this->position_.getY(); }
        inline const Position &getPosition() const { return this->position_; }
        inline std::vector<double> getPositionAsVector() const { return this->position_.toVector(); }
        inline unsigned long getRegion() const { return this->region_; }
        inline bool isAssigned() const { return this->is_assigned_; }

        inline void setRegion(unsigned long region)
        {
            this->region_ = region;
            this->is_assigned_ = true;
        }

        bool operator<(const Node &node);

        std::string prettyPrint();

    private:
        const unsigned long id_;
        const Position position_;
        unsigned long region_;
        bool is_assigned_;
    };
}

#endif