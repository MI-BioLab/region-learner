#pragma once

#ifndef REGION_HPP
#define REGION_HPP

#include <iostream>
#include <list>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <set>

#include "rtabmap/core/graph_clustering/constants.hpp"
#include "rtabmap/core/graph_clustering/position.hpp"
#include "rtabmap/core/graph_clustering/node.hpp"
#include "rtabmap/core/graph_clustering/edge.hpp"
#include "rtabmap/core/graph_clustering/kd_tree.hpp"

namespace graph_clustering
{
    /**
     * Class that represents a cluster of the graph.
     */
    class Region
    {

    public:
        typedef std::shared_ptr<Region> SharedPtr;
        typedef std::shared_ptr<const Region> ConstSharedPtr;

        typedef std::unique_ptr<Region> UniquePtr;
        typedef std::unique_ptr<const Region> ConstUniquePtr;

        typedef std::weak_ptr<Region> WeakPtr;
        typedef std::weak_ptr<const Region> ConstWeakPtr;

        Region(unsigned long id,
             const Constants &constants);

        ~Region();

        inline unsigned long getId() const { return this->id_; }

        std::list<std::shared_ptr<Node>> getNodesAsList() const;
        inline const std::unordered_map<EdgePair, std::shared_ptr<Edge>, EdgeHasher> &getEdges() const { return this->edges_; }
        inline const std::unordered_map<unsigned long, std::shared_ptr<Node>> &getNodes() const { return this->nodes_; } 

        bool addNode(const std::shared_ptr<Node> &node);
        bool addEdge(const std::shared_ptr<Edge> &edge);
        bool addNodeAndEdge(const std::shared_ptr<Node> &node, const std::shared_ptr<Edge> &edge);

        inline int getCardinality() const { return this->cardinality_; }
        inline const Position &getPosition() const { return this->position_; }

        inline float getScattering2() const { return this->scattering2_; }

        inline float getDistanceFromPositionForNode(const std::shared_ptr<Node> &node) const
        {
            if (this->position_distances2_.count(node->getId()))
            {
                return sqrt(this->position_distances2_.at(node->getId()));
            }
            return 0;
        }

        void update(float average_cardinality, float default_scattering2);

        // void traverse(const Node &current, std::unordered_map<unsigned long, bool> &visited);

        std::string prettyPrint();

    private:
        void computeCardinality_();
        void computePosition_();
        void computeKDTree_();
        void computeGap_();
        void computeMesh_();
        void computeEquivalentRadius_();
        void computePositionDistances2_();
        void computeMaximumRadius_();
        void computeScattering2_(float default_scattering2);

        unsigned long id_;

        const Constants constants_;

        int cardinality_;
        Position position_;
        float mesh_;
        float equivalent_radius_;
        float maximum_radius_;
        float scattering2_;

        std::unordered_map<unsigned long, Node::SharedPtr> nodes_;
        std::unordered_map<EdgePair, Edge::SharedPtr, EdgeHasher> edges_;

        std::vector<std::vector<double>> points_;

        std::unordered_map<unsigned long, float> gaps_;
        std::unordered_map<unsigned long, float> position_distances2_;

        KDTree::SharedPtr nodes_tree_;
    };

}

#endif