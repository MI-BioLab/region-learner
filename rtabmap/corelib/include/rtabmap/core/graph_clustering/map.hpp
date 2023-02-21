#pragma once

#ifndef MAP_HPP
#define MAP_HPP

#include <iostream>
#include <list>
#include <unordered_map>
#include <iterator>
#include <memory>

#include "rtabmap/core/graph_clustering/constants.hpp"
#include "rtabmap/core/graph_clustering/position.hpp"
#include "rtabmap/core/graph_clustering/node.hpp"
#include "rtabmap/core/graph_clustering/edge.hpp"
#include "rtabmap/core/graph_clustering/kd_tree.hpp"
#include "rtabmap/core/graph_clustering/region.hpp"

namespace graph_clustering
{

    /**
     * Class that represents a map.
     */
    class Map
    {

    public:
        typedef std::shared_ptr<Map> SharedPtr;
        typedef std::shared_ptr<const Map> ConstSharedPtr;

        typedef std::unique_ptr<Map> UniquePtr;
        typedef std::unique_ptr<const Map> ConstUniquePtr;

        typedef std::weak_ptr<Map> WeakPtr;
        typedef std::weak_ptr<const Map> ConstWeakPtr;

        Map(const Constants &constants,
            float radius_upper_bound,
            int desired_average_cardinality,
            float mesh_shape_factor,
            unsigned long initial_region,
            float alpha_vis,
            float alpha_equ,
            float alpha_pre,
            float alpha_hom,
            float alpha_coh,
            float alpha_reg);

        ~Map();

        void addNode(const Node::SharedPtr &node);
        void SBA(const Node::SharedPtr &node, const Edge::SharedPtr &edge);

        inline const std::unordered_map<unsigned long, Region::SharedPtr> &getRegions() const { return this->regions_; }
        inline const std::unordered_map<unsigned long, Node::SharedPtr> &getNodes() const { return this->nodes_; }
        inline const std::unordered_map<EdgePair, Edge::SharedPtr, EdgeHasher> &getEdges() const { return this->edges_; }
        inline Node::SharedPtr getNode(unsigned long id) const
        {
            if (this->nodes_.count(id))
            {
                return this->nodes_.at(id);
            };

            return std::shared_ptr<Node>();
        }

        bool contains(std::vector<double> node_pos);

        /*
        inline float getTotalScattering() const { return this->total_scattering_; }
        bool isNodeReassignable(const Node::Ptr &node); */
        std::string prettyPrint();

    private:
        void computeAverageCardinality_();
        void computeGaps_();
        void computeTotalMesh_();
        void computeDefaultScattering_();
        void computeThreshold_();

        // void move_(Node &node);

        std::unordered_map<unsigned long, std::shared_ptr<Node>> nodes_;
        std::unordered_map<EdgePair, Edge::SharedPtr, EdgeHasher> edges_;
        std::unordered_map<unsigned long, Region::SharedPtr> regions_;

        const Constants constants_;

        const float radius_upper_bound_;
        const int desired_average_cardinality_;
        const float mesh_shape_factor_;
        const float alpha_vis_;
        const float alpha_equ_;
        const float alpha_pre_;
        const float alpha_hom_;
        const float alpha_coh_;
        const float alpha_reg_;

        float average_cardinality_;
        float total_mesh_;

        const float scattering_1_constants_;
        float default_scattering_;
        float threshold_;

        float total_scattering_;

        std::vector<std::vector<double>> points_;
        KDTree::SharedPtr nodes_tree_;
        std::unordered_map<unsigned long, float> gaps_;

        unsigned long region_id_factory_;

        float base_threshold_;
    };
}

#endif