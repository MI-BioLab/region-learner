#pragma once

/*
 * file: KDTree.hpp
 * author: J. Frederico Carvalho
 *
 * This is an adaptation of the KD-tree implementation in rosetta code
 *  https://rosettacode.org/wiki/K-d_tree
 * It is a reimplementation of the C code using C++.
 * It also includes a few more queries than the original
 *
 */

#ifndef KD_TREE_HPP
#define KD_TREE_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

namespace graph_clustering
{
    using point_t = std::vector<double>;
    using indexArr = std::vector<size_t>;
    using pointIndex = typename std::pair<std::vector<double>, size_t>;

    class KDNode
    {
    public:
        using KDNodePtr = std::shared_ptr<KDNode>;
        size_t index;
        point_t x;
        KDNodePtr left;
        KDNodePtr right;

        // initializer
        KDNode();
        KDNode(const point_t &, const size_t &, const KDNodePtr &,
               const KDNodePtr &);
        KDNode(const pointIndex &, const KDNodePtr &, const KDNodePtr &);
        ~KDNode();

        // getter
        double coord(const size_t &);

        // conversions
        explicit operator bool();
        explicit operator point_t();
        explicit operator size_t();
        explicit operator pointIndex();
    };

    using KDNodePtr = std::shared_ptr<KDNode>;

    KDNodePtr NewKDNodePtr();

    // square euclidean distance
    double dist2(const point_t &, const point_t &);
    double dist2(const KDNodePtr &, const KDNodePtr &);

    // euclidean distance
    double dist(const point_t &, const point_t &);
    double dist(const KDNodePtr &, const KDNodePtr &);

    // Need for sorting
    class comparer
    {
    public:
        size_t idx;
        explicit comparer(size_t idx_);
        inline bool compare_idx(
            const std::pair<std::vector<double>, size_t> &, //
            const std::pair<std::vector<double>, size_t> &  //
        );
    };

    using pointIndexArr = typename std::vector<pointIndex>;

    inline void sort_on_idx(const pointIndexArr::iterator &, //
                            const pointIndexArr::iterator &, //
                            size_t idx);

    using pointVec = std::vector<point_t>;

    class KDTree
    {
        KDNodePtr root;
        KDNodePtr leaf;

        KDNodePtr make_tree(const pointIndexArr::iterator &begin, //
                            const pointIndexArr::iterator &end,   //
                            const size_t &length,                 //
                            const size_t &level                   //
        );

    public:
        typedef std::shared_ptr<KDTree> SharedPtr;
        typedef std::shared_ptr<const KDTree> ConstSharedPtr;

        typedef std::unique_ptr<KDTree> UniquePtr;
        typedef std::unique_ptr<const KDTree> ConstUniquePtr;

        typedef std::weak_ptr<KDTree> WeakPtr;
        typedef std::weak_ptr<const KDTree> ConstWeakPtr;
        KDTree() = default;
        explicit KDTree(pointVec point_array);

    private:
        KDNodePtr nearest_(          //
            const KDNodePtr &branch, //
            const point_t &pt,       //
            const size_t &level,     //
            const KDNodePtr &best,   //
            const double &best_dist, //
            bool exclude_pt);

        // default caller
        KDNodePtr nearest_(const point_t &pt, bool exclude_pt);

    public:
        point_t nearest_point(const point_t &pt, bool exclude_pt);
        size_t nearest_index(const point_t &pt, bool exclude_pt);
        pointIndex nearest_pointIndex(const point_t &pt, bool exclude_pt);

        bool contains(const point_t &pt);

    private:
        pointIndexArr neighborhood_( //
            const KDNodePtr &branch, //
            const point_t &pt,       //
            const double &rad,       //
            const size_t &level      //
        );

    public:
        pointIndexArr neighborhood( //
            const point_t &pt,      //
            const double &rad);

        pointVec neighborhood_points( //
            const point_t &pt,        //
            const double &rad);

        indexArr neighborhood_indices( //
            const point_t &pt,         //
            const double &rad);
    };
}
#endif