#pragma once

#ifndef EDGE_HPP
#define EDGE_HPP

#include <iostream>
#include <memory>

namespace graph_clustering
{

    /**
     * Edge pair to used as key for unordered_map.
     * Edges are not directed. Given two edges E1 = (from_n1, to_n2) and E2 = (from_m1, to_m2), 
     * E1 = E2 if (from_n1 = from_m1 ∧ to_n2 = to_m2) ∨ (from_n1 = to_m2 ∧ to_n2 = from_m1).
    */
    struct EdgePair
    {
        unsigned long from_id;
        unsigned long to_id;

        EdgePair(unsigned long from_id, unsigned long to_id) : from_id(from_id), to_id(to_id) {}

        bool operator==(const EdgePair &other) const
        {
            return ((from_id == other.from_id && to_id == other.to_id) ||
                    (from_id == other.to_id && to_id == other.from_id));
        }
    };

    /**
     * Hash function for edges. 
     * Since edges are not directed, the hash function must be symmetric in order 
     * to respect the conditions in the == operator above.
    */
    struct EdgeHasher
    {
        std::size_t operator()(const EdgePair &k) const
        {
            unsigned long h = std::min(k.from_id, k.to_id);
            h ^= std::hash<unsigned long>()(std::max(k.from_id, k.to_id)) + 0x9e3779b9 + (h << 6) + (h >> 2);

            return h;
        }
    };

    /**
     * Class that represents an edge of the graph.
    */
    class Edge
    {
    public:
        typedef std::shared_ptr<Edge> SharedPtr;
        typedef std::shared_ptr<const Edge> ConstSharedPtr;

        typedef std::unique_ptr<Edge> UniquePtr;
        typedef std::unique_ptr<const Edge> ConstUniquePtr;

        typedef std::weak_ptr<Edge> WeakPtr;
        typedef std::weak_ptr<const Edge> ConstWeakPtr;

        Edge(unsigned long id_from, unsigned long id_to);

        inline unsigned long getIdFrom() const { return this->id_from_; }
        inline unsigned long getIdTo() const { return this->id_to_; }
        inline bool isExternal() const { return this->is_external_; }

        inline void setIdFrom(unsigned long id_from) { this->id_from_ = id_from; }
        inline void setIdTo(unsigned long id_to) { this->id_to_ = id_to; }
        inline void setExternal(bool is_external) { this->is_external_ = is_external; }

        inline const EdgePair toEdgePair() const { return EdgePair(this->id_from_, this->id_to_); }

        std::string prettyPrint();

    private:
        bool is_external_;

        unsigned long id_from_;
        unsigned long id_to_;
    };
}
#endif