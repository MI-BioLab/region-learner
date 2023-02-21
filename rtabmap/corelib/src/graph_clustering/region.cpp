#include "rtabmap/core/graph_clustering/region.hpp"

namespace graph_clustering
{
    Region::Region(unsigned long id,
               const Constants &constants) : id_(id),
                                             constants_(constants),
                                             position_(Position(0, 0)),
                                             cardinality_(0),
                                             mesh_(0),
                                             equivalent_radius_(0),
                                             maximum_radius_(0),
                                             scattering2_(0) {}

    Region::~Region()
    {
        this->nodes_tree_.reset();
    }

    bool Region::addNode(const Node::SharedPtr &node)
    {
        // if the node is not already in the region
        if (!this->nodes_.count(node->getId()))
        {
            this->points_.emplace_back(node->getPositionAsVector());
            this->nodes_.insert({node->getId(), node});
            return true;
        }
        else
        {
            std::cout << "Node already exists in this region!";
            return false;
        }
    }

    bool Region::addEdge(const Edge::SharedPtr &edge)
    {
        EdgePair key = edge->toEdgePair();

        // if the edge is not already in the region
        
        this->edges_.insert({key, edge});
        return true;
           
    }

    bool Region::addNodeAndEdge(const Node::SharedPtr &node, const Edge::SharedPtr &edge)
    {
        if (edge->getIdFrom() != node->getId() && edge->getIdTo() != node->getId())
        {
            std::cout << "Error! This node is not related to this edge!\n";
            return false;
        }
        else
        {
            return this->addNode(node) && this->addEdge(edge);
        }
    }

    std::list<std::shared_ptr<Node>> Region::getNodesAsList() const
    {
        std::list<std::shared_ptr<Node>> nodes;

        for (auto &id_node : this->nodes_)
        {
            nodes.emplace_back(id_node.second);
        }
        return nodes;
    } 

    void Region::update(float average_cardinality, float default_scattering2)
    {
        this->computeCardinality_();
        this->computePosition_();
        this->computeKDTree_();
        this->computeGap_();
        this->computeMesh_();
        this->computeEquivalentRadius_();
        this->computePositionDistances2_();
        this->computeMaximumRadius_();
        this->computeScattering2_(default_scattering2);
    }

    void Region::computeCardinality_()
    {
        this->cardinality_ = this->nodes_.size();
    }

    void Region::computePosition_()
    {
        /*float x_avg;
        float y_avg;
         std::cout << "card " << this->cardinality_ << "\n";
        if (this->cardinality_ <= 1)
        {
            x_avg = this->points_.at(0).at(0);
            y_avg = this->points_.at(0).at(1);
        }
        else
        {
            x_avg = (this->position_->getX() * (this->cardinality_ - 1) + this->points_.at(this->points_.size() - 2).at(0)) / this->cardinality_;
            y_avg = (this->position_->getY() * (this->cardinality_ - 1) + this->points_.at(this->points_.size() - 2).at(1)) / this->cardinality_;
        } */
        /* float x_avg = (this->position_->getX() * (this->cardinality_ - 1) + this->points_.at(this->points_.size() - 1).at(0)) / this->cardinality_;
        float y_avg = (this->position_->getY() * (this->cardinality_ - 1) + this->points_.at(this->points_.size() - 1).at(1)) / this->cardinality_; */
        float x_avg = 0;
        float y_avg = 0;

        for (auto &id_node : this->nodes_)
        {
            x_avg += id_node.second->getX();
            y_avg += id_node.second->getY();
        }

        if (this->cardinality_ > 1)
        {
            x_avg = x_avg / this->cardinality_;
            y_avg = y_avg / this->cardinality_;
        }

        this->position_.setX(x_avg);
        this->position_.setY(y_avg);
    }

    void Region::computeKDTree_()
    {
        this->nodes_tree_ = std::make_shared<KDTree>(this->points_);
    }

    void Region::computeGap_()
    {
        if (this->cardinality_ > 1)
        {
            this->gaps_.clear();

            for (auto &id_node : this->nodes_)
            {
                std::vector<double> point = id_node.second->getPositionAsVector();
                std::vector<double> nearest = this->nodes_tree_->nearest_point(point, true);
                float distance = dist(point, nearest);
                this->gaps_.insert({id_node.first, distance});
            }
        }
    }

    void Region::computeMesh_()
    {
        this->mesh_ = 0;

        if (this->cardinality_ > 1)
        {
            float total_gap = 0;
            for (auto g : this->gaps_)
            {
                total_gap += g.second;
            }

            this->mesh_ = total_gap / this->cardinality_;
        }
    }

    void Region::computeEquivalentRadius_()
    {
        this->equivalent_radius_ = this->constants_.K * this->mesh_ * sqrt(this->cardinality_);
    }

    void Region::computePositionDistances2_()
    {
        this->position_distances2_.clear();
        std::vector<double> position = this->position_.toVector();

        for (auto n : this->nodes_)
        {
            std::vector<double> point = {n.second->getX(), n.second->getY()};
            float distance = dist2(position, point);
            this->position_distances2_.insert({n.first, distance});
        }
    }

    void Region::computeMaximumRadius_()
    {
        float max_distance = 0;
        std::vector<double> position = this->position_.toVector();

        for (auto d : this->position_distances2_)
        {
            if (d.second > max_distance)
            {
                max_distance = d.second;
            }
        }

        this->maximum_radius_ = sqrt(max_distance);
    }

    void Region::computeScattering2_(float default_scattering2)
    {
        if (this->cardinality_ == 1)
        {
            this->scattering2_ = default_scattering2;
        }
        else
        {
            float total_distance = 0;
            for (auto n : this->nodes_)
            {
                total_distance += this->position_distances2_.at(n.first);
            }
            this->scattering2_ = total_distance / (this->equivalent_radius_ + 1e-7);
        }
    }

    std::string Region::prettyPrint()
{
    std::string region_str = "REGION\nregion_id: " + std::to_string(this->id_) + "\n";
    for (auto &id_node : this->nodes_)
    {
        region_str += id_node.second->prettyPrint();
    }

    for (auto &id_edge : this->edges_)
    {
        region_str += id_edge.second->prettyPrint();
    }

    return region_str;
}

}