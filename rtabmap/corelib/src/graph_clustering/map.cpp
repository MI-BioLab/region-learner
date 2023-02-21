#include "rtabmap/core/graph_clustering/map.hpp"

namespace graph_clustering
{

    Map::Map(const Constants &constants,
             float radius_upper_bound,
             int desired_average_cardinality,
             float mesh_shape_factor,
             unsigned long initial_region,
             float alpha_vis,
             float alpha_equ,
             float alpha_pre,
             float alpha_hom,
             float alpha_coh,
             float alpha_reg) : constants_(constants),
                                radius_upper_bound_(radius_upper_bound),
                                desired_average_cardinality_(desired_average_cardinality),
                                mesh_shape_factor_(mesh_shape_factor),
                                region_id_factory_(initial_region),
                                average_cardinality_(0),
                                total_mesh_(0),
                                default_scattering_(0),
                                threshold_(0),
                                /* scattering_1_constants_((desired_average_cardinality + 1) * sqrt(desired_average_cardinality + 1) -
                                                        desired_average_cardinality * sqrt(desired_average_cardinality) - 1), */
                                scattering_1_constants_(desired_average_cardinality * sqrt(desired_average_cardinality)),
                                alpha_vis_(alpha_vis),
                                alpha_equ_(alpha_equ),
                                alpha_pre_(alpha_pre),
                                alpha_hom_(alpha_hom),
                                alpha_coh_(alpha_coh),
                                alpha_reg_(alpha_reg),
                                base_threshold_(4),
                                total_scattering_(0)
    {
    }

    Map::~Map()
    {
        this->nodes_tree_.reset();
    }

    void Map::addNode(const Node::SharedPtr &node)
    {
        if (!this->nodes_.count(node->getId()))
        {
            unsigned long region_id = region_id_factory_++;
            node->setRegion(region_id);
            this->nodes_.insert({node->getId(), node});
            this->points_.emplace_back(node->getPositionAsVector());
            this->nodes_tree_ = std::make_unique<KDTree>(this->points_);

            Region::SharedPtr region;
            if (!this->regions_.count(region_id))
            {
                region = std::make_shared<Region>(region_id, this->constants_);
                region->addNode(node);
                this->regions_.insert({region_id, region});
            }
            else
            {
                region = this->regions_.at(region_id);
                region->addNode(node);
            }

            this->computeAverageCardinality_();
            this->computeGaps_();
            this->computeTotalMesh_();
            this->computeDefaultScattering_();
            this->computeThreshold_();

            region->update(this->average_cardinality_, this->default_scattering_);
        }
    }

    void Map::SBA(const Node::SharedPtr &node, const Edge::SharedPtr &edge)
    {

        EdgePair edge_key = edge->toEdgePair();
        if (!this->edges_.count(edge_key))
        {
            this->edges_.insert({edge_key, edge});
            Node::SharedPtr from_node = this->nodes_.at(edge_key.from_id);

            if (!this->nodes_.count(node->getId()))
            {
                this->nodes_.insert({node->getId(), node});
                this->points_.emplace_back(node->getPositionAsVector());
                this->nodes_tree_ = std::make_shared<KDTree>(this->points_);

                this->computeAverageCardinality_();
                this->computeGaps_();
                this->computeTotalMesh_();
                this->computeDefaultScattering_();
                this->computeThreshold_();

                Region::SharedPtr region = this->regions_.at(from_node->getRegion());
                Region::SharedPtr region_updated = std::make_shared<Region>(*region);
                if (!region_updated->addNodeAndEdge(node, edge))
                {
                    return;
                }

                region_updated->update(this->average_cardinality_, this->default_scattering_);

                if ((region_updated->getScattering2() - region->getScattering2() <= this->threshold_ + this->default_scattering_) &&
                    (region_updated->getDistanceFromPositionForNode(node) <= this->radius_upper_bound_))
                { 

                    node->setRegion(from_node->getRegion());
                    this->regions_.at(region->getId()) = region_updated;
                }
                else
                {
                    Region::SharedPtr new_region = std::make_shared<Region>(region_id_factory_++, this->constants_);

                    node->setRegion(new_region->getId());
                    new_region->addNodeAndEdge(node, edge);
                    new_region->update(this->average_cardinality_, this->default_scattering_);
                    region->addEdge(edge);
                    this->regions_.insert({new_region->getId(), new_region});
                }
            }
        }
}

bool Map::contains(std::vector<double> node_pos){
    return this->nodes_tree_->contains(node_pos);
}

void Map::computeAverageCardinality_()
{
    this->average_cardinality_ = this->nodes_.size() / this->regions_.size();
}

    void Map::computeGaps_()
    {
        if (this->nodes_.size() > 1)
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

void Map::computeTotalMesh_()
{

    float total_gap = 0;
    for (auto g : this->gaps_)
    {
        total_gap += g.second;
    }
    this->total_mesh_ = total_gap / this->nodes_.size();
}

    void Map::computeDefaultScattering_()
    {
        this->default_scattering_ = (this->total_mesh_ * this->mesh_shape_factor_ / this->constants_.K_2_PI);
    }

void Map::computeThreshold_()
{
    this->threshold_ = this->default_scattering_ * this->scattering_1_constants_ / 2;
}

std::string Map::prettyPrint()
{
    std::string map_str = "MAP\n";
    for (auto &id_region : this->regions_)
    {
        map_str += id_region.second->prettyPrint();
    }
    return map_str;
}
}
