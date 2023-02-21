#include "rtabmap/core/graph_clustering/position.hpp"

namespace graph_clustering
{
    Position::Position(float x, float y) : x_(x), y_(y) {}

    std::vector<double> Position::toVector() const
    {
        return {x_, y_};
    }

    std::string Position::prettyPrint() const
    {
        return "POSITION:\nx: " + std::to_string(this->x_) + "\ny: " + std::to_string(this->y_) + "\n";
    }
}
