#pragma once

#ifndef POSITION_HPP
#define POSITION_HPP

#include <iostream>
#include <memory>
#include <vector>

namespace graph_clustering
{

    /**
     * Class that represents a 2D position.
    */
    class Position
    {

    public:
        typedef std::shared_ptr<Position> SharedPtr;
        typedef std::shared_ptr<const Position> ConstSharedPtr;

        typedef std::unique_ptr<Position> UniquePtr;
        typedef std::unique_ptr<const Position> ConstUniquePtr;

        typedef std::weak_ptr<Position> WeakPtr;
        typedef std::weak_ptr<const Position> ConstWeakPtr;

        Position(float x, float y);

        inline float getX() const { return this->x_; }
        inline float getY() const { return this->y_; }

        inline void setX(float x) { this->x_ = x; }
        inline void setY(float y) { this->y_ = y; }

        bool operator==(const Position &other) const
        {
            return (this->x_ == other.getX() && this->y_ == other.getY());
        }

        std::vector<double> toVector() const;

        std::string prettyPrint() const;

    private:
        float x_;
        float y_;
    };
}

#endif