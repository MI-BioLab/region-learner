#pragma once

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <iostream>
#include <memory>
#include <cmath>

namespace graph_clustering
{

    /**
     * Constants of the problem.
     */
    struct Constants
    {
        typedef std::shared_ptr<Constants> SharedPtr;
        typedef std::shared_ptr<const Constants> ConstSharedPtr;

        typedef std::unique_ptr<Constants> UniquePtr;
        typedef std::unique_ptr<const Constants> ConstUniquePtr;

        typedef std::weak_ptr<Constants> WeakPtr;
        typedef std::weak_ptr<const Constants> ConstWeakPtr;

        // 3^(1/4) * (2π)^(-1/2)
        const float K = 0.525;

        // K * 2π
        const float K_2_PI = K * 2 * M_PI;
    };
}

#endif