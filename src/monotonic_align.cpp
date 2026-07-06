#include "monotonic_align.hpp"

#include <algorithm>
#include <stdexcept>

namespace monotonic_align
{
    namespace
    {
        constexpr float kMaxNegativeValue = -1.0e9f;

        void maximumPathEach(
            float* path,
            float* value,
            int64_t tY,
            int64_t tX,
            int64_t yStride,
            int64_t xStride)
        {
            int64_t index = tX - 1;

            for (int64_t y = 0; y < tY; ++y)
            {
                const int64_t xStart = std::max<int64_t>(0, tX + y - tY);
                const int64_t xEnd = std::min<int64_t>(tX, y + 1);
                for (int64_t x = xStart; x < xEnd; ++x)
                {
                    const float vCur = x == y
                        ? kMaxNegativeValue
                        : value[(y - 1) * yStride + x * xStride];

                    float vPrev = kMaxNegativeValue;
                    if (x == 0)
                    {
                        vPrev = y == 0 ? 0.0f : kMaxNegativeValue;
                    }
                    else
                    {
                        vPrev = value[(y - 1) * yStride + (x - 1) * xStride];
                    }

                    value[y * yStride + x * xStride] += std::max(vPrev, vCur);
                }
            }

            for (int64_t y = tY - 1; y >= 0; --y)
            {
                path[y * yStride + index * xStride] = 1.0f;
                if (index != 0 &&
                    (index == y ||
                     value[(y - 1) * yStride + index * xStride] <
                         value[(y - 1) * yStride + (index - 1) * xStride]))
                {
                    --index;
                }
            }
        }
    }

    torch::Tensor maximumPath(
        const torch::Tensor& negCent,
        const torch::Tensor& mask)
    {
        if (negCent.dim() != 3 || mask.dim() != 3)
        {
            throw std::invalid_argument("maximumPath expects negCent and mask with shape [batch, t_t, t_s].");
        }
        if (negCent.sizes() != mask.sizes())
        {
            throw std::invalid_argument("maximumPath expects negCent and mask to have identical shapes.");
        }

        auto valueCpu = negCent.to(torch::kCPU, torch::kFloat32).contiguous();
        auto maskCpu = mask.to(torch::kCPU, torch::kFloat32).contiguous();
        auto pathCpu = torch::zeros_like(valueCpu);

        const int64_t batch = valueCpu.size(0);
        const int64_t tTMax = valueCpu.size(1);
        const int64_t tSMax = valueCpu.size(2);
        auto valueAcc = valueCpu.accessor<float, 3>();
        auto maskAcc = maskCpu.accessor<float, 3>();
        auto pathAcc = pathCpu.accessor<float, 3>();

        for (int64_t b = 0; b < batch; ++b)
        {
            int64_t tT = 0;
            int64_t tS = 0;
            for (int64_t y = 0; y < tTMax; ++y)
            {
                if (maskAcc[b][y][0] > 0.0f)
                {
                    ++tT;
                }
            }
            for (int64_t x = 0; x < tSMax; ++x)
            {
                if (maskAcc[b][0][x] > 0.0f)
                {
                    ++tS;
                }
            }

            if (tT <= 0 || tS <= 0)
            {
                continue;
            }

            maximumPathEach(
                &pathAcc[b][0][0],
                &valueAcc[b][0][0],
                tT,
                tS,
                tSMax,
                1);
        }

        return pathCpu.to(negCent.options());
    }
}
