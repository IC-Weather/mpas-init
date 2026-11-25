#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "atlas/library/Library.h"
#include "atlas/util/KDTree.h"
#include "atlas/util/Point.h"

namespace {

template <typename T>
void read(T* data, std::size_t count) {
    if (!std::cin.read(reinterpret_cast<char*>(data), count * sizeof(T))) {
        throw std::runtime_error("truncated candidate request");
    }
}

template <typename T>
void write(const T* data, std::size_t count) {
    if (!std::cout.write(reinterpret_cast<const char*>(data), count * sizeof(T))) {
        throw std::runtime_error("candidate response write failed");
    }
}

std::uint64_t read_count() {
    std::uint64_t count;
    read(&count, 1);
    return count;
}

void serve() {
    const std::uint32_t endian = 1;
    if (*reinterpret_cast<const char*>(&endian) != 1 || sizeof(double) != 8 ||
        !std::numeric_limits<double>::is_iec559) {
        throw std::runtime_error("protocol requires little-endian IEEE754 binary64");
    }
    std::array<char, 8> magic;
    read(magic.data(), magic.size());
    if (std::string(magic.data(), magic.size()) != "MPASAT01") {
        throw std::runtime_error("unknown candidate protocol");
    }
    const auto count = read_count();
    if (count == 0 || count > static_cast<std::uint64_t>(std::numeric_limits<atlas::idx_t>::max()) ||
        count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid cell count");
    }
    std::vector<std::array<double, 3>> centers(count);
    double max_coordinate = 0.;
    atlas::util::KDTree<std::uint32_t> tree;
    tree.reserve(static_cast<atlas::idx_t>(count));
    for (std::uint64_t cell = 0; cell < count; ++cell) {
        read(centers[cell].data(), 3);
        for (const double value : centers[cell]) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("nonfinite cell coordinate");
            }
            max_coordinate = std::max(max_coordinate, std::abs(value));
        }
        tree.insert(atlas::PointXYZ(centers[cell].data()), static_cast<std::uint32_t>(cell));
    }
    tree.build();
    std::vector<std::uint32_t> order(count), rank(count);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return centers[a] < centers[b]; });
    for (std::uint64_t cell = 0; cell < count; ++cell) {
        rank[order[cell]] = static_cast<std::uint32_t>(cell);
    }
    write("MPASOK01", 8);
    std::cout.flush();

    while (const auto queries = read_count()) {
        if (queries > 65536) {
            throw std::runtime_error("candidate query batch exceeds 65536 blocks");
        }
        std::vector<std::array<double, 4>> blocks(queries);
        // Consume the request before writing a response, so bounded pipe buffers cannot deadlock.
        for (auto& block : blocks) {
            read(block.data(), 4);
        }
        std::vector<std::uint64_t> offsets{0};
        std::vector<std::uint32_t> ids;
        for (const auto& block : blocks) {
            for (const auto value : block) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("nonfinite query geometry");
                }
            }
            if (block[3] < 0.) {
                throw std::runtime_error("negative enclosing radius");
            }
            const atlas::PointXYZ center(block.data());
            double bound = tree.closestPoint(center).distance() + 2. * block[3];
            bound += 32. * std::numeric_limits<double>::epsilon() * (bound + max_coordinate);
            bound = std::nextafter(bound, std::numeric_limits<double>::infinity());
            if (!std::isfinite(bound)) {
                throw std::runtime_error("candidate radius overflow");
            }
            const auto found = tree.closestPointsWithinRadius(center, bound);
            if (found.empty() || ids.size() + found.size() > 16777216) {
                throw std::runtime_error("empty or oversized candidate response");
            }
            const auto begin = ids.size();
            for (const auto& candidate : found) {
                ids.push_back(candidate.payload());
            }
            std::sort(ids.begin() + begin, ids.end(), [&](auto a, auto b) { return rank[a] < rank[b]; });
            offsets.push_back(ids.size());
        }
        const std::array<std::uint64_t, 2> header{queries, ids.size()};
        write(header.data(), header.size());
        write(offsets.data(), offsets.size());
        write(ids.data(), ids.size());
        std::cout.flush();
    }
}

}  // namespace

int main(int argc, char** argv) {
    const auto& library = atlas::Library::instance();
    if (library.version() != MPAS_ATLAS_VERSION || library.gitsha1(40) != MPAS_ATLAS_REVISION) {
        std::cerr << "mpas-atlas-candidates: loaded Atlas differs from its build identity\n";
        return 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "mpas-atlas-candidates protocol=1 atlas=" << MPAS_ATLAS_VERSION
                  << " revision=" << MPAS_ATLAS_REVISION << '\n';
        return 0;
    }
    if (argc != 1) {
        std::cerr << "usage: mpas-atlas-candidates [--version]\n";
        return 2;
    }
    try {
        atlas::initialize(argc, argv);
        serve();
        atlas::finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mpas-atlas-candidates: " << error.what() << '\n';
        return 1;
    }
}
