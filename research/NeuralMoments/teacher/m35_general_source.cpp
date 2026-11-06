// Stock V7 source motion, closed-boundary tessellation, and side selector.
// The supplied design vertices only establish the truncation bounds.
#include "spin/QuadricMomentEvaluator.h"
#include "spin/QuadricSourceMotion.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("target-frame bounding vertices required");
        std::ifstream input(argv[1]);
        int count = 0;
        if (!(input >> count) || count <= 0) throw std::runtime_error("invalid bounding sample size");
        Eigen::MatrixXd design(count, 3);
        for (int i = 0; i < count; ++i)
            for (int j = 0; j < 3; ++j)
                if (!(input >> design(i,j))) throw std::runtime_error("invalid bounding vertex");
        spin::QuadricCoefficients10::Vector raw;
        for (int i = 0; i < 10; ++i)
            if (!(std::cin >> raw(i))) throw std::runtime_error("invalid coefficient");
        const auto q = spin::QuadricCoefficients10::fromUnnormalized(raw);
        spin::QuadricTriangulationSettings settings;
        settings.icosphereSubdivisions = settings.maximumIcosphereSubdivisions = 2;
        settings.algebraicFiniteMap = true;
        settings.retainFixedNormalProvenance = true;
        const auto source = spin::evaluateQuadricSourceMotion(q, design, settings);
        if (!source.defined) throw std::runtime_error(source.diagnostic);
        const auto& tri = source.triangulation;
        const auto& mesh = tri.closedBoundary;
        const bool inside = spin::quadricSolidInteriorIsPositiveRegion(q, tri);
        std::cout << std::setprecision(17) << mesh.vertices.rows() << ' ' << mesh.faces.rows()
                  << ' ' << inside << ' ' << tri.truncationRadius << ' ' << spin::quadricSurfaceTypeToken(tri.type)
                  << ' ' << tri.capFaceCount << ' ' << source.scientificHash << ' ' << tri.hash
                  << ' ' << q.values().transpose() << '\n';
        for (int i = 0; i < mesh.vertices.rows(); ++i) {
            std::cout << mesh.vertices.row(i);
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 10; ++k) std::cout << ' ' << source.velocity.at(i)(j,k);
            std::cout << '\n';
        }
        for (int i = 0; i < mesh.faces.rows(); ++i)
            std::cout << mesh.faces.row(i) << ' ' << tri.faceIsCap[i] << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
