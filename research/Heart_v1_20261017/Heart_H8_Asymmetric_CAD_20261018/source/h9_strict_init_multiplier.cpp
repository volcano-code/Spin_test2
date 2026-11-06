// Offline replay of the initial multiplier projection in QuadricHollowingSolver::evaluateState.
#include "spin/DualMonomialCoordinates.h"
#include "spin/SpinMomentProblem.h"
#include <Eigen/QR>
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    spin::SpinMomentProblem::Vector10 raw, coefficients;
    for (double& value : coefficients) if (!(std::cin >> value)) return 2;
    for (double& value : raw) if (!(std::cin >> value)) return 2;
    const auto quadric = spin::QuadricCoefficients10::fromUnnormalized(coefficients);
    const auto dual = spin::DualMonomialCoordinates10::fromGeometry(quadric);
    const auto objective = spin::SpinMomentProblem::spinningTop(0., spin::SpinningTopObjective::InertiaAndLowering);
    const auto state = objective.evaluate(raw);
    Eigen::MatrixXd projection(10, 5);
    projection.leftCols(4) = -state.constraintJacobian.transpose();
    projection.col(4) = dual.values();
    const auto decomposition = projection.completeOrthogonalDecomposition();
    if (decomposition.rank() != 5) return 3;
    const Eigen::VectorXd multipliers = decomposition.solve(-state.objectiveGradient);
    const double norm = std::hypot((state.objectiveGradient + projection*multipliers).norm(), state.constraints.norm());
    std::cout << std::setprecision(17) << multipliers[4] << ' ' << norm << ' ' << state.objective;
    for (const double component : state.constraints) std::cout << ' ' << component;
    std::cout << '\n';
}
