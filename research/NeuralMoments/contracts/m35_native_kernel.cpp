// Numerical diagnostic for the stock spherical Newton solver. The external
// caller must separately establish ownership of the geometry and its map.
#include "spin/QuadricHollowingSolver.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {
double number(std::istream& in) {
    double x;
    if (!(in >> x) || !std::isfinite(x)) throw std::runtime_error("invalid_number");
    return x;
}
void optional(const std::optional<double>& x) {
    if (x) std::cout << *x; else std::cout << "null";
}
spin::SpinMomentMapEvaluation evaluate(const spin::QuadricCoefficients10& q) {
    std::cout << "QUERY";
    for (double x : q.values()) std::cout << ' ' << x;
    std::cout << std::endl;
    std::string state, error;
    if (!(std::cin >> state)) throw std::runtime_error("missing_evaluator");
    if (state == "FAIL") {
        if (!(std::cin >> error)) throw std::runtime_error("missing_diagnostic");
        throw std::runtime_error(error);
    }
    spin::SpinMomentMapEvaluation evaluation;
    if (state != "MAP" || !(std::cin >> evaluation.geometryHash >> evaluation.topologyHash
            >> evaluation.evaluationMapIdentity >> evaluation.booleanBackendIdentity >> evaluation.derivativeIdentity))
        throw std::runtime_error("invalid_map_header");
    for (int k=0;k<10;++k) evaluation.rawMoments[k] = number(std::cin);
    evaluation.variableRawMoments.emplace();
    for (int k=0;k<10;++k) (*evaluation.variableRawMoments)[k] = number(std::cin);
    for (int i=0;i<10;++i)
        for (int j=0;j<10;++j) evaluation.momentJacobian(i,j) = number(std::cin);
    evaluation.derivativeDefined = true;
    return evaluation;
}
void observe(const spin::SpinSolverEvent& e) {
    std::cout << "EVENT " << e.type << ' ' << e.restart << ' ' << e.iteration
              << ' ' << e.evaluationIndex << ' ' << e.outcome << ' ' << e.reason << ' ';
    optional(e.step);std::cout << ' ';optional(e.rawEquationResidual);std::cout << ' ';
    optional(e.physicalRelativeResidual);std::cout << ' ';optional(e.objective);std::cout << ' ';optional(e.multiplierMu);
    std::cout << std::endl;
}
}
int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(17);
        spin::SpinSolverSettings s;
        s.equationProfile=spin::SpinEquationProfile::DualMonomialRawV1;
        if (argc==2 && std::string(argv[1])=="--starts") {
            for(const auto& q : spin::deterministicInitialQuadrics(spin::SpinMode::SpinningTop,s))
                std::cout << q.values().transpose() << '\n';
            return 0;
        }
        if (argc!=3) throw std::invalid_argument("expected max_iterations max_evaluations");
        s.maximumIterations=std::stoi(argv[1]);
        s.maximumFunctionEvaluationsPerRestart=std::stoi(argv[2]);
        s.maximumRestarts=1;
        spin::QuadricCoefficients10::Vector first;
        for (int k=0;k<10;++k) first[k]=number(std::cin);
        auto out=spin::solveSpinMomentSystem(spin::SpinMomentProblem::spinningTop(1.0),s,evaluate,
                                              {spin::QuadricCoefficients10::fromNormalized(first)},observe);
        std::cout << "TERMINAL " << spin::spinSolverStatusToken(out.status) << ' '
                  << out.functionEvaluations << ' ' << out.restartCount << ' ' << out.trace.size() << ' '
                  << out.eventHash << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
