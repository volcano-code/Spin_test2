// Pinned V7 ellipsoid map: fixed subdivision-two normals, represented algebraic source and its native differential.
#include "spin/QuadricCoefficients.h"
#include "spin/QuadricInverseGaussMap.h"
#include "detail/FixedNormalSphere.h"
#include <Eigen/Geometry>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    using namespace spin;
    QuadricCoefficients10::Vector input;
    for (int k=0;k<10;++k) if (!(std::cin>>input(k))) return 1;
    const auto q=QuadricCoefficients10::fromUnnormalized(input);
    const QuadricInverseGaussMap map(q);
    const auto sphere=detail::fixedNormalSphere(2);
    std::vector<Eigen::Vector3d> points;
    std::cout<<std::setprecision(17)<<sphere.vertices.size()<<" "<<sphere.faces.size()<<"\n";
    for(const auto& n:sphere.vertices) {
        const auto x=map.algebraicPoint(n);
        const auto d=map.algebraicCoefficientDifferential(n);
        if(!x.valid()||!d.coefficientJacobian) throw std::runtime_error("unavailable original finite map/differential");
        points.push_back(*x.point);
        std::cout<<n.transpose()<<" "<<x.point->transpose();
        for(int i=0;i<3;++i) for(int k=0;k<10;++k) std::cout<<" "<<(*d.coefficientJacobian)(i,k);
        std::cout<<"\n";
    }
    // Intersection uses the outward boundary of the finite ellipsoid.
    for(auto f:sphere.faces) {
        const auto normal=(points[f[1]]-points[f[0]]).cross(points[f[2]]-points[f[0]]);
        const auto center=(points[f[0]]+points[f[1]]+points[f[2]])/3;
        if(normal.dot(q.gradient(center))>0.) std::swap(f[1],f[2]);
        std::cout<<f[0]<<" "<<f[1]<<" "<<f[2]<<"\n";
    }
}
