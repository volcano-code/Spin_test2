// Signed rational integration on a transverse plane and three paired finite differences.
#include <gmpxx.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using Q=mpq_class;
using P=std::array<Q,3>;
using Face=std::array<int,3>;
constexpr int bits=160;
constexpr int powers[10][3]={{0,0,0},{1,0,0},{0,1,0},{0,0,1},{2,0,0},{0,2,0},{0,0,2},{0,1,1},{1,0,1},{1,1,0}};
constexpr int pairs[6][2]={{0,0},{1,1},{2,2},{1,2},{0,2},{0,1}};
constexpr int factorial_small[]={1,1,2,6,24,120,720};
int choose(int n,int k){return factorial_small[n]/(factorial_small[k]*factorial_small[n-k]);}
Q powq(const Q& a,int e){Q v=1;while(e--)v*=a;return v;}

// Each term is computed exactly and rounded outwards before summation.
struct Interval {
    mpz_class lo=0, hi=0;
    void add(const Q& q) {
        mpz_class n=q.get_num()<<bits, d=q.get_den(), low, rem;
        mpz_fdiv_qr(low.get_mpz_t(),rem.get_mpz_t(),n.get_mpz_t(),d.get_mpz_t());
        lo+=low;hi+=low+(rem!=0);
    }
};
using Ten=std::array<Interval,10>;
Q triple(const P& a,const P& b,const P& c){
    return a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]);
}
void tetra(Ten& dst,const P& a,const P& b,const P& c){
    Q volume=triple(a,b,c)/6;
    if(volume==0)return;
    P s;dst[0].add(volume);
    for(int k=0;k<3;k++){s[k]=a[k]+b[k]+c[k];dst[k+1].add(volume*s[k]/4);}
    for(int k=0;k<6;k++){
        int i=pairs[k][0],j=pairs[k][1];
        dst[k+4].add(volume*(s[i]*s[j]+a[i]*a[j]+b[i]*b[j]+c[i]*c[j])/20);
    }
}
using Section=std::array<std::array<Interval,10>,10>;
void section(Section& dst,const P& u,const P& v){
    // Signed triangle in xy with corners (0,0), u_xy, v_xy at z=3/4.
    Q det=u[0]*v[1]-u[1]*v[0];
    std::array<std::array<Q,5>,5> xy{};
    for(int a=0;a<=4;a++)for(int b=0;b<=4-a;b++){
        Q val=0;
        for(int i=0;i<=a;i++)for(int j=0;j<=b;j++){
            Q term=powq(u[0],i)*powq(v[0],a-i)*powq(u[1],j)*powq(v[1],b-j);
            val+=term*choose(a,i)*choose(b,j)*factorial_small[i+j]*factorial_small[a+b-i-j];
        }
        xy[a][b]=det*val/factorial_small[a+b+2];
    }
    for(int i=0;i<10;i++)for(int j=0;j<10;j++){
        int a=powers[i][0]+powers[j][0],b=powers[i][1]+powers[j][1],z=powers[i][2]+powers[j][2];
        // dr/dc_j: the cut moves with normal speed phi_j/|grad beta|; native mixed terms carry 2.
        dst[i][j].add(xy[a][b]*powq(Q(3,4),z)*Q(5,4)*(j>=7?2:1));
    }
}
struct Trial { int col; Q delta; Ten sums; };
struct Vertex { P x; bool cut; };
int main(int argc,char** argv) try {
    if(argc!=2)throw std::runtime_error("one rational mesh operand required");
    std::ifstream f(argv[1]);int nv,nf; if(!(f>>nv>>nf)||nv<4||nf<4)throw std::runtime_error("bad header");
    std::vector<P> verts(nv);
    std::vector<Trial> trials={{0,0,{}}};
    for(int col:{0,1,2})for(int e:{20,21})for(int sign:{-1,1})trials.push_back({col,Q(sign,1<<e),{}});
    std::vector<std::vector<Q>> vals(nv,std::vector<Q>(trials.size()));
    Q min_margin=0;
    for(int i=0;i<nv;i++){
        P& p=verts[i]; for(Q& x:p){if(!(f>>x))throw std::runtime_error("bad vertex");x.canonicalize();}
        Q val=Q(4,5)*p[2]-Q(3,5); if(val==0)throw std::runtime_error("source vertex on plane");
        Q margin=abs(val);if(!i||margin<min_margin)min_margin=margin;
        for(size_t t=0;t<trials.size();t++){
            const auto& tr=trials[t];vals[i][t]=val+tr.delta*(tr.col?p[tr.col-1]:Q(1));
            if(sgn(vals[i][t])!=sgn(val))throw std::runtime_error("source vertex side changed");
        }
    }
    int full=0,empty=0,crossings=0;Ten common;Section jac;
    for(int n=0;n<nf;n++){
        Face fi;for(int& i:fi)if(!(f>>i)||i<0||i>=nv)throw std::runtime_error("bad face");
        const int count=(vals[fi[0]][0]>0)+(vals[fi[1]][0]>0)+(vals[fi[2]][0]>0);
        if(count==0){empty++;continue;}
        if(count==3){full++;tetra(common,verts[fi[0]],verts[fi[1]],verts[fi[2]]);continue;}
        crossings++;
        for(size_t t=0;t<trials.size();t++){
            std::array<Vertex,4> poly;int m=0;
            for(int k=0;k<3;k++){
                int a=fi[k],b=fi[(k+1)%3];const Q& va=vals[a][t];const Q& vb=vals[b][t];
                if(va>0)poly[m++]={verts[a],false};
                if(sgn(va)!=sgn(vb)){
                    Q ratio=va/(va-vb);P x;
                    for(int j=0;j<3;j++)x[j]=verts[a][j]+ratio*(verts[b][j]-verts[a][j]);
                    poly[m++]={x,true};
                }
            }
            if(m<3||m>4)throw std::runtime_error("invalid clipped polygon");
            for(int k=1;k+1<m;k++)tetra(trials[t].sums,poly[0].x,poly[k].x,poly[k+1].x);
            int caps=0;
            for(int k=0;k<m;k++){
                const auto& a=poly[k];const auto& b=poly[(k+1)%m];if(!a.cut||!b.cut)continue;
                P anchor={0,0,(Q(3,5)-(trials[t].col==0?trials[t].delta:Q(0)))/Q(4,5)};
                // For z>=h the cap is oriented down, so close its boundary by reversing the cut edge.
                tetra(trials[t].sums,anchor,b.x,a.x);if(t==0)section(jac,a.x,b.x);
                caps++;
            }
            if(caps!=1)throw std::runtime_error("wrong cut boundary count");
        }
    }
    std::string extra;if(f>>extra)throw std::runtime_error("trailing input");
    std::cout<<"header "<<bits<<" "<<nv<<" "<<nf<<" "<<full<<" "<<empty<<" "<<crossings<<" "<<min_margin<<"\n";
    for(const auto& t:trials){
        std::cout<<"r "<<t.col<<" "<<t.delta;
        for(int i=0;i<10;i++)std::cout<<" "<<common[i].lo+t.sums[i].lo<<" "<<common[i].hi+t.sums[i].hi;
        std::cout<<"\n";
    }
    for(int i=0;i<10;i++){std::cout<<"j "<<i;for(const auto& v:jac[i])std::cout<<" "<<v.lo<<" "<<v.hi;std::cout<<"\n";}
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
