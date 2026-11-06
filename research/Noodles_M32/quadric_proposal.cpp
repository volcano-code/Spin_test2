// Proposal only. The independent verifier bounds the actual quadric via rational CGAL meshes.
// Faces are clipped by the linear side planes of an inscribed polygonal frustum.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>
using P=std::array<double,3>;
using T=std::array<P,3>;
static std::vector<T> mesh;
extern "C" int init(const char* path) {
 FILE*f=fopen(path,"rb");if (!f) return 0;T tri;mesh.clear();
 while(fread(&tri,sizeof(T),1,f)==1) mesh.push_back(tri);fclose(f);return int(mesh.size());
}
static void integrate(const P &a,const P &b,const P &c,double x0,double y0,double sx,double sy,double *r) {
 const double area=((b[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(b[1]-a[1]))/2;
 for(int i=0;i<4;++i){
  double f=i==0?1./3:.2;
  double h=f*(a[2]+b[2]+c[2]),x=f*(a[0]+b[0]+c[0]),y=f*(a[1]+b[1]+c[1]);
  if(i){const P &p=i==1?a:i==2?b:c;h+=.4*p[2];x+=.4*p[0];y+=.4*p[1];}
  x+=x0;y+=y0;
  double w=area*(i?25./48:-27./48),h1=w*h,h2=h1*h*.5,h3=h1*h*h/3;
  r[0]+=h1;r[1]+=x*h1+sx*h2;r[2]+=y*h1+sy*h2;r[3]+=h2;
  r[4]+=x*x*h1+2*x*sx*h2+sx*sx*h3;r[5]+=y*y*h1+2*y*sy*h2+sy*sy*h3;r[6]+=h3;
  r[7]+=y*h2+sy*h3;r[8]+=x*h2+sx*h3;r[9]+=x*y*h1+(x*sy+y*sx)*h2+sx*sy*h3;
 }
}
static void cone_tet(P a,P b,P c,P o,double *r){
 P U,V,W;for(int i=0;i<3;i++){U[i]=a[i]-o[i];V[i]=b[i]-o[i];W[i]=c[i]-o[i];}
 double vol=(U[0]*(V[1]*W[2]-V[2]*W[1])+U[1]*(V[2]*W[0]-V[0]*W[2])+U[2]*(V[0]*W[1]-V[1]*W[0]))/6;
 double sum[3];r[0]+=vol;
 for(int i=0;i<3;i++){sum[i]=a[i]+b[i]+c[i]+o[i];r[1+i]+=vol*sum[i]/4;}
 constexpr int pairs[6][2]={{0,0},{1,1},{2,2},{1,2},{0,2},{0,1}};
 for(int l=0;l<6;l++){int i=pairs[l][0],j=pairs[l][1];r[4+l]+=vol/20*(sum[i]*sum[j]+a[i]*a[j]+b[i]*b[j]+c[i]*c[j]+o[i]*o[j]);}
}
extern "C" void cut(int n,double radius,double tx,double ty,double sx,double sy,double slope,double *r){
 const double pi=acos(-1), facet=cos(pi/n), x0=tx-.9*sx, y0=ty-.9*sy;
 double za=slope==0 ? 0 : .9-radius/slope;P apex={x0+sx*za,y0+sy*za,za};
 std::vector<P> normals(n);for(int k=0;k<n;k++){double th=(2*k+1)*pi/n;normals[k]={cos(th),sin(th),0};}
 std::vector<P> poly,after;poly.reserve(n+4);after.reserve(n+4);std::fill(r,r+10,0.);
 for(const auto&t:mesh){
  double minx=1e10,maxx=-1e10,miny=1e10,maxy=-1e10;bool inside=true;double rmax=0,rmin=10;
  poly.clear();for(auto const&p:t){
   double x=p[0]-x0-sx*p[2],y=p[1]-y0-sy*p[2];minx=std::min(x,minx);maxx=std::max(x,maxx);
   miny=std::min(y,miny);maxy=std::max(y,maxy);double rad=radius+slope*(p[2]-.9);if(rad<=0)return; rmin=std::min(rmin,rad);rmax=std::max(rmax,rad);inside &= x*x+y*y <pow(rad*facet,2);poly.push_back({x,y,p[2]});
  }
  if(minx>rmax||miny>rmax||maxx<-rmax||maxy<-rmax)continue;
  double dx=std::max({minx,0.,-maxx}),dy=std::max({miny,0.,-maxy});if(dx*dx+dy*dy>rmax*rmax)continue;
  if(!inside)for(const P&g:normals){
   // Skip planes separated from the projected face AABB.
   double top=g[0]*(g[0]>0?maxx:minx)+g[1]*(g[1]>0?maxy:miny);
   if(top<rmin*facet)continue;
   after.clear();P a=poly.back();double fa=g[0]*a[0]+g[1]*a[1]-(radius+slope*(a[2]-.9))*facet;
   for(const P&b:poly){double fb=g[0]*b[0]+g[1]*b[1]-(radius+slope*(b[2]-.9))*facet;
    if((fa>0)!=(fb>0)){double l=fa/(fa-fb);after.push_back({a[0]+l*(b[0]-a[0]),a[1]+l*(b[1]-a[1]),a[2]+l*(b[2]-a[2])});}
    if(fb<=0)after.push_back(b);a=b;fa=fb;
   }poly.swap(after);if(poly.size()<3)break;
  }
  if (slope == 0) {
   // In sheared coordinates the cylinder is vertical. Integrate each clipped
   // face's vertical flux with the cubic-exact four-point triangle rule.
   for(size_t i=2;i<poly.size();++i)
    integrate(poly[0],poly[i-1],poly[i],x0,y0,sx,sy,r);
  } else {
   // The cone's side faces have zero tetrahedral volume about its apex.
   for(P &p:poly){p[0]+=x0+sx*p[2];p[1]+=y0+sy*p[2];}
   for(size_t i=2;i<poly.size();++i)
    cone_tet(poly[0],poly[i-1],poly[i],apex,r);
  }
 }
}
