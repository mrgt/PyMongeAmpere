// PyMongeAmpere
// Copyright (C) 2014 Quentin Mérigot, CNRS
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef double FT;
typedef Eigen::SparseMatrix<FT> SparseMatrix;
typedef Eigen::SparseVector<FT> SparseVector;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::Vector2d Vector2d;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::MatrixXi MatrixXi;


#include <CGAL/Triangulation_2.h>
#include <CGAL/Triangulation_incremental_builder_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Random.h>

#include <MA/functions.hpp>
#include <MA/kantorovich.hpp>
#include <MA/lloyd.hpp>
#include <MA/rasterization.hpp>

// Triangulation
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::FT FT;
typedef CGAL::Triangulation_vertex_base_with_info_2<size_t, K> Vb;
typedef CGAL::Triangulation_data_structure_2<Vb> Tds;
typedef CGAL::Triangulation_2<K,Tds> T;

// Regular triangulation
typedef CGAL::Regular_triangulation_vertex_base_2<K> RTVbase;
typedef CGAL::Triangulation_vertex_base_with_info_2
<size_t, K, RTVbase> RTVb;
typedef CGAL::Regular_triangulation_face_base_2<K> RTCb;
typedef CGAL::Triangulation_data_structure_2<RTVb,RTCb> RTTds;
typedef CGAL::Regular_triangulation_2<K, RTTds> RT;


typedef RT::Weighted_point Weighted_point;
typedef typename CGAL::Segment_2<K> Segment;
typedef CGAL::Point_2<K> Point;
typedef CGAL::Vector_2<K> Vector;
typedef CGAL::Line_2<K> Line;
typedef CGAL::Ray_2<K> Ray;
typedef CGAL::Segment_2<K> Segment;


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void
python_to_delaunay_2(const MatrixXd& X,
                     const VectorXd& w,
		     RT &dt)
{
  size_t N = X.rows();
  assert(X.cols() == 2);
  assert(w.cols() == 1);
  assert(w.rows() == N);


  // insert points with indices in the regular triangulation
  std::vector<std::pair<Weighted_point,size_t> > Xw(N);
  for (size_t i = 0; i < N; ++i)
    {
      Xw[i] = std::make_pair(Weighted_point(Point(X(i,0), X(i,1)),
					    w(i)), i);
    }
  dt.clear();
  dt.insert(Xw.begin(), Xw.end());
  dt.infinite_vertex()->info() = -1;
}

void restricted_laguerre_edges (const T &t,
				const RT &rt,
				std::vector<Segment> &edges)
{
  typedef RT::Vertex_handle Vertex_handle_RT;
  typedef MA::Voronoi_intersection_traits<K> Traits;
  typedef typename MA::Tri_intersector<T,RT,Traits> Tri_isector;
  typedef typename Tri_isector::Pgon Pgon;
  Tri_isector isector;

  MA::voronoi_triangulation_intersection_raw
    (t,rt, [&] (const Pgon &pgon, typename T::Face_handle f, Vertex_handle_RT v)
     {
       for (size_t i = 0; i < pgon.size(); ++i)
	 {
	   size_t iprev = (i+pgon.size()-1)%pgon.size();
	   size_t inext = (i+1)%pgon.size();
	   Point p = isector.vertex_to_point(pgon[i], pgon[iprev]);
	   Point q = isector.vertex_to_point(pgon[i], pgon[inext]);
	   if (pgon[i].type != Tri_isector::EDGE_DT)
	     continue;
	   edges.push_back(Segment(p,q));
	 }
       });
  }

inline double rand01(CGAL::Random& g)
{ return g.uniform_01<float>();}

class Density_2
{
public:
  T _t;
  typedef MA::Linear_function<K> Function;
  std::map<T::Face_handle, Function> _functions;
  CGAL::Random gen;

  struct Triangle
  {
  public:
    Point a, b, c;
    Triangle (const Point &aa, const Point &bb, const Point &cc) :
      a(aa), b(bb), c(cc) {}

    Point rand(CGAL::Random& g) const
    {
      double r1 = sqrt(rand01(g)), r2 = rand01(g);
      return  CGAL::ORIGIN + ((1 - r1) * (a-CGAL::ORIGIN)  +
			      (r1 * (1 - r2)) * (b-CGAL::ORIGIN) +
			      (r1 * r2) * (c-CGAL::ORIGIN));
    }
  };

  void
  compute_cum_masses(std::vector<Triangle> &triangles,
		     std::vector<double> &cumareas)
  {
    double ta = 0.0;
    cumareas.clear();
    triangles.clear();
    for (T::Finite_faces_iterator it = _t.finite_faces_begin ();
	 it != _t.finite_faces_end(); ++it)
      {
	triangles.push_back(Triangle(it->vertex(0)->point(),
				     it->vertex(1)->point(),
				     it->vertex(2)->point()));
	ta += MA::integrate_centroid<double>(it->vertex(0)->point(),
					     it->vertex(1)->point(),
					     it->vertex(2)->point(),
					     _functions[it]);
	cumareas.push_back(ta);
      }
  }

public:
  Density_2(const MatrixXd& X,
            const VectorXd& f,
            const MatrixXi& tri) : gen(clock())
  {
    size_t N = X.rows();
    assert(X.cols() == 2);
    assert(f.cols() == 1);
    assert(f.rows() == N);
    assert(tri.cols() == 3);


    CGAL::Triangulation_incremental_builder_2<T> builder(_t);
    builder.begin_triangulation();

    // add vertices
    std::vector<T::Vertex_handle> vertices(N);
    for (size_t i = 0; i < N; ++i)
      {
	Point p(X(i,0),X(i,1));
	vertices[i] = builder.add_vertex(Point(X(i,0), X(i,1)));
	vertices[i]->info() = i;
      }

    // add faces
    size_t Nt = tri.rows();
    for (size_t i = 0; i < Nt; ++i)
      {
	int a = tri(i,0), b = tri(i,1), c = tri(i,2);
	/* T::Face_handle fh = builder.add_face(vertices[a], */
	/* 				     vertices[b], */
	/* 				     vertices[c]); */
	builder.add_face(vertices[a], vertices[b], vertices[c]);
      }
    builder.end_triangulation();

    // compute functions
    for (T::Finite_faces_iterator it = _t.finite_faces_begin ();
	 it != _t.finite_faces_end(); ++it)
      {
	size_t a = it->vertex(0)->info();
	size_t b = it->vertex(1)->info();
	size_t c = it->vertex(2)->info();
	_functions[it] = Function(vertices[a]->point(), f[a],
				  vertices[b]->point(), f[b],
				  vertices[c]->point(), f[c]);
      }
  }

  MatrixXi compute_boundary()
  {
    std::vector<std::pair<size_t, size_t> > bd;
    // for (T::All_edges_iterator it = _t.all_edges_begin();
    // 	 it != _t.all_edges_end(); ++it)
    for (T::Finite_edges_iterator it = _t.finite_edges_begin();
	 it != _t.finite_edges_end(); ++it)
      {
	T::Edge e = *it;
	// std::cerr << e.first->vertex(e.second)->info()
	// 	  << " -> ["
	// 	  << e.first->vertex((e.second+1)%3)->info() << ", "
	// 	  << e.first->vertex((e.second+2)%3)->info() << "]\n";
	if (_t.is_infinite(e.first->vertex(e.second)))
	  {
	    int i1= e.first->vertex((e.second+1)%3)->info();
	    int i2= e.first->vertex((e.second+2)%3)->info();
	    bd.push_back(std::make_pair(i1,i2));
	  }
      }
    size_t N = bd.size();
    MatrixXi X(N, 2);
    for (size_t i = 0; i < N; ++i)
      {
	X(i,0) = bd[i].first;
	X(i,1) = bd[i].second;
      }
    return X;
  }

    MatrixXd restricted_laguerre_edges(const MatrixXd& X,
                                       const VectorXd& w)
  {
    RT dt;
    python_to_delaunay_2(X, w, dt);
    std::vector<Segment> edges;
    ::restricted_laguerre_edges(_t,dt,edges);

    size_t N = edges.size();
    MatrixXd Edges(N, 4);
    for (size_t i = 0; i < N; ++i)
      {
	Edges(i,0) = edges[i].source().x();
	Edges(i,1) = edges[i].source().y();
	Edges(i,2) = edges[i].target().x();
	Edges(i,3) = edges[i].target().y();
      }
    return Edges;
  }

  double mass()
  {
    double total(0);
    for (auto f = _t.finite_faces_begin();
	 f != _t.finite_faces_end(); ++f)
      {
	total += MA::integrate_centroid<double>(f->vertex(0)->point(),
						f->vertex(1)->point(),
						f->vertex(2)->point(),
						_functions[f]);
      }
    return total;
  }

  MatrixXd random_sampling (size_t N)
  {
    std::vector<Triangle> triangles;
    std::vector<double> cumareas;
    compute_cum_masses(triangles, cumareas);

    MatrixXd X(N, 2);

    double ta = cumareas.back();
    for (size_t i = 0; i < N; ++i)
      {
	double r = rand01(gen) * ta;
	size_t n = (std::lower_bound(cumareas.begin(),
				     cumareas.end(), r) -
		    cumareas.begin());
	Point p = triangles[n].rand(gen);
	X(i,0) = p.x();
	X(i,1) = p.y();
      }
    return X;
  }
};

std::tuple<double, VectorXd, SparseMatrix>
kantorovich_2(const Density_2 &pl,
	      const MatrixXd &X,
	      const VectorXd &w)
{
  size_t N = X.rows();
  assert(X.cols() == 2);
  assert(w.cols() == 1);
  assert(w.rows() == N);

  VectorXd g(N);

  SparseMatrix h;
  double res = MA::kantorovich(pl._t, pl._functions, X, w, g, h);
  return std::make_tuple(res, g, h);
}


MatrixXi
delaunay_2(const MatrixXd &X,
           const VectorXd& w)
{
  RT dt;
  python_to_delaunay_2(X, w, dt);

  size_t Nt = dt.number_of_faces();

  // convert triangulation to python
  MatrixXi t(Nt, 3);

  size_t f = 0;
  for (RT::Finite_faces_iterator it = dt.finite_faces_begin ();
       it != dt.finite_faces_end(); ++it)
    {
      t(f, 0) = it->vertex(0)->info();
      t(f, 1) = it->vertex(1)->info();
      t(f, 2) = it->vertex(2)->info();
      ++f;
    }
  assert(f == Nt);
  return t;
}

template <class Matrix, class Vector>
void check_points_and_weights(const Matrix &X,
			      const Vector &w)
{
  if(X.cols() != 2)
    {
      PyErr_SetString(PyExc_TypeError,
		      "Point array dimension should be Nx2");
      std::cerr << X.rows() << " " <<  X.cols() << "\n";
      throw py::error_already_set();
    }
  if(w.cols() != 1 || w.rows() != X.rows())
    {
      PyErr_SetString(PyExc_TypeError,
		      "Weight array should be Nx1, where N is "
		      "the number of points");
      throw py::error_already_set();
    }
}

std::tuple<MatrixXd, VectorXd>
lloyd_2(const Density_2 &pl,
	const MatrixXd &X,
	const VectorXd &w)
{
  check_points_and_weights(X, w);

  size_t N = X.rows();
  // create some room for return values: centroids and masses
  MatrixXd c(N, 2);
  VectorXd m(N);

  MA::lloyd(pl._t, pl._functions, X, w, m, c);
  return std::make_tuple(c, m);
}

std::vector<MatrixXd>
rasterize_2(const Density_2 &pl,
	    const MatrixXd &X,
	    const VectorXd &w,
	    const MatrixXd &colors,
	    double x0, double y0, double x1, double y1, // bounding box
	    int ww, int hh)
{
  check_points_and_weights(X, w);
  if (colors.rows() != X.rows())
    {
      PyErr_SetString(PyExc_TypeError,
		      "Color array should be Nxk, where N is "
		      "the number of points and k arbitrary");
      throw py::error_already_set();
    }
  typedef VectorXd Color;
  std::vector<Color> colorv (colors.rows());
  for (int i = 0; i < colors.rows(); ++i)
    colorv[i] = colors.row(i);

  // FIXME: avoid matrix copies
  int nchannels = colors.cols();
  std::vector<MatrixXd> channels(nchannels, MatrixXd::Zero(ww,hh));
  MA::draw_laguerre_diagram(pl._t, pl._functions, X, w, colorv,
			    x0, y0, x1, y1, ww, hh,
			    [&](int i, int j, const Color &col)
			    {
			      for (int k = 0; k < nchannels; ++k)
				channels[k](i,j) += col[k];
			    });

  return channels;
}

template <class K>
bool
object_contains_point(const CGAL::Object &oi, CGAL::Point_2<K> &intp)
{
  if(const CGAL::Point_2<K>* r = CGAL::object_cast< CGAL::Point_2<K> >(&oi))
    {
      intp = *r;
      return true;
    }
  else if(const CGAL::Segment_2<K>* s = CGAL::object_cast< CGAL::Segment_2<K> >(&oi))
    {
      intp = CGAL::midpoint(s->source(), s->target());
      return true;
    }
  return false;
}

template <class K>
bool
edge_dual_and_segment_isect(const CGAL::Object &o,
				   const CGAL::Segment_2<K> &s,
				   CGAL::Point_2<K> &intp)
{
  if (const CGAL::Segment_2<K> *os = CGAL::object_cast< CGAL::Segment_2<K> >(&o))
    return object_contains_point(CGAL::intersection(*os, s), intp);
  if (const CGAL::Line_2<K> *ol = CGAL::object_cast<CGAL::Line_2<K> >(&o))
    return object_contains_point(CGAL::intersection(*ol, s), intp);
  if (const CGAL::Ray_2<K> *orr = CGAL::object_cast< CGAL::Ray_2<K> >(&o))
    return object_contains_point(CGAL::intersection(*orr, s), intp);
  return false;
}

template <class Matrix, class Vector>
void
compute_adjacencies_with_polygon
    (const Matrix &X,
     const Vector &weights,
     const Matrix &polygon,
     std::vector<std::vector<Segment>> &adjedges,
     std::vector<std::vector<size_t>> &adjverts)
{
  auto rt = MA::details::make_regular_triangulation(X,weights);

  int Np = polygon.rows();
  int Nv = X.rows();
  adjedges.assign(Nv, std::vector<Segment>());
  adjverts.assign(Nv, std::vector<size_t>());

  for (int p = 0; p < Np; ++p)
    {
      int pnext = (p + 1) % Np;
      //int pprev = (p + Np - 1) % Np;
      Point source(polygon(p,0), polygon(p,1));
      Point target(polygon(pnext,0), polygon(pnext,1));

      auto u = rt.nearest_power_vertex(source);
      auto v = rt.nearest_power_vertex(target);

      adjverts[u->info()].push_back(p);
      Point pointprev = source;

      auto  uprev = u;
      while (u != v)
	{
	  // find next vertex intersecting with  segment
	  auto c = rt.incident_edges(u), done(c);
	  do
	    {
	      if (rt.is_infinite(c))
		continue;

	      // we do not want to go back to the previous vertex!
	      auto unext = (c->first)->vertex(rt.ccw(c->second));
	      if (unext == uprev)
		continue;

	      // check whether dual edge (which can be a ray, a line
	      // or a segment) intersects with the constraint
	      Point point;
	      if (!edge_dual_and_segment_isect(rt.dual(c),
					       Segment(source,target),
					       point))
		continue;

	      adjedges[u->info()].push_back(Segment(pointprev,point));
	      pointprev = point;
	      uprev = u;
	      u = unext;

	      break;
	    }
	  while(++c != done);
	}

      adjverts[v->info()].push_back(pnext);
      adjedges[v->info()].push_back(Segment(pointprev, target));
    }
}

// Return projection of p on [v,w]
VectorXd projection_on_segment(VectorXd v, VectorXd w, VectorXd p)
{
  double l2 = (v-w).squaredNorm();
  if (l2 <= 1e-10)
    return v;

  // Consider the line extending the segment, parameterized as v + t
  // (w - v).  We find projection of point p onto the line.  It falls
  // where t = [(p-v) . (w-v)] / |w-v|^2
  double t = (p - v).dot(w - v) / l2;
  t = std::min(std::max(t,0.0), 1.0);

  return v + t * (w - v);
}

std::tuple<MatrixXd, VectorXd>
conforming_lloyd_2(const Density_2 &pl,
		   const MatrixXd &X,
		   const VectorXd &w,
		   const MatrixXd &poly)
{
  check_points_and_weights(X, w);

  size_t N = X.rows();
  // create some room for return values: centroids and masses
  VectorXd m(N);
  MatrixXd c(N, 2);

  MA::lloyd(pl._t, pl._functions, X, w, m, c);

  std::vector<std::vector<Segment>> adjedges;
  std::vector<std::vector<size_t>> adjverts;
  compute_adjacencies_with_polygon(X, w, poly, adjedges, adjverts);

  //double lengthbd = 0;
  for (size_t i = 0; i < N; ++i)
    {
      if (adjverts[i].size() != 0)
	c.row(i) = poly.row(adjverts[i][0]);
      if (adjedges[i].size() != 0)
	{
	  double mindist = 1e10;
	  VectorXd proj;
	  for (size_t j = 0; j < adjedges[i].size(); ++j)
	    {
	      Vector2d source (adjedges[i][j].source().x(),
			       adjedges[i][j].source().y());
	      Vector2d dest (adjedges[i][j].target().x(),
			     adjedges[i][j].target().y());
	      //lengthbd += (source-dest).norm();
	      auto p = projection_on_segment(source, dest,
					     c.row(i));
	      double dp = (p - c.row(i)).squaredNorm();
	      if (mindist > dp)
		{
		  mindist = dp;
		  proj = p;
		}
	    }
	  c.row(i) = proj;
	}
    }
  //std::cerr << "length = " << lengthbd << "\n";

  return std::make_tuple(c, m);
}

std::tuple<VectorXd, MatrixXd, MatrixXd>
moments_2(const Density_2 &pl,
	  const MatrixXd &X,
	  const VectorXd &w)
{
  size_t N = X.rows();
  assert(X.cols() == 2);
  assert(w.cols() == 1);
  assert(w.rows() == N);

  // create some room for return values: masses, centroids and inertia
  VectorXd m(N);
  MatrixXd c(N, 2);
  MatrixXd I(N, 3);

  MA::second_moment(pl._t, pl._functions, X, w, m, c, I);
  return std::make_tuple(m, c, I);
}


// This function solves a linear system using a Cholesky
// decomposition. This implementation seems faster and more robust
// than scipy's spsolve.
VectorXd
solve_cholesky(const SparseMatrix &h,
               const VectorXd &b)
{
  assert(b.rows() == h.rows());

  Eigen::SimplicialLLT<SparseMatrix> solver(h);
  VectorXd r = solver.solve(b);
  assert(r.rows() == h.cols());

  return r;
}

PYBIND11_MODULE(MongeAmperePP, m) {
    m.doc() = "Monge Ampère solver using Laguerre diagrams";

  py::class_<Density_2>
    (m, "Density_2")
    .def(py::init<const MatrixXd&,const VectorXd&,
	       const MatrixXi&>())
    //.def_readonly("boundary", &Density_2::boundary)
    //.def("compute_boundary", &Density_2::compute_boundary)
    .def("restricted_laguerre_edges", &Density_2::restricted_laguerre_edges)
    .def("mass", &Density_2::mass)
    .def("random_sampling", &Density_2::random_sampling);

  m.def("kantorovich_2", &kantorovich_2);
  m.def("lloyd_2", &lloyd_2);
  m.def("conforming_lloyd_2", &conforming_lloyd_2);
  m.def("moments_2", &moments_2);
  m.def("delaunay_2", &delaunay_2);
  m.def("rasterize_2", &rasterize_2);
  m.def("solve_cholesky", &solve_cholesky);
}
