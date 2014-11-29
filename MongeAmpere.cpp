#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef double FT;
typedef Eigen::SparseMatrix<FT> SparseMatrix;
typedef Eigen::SparseVector<FT> SparseVector;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;


#include <CGAL/Triangulation_2.h>
#include <CGAL/Triangulation_incremental_builder_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_2<size_t, K> Vb;
typedef CGAL::Triangulation_data_structure_2<Vb> Tds;
typedef CGAL::Triangulation_2<K,Tds> T;

typedef CGAL::Point_2<K> Point;
typedef CGAL::Vector_2<K> Vector;
typedef CGAL::Line_2<K> Line;

#include <MA/functions.hpp>
#include <MA/kantorovich.hpp>
#include <MA/lloyd.hpp>
  
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <boost/scoped_array.hpp>
#include <iostream>
namespace p = boost::python;
namespace np = boost::numpy;

typedef Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> DynamicStride;

template<class FT>
Eigen::Map<Eigen::Matrix<FT,-1,-1>, Eigen::Unaligned,  DynamicStride>
python_to_matrix(const np::ndarray &array)
{
  if (array.get_dtype() != np::dtype::get_builtin<FT>())
    {
      PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
      p::throw_error_already_set();
    }
  if (array.get_nd() != 2)
    {
      PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
      p::throw_error_already_set();
    }
  auto s0 = array.strides(0) / sizeof(FT);
  auto s1 = array.strides(1) / sizeof(FT);
  return Eigen::Map<Eigen::Matrix<FT,-1,-1>,
		    Eigen::Unaligned,  DynamicStride>
    (reinterpret_cast<FT*>(array.get_data()),
     array.shape(0), array.shape(1),
     DynamicStride(s1,s0));
}

template<class FT>
Eigen::Map<Eigen::Matrix<FT,-1,1>, Eigen::Unaligned,  DynamicStride>
python_to_vector(const np::ndarray &array)
{
  if (array.get_dtype() != np::dtype::get_builtin<FT>())
    {
      PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
      p::throw_error_already_set();
    }
  if (array.get_nd() != 1)
    {
      PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
      p::throw_error_already_set();
    }
  auto s0 = array.strides(0) / sizeof(FT);
  return Eigen::Map<Eigen::Matrix<FT,-1,1>,
		    Eigen::Unaligned,  DynamicStride>
    (reinterpret_cast<FT*>(array.get_data()),
     array.shape(0), DynamicStride(0,s0));
}

class Density_2
{
public:
  T _t;
  typedef MA::Linear_function<K> Function;
  std::map<T::Face_handle, Function> _functions;
  
public:
  Density_2(const np::ndarray &npX, 
	    const np::ndarray &npf,
	    const np::ndarray &npT)
  {
    auto X = python_to_matrix<double>(npX);
    auto f = python_to_vector<double>(npf);
    auto tri = python_to_matrix<int>(npT);

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
	T::Face_handle fh = builder.add_face(vertices[a],
					     vertices[b],
					     vertices[c]);
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

  double mass()
  {
    double total(0);
    for (auto f = _t.finite_faces_begin(); 
	 f != _t.finite_faces_end(); ++f)
      {
	total += MA::integrate_centroid(f->vertex(0)->point(),
					f->vertex(1)->point(),
					f->vertex(2)->point(),
					_functions[f]);
      }
    return total;
  }
    
};

p::tuple
sparse_to_python(const SparseMatrix &h)
{
  size_t nnz = h.nonZeros();
  auto gI = np::zeros(p::make_tuple(nnz),
		      np::dtype::get_builtin<int>());
  auto gJ = np::zeros(p::make_tuple(nnz),
		      np::dtype::get_builtin<int>());
  auto gS = np::zeros(p::make_tuple(nnz),
		      np::dtype::get_builtin<double>());
  int i = 0;
  for (int k = 0; k < h.outerSize(); ++k)
    {
      for (SparseMatrix::InnerIterator it(h, k); it; ++it, ++i)
	{
	  gI[i] = it.row();
	  gJ[i] = it.col();
	  gS[i] = it.value();
	}
    }
  return p::make_tuple(gS,p::make_tuple(gI,gJ));
}

p::tuple
kantorovich_2(const Density_2 &pl,
	      const np::ndarray &pX, 
	      const np::ndarray &pw)
{
  auto X = python_to_matrix<double>(pX);
  auto w = python_to_vector<double>(pw);

  size_t N = X.rows();
  assert(X.cols() == 2);
  assert(w.cols() == 1);
  assert(w.rows() == N);
  auto pg = np::zeros(p::make_tuple(N),
		      np::dtype::get_builtin<double>());
  auto g = python_to_vector<double>(pg);//gradient

  SparseMatrix h;
  double res = MA::kantorovich(pl._t, pl._functions, X, w, g, h);
  return p::make_tuple(res, pg, 
		       sparse_to_python(h));
}

np::ndarray
delaunay_2(const np::ndarray &pX, 
	   const np::ndarray &pw)
{
  auto X = python_to_matrix<double>(pX);
  auto w = python_to_vector<double>(pw);

  size_t N = X.rows();
  assert(X.cols() == 2);
  assert(w.cols() == 1);
  assert(w.rows() == N);

  typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
  typedef CGAL::Polygon_2<K> Polygon;
  typedef K::FT FT;
  typedef CGAL::Regular_triangulation_filtered_traits_2<K> RT_Traits;
  typedef CGAL::Regular_triangulation_vertex_base_2<RT_Traits> Vbase;
  typedef CGAL::Triangulation_vertex_base_with_info_2
      <size_t, RT_Traits, Vbase> Vb;
  typedef CGAL::Regular_triangulation_face_base_2<RT_Traits> Cb;
  typedef CGAL::Triangulation_data_structure_2<Vb,Cb> Tds;
  typedef CGAL::Regular_triangulation_2<RT_Traits, Tds> RT;
  
  typedef RT::Vertex_handle Vertex_handle_RT;
  typedef RT::Weighted_point Weighted_point;
  typedef typename CGAL::Point_2<K> Point;
  
  // insert points with indices in the regular triangulation
  std::vector<std::pair<Weighted_point,size_t> > Xw(N);
  for (size_t i = 0; i < N; ++i)
    {
      Xw[i] = std::make_pair(Weighted_point(Point(X(i,0), X(i,1)),
					    w(i)), i);
    }
  RT dt (Xw.begin(), Xw.end());
  dt.infinite_vertex()->info() = -1;
  
  size_t Nt = dt.number_of_faces();

  // convert triangulation to python
  auto pt = np::zeros(p::make_tuple(Nt,3),
		      np::dtype::get_builtin<int>());
  auto t = python_to_matrix<int>(pt);

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
  return pt;
}


p::tuple
lloyd_2(const Density_2 &pl,
	const np::ndarray &pX, 
	const np::ndarray &pw)
{
  auto X = python_to_matrix<double>(pX);
  auto w = python_to_vector<double>(pw);

  size_t N = X.rows();
  assert(X.cols() == 2);
  assert(w.cols() == 1);
  assert(w.rows() == N);

  // create some room for return values: centroids and masses
  auto pm = np::zeros(p::make_tuple(N),
		      np::dtype::get_builtin<double>());
  auto m = python_to_vector<double>(pm); 
  auto pc = np::zeros(p::make_tuple(N,2),
		      np::dtype::get_builtin<double>());
  auto c = python_to_matrix<double>(pc); 

  MA::lloyd(pl._t, pl._functions, X, w, c, m);
  return p::make_tuple(pc, pm);
}

#ifdef MA_USE_SUITESPARSE_QR

#undef Success
// Suitesparse 4.3.1 does not define UF_long, which is expected by the
// Eigen wrapper classes
#include <cs.h>
#ifndef UF_long
#define  UF_long cs_long_t
#endif
#include <Eigen/SPQRSupport>

#endif


void python_to_sparse (const p::object &ph,
		       SparseMatrix &h)
{
  np::ndarray pS = p::extract<np::ndarray>(ph.attr("data"));
  np::ndarray pI = p::extract<np::ndarray>(ph.attr("row"));
  np::ndarray pJ = p::extract<np::ndarray>(ph.attr("col"));
  p::tuple pshape = p::extract<p::tuple>(ph.attr("shape"));
  auto S = python_to_vector<double>(pS);
  auto I = python_to_vector<int>(pI);
  auto J = python_to_vector<int>(pJ);
  size_t nnz = S.rows();
  assert(I.rows() == nnz);
  assert(J.rows() == nnz);

  size_t M = p::extract<size_t>(pshape[0]);
  size_t N = p::extract<size_t>(pshape[1]);

  // build matrix
  typedef Eigen::Triplet<FT> Triplet;
  std::vector<Triplet> triplets(nnz);  
  for (size_t i = 0; i < nnz; ++i)
    {
      // std::cerr << "(i,j,s) = ("
      // 		<< I(i) << ", " << J(i) << ", " << S(i) << ")\n";
      triplets[i] = Triplet(I(i), J(i), S(i));
    }
  h = SparseMatrix(M,N);
  h.setFromTriplets(triplets.begin(), triplets.end());
  h.makeCompressed();
}

#ifdef MA_USE_SUITESPARSE_QR
np::ndarray
solve_spqr(const p::object &ph,
	   np::ndarray &pb)
{
  SparseMatrix h; python_to_sparse(ph,h);
  VectorXd b = python_to_vector<double>(pb);
  assert(b.rows() == h.rows());

  // solve
  Eigen::SPQR<SparseMatrix> solver(h);
  VectorXd r = solver.solve(b);
  assert(r.rows() == h.cols());

  // return result
  auto pr = np::zeros(p::make_tuple(r.rows()),
		      np::dtype::get_builtin<double>());
  for (size_t i = 0; i < r.rows(); ++i)
    pr[i] = r[i];
  return pr;
}
#endif

BOOST_PYTHON_MODULE(MongeAmperePP)
{
  np::initialize();
  p::class_<Density_2>
    ("Density_2",
       p::init<const np::ndarray &,const np::ndarray&,
	       const np::ndarray&>())
    .def("mass", &Density_2::mass);
  p::def("kantorovich_2", &kantorovich_2);
  p::def("lloyd_2", &lloyd_2);
  p::def("delaunay_2", &delaunay_2);
#ifdef MA_USE_SUITESPARSE_QR
  p::def("solve_spqr", &solve_spqr);
#endif
}
