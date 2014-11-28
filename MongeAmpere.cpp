#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef double FT;
typedef Eigen::SparseMatrix<FT> SparseMatrix;
typedef Eigen::SparseVector<FT> SparseVector;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;


#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> T;
typedef CGAL::Point_2<K> Point;
typedef CGAL::Vector_2<K> Vector;
typedef CGAL::Line_2<K> Line;

#include <MA/functions.hpp>
#include <MA/kantorovich.hpp>
  
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

class Density
{
public:
  T _t;
  typedef MA::Linear_function<K> Function;
  std::map<T::Face_handle, Function> _functions;
  
public:
  Density(const np::ndarray &npX, 
	  const np::ndarray &npf)
  {
    auto X = python_to_matrix<double>(npX);
    auto f = python_to_vector<double>(npf);

    size_t N = X.rows();
    assert(X.cols() == 2);
    assert(f.cols() == 1);
    assert(f.rows() == N);

    std::vector<Point> points;
    std::map<Point, double> function;
    for (size_t i = 0; i < N; ++i)
      {
	Point p(X(i,0),X(i,1));
	points.push_back(p);
	function[p] = f(i);
      }
    
    // build triangulation and PL functions
    _t = T(points.begin(), points.end());
    
    for (auto f = _t.finite_faces_begin(); 
	 f != _t.finite_faces_end(); ++f)
      {
	Point p = f->vertex(0)->point();
	Point q = f->vertex(1)->point();
	Point r = f->vertex(2)->point();
	_functions[f] = Function(p, function[p], 
				 q, function[q],
				 r, function[r]);
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
kantorovich(const Density &pl,
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

#undef Success

// Suitesparse 4.3.1 does not define UF_long, which is expected by the
// Eigen wrapper classes
#include <cs.h>
#ifndef UF_long
#define  UF_long cs_long_t
#endif
#include <Eigen/SPQRSupport>

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
  assert(r.rows() = h.cols());

  // return result
  auto pr = np::zeros(p::make_tuple(r.rows()),
		      np::dtype::get_builtin<double>());
  for (size_t i = 0; i < r.rows(); ++i)
    pr[i] = r[i];
  return pr;
}

BOOST_PYTHON_MODULE(MApp)
{
  np::initialize();
  p::class_<Density>
    ("Density",
     p::init<const np::ndarray &,const np::ndarray&>())
    .def("mass", &Density::mass);
  p::def("kantorovich", &kantorovich);
  p::def("solve_spqr", &solve_spqr);
}
