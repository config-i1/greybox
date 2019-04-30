#include <Rcpp.h>

using namespace Rcpp;

/* # Function allows to multiply polinomails */
NumericVector polyMult(NumericVector const &poly1, NumericVector const &poly2){

    int poly1Nonzero = poly1.size()-1;
    int poly2Nonzero = poly2.size()-1;

    NumericVector poly3(poly1Nonzero + poly2Nonzero + 1);

    for(int i = 0; i <= poly1Nonzero; ++i){
        for(int j = 0; j <= poly2Nonzero; ++j){
            poly3[i+j] += poly1[i] * poly2[j];
        }
    }

    return poly3;
}

/* # Function allows to multiply polinomails */
//' This function calculates parameters for the polynomials
//'
//' The function accepts two vectors with the parameters for the polynomials and returns
//' the vector of parameters after their multiplication. This can be especially useful,
//' when working with ARIMA models.
//'
//' @param x The vector of parameters of the first polynomial.
//' @param y The vector of parameters of the second polynomial.
//'
//' @template author
//'
//' @return The function returns a matrix with one column with the parameters for
//' the polynomial, starting from the 0-order.
//'
//' @seealso \link[stats]{convolve}
//'
//' @examples
//'
//' polyprod(c(1,-2,-1),c(1,0.5,0.3))
//'
//' @useDynLib greybox
//' @export
// [[Rcpp::export]]
RcppExport SEXP polyprod(SEXP x, SEXP y){
    NumericVector polyVec1_n(x);
    NumericVector polyVec2_n(y);

    return wrap(polyMult(polyVec1_n, polyVec2_n));
}
