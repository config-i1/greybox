// Friedman's Super-Smoother (1984) matching R's stats::supsmu exactly.
//
// Direct port of R's FORTRAN implementation at
//   src/library/stats/src/ppr.f (subroutines supsmu, smooth, bksupsmu).
//
// Reference:
//   Friedman, J.H. (1984) "A variable span smoother."
//   Technical Report 5, Laboratory for Computational Statistics,
//   Department of Statistics, Stanford University.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace py = pybind11;

namespace {

// Compile-time constants matching the FORTRAN block data bksupsmu.
constexpr double SPANS[3] = {0.05, 0.2, 0.5};
constexpr double BIG = 1.0e20;
constexpr double SML = 1.0e-7;
constexpr double EPS = 1.0e-3;

// One-pass running linear smoother (FORTRAN: subroutine smooth).
//
// When iper > 0, also writes absolute cross-validated residuals into acvr.
// When iper < 0, acvr is written-only (used by supsmu as a scratch).
// |iper| selects periodicity: 1 = non-periodic, 2 = periodic.
void smooth_impl(std::size_t n,
                 const std::vector<double>& x,
                 const std::vector<double>& y,
                 const std::vector<double>& w,
                 double span,
                 int iper,
                 double vsmlsq,
                 std::vector<double>& smo,
                 std::vector<double>& acvr) {
    double xm = 0.0, ym = 0.0, var = 0.0, cvar = 0.0, fbw = 0.0;
    int jper = std::abs(iper);

    // ibw = int(0.5 * span * n + 0.5)
    long ibw = static_cast<long>(0.5 * span * static_cast<double>(n) + 0.5);
    if (ibw < 2) ibw = 2;
    long it = 2 * ibw + 1;
    if (it > static_cast<long>(n)) it = static_cast<long>(n);

    // Initial accumulation (warm-up of running sums).
    for (long i = 1; i <= it; ++i) {  // FORTRAN 1-based loop
        long j;
        double xti;
        if (jper == 2) {
            j = i - ibw - 1;
            if (j >= 1) {
                xti = x[j - 1];
            } else {
                j = static_cast<long>(n) + j;
                xti = x[j - 1] - 1.0;
            }
        } else {
            j = i;
            xti = x[j - 1];
        }
        double wt = w[j - 1];
        double fbo = fbw;
        fbw = fbw + wt;
        if (fbw > 0.0) xm = (fbo * xm + wt * xti) / fbw;
        if (fbw > 0.0) ym = (fbo * ym + wt * y[j - 1]) / fbw;
        double tmp = 0.0;
        if (fbo > 0.0) tmp = fbw * wt * (xti - xm) / fbo;
        var = var + tmp * (xti - xm);
        cvar = cvar + tmp * (y[j - 1] - ym);
    }

    // Main loop: slide the window through each point.
    for (long j = 1; j <= static_cast<long>(n); ++j) {
        long out = j - ibw - 1;
        long in = j + ibw;
        bool skip_update = false;

        if (jper != 2 && (out < 1 || in > static_cast<long>(n))) {
            skip_update = true;
        }

        if (!skip_update) {
            double xto, xti;
            if (out < 1) {
                out = static_cast<long>(n) + out;
                xto = x[out - 1] - 1.0;
                xti = x[in - 1];
            } else if (in > static_cast<long>(n)) {
                in = in - static_cast<long>(n);
                xti = x[in - 1] + 1.0;
                xto = x[out - 1];
            } else {
                xto = x[out - 1];
                xti = x[in - 1];
            }

            double wt = w[out - 1];
            double fbo = fbw;
            fbw = fbw - wt;
            double tmp = 0.0;
            if (fbw > 0.0) tmp = fbo * wt * (xto - xm) / fbw;
            var = var - tmp * (xto - xm);
            cvar = cvar - tmp * (y[out - 1] - ym);
            if (fbw > 0.0) xm = (fbo * xm - wt * xto) / fbw;
            if (fbw > 0.0) ym = (fbo * ym - wt * y[out - 1]) / fbw;

            wt = w[in - 1];
            fbo = fbw;
            fbw = fbw + wt;
            if (fbw > 0.0) xm = (fbo * xm + wt * xti) / fbw;
            if (fbw > 0.0) ym = (fbo * ym + wt * y[in - 1]) / fbw;
            tmp = 0.0;
            if (fbo > 0.0) tmp = fbw * wt * (xti - xm) / fbo;
            var = var + tmp * (xti - xm);
            cvar = cvar + tmp * (y[in - 1] - ym);
        }

        double a = 0.0;
        if (var > vsmlsq) a = cvar / var;
        smo[j - 1] = a * (x[j - 1] - xm) + ym;

        if (iper > 0) {
            double h = 0.0;
            if (fbw > 0.0) h = 1.0 / fbw;
            if (var > vsmlsq) h = h + (x[j - 1] - xm) * (x[j - 1] - xm) / var;
            double a2 = 1.0 - w[j - 1] * h;
            if (a2 > 0.0) {
                acvr[j - 1] = std::abs(y[j - 1] - smo[j - 1]) / a2;
            } else if (j > 1) {
                acvr[j - 1] = acvr[j - 2];
            } else {
                acvr[j - 1] = 0.0;
            }
        }
    }

    // Recompute fitted values as a weighted mean across runs of equal x.
    long j = 1;
    while (j <= static_cast<long>(n)) {
        long j0 = j;
        double sy = smo[j - 1] * w[j - 1];
        double fbw_local = w[j - 1];
        if (j < static_cast<long>(n)) {
            while (x[j] <= x[j - 1]) {  // x[j] is FORTRAN x(j+1) (0-based j)
                j = j + 1;
                sy = sy + w[j - 1] * smo[j - 1];
                fbw_local = fbw_local + w[j - 1];
                if (j >= static_cast<long>(n)) break;
            }
        }
        if (j > j0) {
            double a = 0.0;
            if (fbw_local > 0.0) a = sy / fbw_local;
            for (long i = j0; i <= j; ++i) {
                smo[i - 1] = a;
            }
        }
        j = j + 1;
    }
}

// Friedman's super-smoother (FORTRAN: subroutine supsmu).
//
// Inputs:
//   x : abscissa values, sorted ascending.
//   y : ordinate values.
//   w : weights.
//   iper : 1 (non-periodic) or 2 (periodic in [0,1]).
//   span : 0 for "cv" (cross-validated automatic), else fixed span in (0,1).
//   alpha : bass tone (0..10). 0 or out-of-range disables.
// Output: smoothed y (length n).
std::vector<double> supsmu_impl(const std::vector<double>& x,
                                const std::vector<double>& y,
                                const std::vector<double>& w,
                                int iper,
                                double span,
                                double alpha) {
    std::size_t n = x.size();
    std::vector<double> smo(n, 0.0);
    if (n == 0) return smo;

    // Boundary case: x(n) <= x(1)
    if (x[n - 1] <= x[0]) {
        double sy = 0.0, sw = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            sy += w[j] * y[j];
            sw += w[j];
        }
        double a = (sw > 0.0) ? sy / sw : 0.0;
        std::fill(smo.begin(), smo.end(), a);
        return smo;
    }

    // IQR-based scale for vsmlsq (FORTRAN: i = n/4; j = 3i).
    long i_iqr = static_cast<long>(n) / 4;
    long j_iqr = 3 * i_iqr;
    double scale = 0.0;
    // FORTRAN 1-based indices: x(j) - x(i) → x[j_iqr-1] - x[i_iqr-1].
    // Guard against i_iqr == 0 (which would access x[-1]).
    if (i_iqr < 1) i_iqr = 1;
    if (j_iqr < 1) j_iqr = 1;
    if (j_iqr > static_cast<long>(n)) j_iqr = static_cast<long>(n);
    scale = x[j_iqr - 1] - x[i_iqr - 1];
    while (scale <= 0.0) {
        if (j_iqr < static_cast<long>(n)) j_iqr = j_iqr + 1;
        if (i_iqr > 1) i_iqr = i_iqr - 1;
        scale = x[j_iqr - 1] - x[i_iqr - 1];
        if (i_iqr == 1 && j_iqr == static_cast<long>(n) && scale <= 0.0) break;
    }
    double vsmlsq = (EPS * scale) * (EPS * scale);

    int jper = iper;
    if (iper == 2 && (x[0] < 0.0 || x[n - 1] > 1.0)) jper = 1;
    if (jper < 1 || jper > 2) jper = 1;

    // Fixed-span branch.
    if (span > 0.0) {
        std::vector<double> acvr(n, 0.0);
        smooth_impl(n, x, y, w, span, jper, vsmlsq, smo, acvr);
        return smo;
    }

    // Cross-validated span selection.
    // sc has 7 columns; we use 7 vectors of length n.
    std::vector<std::vector<double>> sc(7, std::vector<double>(n, 0.0));
    std::vector<double> h(n, 0.0);

    // For each candidate span, smooth y; then smooth |cv-residual| with the middle span.
    for (int s = 0; s < 3; ++s) {
        // sc(:, 2s)  in FORTRAN (1-based 2*i-1) = sc[2s]   (0-based) -> smoothed y at span s
        // sc(:, 7)   = sc[6] -> absolute cv residuals
        smooth_impl(n, x, y, w, SPANS[s], jper, vsmlsq, sc[2 * s], sc[6]);
        // Now smooth those residuals with the middle span, with iper negated
        // (-jper signals smooth to skip acvr computation).
        // FORTRAN: sc(1,2*i) ← smoothed residuals (0-based sc[2s+1])
        smooth_impl(n, x, sc[6], w, SPANS[1], -jper, vsmlsq, sc[2 * s + 1], h);
    }

    // For each point j, pick the span minimising the smoothed residual.
    // sc[6] (column 7) gets the selected span at each point.
    for (std::size_t k = 0; k < n; ++k) {
        double resmin = BIG;
        for (int s = 0; s < 3; ++s) {
            double rval = sc[2 * s + 1][k];
            if (rval < resmin) {
                resmin = rval;
                sc[6][k] = SPANS[s];
            }
        }
        // Bass-tone adjustment.
        if (alpha > 0.0 && alpha <= 10.0 &&
            resmin < sc[5][k] && resmin > 0.0) {
            double ratio = resmin / sc[5][k];
            double base = (ratio > SML) ? ratio : SML;
            sc[6][k] = sc[6][k] +
                       (SPANS[2] - sc[6][k]) * std::pow(base, 10.0 - alpha);
        }
    }

    // Smooth the chosen-span sequence with the middle span.
    smooth_impl(n, x, sc[6], w, SPANS[1], -jper, vsmlsq, sc[1], h);

    // Clamp and linearly interpolate between SPANS values; result in sc[3].
    for (std::size_t k = 0; k < n; ++k) {
        if (sc[1][k] <= SPANS[0]) sc[1][k] = SPANS[0];
        if (sc[1][k] >= SPANS[2]) sc[1][k] = SPANS[2];
        double f = sc[1][k] - SPANS[1];
        if (f < 0.0) {
            f = -f / (SPANS[1] - SPANS[0]);
            sc[3][k] = (1.0 - f) * sc[2][k] + f * sc[0][k];
        } else {
            f = f / (SPANS[2] - SPANS[1]);
            sc[3][k] = (1.0 - f) * sc[2][k] + f * sc[4][k];
        }
    }

    // Final pass: smooth the interpolated estimates with the smallest span.
    smooth_impl(n, x, sc[3], w, SPANS[0], -jper, vsmlsq, smo, h);
    return smo;
}

py::array_t<double> supsmu_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::object wt,
    double span,
    double bass,
    bool periodic) {
    auto bx = x.request();
    auto by = y.request();
    if (bx.ndim != 1 || by.ndim != 1) {
        throw std::runtime_error("supsmu: x and y must be 1-D arrays");
    }
    if (bx.size != by.size) {
        throw std::runtime_error("supsmu: x and y must have the same length");
    }
    std::size_t n = static_cast<std::size_t>(bx.size);

    std::vector<double> xv(static_cast<double*>(bx.ptr),
                           static_cast<double*>(bx.ptr) + n);
    std::vector<double> yv(static_cast<double*>(by.ptr),
                           static_cast<double*>(by.ptr) + n);

    std::vector<double> wv(n, 1.0);
    if (!wt.is_none()) {
        auto wt_arr = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(wt);
        auto bw = wt_arr.request();
        if (bw.ndim != 1 || static_cast<std::size_t>(bw.size) != n) {
            throw std::runtime_error("supsmu: wt must be a 1-D array of length n");
        }
        wv.assign(static_cast<double*>(bw.ptr),
                  static_cast<double*>(bw.ptr) + n);
    }

    int iper = periodic ? 2 : 1;
    double sp = (span < 0.0) ? 0.0 : span;
    std::vector<double> out = supsmu_impl(xv, yv, wv, iper, sp, bass);

    py::array_t<double> result(static_cast<py::ssize_t>(n));
    auto buf = result.mutable_unchecked<1>();
    for (std::size_t k = 0; k < n; ++k) {
        buf(static_cast<py::ssize_t>(k)) = out[k];
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_native_supsmu, m) {
    m.doc() = "Friedman's super-smoother matching R's stats::supsmu.";
    m.def(
        "supsmu",
        &supsmu_wrapper,
        py::arg("x"),
        py::arg("y"),
        py::arg("wt") = py::none(),
        py::arg("span") = 0.0,
        py::arg("bass") = 0.0,
        py::arg("periodic") = false,
        R"pbdoc(Friedman's super-smoother. Mirrors R's stats::supsmu.)pbdoc"
    );
}
