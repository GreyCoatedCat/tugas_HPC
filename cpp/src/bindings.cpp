// Binding pybind11 untuk modul rag_hpc (dipakai oleh rag/hpc.py).
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bm25_omp.h"
#include "topk_cuda.h"

namespace py = pybind11;
using namespace rag_hpc;

namespace {

using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using I32 = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using I64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

class PyCudaTopK {
public:
    PyCudaTopK(const F32& emb, bool fp16, int device) {
        if (emb.ndim() != 2) throw std::invalid_argument("embeddings harus 2D [n, d]");
        const int64_t n = emb.shape(0);
        const int d = (int)emb.shape(1);
        const float* ptr = emb.data();
        py::gil_scoped_release nogil;
        impl_ = std::make_unique<CudaTopK>(ptr, n, d, fp16, device);
    }

    // queries [m, d] atau [d] -> (idx [m, k], scores [m, k], timings dict)
    py::tuple search(const F32& queries, int k) {
        const int d = impl_->dim();
        int m;
        if (queries.ndim() == 1) {
            if (queries.shape(0) != d) throw std::invalid_argument("dimensi kueri tidak cocok");
            m = 1;
        } else if (queries.ndim() == 2) {
            if (queries.shape(1) != d) throw std::invalid_argument("dimensi kueri tidak cocok");
            m = (int)queries.shape(0);
        } else {
            throw std::invalid_argument("kueri harus 1D atau 2D");
        }
        const int k_eff = (int)std::max<int64_t>(0, std::min<int64_t>(k, impl_->size()));
        I64 idx({m, k_eff});
        F32 scores({m, k_eff});
        CudaTimings t;
        if (m > 0 && k_eff > 0) {
            const float* q = queries.data();
            int64_t* pi = idx.mutable_data();
            float* ps = scores.mutable_data();
            py::gil_scoped_release nogil;  // thread Python lain (mis. BM25 di CPU) bisa jalan bersamaan
            impl_->search(q, m, k_eff, pi, ps, &t);
        }
        py::dict td;
        td["h2d_ms"] = t.h2d_ms;
        td["score_ms"] = t.score_ms;
        td["select_ms"] = t.select_ms;
        td["d2h_ms"] = t.d2h_ms;
        return py::make_tuple(idx, scores, td);
    }

    int64_t size() const { return impl_->size(); }
    int dim() const { return impl_->dim(); }
    bool fp16() const { return impl_->fp16(); }
    size_t matrix_bytes() const { return impl_->matrix_bytes(); }
    int max_batch() const { return impl_->max_batch(); }

private:
    std::unique_ptr<CudaTopK> impl_;
};

class PyOpenMPBM25 {
public:
    PyOpenMPBM25(const F32& data, const I32& indices, const I64& indptr, int64_t n_docs, int threads)
        : data_(data), indices_(indices), indptr_(indptr) {
        if (indptr_.ndim() != 1 || indptr_.shape(0) < 1) throw std::invalid_argument("indptr tidak valid");
        const int64_t n_terms = indptr_.shape(0) - 1;
        impl_ = std::make_unique<OpenMPBM25>(data_.data(), indices_.data(), indptr_.data(), n_docs, n_terms, threads);
    }

    py::tuple search(const I32& cols, int k) {
        std::vector<int64_t> idx((size_t)std::max(k, 0));
        std::vector<float> sc((size_t)std::max(k, 0));
        BM25Timings t;
        int n = 0;
        {
            const int32_t* c = cols.data();
            const int nc = (int)cols.size();
            py::gil_scoped_release nogil;
            n = impl_->search(c, nc, k, idx.data(), sc.data(), &t);
        }
        I64 out_i(n);
        F32 out_s(n);
        std::copy(idx.begin(), idx.begin() + n, out_i.mutable_data());
        std::copy(sc.begin(), sc.begin() + n, out_s.mutable_data());
        py::dict td;
        td["parallel_ms"] = t.parallel_ms;
        td["merge_ms"] = t.merge_ms;
        return py::make_tuple(out_i, out_s, td);
    }

    int threads() const { return impl_->threads(); }
    int64_t n_docs() const { return impl_->n_docs(); }

private:
    F32 data_;      // referensi dijaga agar pointer di impl_ tetap valid
    I32 indices_;
    I64 indptr_;
    std::unique_ptr<OpenMPBM25> impl_;
};

py::dict py_device_info(int device) {
    DeviceInfo i = device_info(device);
    py::dict d;
    d["name"] = std::string(i.name);
    d["compute_capability"] = std::to_string(i.major) + "." + std::to_string(i.minor);
    d["sm_count"] = i.sm_count;
    d["total_mem_mb"] = (double)i.total_mem / (1024.0 * 1024.0);
    d["free_mem_mb"] = (double)i.free_mem / (1024.0 * 1024.0);
    d["cuda_runtime"] = i.runtime_version;
    d["cuda_driver"] = i.driver_version;
    return d;
}

int cuda_device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return n;
}

}  // namespace

PYBIND11_MODULE(rag_hpc, m) {
    m.doc() = "Akselerasi pipeline RAG: pencarian vektor CUDA (similarity + top-k) dan BM25 OpenMP";
    m.attr("__version__") = "0.1.0";

    py::class_<PyCudaTopK>(m, "CudaTopK")
        .def(py::init<const F32&, bool, int>(), py::arg("embeddings"), py::arg("fp16") = true, py::arg("device") = 0)
        .def("search", &PyCudaTopK::search, py::arg("queries"), py::arg("k"))
        .def_property_readonly("size", &PyCudaTopK::size)
        .def_property_readonly("dim", &PyCudaTopK::dim)
        .def_property_readonly("fp16", &PyCudaTopK::fp16)
        .def_property_readonly("matrix_bytes", &PyCudaTopK::matrix_bytes)
        .def_property_readonly("max_batch", &PyCudaTopK::max_batch);

    py::class_<PyOpenMPBM25>(m, "OpenMPBM25")
        .def(py::init<const F32&, const I32&, const I64&, int64_t, int>(), py::arg("data"), py::arg("indices"),
             py::arg("indptr"), py::arg("n_docs"), py::arg("threads") = 0)
        .def("search", &PyOpenMPBM25::search, py::arg("cols"), py::arg("k"))
        .def_property_readonly("threads", &PyOpenMPBM25::threads)
        .def_property_readonly("n_docs", &PyOpenMPBM25::n_docs);

    m.def("device_info", &py_device_info, py::arg("device") = 0);
    m.def("cuda_device_count", &cuda_device_count);
    m.def("omp_max_threads", []() { return omp_get_max_threads(); });
}
