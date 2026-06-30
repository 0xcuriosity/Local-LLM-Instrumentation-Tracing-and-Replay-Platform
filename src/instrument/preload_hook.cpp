#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

// Dummy ggml types for demonstration
struct ggml_tensor {
    int type;
    int n_dims;
    int64_t ne[4];
    size_t nb[4];
    int op;
    int op_params[16];
    int32_t flags;
    void * data;
    char name[64];
};

struct ggml_cgraph {
    int n_nodes;
    int n_leafs;
    struct ggml_tensor ** nodes;
    struct ggml_tensor ** leafs;
};

// Original function pointer types
typedef int (*ggml_graph_compute_t)(struct ggml_cgraph * cgraph, void * cplan);
typedef void* (*dlopen_t)(const char*, int);

static ggml_graph_compute_t real_ggml_graph_compute = nullptr;
static dlopen_t real_dlopen = nullptr;

extern "C" {

// Intercept dlopen just to show we are loaded
void* dlopen(const char* filename, int flags) {
    if (!real_dlopen) {
        real_dlopen = (dlopen_t)dlsym(RTLD_NEXT, "dlopen");
    }
    
    if (filename) {
        std::cerr << "[NeuralScope Preload] dlopen intercepted: " << filename << std::endl;
    }
    
    return real_dlopen(filename, flags);
}

// Intercept ggml_graph_compute to hook into model execution
int ggml_graph_compute(struct ggml_cgraph * cgraph, void * cplan) {
    if (!real_ggml_graph_compute) {
        real_ggml_graph_compute = (ggml_graph_compute_t)dlsym(RTLD_NEXT, "ggml_graph_compute");
    }
    
    std::cerr << "[NeuralScope Preload] Graph compute intercepted. Nodes: " << cgraph->n_nodes << std::endl;
    
    // Call the original function to let the model run, forwarding the cplan and capturing status
    int status = real_ggml_graph_compute(cgraph, cplan);
    
    std::cerr << "[NeuralScope Preload] Graph compute finished. Status: " << status << std::endl;
    
    return status;
}

} // extern "C"

// GCC constructor to run when the library is loaded
__attribute__((constructor))
static void ns_preload_init() {
    std::cerr << "========================================================\n";
    std::cerr << "  NeuralScope LD_PRELOAD Instrumentation Loaded\n";
    std::cerr << "========================================================\n";
}
