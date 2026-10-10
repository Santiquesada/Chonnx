#include "chugin.h"
#include <onnxruntime_cxx_api.h>
#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

//-----------------------------------------------------------------------------
// CHONNX — Generic float32 ONNX inference engine for ChucK
//
// Fully model-agnostic. Supports single or multiple float32 inputs/outputs.
// Seamlessly handles dynamic dimensions and maintains backward compatibility.
//-----------------------------------------------------------------------------

t_CKINT chonnx_data_offset = 0;
static Ort::Env* g_ort_env = nullptr;

struct ChonnxData {
    int           initialized;
    Ort::Session* session;

    size_t                             num_inputs;
    std::vector<std::string>           input_names;
    std::vector<std::vector<int64_t>>  input_shapes;
    std::vector<std::vector<float>>    input_buffers; // One flat buffer per input node

    size_t                             num_outputs;
    std::vector<std::string>           output_names;
    std::vector<std::vector<float>>    output_buffers; // One flat buffer per output node
};

// Functions declarations
CK_DLL_CTOR(chonnx_ctor);
CK_DLL_DTOR(chonnx_dtor);
CK_DLL_MFUN(chonnx_load_model);
CK_DLL_MFUN(chonnx_run);

// Overloaded functions for backward compatibility (Single Input / Output)
CK_DLL_MFUN(chonnx_set_input_single);
CK_DLL_MFUN(chonnx_get_output_single);

// New functions for Multi-Input / Multi-Output architectures (like DDSP)
CK_DLL_MFUN(chonnx_set_input_multi);
CK_DLL_MFUN(chonnx_get_output_multi);

//-----------------------------------------------------------------------------
// CHUCK CLASS REGISTRATION
//-----------------------------------------------------------------------------
CK_DLL_QUERY(Chonnx)
{
    QUERY->setname(QUERY, "Chonnx");
    QUERY->begin_class(QUERY, "Chonnx", "Object");
    QUERY->doc_class(QUERY, "Generic multi-tensor float32 ONNX inference engine for ChucK.");

    QUERY->add_ctor(QUERY, chonnx_ctor);
    QUERY->add_dtor(QUERY, chonnx_dtor);

    // load_model(string path) — loads .onnx and auto-detects architecture structures
    QUERY->add_mfun(QUERY, chonnx_load_model, "void", "load_model");
    QUERY->add_arg(QUERY, "string", "path");

    // BACKWARD COMPATIBILITY: set_input(int index, float value) -> Maps to Input 0
    QUERY->add_mfun(QUERY, chonnx_set_input_single, "void", "set_input");
    QUERY->add_arg(QUERY, "int", "index");
    QUERY->add_arg(QUERY, "float", "value");

    // MULTI-INPUT: set_input(int input_idx, int sample_idx, float value)
    QUERY->add_mfun(QUERY, chonnx_set_input_multi, "void", "set_input");
    QUERY->add_arg(QUERY, "int", "input_idx");
    QUERY->add_arg(QUERY, "int", "sample_idx");
    QUERY->add_arg(QUERY, "float", "value");

    // run() — executes inference, returns 1 on success, 0 on failure
    QUERY->add_mfun(QUERY, chonnx_run, "int", "run");

    // BACKWARD COMPATIBILITY: get_output(int index) -> Maps to Output 0
    QUERY->add_mfun(QUERY, chonnx_get_output_single, "float", "get_output");
    QUERY->add_arg(QUERY, "int", "index");

    // MULTI-OUTPUT: get_output(int output_idx, int sample_idx)
    QUERY->add_mfun(QUERY, chonnx_get_output_multi, "float", "get_output");
    QUERY->add_arg(QUERY, "int", "output_idx");
    QUERY->add_arg(QUERY, "int", "sample_idx");

    chonnx_data_offset = QUERY->add_mvar(QUERY, "int", "@cx_data", false);
    QUERY->end_class(QUERY);
    return TRUE;
}

//-----------------------------------------------------------------------------
// CONSTRUCTOR / DESTRUCTOR
//-----------------------------------------------------------------------------
CK_DLL_CTOR(chonnx_ctor)
{
    ChonnxData* cxdata = new ChonnxData;
    cxdata->initialized = 1;
    cxdata->session = nullptr;
    cxdata->num_inputs = 0;
    cxdata->num_outputs = 0;
    OBJ_MEMBER_INT(SELF, chonnx_data_offset) = (t_CKINT)cxdata;
    std::cout << "[CHONNX] Universal Instance allocated." << std::endl;
}

CK_DLL_DTOR(chonnx_dtor)
{
    ChonnxData* cxdata = (ChonnxData*)OBJ_MEMBER_INT(SELF, chonnx_data_offset);
    if (cxdata) {
        if (cxdata->session) delete cxdata->session;
        delete cxdata;
        OBJ_MEMBER_INT(SELF, chonnx_data_offset) = 0;
    }
    std::cout << "[CHONNX] Universal Instance freed." << std::endl;
}

//-----------------------------------------------------------------------------
// LOAD MODEL — dynamic inspection of nodes, shapes and capacities
//-----------------------------------------------------------------------------
CK_DLL_MFUN(chonnx_load_model)
{
    ChonnxData* cxdata = (ChonnxData*)OBJ_MEMBER_INT(SELF, chonnx_data_offset);
    Chuck_String* ck_str = *(Chuck_String**)ARGS;
    if (!ck_str || !cxdata) return;

    const char* raw_path = API->object->str(ck_str);
    if (!raw_path || strlen(raw_path) == 0) return;

    try {
        if (!g_ort_env)
            g_ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ChonnxEnv");

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

        std::string path_str(raw_path);
        int wlen = MultiByteToWideChar(CP_UTF8, 0, path_str.c_str(), -1, NULL, 0);
        std::wstring wide_path(wlen, 0);
        MultiByteToWideChar(CP_UTF8, 0, path_str.c_str(), -1, &wide_path[0], wlen);

        if (cxdata->session) delete cxdata->session;
        cxdata->session = new Ort::Session(*g_ort_env, wide_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        // --- Inspect Inputs ---
        cxdata->num_inputs = cxdata->session->GetInputCount();
        cxdata->input_names.resize(cxdata->num_inputs);
        cxdata->input_shapes.resize(cxdata->num_inputs);
        cxdata->input_buffers.resize(cxdata->num_inputs);

        std::cout << "[CHONNX] Model inputs discovered: " << cxdata->num_inputs << std::endl;
        for (size_t i = 0; i < cxdata->num_inputs; i++) {
            auto name_ptr = cxdata->session->GetInputNameAllocated(i, allocator);
            cxdata->input_names[i] = std::string(name_ptr.get());

            Ort::TypeInfo type_info = cxdata->session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            cxdata->input_shapes[i] = tensor_info.GetShape();

            size_t input_flat_size = 1;
            for (int64_t dim : cxdata->input_shapes[i]) {
                input_flat_size *= (dim > 0 ? (size_t)dim : 1); // Dynamic dims (-1) treated as 1
            }
            cxdata->input_buffers[i].assign(input_flat_size, 0.0f);

            std::cout << "  -> Input [" << i << "]: name=\"" << cxdata->input_names[i]
                << "\" allocation_size=" << input_flat_size << " floats." << std::endl;
        }

        // --- Inspect Outputs ---
        cxdata->num_outputs = cxdata->session->GetOutputCount();
        cxdata->output_names.resize(cxdata->num_outputs);
        cxdata->output_buffers.resize(cxdata->num_outputs);

        std::cout << "[CHONNX] Model outputs discovered: " << cxdata->num_outputs << std::endl;
        for (size_t i = 0; i < cxdata->num_outputs; i++) {
            auto name_ptr = cxdata->session->GetOutputNameAllocated(i, allocator);
            cxdata->output_names[i] = std::string(name_ptr.get());
            std::cout << "  -> Output [" << i << "]: name=\"" << cxdata->output_names[i] << "\"" << std::endl;
        }

        std::cout << "[CHONNX SUCCESS] Universal Graph mapped and locked." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[CHONNX ERROR] load_model initialization crashed: " << e.what() << std::endl;
        if (cxdata->session) { delete cxdata->session; cxdata->session = nullptr; }
    }
}

//-----------------------------------------------------------------------------
// SET INPUTS (Overloads)
//-----------------------------------------------------------------------------
CK_DLL_MFUN(chonnx_set_input_single)
{
    ChonnxData* cxdata = (ChonnxData*)OBJ_MEMBER_INT(SELF, chonnx_data_offset);
    t_CKINT index = GET_NEXT_INT(ARGS);
    t_CKFLOAT value = GET_NEXT_FLOAT(ARGS);

    if (!cxdata || !cxdata->session || cxdata->num_inputs == 0) return;
    if (index < 0 || (size_t)index >= cxdata->input_buffers[0].size()) return;

    cxdata->input_buffers[0][(size_t)index] = (float)value;
}

CK_DLL_MFUN(chonnx_set_input_multi)
{
    ChonnxData* cxdata = (ChonnxData*)OBJ_MEMBER_INT(SELF, chonnx_data_offset);
    t_CKINT input_idx = GET_NEXT_INT(ARGS);
    t_CKINT sample_idx = GET_NEXT_INT(ARGS);
    t_CKFLOAT value = GET_NEXT_FLOAT(ARGS);

    if (!cxdata || !cxdata->session) return;
    if (input_idx < 0 || (size_t)input_idx >= cxdata->num_inputs) return;
    if (sample_idx < 0 || (size_t)sample_idx >= cxdata->input_buffers[input_idx].size()) return;

    cxdata->input_buffers[input_idx][(size_t)sample_idx] = (float)value;
}

//-----------------------------------------------------------------------------
// RUN — Multi-Tensor Parallel Inference Execution Loop
//-----------------------------------------------------------------------------
CK_DLL_MFUN(chonnx_run)
{
    ChonnxData* cxdata = (ChonnxData*)OBJ_MEMBER_INT(SELF, chonnx_data_offset);
    if (!cxdata || !cxdata->session) {
        RETURN->v_int = 0;
        return;
    }

    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_node_names;

        // Build dynamic container allocations for all active inputs
        for (size_t i = 0; i < cxdata->num_inputs; i++) {
            std::vector<int64_t> runtime_shape = cxdata->input_shapes[i];
            for (auto& dim : runtime_shape) {
                if (dim < 0) dim = 1; // Safeguard dynamic nodes
            }

            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                cxdata->input_buffers[i].data(),
                cxdata->input_buffers[i].size(),
                runtime_shape.data(),
                runtime_shape.size()
            ));
            input_node_names.push_back(cxdata->input_names[i].c_str());
        }

        std::vector<const char*> output_node_names;
        std::vector<Ort::Value> output_tensors;
        for (size_t i = 0; i < cxdata->num_outputs; i++) {
            output_node_names.push_back(cxdata->output_names[i].c_str());
            output_tensors.push_back(Ort::Value{ nullptr });
        }

        Ort::RunOptions run_options;
        cxdata->session->Run(
            run_options,
            input_node_names.data(), input_tensors.data(), cxdata->num_inputs,
            output_node_names.data(), output_tensors.data(), cxdata->num_outputs
        );

        // Extract multi-tensor output arrays into C++ safe local layers
        for (size_t i = 0; i < cxdata->num_outputs; i++) {
            auto out_info = output_tensors[i].GetTensorTypeAndShapeInfo();
            size_t out_count = out_info.GetElementCount();
            cxdata->output_buffers[i].resize(out_count);

            ONNXTensorElementDataType out_type = out_info.GetElementType();
            if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                float* raw_data = output_tensors[i].GetTensorMutableData<float>();
                std::copy(raw_data, raw_data + out_count, cxdata->output_buffers[i].begin());
            }
            else if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                int64_t* raw_data = output_tensors[i].GetTensorMutableData<int64_t>();
                for (size_t j = 0; j < out_count; j++) {
                    cxdata->output_buffers[i][j] = static_cast<float>(raw_data[j]);
                }
            }
        }
        RETURN->v_int = 1;
    }
    catch (const std::exception& e) {
        std::cerr << "[CHONNX RUN CRASH] Inference engine failed: " << e.what() << std::endl;
        RETURN->v_int = 0;
    }
}

//-----------------------------------------------------------------------------
// GET OUTPUTS (Overloads)
//-----------------------------------------------------------------------------
CK_DLL_MFUN(chonnx_get_output_single)
{
    ChonnxData* cxdata = (ChonnxData*)OBJ_MEMBER_INT(SELF, chonnx_data_offset);
    t_CKINT index = GET_NEXT_INT(ARGS);

    if (!cxdata || cxdata->num_outputs == 0 || cxdata->output_buffers[0].empty()) {
        RETURN->v_float = 0.0; return;
    }
    if (index < 0 || (size_t)index >= cxdata->output_buffers[0].size()) {
        RETURN->v_float = 0.0; return;
    }
    RETURN->v_float = (t_CKFLOAT)cxdata->output_buffers[0][(size_t)index];
}

CK_DLL_MFUN(chonnx_get_output_multi)
{
    ChonnxData* cxdata = (ChonnxData*)OBJ_MEMBER_INT(SELF, chonnx_data_offset);
    t_CKINT output_idx = GET_NEXT_INT(ARGS);
    t_CKINT sample_idx = GET_NEXT_INT(ARGS);

    if (!cxdata || (size_t)output_idx >= cxdata->num_outputs || cxdata->output_buffers[output_idx].empty()) {
        RETURN->v_float = 0.0; return;
    }
    if (sample_idx < 0 || (size_t)sample_idx >= cxdata->output_buffers[output_idx].size()) {
        RETURN->v_float = 0.0; return;
    }
    RETURN->v_float = (t_CKFLOAT)cxdata->output_buffers[output_idx][(size_t)sample_idx];
}