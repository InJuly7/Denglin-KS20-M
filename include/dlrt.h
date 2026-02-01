#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <dlnne/dlnne.h>
#include "op_gpu.h"
namespace dlrt {

struct EngineContext {
    dl::nne::Engine* engine = nullptr;
    dl::nne::ExecutionContext* context = nullptr;
    int64_t slz_length = 0;
};

struct Buffer {
    float* device_ptr = nullptr;
    size_t size = 0;
    std::vector<int> shape;
    std::string name = "";
    bool is_input = true;
    int index = -1;
};

struct dlrtParam {
    std::string model_name = "";
    std::string onnx_path = "";
    std::string rlym_path = "";
    std::string quantize_rlym_path = "";
    std::string mlir_path = "";
    std::string quantize_mlir_path = "";
    std::string engine_path = "";
    std::string quantize_engine_path = "";
    int64_t slz_length = 0;
    struct EngineContext engine_context;
    std::vector<Buffer> input_buffers;
    std::vector<Buffer> output_buffers;
    int input_num = 0;
    int output_num = 0;
    std::vector<float*> bindings;
    bool is_debug = false;
};

inline void PrintBuffers(const std::vector<Buffer>& buffers, const std::string& title = "buffers") {
    std::cout << title << " size: " << buffers.size() << std::endl;
    for (size_t i = 0; i < buffers.size(); ++i) {
        const auto& buffer = buffers[i];
        std::cout << "[" << i << "] name=" << buffer.name
                  << " is_input=" << buffer.is_input
                  << " index=" << buffer.index
                  << " size=" << buffer.size
                  << " device_ptr=" << buffer.device_ptr
                  << " dims=[";
        for (size_t d = 0; d < buffer.shape.size(); ++d) {
            std::cout << buffer.shape[d];
            if (d + 1 < buffer.shape.size()) std::cout << ",";
        }
        std::cout << "]" << std::endl;

    }
}

inline void BufferInit(dlrtParam& param, const EngineContext& engine_context, int input_num, int output_num) {
    // 输入输出buffer初始化
    int nb_bindings = engine_context.engine->GetNbBindings();
    int bidding_num = input_num + output_num;
    assert(nb_bindings == bidding_num);
    for (int i = 0; i < nb_bindings; ++i) {
        auto shape = engine_context.engine->GetBindingDimensions(i);
        auto name = engine_context.engine->GetBindingName(i);
        auto data_type = engine_context.engine->GetBindingDataType(i);
        size_t size = 1;
        Buffer buffer;
        buffer.name = name;
        // 记录是否是输入 tensor, 以及第几个输入 tensor
        buffer.is_input = (i < input_num) ? true : false;
        buffer.index = (i < input_num) ? i : (i - input_num);
        buffer.shape.resize(shape.nbDims);
        for (int j = 0; j < shape.nbDims; ++j) {
            buffer.shape[j] = shape.d[j];
            size *= shape.d[j];
        }
        int dtype_size = 1;
        if (data_type == dl::nne::kFLOAT32) {
            dtype_size = 4;
        }
        buffer.size = size * dtype_size;
        CHECK(cudaMalloc(&buffer.device_ptr, buffer.size));
        param.bindings.push_back(buffer.device_ptr);
        if (buffer.is_input) {
            param.input_buffers.emplace_back(std::move(buffer));
        } else {
            param.output_buffers.emplace_back(std::move(buffer));
        }
    }
    // PrintBuffers(param.input_buffers, "Input Buffers");
    // PrintBuffers(param.output_buffers, "Output Buffers");
}
inline void EngineContextInit(EngineContext& engine_context, const std::string& rlym_file, const std::string& engine_file,
                              int64_t& slz_length, bool is_debug = false) {
    std::ifstream slz(engine_file, std::ios::in | std::ios::binary);
    // 如果引擎文件不存在，则从rlym文件构建引擎并序列化保存
    if (!slz.is_open()) {
        if (is_debug) std::cout << "Build serialize engine " << std::endl;
        auto builder = dl::nne::CreateInferBuilder();
        auto network = builder->CreateNetwork();
        auto parser = dl::nne::CreateParser();
        parser->Parse(rlym_file.c_str(), *network);
        dl::nne::Engine* engine = nullptr;
        engine = builder->BuildEngine(*network);
        parser->Destroy();
        network->Destroy();
        builder->Destroy();
        auto ser_res = engine->Serialize();
        std::ofstream new_slz(engine_file);
        new_slz.write(static_cast<char*>(ser_res->Data()), static_cast<int64_t>(ser_res->Size()));
        new_slz.close();
        ser_res->Destroy();
        slz.open(engine_file, std::ios::in | std::ios::binary);
    }
    // 读取序列化引擎文件
    if (is_debug) std::cout << "Load engine from file: " << engine_file << std::endl;
    slz.seekg(0, std::ios::end);
    slz_length = static_cast<uint64_t>(slz.tellg());
    slz.seekg(0, std::ios::beg);
    std::vector<uint8_t> slz_data;
    slz_data.resize(slz_length);
    slz.read(reinterpret_cast<char*>(slz_data.data()), static_cast<int64_t>(slz_length));
    slz.close();

    engine_context.engine = dl::nne::Deserialize(reinterpret_cast<char*>(slz_data.data()), slz_length);
    assert(engine_context.engine != nullptr);
    if (is_debug) std::cout << "Model Deserialize finished" << std::endl;

    engine_context.context = engine_context.engine->CreateExecutionContext();
    assert(engine_context.context != nullptr);
    if (is_debug) std::cout << "Model CreateExecutionContext finished" << std::endl;
    engine_context.slz_length = slz_length;
}

inline void dlrtInit(const std::string& model_name, dlrtParam& param, int input_num, int output_num, bool is_debug = false) {
    const char* kModelDir = "/data/model/yolov11s_seg/";
    param.model_name = model_name;
    param.onnx_path = kModelDir + model_name + ".onnx";
    param.rlym_path = kModelDir + model_name + ".rlym";
    param.quantize_rlym_path = kModelDir + model_name + ".quantized.rlym";
    param.mlir_path = kModelDir + model_name + ".mlir";
    param.quantize_mlir_path = kModelDir + model_name + ".quantized.mlir";
    param.engine_path = kModelDir + model_name + ".slz";
    param.quantize_engine_path = kModelDir + model_name + ".quantized.slz";
    param.is_debug = is_debug;
    param.input_num = input_num;
    param.output_num = output_num;
    EngineContextInit(param.engine_context, param.rlym_path, param.engine_path, param.slz_length, param.is_debug);
    BufferInit(param, param.engine_context, param.input_num, param.output_num);
}

inline void BufferFree(dlrtParam& param) {
    for (auto& input_buffer : param.input_buffers) {
        if (input_buffer.device_ptr) {
            cudaFree(input_buffer.device_ptr);
            input_buffer.device_ptr = nullptr;
        }
    }
    for (auto& output_buffer : param.output_buffers) {
        if (output_buffer.device_ptr) {
            cudaFree(output_buffer.device_ptr);
            output_buffer.device_ptr = nullptr;
        }
    }
}

inline void check_engine_info(const EngineContext& engine_context) {
    std::cout << "the engine's info:" << std::endl;
    int nb_bindings = engine_context.engine->GetNbBindings();
    for (int i = 0; i < nb_bindings; ++i) {
        auto shape = engine_context.engine->GetBindingDimensions(i);
        auto name = engine_context.engine->GetBindingName(i);
        auto data_type = engine_context.engine->GetBindingDataType(i);
        std::cout << name << "  " << data_type << std::endl;
        for (int j = 0; j < shape.nbDims; ++j) {
            std::cout << shape.d[j] << "  ";
        }
        std::cout << std::endl;
    }

    // auto input_dims = engine_context.context->GetBindingDimensions(0);
    // assert(batch_size == input_dims.d[0]);
    // assert(channel == input_dims.d[1]);
    // assert(dst_h == input_dims.d[2]);
    // assert(dst_w == input_dims.d[3]);

    // auto output_dims = engine_context.context->GetBindingDimensions(1);
    // assert(mask_channel == output_dims.d[1]);
    // assert(mask_grid_h == output_dims.d[2]);
    // assert(mask_grid_w == output_dims.d[3]);

    // auto boxes_dims = engine_context.context->GetBindingDimensions(2);
    // assert(boxes_num == boxes_dims.d[1]);
    // assert(boxes_width == boxes_dims.d[2]);
}

inline bool infer(const dlrtParam& param) {
    // 执行推理
    if(param.is_debug) PrintBuffers(param.input_buffers, "Input Buffers before infer");
    if(param.is_debug) PrintBuffers(param.output_buffers, "Output Buffers before infer");
    if (param.is_debug)
        std::cout << "bingdings: " << param.bindings[0] << " " << param.bindings[1] << " " << param.bindings[2] << std::endl;
    bool success = param.engine_context.context->Execute(1, (void**)param.bindings.data());
    assert(success == true);
    if(param.is_debug) std::cout << "Inference executed successfully." << std::endl;
    return success;
}

}  // namespace dlrt