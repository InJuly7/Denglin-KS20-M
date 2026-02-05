#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace ort {

struct Buffer {
    size_t size = 0;  // bytes
    std::vector<int64_t> shape;
    std::string name = "";
    bool is_input = true;
    int index = -1;
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    bool is_dynamic = false;
    bool is_from_gpu = false;
    float* device_ptr = nullptr;
    bool external = false;  // true if memory is owned externally
};

struct SessionContext {
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session{nullptr};
    Ort::RunOptions run_options;
};

struct OrtParam {
    OrtParam()
        : model_name(""),
          model_path(""),
          session_context(),
          allocator(),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          input_shapes(),
          input_tensors(),
          input_names(),
          output_names(),
          input_name_storage(),
          output_name_storage(),
          input_buffers(),
          output_buffers(),
          input_num(0),
          output_num(0),
          is_debug(false),
          is_dynamic(true) {}

    std::string model_name = "";
    std::string model_path = "";
    SessionContext session_context;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info{nullptr};
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<Ort::Value> input_tensors;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::string> input_name_storage;
    std::vector<std::string> output_name_storage;
    
    std::vector<Buffer> input_buffers;
    std::vector<Buffer> output_buffers;    
    int input_num = 0;
    int output_num = 0;
    bool is_debug = false;
    bool is_dynamic = true;
};

static size_t ElementSize(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return sizeof(float);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return sizeof(uint8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return sizeof(int8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return sizeof(uint16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return sizeof(int16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return sizeof(int32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return sizeof(int64_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return sizeof(bool);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return sizeof(double);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return sizeof(uint32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return sizeof(uint64_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return sizeof(uint16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return sizeof(uint16_t);
        default:
            return 0;
    }
}

// 辅助函数：将 ONNXTensorElementDataType 转换为字符串
inline std::string DataTypeToString(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "string";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
        default: return "unknown";
    }
}

// 辅助函数：将 shape 转换为字符串（支持动态维度）
inline std::string ShapeToString(const std::vector<int64_t>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            result += "dynamic";  // 动态维度
        } else {
            result += std::to_string(shape[i]);
        }
        if (i < shape.size() - 1) {
            result += ", ";
        }
    }
    result += "]";
    return result;
}

// 打印模型输入信息
inline void PrintInputInfo(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session.GetInputCount();
    
    std::cout << "=== Model Input Information ===" << std::endl;
    std::cout << "Number of inputs: " << num_input_nodes << std::endl << std::endl;
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        // 获取输入名称
        auto input_name = session.GetInputNameAllocated(i, allocator);
        std::cout << "Input " << i << ":" << std::endl;
        std::cout << "  Name: " << input_name.get() << std::endl;
        
        // 获取输入类型信息
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        // 获取数据类型
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::cout << "  Data Type: " << DataTypeToString(type) << std::endl;
        
        // 获取shape
        std::vector<int64_t> shape = tensor_info.GetShape();
        std::cout << "  Shape: " << ShapeToString(shape) << std::endl;
        
        // 获取维度数量
        size_t num_dims = tensor_info.GetDimensionsCount();
        std::cout << "  Number of Dimensions: " << num_dims << std::endl;
        
        std::cout << std::endl;
    }
}

// 打印模型输出信息
inline void PrintOutputInfo(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_output_nodes = session.GetOutputCount();
    
    std::cout << "=== Model Output Information ===" << std::endl;
    std::cout << "Number of outputs: " << num_output_nodes << std::endl << std::endl;
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        // 获取输出名称
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        std::cout << "Output " << i << ":" << std::endl;
        std::cout << "  Name: " << output_name.get() << std::endl;
        
        // 获取输出类型信息
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        // 获取数据类型
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::cout << "  Data Type: " << DataTypeToString(type) << std::endl;
        
        // 获取shape
        std::vector<int64_t> shape = tensor_info.GetShape();
        std::cout << "  Shape: " << ShapeToString(shape) << std::endl;
        
        // 获取维度数量
        size_t num_dims = tensor_info.GetDimensionsCount();
        std::cout << "  Number of Dimensions: " << num_dims << std::endl;
        
        std::cout << std::endl;
    }
}

inline size_t TensorByteSize(const std::vector<int64_t>& shape, ONNXTensorElementDataType type) {
    size_t elems = 1;
    for (auto d : shape) {
        elems *= static_cast<size_t>(d);
    }
    return elems * ElementSize(type);
}

inline void PrintBuffers(const std::vector<Buffer>& buffers, const std::string& title = "buffers") {
    std::cout << title << " size: " << buffers.size() << std::endl;
    for (size_t i = 0; i < buffers.size(); ++i) {
        const auto& buffer = buffers[i];
        std::cout << "[" << i << "] name=" << buffer.name << " is_input=" << buffer.is_input
                  << " index=" << buffer.index << " size=" << buffer.size
                  << " device_ptr=" << buffer.device_ptr << " dims=[";
        for (size_t d = 0; d < buffer.shape.size(); ++d) {
            std::cout << buffer.shape[d];
            if (d + 1 < buffer.shape.size()) std::cout << ",";
        }
        std::cout << "]" << std::endl;
    }
}

inline void SessionContextInit(SessionContext& session_context, const std::string& model_path, bool is_debug = false) {
    session_context.run_options = Ort::RunOptions{};
    session_context.env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ModelInspector");
    session_context.session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);    
    session_context.session = Ort::Session(session_context.env, model_path.c_str(), session_context.session_options);
}


inline std::vector<int64_t> ResolveShape(const std::vector<int64_t>& shape, int64_t dynamic_dim = 1) {
    std::vector<int64_t> resolved = shape;
    for (auto& d : resolved) {
        if (d <= 0) {
            d = dynamic_dim;
        }
    }
    return resolved;
}

inline bool is_dynamic_shape(const std::vector<int64_t>& shape) {
    for (auto d : shape) {
        if (d <= 0) {
            return true;
        }
    }
    return false;
}

inline void BufferInit(OrtParam& param, std::vector<std::vector<int64_t>> shape, std::vector<bool> is_dynamic, std::vector<bool> is_from_gpu,
                       std::vector<float*> input_ptrs) {
    // ORT 推理静态模型
    if(param.is_dynamic) {
        // 待完成
    } else {
        // ORT 推理动态模型, 如sam2 decoder
        param.input_shapes.clear();
        param.input_tensors.clear();
        param.input_names.clear();
        param.output_names.clear();
        param.input_name_storage.clear();
        param.output_name_storage.clear();
        param.input_buffers.clear();
        assert(param.input_num == is_dynamic.size());
        assert(param.input_num == is_from_gpu.size());
        assert(param.input_num == input_ptrs.size());
        assert(param.input_num == param.session_context.session.GetInputCount());
        int num_inputs = param.input_num;
        param.input_name_storage.reserve(num_inputs);
        param.input_names.reserve(num_inputs);
        for (int i = 0; i < num_inputs; ++i) {
            Ort::AllocatedStringPtr name_ptr = param.session_context.session.GetInputNameAllocated(i, param.allocator);
            Buffer buffer;
            if (!name_ptr || name_ptr.get() == nullptr || std::strlen(name_ptr.get()) == 0) {
                std::cerr << "BufferInit input name is empty at index " << i << std::endl;
            }
            param.input_name_storage.emplace_back(name_ptr.get());
            buffer.name = param.input_name_storage.back();
            buffer.is_input = true;
            buffer.index = i;
            buffer.is_from_gpu = is_from_gpu[i];
            param.input_names.push_back(param.input_name_storage.back().c_str());

            auto type_info = param.session_context.session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto elem_type = tensor_info.GetElementType();
            auto raw_shape = tensor_info.GetShape();
            buffer.type = elem_type;
            std::vector<int64_t> resolved_shape;
            if(is_dynamic_shape(raw_shape)) { // 动态shape输入
                assert(is_dynamic[i]);
                buffer.is_dynamic = true;
                for(size_t j = 0; j < raw_shape.size(); ++j) {
                    if (raw_shape[j] > 0) {
                        assert(raw_shape[j] == shape[i][j]);
                        resolved_shape.push_back(raw_shape[j]);
                    } else {
                        assert(shape[i][j] > 0);
                        assert(raw_shape[j] <= 0);
                        resolved_shape.push_back(shape[i][j]);
                    }
                }
            } else { // 静态shape输入
                resolved_shape = raw_shape;
                buffer.is_dynamic = false;
            }
            param.input_shapes.push_back(resolved_shape);
            int64_t numel = 1;
            for (auto d : resolved_shape) {
                numel *= d;
            }
            const size_t elem_size = ElementSize(elem_type);
            buffer.size = static_cast<size_t>(numel) * elem_size;
            if(is_from_gpu[i] && input_ptrs[i] != nullptr) {
                assert(input_ptrs[i] != nullptr);
                buffer.device_ptr = (float*)malloc(buffer.size);
                buffer.external = false;
                cudaMemcpy(buffer.device_ptr, input_ptrs[i], buffer.size, cudaMemcpyDeviceToHost);
            } else if (is_from_gpu[i] == false && input_ptrs[i] != nullptr) {
                buffer.device_ptr = input_ptrs[i];
                buffer.external = true;
            } else {
                std::cerr << "BufferInit input buffer error!" << std::endl;
            }

            param.input_tensors.emplace_back(Ort::Value::CreateTensor(param.memory_info, buffer.device_ptr, buffer.size, resolved_shape.data(),
                                                                resolved_shape.size(), elem_type));
            param.input_buffers.emplace_back(std::move(buffer));
        }

        const size_t num_outputs = param.session_context.session.GetOutputCount();
        assert(param.output_num == num_outputs);
        param.output_name_storage.reserve(num_outputs);
        param.output_names.reserve(num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            Ort::AllocatedStringPtr name_ptr = param.session_context.session.GetOutputNameAllocated(i, param.allocator);
            if (!name_ptr || name_ptr.get() == nullptr || std::strlen(name_ptr.get()) == 0) {
                std::cerr << "BufferInit output name is empty at index " << i << std::endl;
            }
            param.output_name_storage.emplace_back(name_ptr.get());
            param.output_names.push_back(param.output_name_storage.back().c_str());
        }
    }
}

inline std::string NormalizeModelDir(const std::string& model_dir) {
    std::string dir = model_dir.empty() ? "/data/model/" : model_dir;
    if (!dir.empty() && dir.front() != '/') {
        dir = std::string("/data/model/") + dir;
    }
    if (!dir.empty() && dir.back() != '/') {
        dir.push_back('/');
    }
    return dir;
}

inline void OrtInit(const std::string& model_name, OrtParam& param, const int input_num, const int output_num, bool is_debug = false,
                    bool is_dynamic = true, const std::string& model_dir = "") {
    const std::string kModelDir = NormalizeModelDir(model_dir);
    param.model_name = model_name;
    param.model_path = kModelDir + model_name + ".onnx";
    param.input_num = input_num;
    param.output_num = output_num;
    param.is_debug = is_debug;
    param.is_dynamic = is_dynamic;
    param.memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    SessionContextInit(param.session_context, param.model_path, is_debug);
    // 打印输入输出信息
    PrintInputInfo(param.session_context.session);
    PrintOutputInfo(param.session_context.session);
}

inline void infer(OrtParam& param) {
    auto outputs =
        param.session_context.session.Run(param.session_context.run_options, param.input_names.data(), param.input_tensors.data(),
                                          param.input_tensors.size(), param.output_names.data(), param.output_names.size());

    // 清理旧输出
    for (auto& buffer : param.output_buffers) {
        if (!buffer.external && buffer.device_ptr) {
            free(buffer.device_ptr);
            buffer.device_ptr = nullptr;
        }
    }
    param.output_buffers.clear();

    const size_t num_outputs = outputs.size();
    param.output_buffers.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto& output = outputs[i];
        auto tensor_info = output.GetTensorTypeAndShapeInfo();
        auto elem_type = tensor_info.GetElementType();
        auto shape = tensor_info.GetShape();
        const size_t byte_size = TensorByteSize(shape, elem_type);

        Buffer buffer;
        buffer.is_input = false;
        buffer.index = static_cast<int>(i);
        if (i < param.output_names.size() && param.output_names[i]) {
            buffer.name = param.output_names[i];
        }
        buffer.type = elem_type;
        buffer.shape = shape;
        buffer.size = byte_size;
        buffer.is_from_gpu = false;
        buffer.is_dynamic = is_dynamic_shape(shape);
        buffer.external = false;

        void* src = output.GetTensorMutableData<void>();
        buffer.device_ptr = static_cast<float*>(malloc(byte_size));
        if (buffer.device_ptr && src) {
            std::memcpy(buffer.device_ptr, src, byte_size);
        }
        param.output_buffers.emplace_back(std::move(buffer));
    }
}

inline void BufferFree(OrtParam& param) {
    for (auto& buffer : param.input_buffers) {
        if (!buffer.external && buffer.device_ptr) {
            free(buffer.device_ptr);
            buffer.device_ptr = nullptr;
        }
    }
    for (auto& buffer : param.output_buffers) {
        if (!buffer.external && buffer.device_ptr) {
            free(buffer.device_ptr);
            buffer.device_ptr = nullptr;
        }
    }
}

}  // namespace ort