#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>

// 辅助函数：将 ONNXTensorElementDataType 转换为字符串
std::string DataTypeToString(ONNXTensorElementDataType type) {
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
std::string ShapeToString(const std::vector<int64_t>& shape) {
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

static size_t ElementSize(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return sizeof(float);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return sizeof(uint8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return sizeof(int8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return sizeof(uint16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return sizeof(int16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return sizeof(int32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return sizeof(int64_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return sizeof(bool);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return sizeof(double);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return sizeof(uint32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return sizeof(uint64_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return sizeof(uint16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return sizeof(uint16_t);
        default: return 0;
    }
}

static std::vector<int64_t> ResolveShape(const std::vector<int64_t>& shape, int64_t dynamic_dim = 1) {
    std::vector<int64_t> resolved = shape;
    for (auto& d : resolved) {
        if (d <= 0) {
            d = dynamic_dim;
        }
    }
    return resolved;
}

// 打印模型输入信息
void PrintInputInfo(Ort::Session& session) {
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
void PrintOutputInfo(Ort::Session& session) {
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

// 完整示例
int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <model.onnx> [warmup=5] [iters=50]" << std::endl;
            return 1;
        }
        const std::string model_path = argv[1];
        const int warmup = (argc > 2) ? std::stoi(argv[2]) : 5;
        const int iters = (argc > 3) ? std::stoi(argv[3]) : 50;

        // 初始化环境
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelInspector");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // 加载模型
        Ort::Session session(env, model_path.c_str(), session_options);
        
        // 打印输入输出信息
        PrintInputInfo(session);
        PrintOutputInfo(session);

        // 构建输入
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        const size_t num_inputs = session.GetInputCount();
        std::vector<Ort::Value> input_tensors;
        std::vector<std::vector<int64_t>> input_shapes;
        std::vector<std::vector<uint8_t>> input_buffers;
        std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
        std::vector<const char*> input_names;

        input_tensors.reserve(num_inputs);
        input_shapes.reserve(num_inputs);
        input_name_ptrs.reserve(num_inputs);
        input_names.reserve(num_inputs);
        input_buffers.reserve(num_inputs);

        for (size_t i = 0; i < num_inputs; ++i) {
            input_name_ptrs.emplace_back(session.GetInputNameAllocated(i, allocator));
            input_names.push_back(input_name_ptrs.back().get());

            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto elem_type = tensor_info.GetElementType();

            auto raw_shape = tensor_info.GetShape();
            auto resolved_shape = ResolveShape(raw_shape, 1);
            input_shapes.push_back(resolved_shape);

            int64_t numel = 1;
            for (auto d : resolved_shape) {
                numel *= d;
            }

            const size_t elem_size = ElementSize(elem_type);
            if (numel <= 0 || elem_size == 0) {
                throw std::runtime_error("Unsupported input tensor type or invalid shape.");
            }
            input_buffers.emplace_back(static_cast<size_t>(numel) * elem_size, 0);
            void* buffer = input_buffers.back().data();

            switch (elem_type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                    float* p = reinterpret_cast<float*>(buffer);
                    std::fill(p, p + numel, 0.01f);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
                    double* p = reinterpret_cast<double*>(buffer);
                    std::fill(p, p + numel, 0.01);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                    int64_t* p = reinterpret_cast<int64_t*>(buffer);
                    std::fill(p, p + numel, 1);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                    int32_t* p = reinterpret_cast<int32_t*>(buffer);
                    std::fill(p, p + numel, 1);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
                    uint8_t* p = reinterpret_cast<uint8_t*>(buffer);
                    std::fill(p, p + numel, 1);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
                    bool* p = reinterpret_cast<bool*>(buffer);
                    std::fill(p, p + numel, true);
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported input tensor data type for demo.");
            }

            input_tensors.emplace_back(Ort::Value::CreateTensor(
                memory_info, buffer, input_buffers.back().size(), resolved_shape.data(), resolved_shape.size(), elem_type));
        }

        // 输出名
        const size_t num_outputs = session.GetOutputCount();
        std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
        std::vector<const char*> output_names;
        output_name_ptrs.reserve(num_outputs);
        output_names.reserve(num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            output_name_ptrs.emplace_back(session.GetOutputNameAllocated(i, allocator));
            output_names.push_back(output_name_ptrs.back().get());
        }

        // 预热
        for (int i = 0; i < warmup; ++i) {
            auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
                                       input_tensors.size(), output_names.data(), output_names.size());
            (void)outputs;
        }

        // 计时推理
        std::vector<double> times_ms;
        times_ms.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
                                       input_tensors.size(), output_names.data(), output_names.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dt = t1 - t0;
            times_ms.push_back(dt.count());
            (void)outputs;
        }

        double avg = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
        std::cout << "Average inference time: " << avg << " ms (" << iters << " runs)" << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}