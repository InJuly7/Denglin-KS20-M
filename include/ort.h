#pragma once

#include <memory>

#include <onnxruntime_cxx_api.h>

namespace ort {

struct SessionContext {
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info;

    explicit SessionContext(const char* name = "SAM2")
        : env(ORT_LOGGING_LEVEL_WARNING, name), session_options(), session(nullptr),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}
};

}  // namespace ort