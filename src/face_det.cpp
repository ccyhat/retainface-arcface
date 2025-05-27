#include "face_det.h"
#include<thread>
int RETINA::LoadModel(const std::string& model_dir) {
    try {

        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "FaceDetection");
        session_options = Ort::SessionOptions();
        session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
       
        session = std::make_unique<Ort::Session>(env, model_dir.c_str(), session_options);

 

        std::cout << "[INFO] ONNXRuntime environment created successfully." << std::endl;
    }
    catch (const std::exception& ex) {
        std::cerr << "[ERROR] ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
int RETINA::Run(cv::Mat& img, std::vector<FACEPredictResult>& FACEres,
    std::vector<double>& times) {
  

    cv::Mat resize_img;
    img.copyTo(resize_img);
    auto preprocess_start = std::chrono::steady_clock::now();
    this->normalize_op_.Run(&resize_img,this->mean_);
    std::vector<float> srcInputTensorValues(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
    this->permute_op_.Run(&resize_img, srcInputTensorValues.data());
    auto preprocess_end = std::chrono::steady_clock::now();

    const size_t numInputNodes = this->session->GetInputCount();
    std::vector<char*> InputNodeNames;
    std::vector<std::vector<int64_t>> InputNodeShapes;
    Ort::AllocatorWithDefaultOptions allocator;
    InputNodeNames.reserve(numInputNodes);
    for (size_t i = 0; i < numInputNodes; i++)
    {
        InputNodeNames.emplace_back(strdup(session->GetInputNameAllocated(i, allocator).get()));
        InputNodeShapes.emplace_back(session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    InputNodeShapes[0] = { 1,3,resize_img.rows,resize_img.cols };
    std::vector<char*> OutputNodeNames;
    std::vector<std::vector<int64_t>> OutputNodeShapes;
    const size_t numOutputNodes = session->GetOutputCount();
    OutputNodeNames.reserve(numOutputNodes);
    for (size_t i = 0; i < numOutputNodes; i++)
    {
        OutputNodeNames.emplace_back(strdup(session->GetOutputNameAllocated(i, allocator).get()));
        OutputNodeShapes.emplace_back(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
    auto inference_start = std::chrono::steady_clock::now();
    // Inference.
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, srcInputTensorValues.data(), srcInputTensorValues.size(), \
        InputNodeShapes[0].data(), InputNodeShapes[0].size()
    ));
    auto output_tensors = session->Run(Ort::RunOptions{ nullptr }, InputNodeNames.data(), input_tensors.data(), \
        input_tensors.size(), OutputNodeNames.data(), OutputNodeNames.size());
    std::vector<cv::Mat> out_datas;
    for (size_t i = 0; i < numOutputNodes; i++)
    {
        auto output_shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        float* pOutputData = (float*)output_tensors[i].GetTensorMutableData<float>();
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
            std::multiplies<int>());
        cv::Mat out_data(output_shape[1], output_shape[2], CV_32FC1, pOutputData);
        out_datas.push_back(out_data);
    }
   
  

    
    auto inference_end = std::chrono::steady_clock::now();

    auto postprocess_start = std::chrono::steady_clock::now();

    post_processor_.BoxesFromRETINA(out_datas, FACEres, this->confidence, this->NMSthreshold, img.cols, img.rows);

    auto postprocess_end = std::chrono::steady_clock::now();

    std::chrono::duration<float> preprocess_diff =
        preprocess_end - preprocess_start;
    times.push_back(double(preprocess_diff.count() * 1000));
    std::chrono::duration<float> inference_diff = inference_end - inference_start;
    times.push_back(double(inference_diff.count() * 1000));
    std::chrono::duration<float> postprocess_diff =
        postprocess_end - postprocess_start;
    times.push_back(double(postprocess_diff.count() * 1000));
    return 0;
}