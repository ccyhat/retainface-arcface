#include "face_rec.h"
#include<thread>
int ARCFACE::LoadModel(const std::string& model_dir) {
    try {

        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "FaceRecognization");
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
int ARCFACE::Run(std::vector<cv::Mat>& imgs, std::vector<FACEPredictResult>& FACEres) {
    float ratio{};
    int x_off{};
    int y_off{};
    std::vector<Ort::Value> input_tensors;
    auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
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
    std::vector<cv::Mat>resize_imgs;
    for (auto img : imgs) {
        cv::Mat resize_img;
        img.copyTo(resize_img);
        
        this->resize_op_.Run(img, resize_img, this->img_size,
            ratio, x_off, y_off);
        this->normalize_op_.Run(&resize_img);
        resize_imgs.push_back(resize_img);
    }
    std::vector<float> srcInputTensorValues(imgs.size() * 3 * resize_imgs[0].rows * resize_imgs[0].cols, 0.0f);
    this->permute_op_.Run(resize_imgs, srcInputTensorValues.data());

    InputNodeShapes[0] = {(int)imgs.size(),3,resize_imgs[0].rows,resize_imgs[0].cols };
   
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, srcInputTensorValues.data(), srcInputTensorValues.size(), \
        InputNodeShapes[0].data(), InputNodeShapes[0].size()
    ));

    std::vector<char*> OutputNodeNames;
    std::vector<std::vector<int64_t>> OutputNodeShapes;
    const size_t numOutputNodes = session->GetOutputCount();
    OutputNodeNames.reserve(numOutputNodes);
    for (size_t i = 0; i < numOutputNodes; i++)
    {
        OutputNodeNames.emplace_back(strdup(session->GetOutputNameAllocated(i, allocator).get()));
        OutputNodeShapes.emplace_back(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    // Inference.
    auto output_tensors = session->Run(Ort::RunOptions{ nullptr }, InputNodeNames.data(), input_tensors.data(), \
        input_tensors.size(), OutputNodeNames.data(), OutputNodeNames.size());
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    float* pOutputData = (float*)output_tensors[0].GetTensorMutableData<float>();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,std::multiplies<int>());
    cv::Mat out_data(output_shape[0], output_shape[1], CV_32FC1, pOutputData);

    for (int i = 0; i < out_data.rows; i++) {
        // 取出当前特征并归一化
        cv::Mat tempfeature = out_data.row(i); // 1x128
        cv::normalize(tempfeature, tempfeature);

        float best_score = -1.0f;
        std::string best_name = "unknown";
        for (auto& it : this->feature) {
            // 保证 dbfeature 也是 1x128 行向量
            cv::Mat dbfeature = cv::Mat(it.feature);
            if (dbfeature.rows != 1) dbfeature = dbfeature.reshape(1, 1);
            cv::normalize(dbfeature, dbfeature);

            // 计算余弦相似度
            float cosine = tempfeature.dot(dbfeature);
            if (cosine > best_score) {
                best_score = cosine;
                best_name = it.name;
            }
        }
        // 阈值可根据实际情况调整（如 0.4~0.5）
        if (best_score > 0.5f) {
            FACEres[i].face_name = best_name;
            FACEres[i].score = best_score;
        } else {
            FACEres[i].face_name = "unknown";
            FACEres[i].score = best_score;
        }
    }
    return 0;
}
void ARCFACE::GetFeature(std::vector<std::string>& path,std::vector<cv::Mat> imgs) {
    float ratio{};
    int x_off{};
    int y_off{};
    std::vector<Ort::Value> input_tensors;
    auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
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
    std::vector<cv::Mat>resize_imgs;
    for (auto img : imgs) {
        cv::Mat resize_img;
        img.copyTo(resize_img);

        this->resize_op_.Run(img, resize_img, this->img_size,
            ratio, x_off, y_off);
        this->normalize_op_.Run(&resize_img);
        resize_imgs.push_back(resize_img);
    }
    std::vector<float> srcInputTensorValues(imgs.size() * 3 * resize_imgs[0].rows * resize_imgs[0].cols, 0.0f);
    this->permute_op_.Run(resize_imgs, srcInputTensorValues.data());

    InputNodeShapes[0] = { (int)imgs.size(),3,resize_imgs[0].rows,resize_imgs[0].cols };


    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, srcInputTensorValues.data(), srcInputTensorValues.size(), \
        InputNodeShapes[0].data(), InputNodeShapes[0].size()
    ));
  
    std::vector<char*> OutputNodeNames;
    std::vector<std::vector<int64_t>> OutputNodeShapes;
    const size_t numOutputNodes = session->GetOutputCount();
    OutputNodeNames.reserve(numOutputNodes);
    for (size_t i = 0; i < numOutputNodes; i++)
    {
        OutputNodeNames.emplace_back(strdup(session->GetOutputNameAllocated(i, allocator).get()));
        OutputNodeShapes.emplace_back(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    // Inference.
    auto output_tensors = session->Run(Ort::RunOptions{ nullptr }, InputNodeNames.data(), input_tensors.data(), \
        input_tensors.size(), OutputNodeNames.data(), OutputNodeNames.size());
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    float* pOutputData = (float*)output_tensors[0].GetTensorMutableData<float>();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    cv::Mat out_data(output_shape[0], output_shape[1], CV_32FC1, pOutputData);
    for (int i = 0; i < out_data.rows; i++) {
        FaceData obj;
        obj.id = i;
        std::string temp= Utility::basename(path[i]);
        size_t dotPos = temp.find_last_of('.');
        if (dotPos != std::string::npos) {
            // 去掉点及其后面的部分
            temp = temp.substr(0, dotPos);
        }
        obj.name = temp;
        obj.feature = out_data.rowRange(i,i+1);
        this->feature.push_back(obj);
    }
}