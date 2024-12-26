#include <iostream>
#include <vector>

// Se no PyTorch >= 2.0 algumas cabeçalhos mudaram, mas normalmente:
#include <torch/script.h> // Para torch::jit::load, etc.
#include <torch/torch.h>

// DBoW2
#include "DBoW2.h"
#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h> // defines macros CVXX
#include <DVision/DVision.h>

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace DBoW2;
using namespace DUtils;
using namespace std;

// ---------------------------------------------------------
// Função nms: extrai pontos a partir de uma matriz 'det' e
// descript. Implementação permanece inalterada
// ---------------------------------------------------------
void nms(cv::Mat det, cv::Mat desc, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors,
         int border, int dist_thresh, int img_width, int img_height)
{
    std::vector<cv::Point2f> pts_raw;
    for (int i = 0; i < det.rows; i++){
        int u = (int) det.at<float>(i, 0);
        int v = (int) det.at<float>(i, 1);
        pts_raw.push_back(cv::Point2f(u, v));
    }

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    grid.setTo(0);
    inds.setTo(0);

    for (int i = 0; i < (int)pts_raw.size(); i++)
    {
        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;
        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;
    }

    cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

    for (int i = 0; i < (int)pts_raw.size(); i++)
    {
        int uu = (int) pts_raw[i].x + dist_thresh;
        int vv = (int) pts_raw[i].y + dist_thresh;
        if (grid.at<char>(vv, uu) != 1) continue;
        for(int k = -dist_thresh; k < (dist_thresh+1); k++){
            for(int j = -dist_thresh; j < (dist_thresh+1); j++){
                if(j==0 && k==0) continue;
                grid.at<char>(vv + k, uu + j) = 0;
            }
        }
        grid.at<char>(vv, uu) = 2;
    }

    // Coletar pontos válidos
    std::vector<int> select_indice;
    for (int v = 0; v < (img_height + dist_thresh); v++){
        for (int u = 0; u < (img_width + dist_thresh); u++){
            if ( (u-dist_thresh) >= (img_width - border) || (u-dist_thresh) < border ||
                 (v-dist_thresh) >= (img_height - border) || (v-dist_thresh) < border)
            {
                continue;
            }
            if (grid.at<char>(v,u) == 2)
            {
                int select_ind = (int) inds.at<unsigned short>(v-dist_thresh, u-dist_thresh);
                pts.push_back(cv::KeyPoint(pts_raw[select_ind], 1.0f));
                select_indice.push_back(select_ind);
            }
        }
    }

    // Copiar descritores
    descriptors.create((int)select_indice.size(), 32, CV_8U);
    for (int i = 0; i < (int)select_indice.size(); i++){
        for (int j = 0; j < 32; j++){
            descriptors.at<unsigned char>(i, j) = desc.at<unsigned char>(select_indice[i], j);
        }
    }
}

// ---------------------------------------------------------
void loadFeatures(vector<vector<cv::Mat> > &features, cv::Mat descriptors);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void createVocabularyFile(OrbVocabulary &voc, std::string &fileName, const vector<vector<cv::Mat > > &features);

// ---------------------------------------------------------
int main()
{
    // Se sua GPU suportar, use CUDA. Caso contrário, pode usar CPU
    torch::DeviceType device_type = torch::kCUDA;
    if(!torch::cuda::is_available()){
        std::cout << "CUDA not available! Using CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    // Carregando modelo com a API moderna
    // Se preferir, use torch::jit::Module module; etc.
    torch::jit::Module module;
    try {
        module = torch::jit::load("/home/t/Dropbox/gcn2.pt", device);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    // Pegar lista de imagens
    std::vector<cv::String> fn;
    cv::glob("/home/t/Workspace/GCNV2/SUN3D/mono1/*.png", fn, false);
    size_t count = fn.size(); 

    // Vetor que guardará features
    vector<vector<cv::Mat> > features;

    int index = 0;
    for (size_t i = 0; i < count; i++)
    {
        index++;
        if (index > 0)
        {
            std::cout << fn[i] << std::endl;
            cv::Mat image_now = cv::imread(fn[i], cv::IMREAD_UNCHANGED);

            if(image_now.empty()){
                std::cout << "Could not open image: " << fn[i] << std::endl;
                continue;
            }

            cv::Mat img1;
            image_now.convertTo(img1, CV_32FC1, 1.f / 255.f , 0);

            int img_width = 320;
            int img_height = 240;

            // Cria tensor a partir do cv::Mat
            // Observação: em PyTorch 2.x, você pode usar `torch::from_blob(...)`
            // ou `torch::tensor(...)`. A seguir, usando from_blob:
            auto input_tensor = torch::from_blob(
                  img1.data, {1, img_height, img_width, 1}, torch::kFloat32
            );

            // Permute para [N, C, H, W]
            input_tensor = input_tensor.permute({0,3,1,2}).to(device);

            // Forward
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);

            // module.forward retorna um IValue
            torch::jit::IValue iv_out = module.forward(inputs);

            // Supondo que o forward retorne um tuple de (pts,desc)
            // A API moderna aceita .toTuple() e elements()...
            auto tuple_ptr = iv_out.toTuple();
            auto elements = tuple_ptr->elements();

            // 0: pts, 1: desc
            at::Tensor pts  = elements[0].toTensor().to(torch::kCPU).squeeze();
            at::Tensor desc = elements[1].toTensor().to(torch::kCPU).squeeze();

            // Copiamos para cv::Mat
            cv::Mat pts_mat((int)pts.size(0), 3, CV_32FC1, pts.data_ptr<float>());
            cv::Mat desc_mat((int)pts.size(0), 32, CV_8UC1, desc.data_ptr<uint8_t>());

            int border = 8;
            int dist_thresh = 4;
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            nms(pts_mat, desc_mat, keypoints, descriptors, border, dist_thresh, img_width, img_height);

            // Adiciona no array de features
            loadFeatures(features, descriptors);
        }
    }

    std::cout << "... Extraction done!" << std::endl;

    // define vocabulary
    const int nLevels = 6;
    const int k = 10; // branching factor
    const WeightingType weight = TF_IDF;
    const ScoringType  score  = L1_NORM;
    OrbVocabulary voc(k, nLevels, weight, score);

    std::string vocName = "vocGCN.bin";
    createVocabularyFile(voc, vocName, features);
    std::cout << "--- THE END ---" << std::endl;

    return 0;
}

// ---------------------------------------------------------
void loadFeatures(vector<vector<cv::Mat>> &features, cv::Mat descriptors)
{
    features.push_back(vector<cv::Mat>());
    changeStructure(descriptors, features.back());
}

// ---------------------------------------------------------
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
    out.resize(plain.rows);
    for(int i = 0; i < plain.rows; ++i)
    {
        out[i] = plain.row(i);
    }
}

// ---------------------------------------------------------
void createVocabularyFile(OrbVocabulary &voc, std::string &fileName,
                          const vector<vector<cv::Mat > > &features)
{
    std::cout << "> Creating vocabulary. May take some time ..." << std::endl;
    voc.create(features);
    std::cout << "... done!" << std::endl;

    std::cout << "> Vocabulary information: " << std::endl
              << voc << std::endl << std::endl;

    std::cout << std::endl << "> Saving vocabulary..." << std::endl;
    voc.saveToBinaryFile(fileName);
    std::cout << "... saved to file: " << fileName << std::endl;
}
// ---------------------------------------------------------

