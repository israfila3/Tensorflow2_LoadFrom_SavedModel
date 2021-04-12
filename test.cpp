#include <stdio.h>
#include <tensorflow/c/c_api.h>
//#include "pch.h"
#include <iostream>
#include <codecvt>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>
#include <sstream>
#include <atlstr.h>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
std::string filename1;

void DeallocateBuffer(void* data, size_t) {
	std::free(data);
}
TF_Buffer* ReadBufferFromFile(std::string file) {
	std::ifstream f(file, std::ios::binary);
	if (f.fail() || !f.is_open()) {
		return nullptr;
	}

	f.seekg(0, std::ios::end);
	const auto fsize = f.tellg();
	f.seekg(0, std::ios::beg);

	if (fsize < 1) {
		f.close();
		return nullptr;
	}

	char* data = static_cast<char*>(std::malloc(fsize));
	f.read(data, fsize);
	f.close();

	TF_Buffer* buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fsize;
	buf->data_deallocator = DeallocateBuffer;
	return buf;
}

typedef struct {
	void* buffer;
	int32_t width;
	int32_t height;
	int32_t stride;
	int32_t bit_depth;
}ImageDataRecord;


class Network {

public:
	void LoadGraph(std::string modelPath);
	void Detect(cv::Mat  image_record);
	static void Deallocator(void* data, size_t length, void* arg);
	Network();
private:
	TF_Session* session;
	TF_Graph* graph;
};

Network::Network()
{
	std::cout << "Created a new network";
}

void Network::LoadGraph(std::string modelPath)
{
	graph = TF_NewGraph();
	TF_Status* Status = TF_NewStatus();
	TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
	TF_Buffer* RunOpts = NULL;
	const char* tags = "serve";
	int ntags = 1;
	char* path = const_cast<char*>(modelPath.c_str());

	session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, path, &tags, ntags, graph, NULL, Status);
	if (TF_GetCode(Status) == TF_OK)
	{
		printf("Model Is loaded succesfully");
	}
	else
	{
		printf("%s", TF_Message(Status));
	}
	TF_DeleteSessionOptions(SessionOpts);
	TF_DeleteStatus(Status);
}

CString probab("");
void Network::Detect(cv::Mat image)
{
	std::vector<TF_Output> 	input_tensors, output_tensors;
	std::vector<TF_Tensor*> input_values, output_values;
	clock_t tStart1 = clock();

	int num_dims = 4;
	std::int64_t input_dims[4] = { 1, image.rows, image.cols, 3 };
	int num_bytes_in = image.cols * image.rows * 3;

	TF_Output t0 = { TF_GraphOperationByName(graph, "StatefulPartitionedCall"),0 };
	
	size_t pos = 0;
	TF_Operation* oper;

	while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
		printf(TF_OperationName(oper));
		printf("\n");
	}
	input_tensors.push_back({ TF_GraphOperationByName(graph, "serving_default_input_tensor"),0 });
	input_values.push_back(TF_NewTensor(TF_UINT8, input_dims, num_dims, image.data, num_bytes_in, &Deallocator, 0));


	output_tensors.push_back({ TF_GraphOperationByName(graph, "StatefulPartitionedCall"),1 });
	output_values.push_back(nullptr);

	output_tensors.push_back({ TF_GraphOperationByName(graph, "StatefulPartitionedCall"),2 });
	output_values.push_back(nullptr);

	output_tensors.push_back({ TF_GraphOperationByName(graph, "StatefulPartitionedCall"),4 });
	output_values.push_back(nullptr);

	output_tensors.push_back({ TF_GraphOperationByName(graph, "StatefulPartitionedCall"),5 });
	output_values.push_back(nullptr);

	TF_Status* status = TF_NewStatus();
	TF_SessionRun(session, nullptr,
		&input_tensors[0], &input_values[0], input_values.size(),
		&output_tensors[0], &output_values[0], output_values.size(),
		nullptr, 0, nullptr, status
	);	
	if (TF_GetCode(status) != TF_OK)
	{
		printf("ERROR: SessionRun: %s", TF_Message(status));
	}
	auto detection_classes = static_cast<float_t*>(TF_TensorData(output_values[1]));
	auto detection_scores = static_cast<float_t*>(TF_TensorData(output_values[2]));
	auto detection_boxes = static_cast<float_t*>(TF_TensorData(output_values[0]));
	auto num_detections = static_cast<float_t*>(TF_TensorData(output_values[3]));
	int number_detections = (int)num_detections[0];
	std::cout << "number_detections: " << number_detections << " ";

	if (number_detections > 10) {
		number_detections = 10; //Total number of detection to display
	}


	for (int i = 0; i < number_detections; i++) {
		if (detection_scores[i] > 0.70) {

			int xmin = (int)(detection_boxes[i * 4 + 1] * image.cols);
			int ymin = (int)(detection_boxes[i * 4 + 0] * image.rows);
			int xmax = (int)((detection_boxes[i * 4 + 3] - detection_boxes[i * 4 + 1]) * image.cols);
			int ymax = (int)((detection_boxes[i * 4 + 2] - detection_boxes[i * 4 + 0]) * image.rows);
			std::cout << "X Y : " << xmin << ":" << ymin << ":" << xmax << ":" << ymax << ":";

			std::string ng_score = std::to_string(float(detection_scores[i]));
			std::string ng_id = std::to_string(int(detection_classes[i]));
			std::string display_info = "ID:" + ng_id + "[P:" + ng_score + "]";
			cv::putText(image, display_info, cv::Point2f(xmin - 15, ymin - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(0, 255, 255), 1);
			//cv::rectangle(image, CvPoint(xmin, ymin), CvPoint(xmax, ymax), Scalar(0, 0, 255));
			cv::Rect rect2 = cv::Rect(xmin, ymin, xmax, ymax);
			rectangle(image, rect2, cv::Scalar(0, 0, 255), 2, 8, 0);
			
			
		}
	}
	
	clock_t netforward = clock();
	double netforwardtime = (double)(netforward - tStart1) / CLOCKS_PER_SEC;
	std::cout << "Network time : " << netforwardtime << std::endl;
	cv::imshow("test", image);
	cv::waitKey(0);


}
void Network::Deallocator(void* data, size_t length, void* arg)
{
	std::free(data);
}



int main() {

	std::string modelPath = "D:/AI/TrainingAlgo/Tensorflow2/output_inference_graph/saved_model";
	
	Network TFnetwork;
	TFnetwork.LoadGraph(modelPath);
	std::ofstream myfile;
	while (true) {
		std::string filename;
		std::cout << "input image filename: ";
		std::cin >> filename;
		filename1 = filename;
		if (filename.size() == 0) break;
		cv::Mat image = cv::imread(filename);
		//clock_t tStart1 = clock();
		image.convertTo(image, CV_8UC3);
		//std::cout << image.channels;
		TFnetwork.Detect(image);

	}

}
