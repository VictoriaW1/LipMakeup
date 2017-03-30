/*#include <cv.h>
#include <highgui.h>
#include <videoio.hpp>
#include <imgproc\imgproc.hpp>*/
#include <opencv.hpp>
#include <core/mat.hpp>
#include <iostream>
#include <string>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace cv;

String IMG_DIR  = "D:\\visual studio 2015\\Projects\\Lip_Makeup\\images\\";
String IMG_PATH ;
String IMG_FN = "mouth_men2";
String EXTENSION = ".jpg";
String VIDEO_DIR = "D:\\visual studio 2015\\Projects\\Lip_Makeup\\videos\\";
String VIDEO_PATH;
const int MAX_ITER = 500;
const float TIME_STEP = 10;
const float MU = 0.5;
const float V = 0.0;
const float R_LAMBDA1 = 1;
const float R_LAMBDA2 = 1;
const float G_LAMBDA1 = 0.5;
const float G_LAMBDA2 = 0.5;
const float B_LAMBDA1 = 0.8;
const float B_LAMBDA2 = 0.8;
const float LAMBDA = 0.8;
const double EPSILON = 1;
const float X_CENTER = 100;
const float Y_CENTER = 100;
const float RADIUS = 100;
const int MAX_LENGTH_OF_IMAGE = 10000000;
const float BAND = 2;


int splitIntoFrames(string file_path) {
	CvCapture *capture = cvCaptureFromFile(file_path.c_str());
	if (!capture) {
		cerr << "cvCaptureFromVideo failed!!" << endl;
	}
	IplImage* frame = NULL;
	int frame_id = 0;
	while (frame = cvQueryFrame(capture)) {
		frame_id++;
		string curr_filename(IMG_DIR);
		curr_filename += to_string(frame_id) ;
		curr_filename += ".jpg";
		cvSaveImage(curr_filename.c_str(), frame);
		cout << "save images." << curr_filename << endl;
	}
	return frame_id;
}

void mergeIntoVideo(string imgs_path, string video_fn) {

	
}

/*
Detect lip region based on RGB color space
*/
void RGBDetectLipRegion(string file_path) {
	Mat img = imread(file_path);
	int rows = img.rows;
	int cols = img.cols;
	uchar* p;
	int b, g, r;
	vector<float> YIQ;
	for (int i = 0; i < rows; ++i) {
		p = img.ptr<uchar>(i);
		for (int j = 0; j < cols; ++j) {
			b = p[j * 3];
			g = p[j * 3 + 1];
			r = p[j * 3 + 2];
			//cout << "bgr" << b << g << r << endl;
			
			if (log10(g / (pow(b, 0.391) * pow(r, 0.609))) < -0.15) {
				p[j * 3] = 0;
				p[j * 3 + 1] = 255;
				p[j * 3 + 2] = 0;
			}
		}
	}
	namedWindow("RGB Detect Lip Region");
	imshow("RGB Detect Lip Region", img);
	waitKey(50000);
	imwrite(IMG_DIR + IMG_FN + "_RGBRegion.jpg", img);
}
/*
Detect lip region based on YIQ statistic
*/
vector<float> RGB2YIQ(float R, float G, float B) {
	float Y = 0.299 * R + 0.587 * G + 0.114 * B;
	float I = 0.596 * R - 0.275 * G - 0.321 * B;
	float Q = 0.212 * R - 0.523 * G + 0.311 * B;
	vector<float> YIQ;
	YIQ.push_back(Y);
	YIQ.push_back(I);
	YIQ.push_back(Q);
	return YIQ;
}

void YIQDetectLipRegion(string file_path) {
	Mat img = imread(file_path, 1);//BGR
	int rows = img.rows;
	int cols = img.cols;
	uchar* p;
	int b, g, r;
	vector<float> YIQ;
	for (int i = 0; i < rows; ++i) {
		p = img.ptr<uchar>(i);
		for (int j = 0; j < cols; ++j) {
			b = p[j * 3];
			g = p[j * 3 + 1];
			r = p[j * 3 + 2];
			//cout << "bgr" << b << g << r << endl;
			YIQ = RGB2YIQ(r, g, b);
			if (YIQ[0] >= 80 && YIQ[0] <= 220 && YIQ[1] >= 12 && YIQ[1] <= 78 && YIQ[2] >= 7 && YIQ[2] <= 25) {
				p[j * 3] = 255;
				p[j * 3 + 1] = 0;
				p[j * 3 + 2] = 255;
			}
		}
	}
	namedWindow("YIQ Detect Lip Region");
	imshow("YIQ Detect Lip Region", img);
	waitKey(50000);
	imwrite(IMG_DIR + IMG_FN + "_YIQRegion.jpg", img);
}


/*
Algorithm in Active Contour Without Edge.
*/
vector<Mat> gradient(Mat f) {
	int rows = f.rows;
	int cols = f.cols;
	Mat fx(rows, cols, CV_32FC1);
	Mat fy(rows, cols, CV_32FC1);
	vector<Mat> fxy;
	fx.col(0) = f.col(1) - f.col(0);
	fx.col(cols-1) = f.col(cols-1) - f.col(cols - 2);
	fy.row(0) = f.row(1) - f.row(0);
	fy.row(rows-1) = f.row(rows-1) - f.row(rows - 2);
	for (int i = 1; i < rows - 1; i++) {
		fy.row(i) = (f.row(i + 1) - f.row(i - 1)) / 2;
	}
	for (int i = 1; i < cols - 1; ++i) {
		fx.col(i) = (f.col(i + 1) - f.col(i - 1)) / 2;
	}
	fxy.push_back(fx);
	fxy.push_back(fy);
	return fxy;
}
Mat curvatureCentral(Mat phi) {
	/*Mat phi_x;
	Mat phi_y;
	Sobel(phi, phi_x, CV_32F, 1, 0, 3);
	Sobel(phi, phi_y, CV_32F, 0, 1, 3);
	Mat norm;
	pow(phi_x.mul(phi_x) + phi_y.mul(phi_y), 0.5, norm);
	Mat xx;
	Mat yy;
	Sobel(phi_x / norm, xx, CV_32F, 1, 0, 3);
	Sobel(phi_y / norm, yy, CV_32F, 0, 1, 3);
	return (xx+yy);*/
	vector<Mat> fxy = gradient(phi);
	Mat phi_x = fxy[0];
	Mat phi_y = fxy[1];
	Mat norm; 
	pow(phi_x.mul(phi_x) + phi_y.mul(phi_y), 0.5, norm);
	vector<Mat> phix_xy = gradient(phi_x / norm);
	vector<Mat> phiy_xy = gradient(phi_y / norm);
	Mat curvature = phix_xy[0] + phiy_xy[1];
	return curvature;
}

Mat curvatureCentral_Sobel(Mat phi) {
	Mat phi_x;
	Mat phi_y;
	Sobel(phi, phi_x, CV_32F, 1, 0, 3);
	Sobel(phi, phi_y, CV_32F, 0, 1, 3);
	Mat norm;
	pow(phi_x.mul(phi_x) + phi_y.mul(phi_y), 0.5, norm);
	Mat xx;
	Mat yy;
	Sobel(phi_x / norm, xx, CV_32F, 1, 0, 3);
	Sobel(phi_y / norm, yy, CV_32F, 0, 1, 3);
	return (xx + yy);
}

void neumannBoundCond(Mat f) {
	Mat g;
	f.copyTo(g);
	int rows = f.rows;
	int cols = f.cols;
	g.at<float>(0, 0) = g.at<float>(2, 2);
	g.at<float>(0, cols-1) = g.at<float>(2, cols-3);
	g.at<float>(rows - 1, 0) = g.at<float>(rows = 3, 2);
	g.at<float>(rows - 1, cols - 1) = g.at<float>(rows - 3, cols - 3);
	g.row(0) = g.row(2);
	g.row(rows - 1) = g.row(rows - 3);
	g.col(0) = g.col(2);
	g.col(cols - 1) = g.col(cols - 3);
	f = g;
}
void ACWE(Mat color_img, Mat img, Mat phi, int max_iter, float time_step, float mu, float v, float r_lambda1, float r_lambda2, 
	float g_lambda1, float g_lambda2, float b_lambda1, float b_lambda2, float lambda, double epsilon) {
	Mat original_img;
	img.convertTo(original_img, CV_8UC1);
	//Mat phi = phi0;
	int rows = img.rows;
	int cols = img.cols;
	Mat Heaviside(rows, cols, CV_32FC1);
	Mat Delta(rows, cols, CV_32FC1);
	Mat internal_force(rows, cols, CV_32FC1);
	Mat image_force(rows, cols, CV_32FC1);
	Mat inside_coor_of_curve;
	Mat outside_coor_of_curve;
	int* x_of_curve_neighbor = new int[MAX_LENGTH_OF_IMAGE];
	int* y_of_curve_neighbor = new int[MAX_LENGTH_OF_IMAGE];
	int* x_inside = new int[MAX_LENGTH_OF_IMAGE];
	int* y_inside = new int[MAX_LENGTH_OF_IMAGE];
	int* x_outside = new int[MAX_LENGTH_OF_IMAGE];
	int* y_outside = new int[MAX_LENGTH_OF_IMAGE];

	
	vector<vector<Point>> contours;
	Mat phi_8UC1;
	string window_name = "Lip Detection";
	//cvNamedWindow(window_name.c_str(), CV_WINDOW_AUTOSIZE);
	//Draw contour according to phi=0
	phi.convertTo(phi_8UC1, CV_8UC1);
	findContours(phi_8UC1, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	drawContours(original_img, contours, 0, CV_RGB(255, 0, 0));

	//imshow(window_name, original_img);
	//cvWaitKey(0.1);

	int fps = 15;
	CvSize size = cvSize(cols, rows);
	
	VideoWriter writer((VIDEO_PATH).c_str(), CV_FOURCC('M', 'P', '4', '2'), fps, size, 0);

	for (int itr = 0; itr < max_iter; ++itr) {
		neumannBoundCond(phi);
		cout << "ACWE iter: " << itr << endl;

		float cr1 = 0;
		float cr2 = 0;
		float cg1 = 0;
		float cg2 = 0;
		float cb1 = 0;
		float cb2 = 0;
		int neighbor_num = 0;
		int inside_num = 0;
		int outside_num = 0;

		uchar* p;
		for (int i = 0; i < rows; ++i) {
			p = color_img.ptr<uchar>(i);
			for (int j = 0; j < cols; ++j) {
				int b = p[3 * j];
				int g = p[3 * j + 1];
				int r = p[3 * j + 2];
				//Neigbor of curve phi
				if (phi.at<float>(i,j) < BAND & phi.at<float>(i, j) > -BAND) {
					*(x_of_curve_neighbor + neighbor_num) = j;
					*(y_of_curve_neighbor + neighbor_num) = i;
					++neighbor_num;
				}

				//Mean value inside curve of img
				if (0 <= phi.at<float>(i, j)) {
					//c1 += img.at<float>(i, j)
					cr1 += r;
					cg1 += g;
					cb1 += b;
					*(x_inside + inside_num) = j;
					*(y_inside + inside_num) = i;
					++inside_num;
				}
				//Mean value outside curve of img
				if (phi.at<float>(i, j) < 0) {
					cr2 += r;
					cg2 += g;
					cb2 += b;
					//c2 += img.at<float>(i, j);
					*(x_outside + outside_num) = j;
					*(y_outside + outside_num) = i;
					++outside_num;
				}
				//Compute H function and Delta function
				Heaviside.at<float>(i,j) = 0.5 * (1 + 2 / M_PI * atan(phi.at<float>(i, j) / epsilon));
				Delta.at<float>(i,j) = (epsilon / M_PI) / (pow(epsilon, 2) + pow(phi.at<float>(i, j), 2));
			}
			
		}

		if (1) {
			//Internal force
			internal_force = curvatureCentral(phi);

			//Image force
			//c1 /= (inside_num + 0.00000001);
			//c2 /= (outside_num + 0.000000001);
			cr1 /= (inside_num + 0.00000001);
			cr2 /= (outside_num + 0.000000001);
			cg1 /= (inside_num + 0.00000001);
			cg2 /= (outside_num + 0.000000001);
			cb1 /= (inside_num + 0.00000001);
			cb2 /= (outside_num + 0.000000001);
			//image_force = -lambda1 * (img - c1).mul(img - c1) + lambda2 * (img - c2).mul(img - c2);
			Mat channels[3];
			split(color_img, channels);
			image_force = -( b_lambda1 * (channels[0] - cb1).mul(channels[0] - cb1)
				+ g_lambda1 * (channels[1] - cg1).mul(channels[1] - cg1)
				+ r_lambda1 * (channels[2] - cr1).mul(channels[2] - cr1)) / 3
				+ (b_lambda2 * (channels[0] - cb2).mul(channels[0] - cb2)
					+ g_lambda2 * (channels[1] - cg2).mul(channels[1] - cg2)
					+ r_lambda2 * (channels[2] - cr2).mul(channels[2] - cr2)) / 3;
			double* max_image_force = new double;
			minMaxIdx(image_force, NULL, max_image_force);
			//cout << "max image_force: " << lambda * *max_image_force << endl;
			//Mat gradient = Delta.mul(mu * (internal_force) - v + (image_force) / *max_image_force);
			Mat gradient = mu * internal_force + image_force / *max_image_force;
			double* max_internal_force = new double;
			minMaxIdx(internal_force, NULL, max_internal_force);
			//cout << "max internal force: " << mu * (*max_internal_force) << endl;

			cout << "max image_force term: " << *max_image_force << endl;
			cout << "max internal force term: " << mu * (*max_internal_force) << endl;

			double* max_delta = new double;
			minMaxIdx(Delta, NULL, max_delta);
			cout << "max delta: " << (*max_delta) << endl;

			double* max_gradient = new double;
			minMaxIdx(gradient, NULL, max_gradient);
			cout << "max gradient: " << (*max_gradient) << endl;

			//cout << "gradient rows and cols: " << gradient.rows << gradient.cols << endl;
			
			cout << "neighbr_num: " << neighbor_num << endl;
			//Update neighbor of curve in phi
			/*for (int i = 0; i<neighbor_num; ++i) {
				int x = *(x_of_curve_neighbor + i);
				int y = *(y_of_curve_neighbor + i);
				cout << "phi(r, c)" << phi.at<float>(y, x) << "gradient" << gradient.at<float>(y, x) << endl;
				phi.at<float>(y, x) += time_step * gradient.at<float>(y, x);
			}*/
			
			//Update all pixel in phi
			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < cols; ++j) {
					phi.at<float>(i, j) += time_step * gradient.at<float>(i, j);
				}
			}

			double* max_phi = new double;
			minMaxIdx(phi, NULL, max_phi);
			cout << "max phi after update: " << (*max_phi) << endl;

			//Draw contour according to phi=0
			phi.convertTo(phi_8UC1, CV_8UC1, 255 / *max_phi);
			findContours(phi_8UC1, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
			img.convertTo(original_img, CV_8UC1);
			drawContours(original_img, contours, -1, CV_RGB(255, 0, 0));
			imwrite((IMG_DIR + IMG_FN + "_rgbcontour.jpg").c_str(), original_img);
			cout << "contour size: " << contours.size() << endl;
			//imshow(window_name, original_img);
			//cvWaitKey(0.1);
			Mat gradient_U8;
			gradient.convertTo(gradient_U8, CV_8UC1, 255 / *max_gradient);
			//imshow(window_name, gradient_U8);
			//cvWaitKey(0.00001);
			Mat delta_U8;
			Delta.convertTo(delta_U8, CV_8UC1, 255 / *max_delta);
			//imshow(window_name, delta_U8);
			//cvWaitKey(0.01);
			writer.write(original_img);
			delete max_phi;
			delete max_delta;
			delete max_internal_force;
			delete max_image_force;
			delete max_gradient;
		}
		
	}
	//cvDestroyWindow(window_name.c_str());
	cout << "finish!!" << endl;
	delete[] x_of_curve_neighbor;
	delete[] y_of_curve_neighbor;
	delete[] x_inside;
	delete[] y_inside;
	delete[] x_outside;
	delete[] y_outside;
	

	//destroyWindow(window_name);
	writer.release();
	cout << "hahhah" << endl;

}

Mat initializePhi(int rows, int cols, float x_center, float y_center, float radius) {
	Mat phi0(rows, cols, CV_32FC1);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {	
			float value = -sqrt(pow((j - x_center), 2) + pow((i - y_center), 2)) + radius;
			if (value < 0.1 & value > -0.1) {
				//cout << i << " " << j << endl;
				phi0.at<float>(i, j) = 0;
			}
			else {
				phi0.at<float>(i, j) = value; 
			}			
		}
	}
	return phi0;
}
void detectLipContInVideo(string file_path, bool show_contour = true) {
	//int frames_number = splitIntoFrames(file_path);
	int frames_number = 1;

	for (int frame_id = 0; frame_id < frames_number; ++frame_id) {
		Mat img = imread(IMG_DIR + to_string(frame_id+1) + ".jpg", 0);
		Mat img0;
		img.convertTo(img0, CV_32F);
		/*/
		for (int i = 0; i < 100; i++) {
			for (int j = 0; j < 100; j++) {
				cout << img.at<float>(i, j) << endl;
			}
		}*/
		cerr << IMG_DIR + to_string(frame_id+1) + ".jpg" << endl;
		cout << "rows:" << img.rows << endl;
		cout << "cols: " << img.cols << endl;
		Mat phi0 = initializePhi(img.rows, img.cols, X_CENTER, Y_CENTER, RADIUS);
		//ACWE(img0, phi0, MAX_ITER, TIME_STEP, MU, V, LAMBDA1, LAMBDA2, LAMBDA, EPSILON);	
	}

}

void detectLipContInImage(string img_path, bool show_contour = true) {
	VIDEO_PATH = VIDEO_DIR + IMG_FN + "_rgbphi.avi";
	cout << "video path:" << VIDEO_PATH << endl;
	Mat img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
	Mat color_img = imread(img_path, CV_LOAD_IMAGE_COLOR);
	color_img.convertTo(color_img, CV_32F);
	Mat img0;
	img.convertTo(img0, CV_32F);
	/*/
	for (int i = 0; i < 100; i++) {
	for (int j = 0; j < 100; j++) {
	cout << img.at<float>(i, j) << endl;
	}
	}*/
	cout << "rows:" << img.rows << endl;
	cout << "cols: " << img.cols << endl;
	Mat phi0 = initializePhi(img.rows, img.cols, X_CENTER, Y_CENTER, RADIUS);
	ACWE(color_img, img0, phi0, MAX_ITER, TIME_STEP, MU, V, R_LAMBDA1, R_LAMBDA2, G_LAMBDA1, G_LAMBDA2, B_LAMBDA1, B_LAMBDA2, LAMBDA, EPSILON);
	cout << "jijiiji" << endl;


}
int main() {
	cout << "split video into frames" << endl;
	string video_path("F:\\Project\\LipMakeup\\video\\video.mp4");
	//detectLipContInVideo(video_path);
	//YIQDetectLipRegion(IMG_DIR + IMG_FN + EXTENSION);
	RGBDetectLipRegion(IMG_DIR + IMG_FN + EXTENSION);
	//detectLipContInImage(IMG_DIR + IMG_FN + EXTENSION);
}