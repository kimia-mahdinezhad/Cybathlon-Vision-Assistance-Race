#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"


class Footpath : public rclcpp::Node
{
public:
    Footpath() : Node("Footpath")
    {
        kinect_sub = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw", 10,
            [this](const sensor_msgs::msg::Image::SharedPtr msg)
            {
                this->LineDetector(msg);
            });

        // Initialize the video writer
        videoWriter.open("/home/redha/humble_ws/src/output3.mp4", cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), 30, cv::Size(640, 480));
        if (!videoWriter.isOpened())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open video file for writing");
        }
    }

private:
    void LineDetector(sensor_msgs::msg::Image::SharedPtr msg)
    {

        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        videoWriter.write(cv_ptr->image);

        // // Convert ROS image message to OpenCV Mat
        // cv::Mat image(msg->height, msg->width, CV_8UC3, const_cast<unsigned char*>(msg->data.data()));

        // // Convert BGR to grayscale
        // cv::Mat grayImage;
        // cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

        // cv::Mat blurImage;
        // cv::GaussianBlur(grayImage, blurImage, cv::Size(3, 3), 0);

        // // Apply Canny edge detection
        // cv::Mat edges;
        // cv::Canny(blurImage, edges, 120, 250);

        // // Apply Hough Transform for line detection
        // std::vector<cv::Vec2f> lines;
        // cv::HoughLines(edges, lines, 1, CV_PI / 180, 100);

        // // Draw the detected lines on the original image
        // cv::Mat resultImage = edges.clone();
        // for (size_t i = 0; i < lines.size(); i++)
        // {
        //     float rho = lines[i][0];
        //     float theta = lines[i][1];
        //     cv::Point pt1, pt2;
        //     double a = cos(theta), b = sin(theta);
        //     double x0 = a * rho, y0 = b * rho;
        //     pt1.x = cvRound(x0 + 1000 * (-b));
        //     pt1.y = cvRound(y0 + 1000 * (a));
        //     pt2.x = cvRound(x0 - 1000 * (-b));
        //     pt2.y = cvRound(y0 - 1000 * (a));
        //     cv::line(resultImage, pt1, pt2, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        // }

        // // Display the edges (optional)
        // cv::imshow("Edges", edges);

        // // Display the result image with detected lines
        // cv::imshow("Detected Lines", resultImage);
        // cv::waitKey(1);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr kinect_sub;
    cv::VideoWriter videoWriter;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Footpath>());
    rclcpp::shutdown();
    return 0;
}
