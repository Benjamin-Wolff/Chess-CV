/*
  Author: Benjamin Wolff
  Date: April 16, 2024
  
  Implementations of the operations related to processing the chess board image.
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

#include "processingOps.hpp"


/**
 * Custom comparison function used to sort the points by their y and x values.
 *      Sorts first by y values (with some slack) and then by x values.
 * @param p1    Point2f representing the first point to compare
 * @param p2    Point2f representing the second point to compare
 * 
 * @returns true if p1 should be evaluated as "less", false if p2 should be evaluated as "less"
*/
bool comparePoints(const cv::Point2f &p1, const cv::Point2f &p2) {
    // float xSlack = 20.0;
    float ySlack = 8.0;
    // Sort by y-coordinate first (check if p2 is in next row)
    if (p1.y + ySlack < p2.y) {
        return true;
    }
    // If y-coordinates are (approximately) equal, sort by x-coordinate (check if in same row, but p1 lower)
    // Otherwise, p1 should come after p2 (if in same row, but p2 lower)
    return (p1.y + ySlack > p2.y && p1.y < p2.y + ySlack && p1.x < p2.x);
}


/**
 * Checks if there is an intersection between the points represented by the lines
 * @param line1     cv::Vec4i representing the first line
 * @param line2     cv::Vec4i representing the secont line
 * @param imageSize cv::Size representing the size of the image (to check bounds)
 * @param r         resulting intersection (if exists)
 * 
 * @returns true if there is an interesection, false otherwise
*/
bool checkIntersection(cv::Vec4i &line1, cv::Vec4i &line2, cv::Size imageSize,
                      cv::Point2f &r) {
    cv::Point2f o1(line1[0], line1[1]);
    cv::Point2f p1(line1[2], line1[3]);
    cv::Point2f o2(line2[0], line2[1]);
    cv::Point2f p2(line2[2], line2[3]);

    if (o1.x < 0 || o1.y < 0  || p1.x < 0 || p1.y < 0 || o2.x < 0 || o2.y < 0
     || p2.x < 0  || p2.y < 0) {
        printf("We got negatives\n");
     }
    cv::Point2f x = o2 - o1;
    cv::Point2f d1 = p1 - o1;
    cv::Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (std::abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return (r.x >= 0 && r.y >= 0 && r.x < imageSize.width && r.y < imageSize.height);
}

/**
 * Checks if points are close to the current point in the image to remove potential duplicates.
 * @param point     a cv::Point2f representing the point of interest
 * @param points    a vector of cv::Point2f's representing the points to check
 * @param distance  a float for the threshold to determine if the points are close enough to be considered "duplicates"
 * 
 * @returns true if any point is within the specificed distance from the point, false otherwise
*/
bool arePointsNearby(const cv::Point2f& point, const std::vector<cv::Point2f>& points, float distance) {
    // Iterate over each point in the vector
    for (const auto& p : points) {
        float dist = distMacro(point, p);
        // printf("dist: %f\n", dist);
        // Check if the distance is within the specified threshold
        if (dist <= distance) {
            return true; // Return true if any point is within the specified distance
        }
    }
    return false; // Return false if no point is within the specified distance
}


/**
 * Calculates the Hough lines for the source image
 *      First resizes and converts to grayscale, applies gaussian blur, and applies Canny.
 * @param src       cv::Mat representing the source image
 * @param resized   cv::Mat representing the the resized image, to be used for its size
 * @param newSize   cv::Size representing the size of the resized image
 * @param lines     vector of cv::Vec4i's representing the resulting hough lines calculated
 * @param showCanny a bool representing if Canny intermediate results should be shown
 * 
 * @returns 0 if the function returns successfully.
*/
int calcHoughLines(cv::Mat &src, cv::Mat &resized, cv::Size newSize, std::vector<cv::Vec4i> &lines, bool showCanny) {
        cv::Mat temp;
        // Resize image
        
        cv::resize(src, resized, newSize, 0, 0, cv::INTER_AREA);
        // src.copyTo(resized);

        // Convert to grayscale
        cv::cvtColor(resized, temp, cv::COLOR_BGR2GRAY);

        // Gaussian blur
        int kernelSize = 5;
        cv::GaussianBlur(temp, temp, cv::Size(kernelSize, kernelSize), 0);
        // cv::imshow("gausian blur", temp);
        // cv::waitKey(0);

        // Applies Canny
        cv::Canny(temp, temp, 10, 250, 3);
        if (showCanny) {
            cv::imshow("Canny", temp);
            cv::waitKey(0);
        }

        // Gets Hough Lines
        cv::HoughLinesP(temp, lines, 0.5, CV_PI/180, 50, 30, 100);

        return 0;
}


/**
 * Scale the points back to the original size so we can work with bigger images.
 * @param image         cv::Mat for the original size image
 * @param points        vector of cv::Point2f's for the intersections on the resized image
 * @param originalSize  cv::Size of the original size of the image
 * @param smallerSize   cv::Size of the smaller, resized image
 * @param showPoints    bool flag for if we want to visualize the points or not on the original image
 * 
 * @returns a vector of cv::Point2f's representing the points, but in the scale of the original image
*/
std::vector<cv::Point2f> scalePointsToOriginal(cv::Mat &image, std::vector<cv::Point2f> &points, const cv::Size &originalSize, 
                                               const cv::Size &smallerSize, bool showPoints) {
    std::vector<cv::Point2f> scaledPoints;
    float scaleX = static_cast<float>(originalSize.width) / smallerSize.width;
    float scaleY = static_cast<float>(originalSize.height) / smallerSize.height;

    int current = 0;
    for (const cv::Point2f& point : points) {
        cv::Point2f scaledPoint(point.x * scaleX, point.y * scaleY);
        scaledPoints.push_back(scaledPoint);
        if (showPoints) {
            cv::circle(image, scaledPoint, 15, cv::Scalar(0, 0, 255), -1);
            cv::putText(image, //target image
                        std::to_string(current),
                        cv::Point(scaledPoint.x, scaledPoint.y - 20),
                        cv::FONT_HERSHEY_DUPLEX,
                        3.0,
                        CV_RGB(65, 105, 225), //font color
                        5);
            current++;
        }
        
    }

    return scaledPoints;
}

/**
 * Calculates the intersections of the lines provided, and draws circles on the destination image.
 * @param dst               cv::Mat representing the destination image
 * @param lines             vector of cv::Vec4i's representing the resulting hough lines calculated
 * @param imageSize         cv::Size of the image size
 * @param intersections     vector of cv::Point2f's representing the intersections between the Hough lines
 * @param showIntersections bool flag to represent if the output image should include text and circles of the intersections
 * 
 * @returns 0 if the function returns successfully.
*/
int getIntersections(cv::Mat &dst, std::vector<cv::Vec4i> &lines, cv::Size imageSize, std::vector<cv::Point2f> &intersections,
                     bool showIntersections) {
    // compute intersections based on combinations of lines
    for (size_t i = 0; i < lines.size(); i++) {
        for (size_t j = i + 1; j < lines.size(); j++) {
            cv::Vec4i line1 = lines[i];
            cv::Vec4i line2 = lines[j];
            
            // Compute intersection point
            cv::Point2f intersection;
            bool hasIntersection = checkIntersection(
                line1, line2, imageSize,
                intersection);
            
            // If there is an intersection, draw it on the image
            if (hasIntersection && (intersection.x > 25 && intersection.x < dst.cols - 25 && intersection.y > 25 && intersection.y < dst.rows - 25)) {
                bool nearby = arePointsNearby(intersection, intersections, 30);
                if (!nearby) {
                    intersections.push_back(intersection);
                    
                }
            }
        }
    }
    // sort to make the intersections easy to understand/use
    std::sort(intersections.begin(), intersections.end(), comparePoints);

    // display intersections and text on the dst image if showIntersections is true
    if (showIntersections) {
        int current = 0;
        for (cv::Point2f intersection : intersections) {
            // printf("Intesection %d: %f %f\n", current, intersection.x, intersection.y);
            cv::circle(dst, intersection, 5, cv::Scalar(0, 0, 255), -1);
            cv::putText(dst, //target image
                        std::to_string(current),
                        cv::Point(intersection.x, intersection.y - 10),
                        cv::FONT_HERSHEY_DUPLEX,
                        0.5,
                        CV_RGB(65, 105, 225), //font color
                        2);
            current++;
        }
    }
    printf("Number of found intersections: %zu\n", intersections.size());

    return 0;
}


/**
 * Creates the rectangles representing each square based on the (sorted) list of intersections
 * @param dst               cv::Mat representing the destination image
 * @param intersections     vector of cv::Point2f's representing the intersections between the Hough lines
 * @param rectangles        vector of cv::Rect's representing the rectangles for each space on the board
 * @param showRectangles    bool flag to represent if the output image should include rectangles numbered and highlighted
 * 
 * @returns 0 if the function returns successfully.
*/
int setRectangles(cv::Mat &dst, std::vector<cv::Point2f> &intersections, std::vector<cv::Rect> &rectangles, bool showRectangles) {
    // creates rectangles based on the top left corner and bottom right corner. With 9 corners in each row, this would be the i + 10 corner for the bottom right
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            cv::Point2f topLeft = intersections[row * 9 + col];
            cv::Point2f bottomRight = intersections[(row + 1) * 9 + (col + 1)];

            rectangles.push_back(cv::Rect(topLeft, bottomRight));
        }
    }

    // display rectangles on image if desired
    if (showRectangles) {
        int current = 0;
        for (cv::Rect currentRect : rectangles) {
            cv::rectangle(dst, currentRect, cv::Scalar(0, 255, 0), 5);
            cv::putText(dst, //target image
                    std::to_string(current),
                    cv::Point(currentRect.x + (0.5 * currentRect.width), currentRect.y + (0.5 * currentRect.height)),
                    cv::FONT_HERSHEY_DUPLEX,
                    3.0,
                    CV_RGB(0, 255, 0), //font color
                    5);
            current++;
        }
    }

    printf("Number of squares found: %zu\n", rectangles.size());

    return 0;
}

/**
 * Display the lines on the given destination image.
 * @param dst   a cv::Mat of the destination to display the lines
 * @param lines a vector of cv::Vec4i's representing the existing lines from Hough.
*/
void displayLines(cv::Mat &dst, std::vector<cv::Vec4i> &lines) {
    for (size_t i = 0; i < lines.size(); i++ ){
            cv::Vec4i l = lines[i];
            cv::line(dst, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, cv::LINE_AA);
        }
}

