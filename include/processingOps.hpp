/*
  Author: Benjamin Wolff
  Date: April 16, 2024
  
  Headers for the operations related to processing the chess board image.
*/

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>


#define SSD(a, b) ((a[0] - b[0]) * (a[0] - b[0])) + ((a[1] - b[1]) * (a[1] - b[1]))  + ((a[2] - b[2]) * (a[2] - b[2]))
#define distMacro(a, b) sqrtf(((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y)))

/**
 * Custom comparison function used to sort the points by their y and x values.
 *      Sorts first by y values (with some slack) and then by x values.
 * @param p1    Point2f representing the first point to compare
 * @param p2    Point2f representing the second point to compare
 * 
 * @returns true if p1 should be evaluated as "less", false if p2 should be evaluated as "less"
*/
bool comparePoints(const cv::Point2f &p1, const cv::Point2f &p2);


/**
 * Checks if there is an intersection between the points represented by the lines
 * @param line1     cv::Vec4i representing the first line
 * @param line2     cv::Vec4i representing the secont line
 * @param imageSize cv::Size representing the size of the image (to check bounds)
 * @param r         resulting intersection (if exists)
 * 
 * @returns true if there is an interesection, false otherwise
*/
bool checkIntersection(cv::Vec4i &line1, cv::Vec4i &line2, cv::Size imageSize, cv::Point2f &r);


/**
 * Checks if points are close to the current point in the image to remove potential duplicates.
 * @param point     a cv::Point2f representing the point of interest
 * @param points    a vector of cv::Point2f's representing the points to check
 * @param distance  a float for the threshold to determine if the points are close enough to be considered "duplicates"
 * 
 * @returns true if any point is within the specificed distance from the point, false otherwise
*/
bool arePointsNearby(const cv::Point2f& point, const std::vector<cv::Point2f>& points, float distance=10.0);


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
int calcHoughLines(cv::Mat &src, cv::Mat &resized, cv::Size newSize, std::vector<cv::Vec4i> &lines, bool showCanny=false);

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
                     bool showIntersections=false);


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
                                               const cv::Size &smallerSize, bool showPoints=false);

/**
 * Creates the rectangles representing each square based on the (sorted) list of intersections
 * @param dst               cv::Mat representing the destination image
 * @param intersections     vector of cv::Point2f's representing the intersections between the Hough lines
 * @param rectangles        vector of cv::Rect's representing the rectangles for each space on the board
 * @param showRectangles    bool flag to represent if the output image should include rectangles numbered and highlighted
 * 
 * @returns 0 if the function returns successfully.
*/
int setRectangles(cv::Mat &dst, std::vector<cv::Point2f> &intersections, std::vector<cv::Rect> &rectangles, bool showRectangles=false);

/**
 * Display the lines on the given destination image.
 * @param dst   a cv::Mat of the destination to display the lines
 * @param lines a vector of cv::Vec4i's representing the existing lines from Hough.
*/
void displayLines(cv::Mat &dst, std::vector<cv::Vec4i> &lines);

