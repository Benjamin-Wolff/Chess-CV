/*
  Author: Benjamin Wolff
  Date: April 18, 2024
  
  Code for the operations related to analyzing the chess position and working with chess-related APIs
*/

#include "chessAnalysis.hpp"
#include <opencv2/imgproc.hpp>

std::unordered_map<std::string, std::string> pieceToFen = {
    {"wp", "P"},
    {"wn", "N"},
    {"wb", "B"},
    {"wr", "R"},
    {"wq", "Q"},
    {"wk", "K"},
    {"bp", "p"},
    {"bn", "n"},
    {"bb", "b"},
    {"br", "r"},
    {"bq", "q"},
    {"bk", "k"}
};

std::unordered_map<std::string, int> squareNameToIndex = {
    {"a8", 0}, {"b8", 1}, {"c8", 2}, {"d8", 3}, {"e8", 4}, {"f8", 5}, {"g8", 6}, {"h8", 7},
    {"a7", 8}, {"b7", 9}, {"c7", 10}, {"d7", 11}, {"e7", 12}, {"f7", 13}, {"g7", 14}, {"h7", 15},
    {"a6", 16}, {"b6", 17}, {"c6", 18}, {"d6", 19}, {"e6", 20}, {"f6", 21}, {"g6", 22}, {"h6", 23},
    {"a5", 24}, {"b5", 25}, {"c5", 26}, {"d5", 27}, {"e5", 28}, {"f5", 29}, {"g5", 30}, {"h5", 31},
    {"a4", 32}, {"b4", 33}, {"c4", 34}, {"d4", 35}, {"e4", 36}, {"f4", 37}, {"g4", 38}, {"h4", 39},
    {"a3", 40}, {"b3", 41}, {"c3", 42}, {"d3", 43}, {"e3", 44}, {"f3", 45}, {"g3", 46}, {"h3", 47},
    {"a2", 48}, {"b2", 49}, {"c2", 50}, {"d2", 51}, {"e2", 52}, {"f2", 53}, {"g2", 54}, {"h2", 55},
    {"a1", 56}, {"b1", 57}, {"c1", 58}, {"d1", 59}, {"e1", 60}, {"f1", 61}, {"g1", 62}, {"h1", 63}
};

/**
 * Gets the indices for the squares of the best move by parsing the StockFish API response.
 * @param fullString    the full string of the 'bestmove' json value from the stockfish API
 * 
 * @returns a pair of ints, where the first index is the current piece position and the second index is where it should be moved to
*/
std::pair<int, int> getBestMove(std::string fullString) {
    std::string bestMove = "";
    std::string firstSquare, secondSquare;
    int firstIndex, secondIndex;

    // just check if string is what we expect, with bestmove followed by a space and 4 strings for the best move
    if (fullString.rfind("bestmove", 0) == 0 && fullString.size() >= 13) {
        // get the best move as a string
        bestMove = fullString.substr(9, 4);
        // printf("bestMove var: %s\n", bestMove.c_str());
        firstSquare = bestMove.substr(0, 2);
        secondSquare = bestMove.substr(2, 2);
        printf("Best move: %s to %s\n", firstSquare.c_str(), secondSquare.c_str());
        if (squareNameToIndex.find(firstSquare) != squareNameToIndex.end() && squareNameToIndex.find(secondSquare) != squareNameToIndex.end()) {
            firstIndex = squareNameToIndex[firstSquare];
            secondIndex = squareNameToIndex[secondSquare];
            
            return std::pair<int, int>(firstIndex, secondIndex);
        }
    }

    return std::pair<int, int>(0, 0);
}


/**
 * Makes an API call to Stockfish chess engine based on the fen, and displays the evaluation and best move if obtained.
 * @param dst       cv::Mat representing the image
 * @param fen       string of the 'fen' representation of the board's pieces
 * @param squares   vector of cv::Rect's representing the squares so the best move can be displayed.
 * 
 * @returns 0 if the function returns successfully
*/
int getChessAnalysis(cv::Mat image, std::string fen, std::vector<cv::Rect> squares) {
    std::string API_URL = "https://stockfish.online/api/s/v2.php";
    // Make a GET request to the API endpoint
    printf("Awaiting Stockfish server response...\n");
    cpr::Response response = cpr::Get(cpr::Url{API_URL},
                                      cpr::Parameters{{"fen", fen.c_str()}, {"depth", "10"}});

    // Check if the request was successful
    if (response.status_code == 200) {
        // Print the response body
        std::cout << "API Response: " << response.text << std::endl;
    } else {
        // Print an error message
        std::cerr << "Error: Failed to fetch API data. Status code: " << response.status_code << std::endl;
    }

    nlohmann::json j = nlohmann::json::parse(response.text);
    float eval;
    std::string fullString;
    std::pair<int, int> bestMove;
    if (j.find("evaluation") != j.end()) {
        eval = j["evaluation"];
        printf("Eval: %f\n", eval);
        cv::putText(image, //target image
                    "Eval: " + static_cast<std::string>((eval > 0 ? "+" : "")) + std::to_string(eval),
                    cv::Point(10, 90),
                    cv::FONT_HERSHEY_DUPLEX,
                    3.0,
                    CV_RGB(65, 105, 225), //font color
                    5);
    }

    if (j.find("bestmove") != j.end()) {
        fullString = j["bestmove"];
        bestMove = getBestMove(fullString);
        // printf("bestMove: %d %d\n", bestMove.first, bestMove.second);
        if (bestMove.first != bestMove.second) {
            cv::Rect startSquare, endSquare;
            startSquare = squares[bestMove.first];
            endSquare = squares[bestMove.second];

            cv::Point2f start(startSquare.x + (0.5 * startSquare.width), startSquare.y + (0.5 * startSquare.height));
            cv::Point2f end(endSquare.x + (0.5 * endSquare.width), endSquare.y + (0.5 * endSquare.height));
            cv::arrowedLine(image, start, end, CV_RGB(255, 0, 255), 10);
        }
    }
    
    return 0;
}

/**
 * Converts the labels of the chessboard to the chess "fen" format, a format that an API related to chess can read
 * @param squareLabels  vector of strings representing the labels for each square on the board
 * 
 * @returns a string in the "fen" format
*/
std::string getFenFromLabels(std::vector<std::string> squareLabels) {
    std::string fen = "";

    if (squareLabels.size() != 64) {
        printf("Improper size of labels parameter. Should be 64 squares. Size is: %zu\n", squareLabels.size());
        return "";
    }

    int currentEmpty = 0;

    for (int i = 0; i < squareLabels.size(); i++) {

        // first, we need to check if we are at a new row to add a '/'
        if (i % 8 == 0 && i != 0) {
            if (currentEmpty > 0) {
                fen += std::to_string(currentEmpty);
                currentEmpty = 0;
            }
            fen += "/";
        }

        // if there is an empty space, keep counting
        if (squareLabels[i] == "ee") {
                currentEmpty++;
        }
        else {
            // if we are currently counting empty spaces, but can stop now
            if (currentEmpty > 0) {
                fen += std::to_string(currentEmpty);
                currentEmpty = 0;
            }

            fen += pieceToFen.at(squareLabels[i]);
        }
    }
    fen += " ";

    std::string turn = "";
    while (turn != "w" && turn != "b") {
        printf("Please enter 'w' if it's white's turn and 'b' if it's black's turn:\n");
        std::cin >> turn;
    }

    fen += turn;
    fen += " - - 0 0";

    printf("Resulting fen: %s\n", fen.c_str());

    return fen;
}


