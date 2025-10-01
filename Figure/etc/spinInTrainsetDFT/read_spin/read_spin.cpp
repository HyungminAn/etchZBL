#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


std::string get_last_word(std::stringstream& ss) {
    std::string word, lastWord;

    while (ss >> word) {
        lastWord = word;
    }

    return lastWord;
}


double get_max_abs(double* arr, int arr_size) {
    double max_val = 0 ;
    for (int i = 0; i < arr_size; i++) {
        if (std::abs(arr[i]) > max_val) {
            max_val = std::abs(arr[i]);
        }
    }

    return max_val;
}


void get_maxspin_from_OUTCAR(std::string path_outcar) {
    std::ifstream inputFile(path_outcar);

    // Check if the file is opened successfully
    if (!inputFile.is_open()) {
        std::cerr << "Could not open the file." << std::endl;
        return; // Exit with an error code
    }

    std::string keyword = "NIONS";
    std::string line;
    std::string word;
    std::stringstream ss;
    int nions;
    size_t pos;

    while (std::getline(inputFile, line)) {
        pos = line.find(keyword);

        if (pos != std::string::npos) {
            ss.clear();
            ss << line;
            nions = std::stoi(get_last_word(ss));
            break;
        }
    }

    double* spin = (double*)malloc(nions * sizeof(double));
    keyword = "magnetization (x)";
    std::getline(inputFile, line);

    while (std::getline(inputFile, line)) {
        pos = line.find(keyword);

        if (pos != std::string::npos) {
            break;
        }
    }

    for (int i = 0; i < 3; i++) {
        std::getline(inputFile, line);
    }

    for (int i = 0; i < nions; i++) {
        std::getline(inputFile, line);
        ss.clear();
        ss << line;
        spin[i] = std::stod(get_last_word(ss));
    }

    double max_spin = get_max_abs(spin, nions);
    if (max_spin > 0.0) {
        std::cout << "OUTCAR path " << path_outcar << " , Max spin : " << max_spin << std::endl;
    }

    inputFile.close();
}


int main() {
    std::ifstream inputFile("outcar_list");

    // Check if the file is opened successfully
    if (!inputFile.is_open()) {
        std::cerr << "Could not open the file." << std::endl;
        return 1; // Exit with an error code
    }

    std::string outcar_path;

    while (std::getline(inputFile, outcar_path)) {
        get_maxspin_from_OUTCAR(outcar_path);
    }

}
