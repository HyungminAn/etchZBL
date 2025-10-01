#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cstdlib>

#define ATOM_NUM_SI 1

namespace fs = std::filesystem;
using namespace std;

// returns a vector of strings split by whitespace
vector<string> split(const string& s) {
    vector<string> tokens;
    istringstream iss(s);
    string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

// Read the .coo file and count the number of Si atoms
int countSiAtoms(const string& filename) {
    ifstream inFile(filename);
    if (!inFile.is_open()) {
        cerr << "Cannot open file." << filename << endl;
        return 0;
    }

    string line;
    bool inAtomsSection = false;
    int siCount = 0;
    while (getline(inFile, line)) {
        if (line.find("Velocities") != string::npos) {
            break; // stop reading
        }
        // Find "Atoms" section
        if (!inAtomsSection) {
            if (line.find("Atoms") != string::npos) {
                inAtomsSection = true;
                continue;
            }
        } else {
            // Skip empty lines and lines not starting with a digit
            if (line.empty()) {
                continue;
            }
            istringstream iss(line);
            int id, type;
            double x, y, z;
            // Read id, type, x, y, z (ignore other fields)
            if (!(iss >> id >> type >> x >> y >> z)) {
                continue; // skip if read fails
            }
            if (type == ATOM_NUM_SI) {
                siCount++;
            }
        }
    }
    inFile.close();
    return siCount;
}

double getNormalizationFactor(const string& filepath) {
    ifstream inFile(filepath);
    if (!inFile.is_open()) {
        cerr << "Cannot open file (normalize factor)" << filepath << endl;
        return 0.0;
    }
    vector<string> lines;
    string line;
    while (getline(inFile, line)) {
        lines.push_back(line);
    }
    inFile.close();

    if (lines.size() < 7) {
        cerr << "Insufficient line counts" << filepath << endl;
        return 0.0;
    }

    // Read 6th and 7th lines
    vector<string> tokens5 = split(lines[5]);
    vector<string> tokens6 = split(lines[6]);
    if (tokens5.size() < 2 || tokens6.size() < 2) {
        cerr << "Format error" << filepath << endl;
        return 0.0;
    }

    int lat_x = stoi(tokens5[1]);
    int lat_y = stoi(tokens6[1]);

    if(lat_x == 0 || lat_y == 0) {
        cerr << "The lattice size is 0." << endl;
        return 0.0;
    }

    return 1.0 / (lat_x * lat_y);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " [interval] [dst] [src1] [src2] ..." << endl;
        return 1;
    }

    int interval = atoi(argv[1]);
    string dst = argv[2];
    vector<string> srcDirs;
    for (int i = 3; i < argc; ++i) {
        srcDirs.push_back(argv[i]);
    }

    // Search files, and store the file paths
    // nowFiles: "rm_byproduct_str_shoot_*.coo"
    // prevFiles: "str_shoot_*_after_mod.coo"
    map<int, string> nowFiles;
    map<int, string> prevFiles;

    for (const auto& folder : srcDirs) {
        if (!fs::exists(folder) || !fs::is_directory(folder)) {
            cerr << "Folder does not exist or is not a directory." << folder << endl;
            continue;
        }
        for (const auto& entry : fs::directory_iterator(folder)) {
            if (!entry.is_regular_file())
                continue;
            string filename = entry.path().filename().string();
            if (entry.path().extension() != ".coo")
                continue;
            if (filename.find("_before_anneal") != string::npos)
                continue;

            const string prefixNow = "rm_byproduct_str_shoot_";
            if (filename.find(prefixNow) != string::npos) {
                // Removing prefix
                size_t pos = filename.find(prefixNow);
                if (pos != 0) {
                    continue;
                }
                string idxStr = filename.substr(prefixNow.size());
                size_t extPos = idxStr.rfind(".coo");
                if (extPos != string::npos) {
                    idxStr = idxStr.substr(0, extPos);
                }
                try {
                    int idx = stoi(idxStr);
                    nowFiles[idx] = entry.path().string();
                } catch (...) {
                    cerr << "Failed to convert number." << filename << endl;
                    continue;
                }
            }
            else if (filename.find("after_mod") != string::npos) {
                const string prefixPrev = "str_shoot_";
                const string suffixPrev = "_after_mod.coo";
                if (filename.find(prefixPrev) != 0 || filename.find(suffixPrev) == string::npos)
                    continue;
                string idxStr = filename.substr(prefixPrev.size());
                size_t pos = idxStr.find(suffixPrev);
                if (pos != string::npos) {
                    idxStr = idxStr.substr(0, pos);
                }
                try {
                    int idx = stoi(idxStr);
                    prevFiles[idx] = entry.path().string();
                } catch (...) {
                    cerr << "Failed to convert number." << filename << endl;
                    continue;
                }
            }
        } // for entry
    } // for folder

    if (nowFiles.empty()) {
        cerr << "Cannot find rm_byproduct_str_shoot_*.coo" << endl;
        return 1;
    }

    int n_traj = nowFiles.rbegin()->first;
    cout << "n_traj: " << n_traj << endl;

    vector<int> diffList;
    diffList.push_back(0);  // index 0: initial value

    // Calculate the number of Si atoms in each trajectory
    for (int i = 1; i <= n_traj; ++i) {
        // now: index i
        if (nowFiles.find(i) == nowFiles.end()) {
            cerr << "now MISSING, index: " << i << endl;
            continue;
        }
        // previous: index i-1
        if (prevFiles.find(i - 1) == prevFiles.end()) {
            cerr << "prev MISSING, index: " << (i - 1) << endl;
            continue;
        }

        string nowFile = nowFiles[i];
        string prevFile = prevFiles[i - 1];

        int countNow = countSiAtoms(nowFile);
        int countPrev = countSiAtoms(prevFile);

        int n_diff = (countPrev > countNow) ? (countPrev - countNow) : 0;
        diffList.push_back(n_diff);
        cout << nowFile << " Done, deleted Si: " << n_diff << endl;
    }

    // (cumulative sum) (Python np.cumsum)
    vector<int> cumSum;
    if (!diffList.empty()) {
        cumSum.push_back(diffList[0]);
        for (size_t i = 1; i < diffList.size(); ++i) {
            cumSum.push_back(cumSum[i - 1] + diffList[i]);
        }
    }

    // interval average etch yield
    // For each idx_end, idx_start = max(0, idx_end - interval)
    // etch yield = (cumSum[idx_end] - cumSum[idx_start]) / (idx_end - idx_start)
    vector<double> etchYield;
    etchYield.push_back(0.0);
    for (size_t idx_end = 1; idx_end < cumSum.size(); ++idx_end) {
        size_t idx_start = (idx_end < (size_t)interval) ? 0 : (idx_end - interval);
        int diff = cumSum[idx_end] - cumSum[idx_start];
        size_t steps = idx_end - idx_start;
        double avg = static_cast<double>(diff) / steps;
        etchYield.push_back(avg);
    }

    string normFile;
    {
        string firstDir = srcDirs.front();
        fs::path p1 = fs::path(firstDir) / "str_shoot_0_after_mod.coo";
        fs::path p2 = fs::path(firstDir) / "str_shoot_0.coo";
        if (fs::exists(p1)) {
            normFile = p1.string();
        } else if (fs::exists(p2)) {
            normFile = p2.string();
        } else {
            cerr << "Normalization MISSING in folder: " << firstDir << endl;
            return 1;
        }
    }
    double normFactor = getNormalizationFactor(normFile);

    // 결과 출력
    cout << "\n=== Results ===" << endl;
    cout << "Normalization Factor: " << normFactor << endl;
    cout << "\nTotal Si Etched:" << endl;
    for (size_t i = 0; i < cumSum.size(); ++i) {
        cout << "Index " << i << ": " << cumSum[i] << endl;
    }
    cout << "\nEtch Yield (interval average):" << endl;
    for (size_t i = 0; i < etchYield.size(); ++i) {
        cout << "Index " << i << ": " << etchYield[i] << endl;
    }

    // save results
    string outFilename = dst + ".dat";
    ofstream outFile(outFilename);
    if (!outFile.is_open()) {
        cerr << "Cannot open dst: " << outFilename << endl;
        return 1;
    }
    outFile << "# Normalization Factor: " << normFactor << "\n";
    outFile << "# Index   Cum_Si_Etched   Etch_Yield\n";
    for (size_t i = 0; i < cumSum.size() && i < etchYield.size(); ++i) {
        outFile << i << "\t" << cumSum[i] << "\t" << etchYield[i] << "\n";
    }
    outFile.close();
    cout << "\nResults have saved to " << outFilename << ". FINISHED" << endl;

    return 0;
}

