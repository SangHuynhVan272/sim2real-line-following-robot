// Host-side parity test for firmware/reference/line_following_policy.h.
// Compile with a desktop C++ compiler; it is not flashed to the ESP32.

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef LINE_FOLLOWING_POLICY_HEADER
#define LINE_FOLLOWING_POLICY_HEADER "../reference/line_following_policy.h"
#endif

#include LINE_FOLLOWING_POLICY_HEADER

static bool approximately_equal(float actual, float expected) {
    return std::fabs(actual - expected) <= 1e-5f;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "firmware/reference/line_following_policy_vectors.csv";
    std::ifstream source(path);
    if (!source) {
        std::cerr << "Cannot open " << path << "\n";
        return 2;
    }
    std::string line;
    if (!std::getline(source, line)) {
        std::cerr << "Empty vector file\n";
        return 2;
    }
    size_t line_number = 1;
    while (std::getline(source, line)) {
        ++line_number;
        std::stringstream row(line);
        std::string field;
        std::vector<float> values;
        while (std::getline(row, field, ',')) {
            values.push_back(std::stof(field));
        }
        if (values.size() != 11) {
            std::cerr << "Malformed row " << line_number << "\n";
            return 2;
        }
        const line_following_observation_t observation = {
            values[0], values[1], values[2], values[3], values[4], values[5], values[6],
        };
        const line_following_action_t target = lf_policy_target(&observation);
        const line_following_action_t step = lf_policy_step(&observation);
        if (!approximately_equal(target.duty_left, values[7])
            || !approximately_equal(target.duty_right, values[8])
            || !approximately_equal(step.duty_left, values[9])
            || !approximately_equal(step.duty_right, values[10])) {
            std::cerr << "Parity failure at row " << line_number
                      << ": target=(" << target.duty_left << ", " << target.duty_right
                      << ") step=(" << step.duty_left << ", " << step.duty_right << ")\n";
            return 1;
        }
    }
    std::cout << "Policy vectors PASS\n";
    return 0;
}
