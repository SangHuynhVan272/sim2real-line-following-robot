// Host-side smoke test for the allocation-free ESP32 classical perception port.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#ifndef LINE_FOLLOWING_POLICY_HEADER
#define LINE_FOLLOWING_POLICY_HEADER "../reference/line_following_policy.h"
#endif

#include LINE_FOLLOWING_POLICY_HEADER
#include "../esp32s3_line_following/classic_line_perception.h"

static void draw_tape(std::vector<uint8_t> *image, float slope_px_per_row) {
    image->assign((size_t)LF_CAMERA_WIDTH_PX * (size_t)LF_CAMERA_HEIGHT_PX, 220);
    for (int row = 0; row < LF_CAMERA_HEIGHT_PX; ++row) {
        const int center = (int)lroundf(159.5f + slope_px_per_row * (float)(row - 180));
        for (int offset = -4; offset <= 4; ++offset) {
            const int column = center + offset;
            if (column >= 0 && column < LF_CAMERA_WIDTH_PX) {
                (*image)[(size_t)row * (size_t)LF_CAMERA_WIDTH_PX + (size_t)column] = 20;
            }
        }
    }
}

int main() {
    std::vector<uint8_t> image;
    draw_tape(&image, 0.0f);
    lf_classic_perception_t centered = lf_classic_perceive(
        image.data(), image.size(), LF_CAMERA_WIDTH_PX, LF_CAMERA_HEIGHT_PX
    );
    if (!centered.valid || std::fabs(centered.e_y) > 0.02f || std::fabs(centered.e_theta_rad) > 0.02f) {
        std::cerr << "Centered tape did not produce a centered observation\n";
        return 1;
    }
    draw_tape(&image, 0.25f);
    lf_classic_perception_t slanted = lf_classic_perceive(
        image.data(), image.size(), LF_CAMERA_WIDTH_PX, LF_CAMERA_HEIGHT_PX
    );
    if (!slanted.valid || slanted.e_theta_rad >= -0.1f || slanted.trace_points < LF_VISION_MIN_CENTERLINE_POINTS) {
        std::cerr << "Slanted tape did not produce the expected heading\n";
        return 1;
    }
    image.assign((size_t)LF_CAMERA_WIDTH_PX * (size_t)LF_CAMERA_HEIGHT_PX, 220);
    lf_classic_perception_t blank = lf_classic_perceive(
        image.data(), image.size(), LF_CAMERA_WIDTH_PX, LF_CAMERA_HEIGHT_PX
    );
    if (blank.valid) {
        std::cerr << "Uniform floor was accepted as tape\n";
        return 1;
    }
    std::cout << "Classic perception smoke PASS\n";
    return 0;
}
