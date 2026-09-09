#ifndef CLASSIC_LINE_PERCEPTION_H
#define CLASSIC_LINE_PERCEPTION_H

/*
 * Allocation-free port of the simulator's grayscale/Otsu/centerline block.
 *
 * It deliberately consumes the OV2640 grayscale Y plane directly.  That
 * avoids an RGB565 byte-order assumption and keeps the normal tracking frame
 * below 80 KiB.  The constants come from generated/line_following_policy.h,
 * which must be included before this header.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    LF_CLASSIC_PERCEPTION_ACCEPTED = 0,
    LF_CLASSIC_PERCEPTION_FRAME_SHAPE,
    LF_CLASSIC_PERCEPTION_EMPTY_HISTOGRAM,
    LF_CLASSIC_PERCEPTION_SINGLE_POPULATION,
    LF_CLASSIC_PERCEPTION_CONTRAST_GUARD,
    LF_CLASSIC_PERCEPTION_DARK_FRACTION_GUARD,
    LF_CLASSIC_PERCEPTION_BRIGHT_FRACTION_GUARD,
    LF_CLASSIC_PERCEPTION_CENTERLINE_GAP,
    LF_CLASSIC_PERCEPTION_FIT_FAILED,
} lf_classic_perception_state_t;

typedef struct {
    bool valid;
    float e_y;
    float e_theta_rad;
    uint8_t threshold_px;
    uint8_t trace_points;
    lf_classic_perception_state_t state;
} lf_classic_perception_t;

static inline float lf_perception_clip(float value, float low, float high) {
    return fminf(high, fmaxf(low, value));
}

static inline lf_classic_perception_state_t lf_adaptive_otsu_threshold(
    const uint8_t *luma,
    size_t pixel_count,
    uint8_t *threshold_px
) {
    uint32_t histogram[256] = {0};
    uint64_t total_sum = 0;
    for (size_t index = 0; index < pixel_count; ++index) {
        const uint8_t value = luma[index];
        ++histogram[value];
        total_sum += value;
    }
    if (pixel_count == 0) {
        return LF_CLASSIC_PERCEPTION_EMPTY_HISTOGRAM;
    }

    uint32_t weight_low = 0;
    uint64_t sum_low = 0;
    double best_between = -1.0;
    int best_threshold = 0;
    double best_mean_low = 0.0;
    double best_mean_high = 0.0;
    uint32_t best_weight_low = 0;
    bool found_split = false;
    for (int level = 0; level < 256; ++level) {
        weight_low += histogram[level];
        sum_low += (uint64_t)histogram[level] * (uint64_t)level;
        const uint32_t weight_high = (uint32_t)pixel_count - weight_low;
        if (weight_low == 0 || weight_high == 0) {
            continue;
        }
        found_split = true;
        const double mean_low = (double)sum_low / (double)weight_low;
        const double mean_high = (double)(total_sum - sum_low) / (double)weight_high;
        const double between = (double)weight_low * (double)weight_high
            * (mean_low - mean_high) * (mean_low - mean_high);
        if (between > best_between) {
            best_between = between;
            best_threshold = level;
            best_mean_low = mean_low;
            best_mean_high = mean_high;
            best_weight_low = weight_low;
        }
    }
    if (!found_split) {
        return LF_CLASSIC_PERCEPTION_SINGLE_POPULATION;
    }
    if (best_mean_high - best_mean_low < LF_VISION_ADAPTIVE_MIN_CONTRAST_PX) {
        return LF_CLASSIC_PERCEPTION_CONTRAST_GUARD;
    }
    const float dark_fraction = (float)best_weight_low / (float)pixel_count;
    if (dark_fraction < LF_VISION_ADAPTIVE_MIN_DARK_FRACTION) {
        return LF_CLASSIC_PERCEPTION_DARK_FRACTION_GUARD;
    }
    if (1.0f - dark_fraction < LF_VISION_ADAPTIVE_MIN_BRIGHT_FRACTION) {
        return LF_CLASSIC_PERCEPTION_BRIGHT_FRACTION_GUARD;
    }
    best_threshold = (int)lf_perception_clip(
        (float)best_threshold,
        LF_VISION_ADAPTIVE_THRESHOLD_MIN_PX,
        LF_VISION_ADAPTIVE_THRESHOLD_MAX_PX
    );
    *threshold_px = (uint8_t)best_threshold;
    return LF_CLASSIC_PERCEPTION_ACCEPTED;
}

static inline bool lf_is_dark_in_band(
    const uint8_t *luma,
    int width,
    int height,
    int x,
    int row,
    int band,
    uint8_t threshold_px
) {
    const int first_row = row - band < 0 ? 0 : row - band;
    const int last_row = row + band >= height ? height - 1 : row + band;
    for (int y = first_row; y <= last_row; ++y) {
        if (luma[(size_t)y * (size_t)width + (size_t)x] <= threshold_px) {
            return true;
        }
    }
    return false;
}

static inline bool lf_find_tape_run(
    const uint8_t *luma,
    int width,
    int height,
    int row,
    uint8_t threshold_px,
    bool has_anchor,
    float anchor,
    int *run_start,
    int *run_end
) {
    bool inside = false;
    int start = 0;
    bool found = false;
    int chosen_start = 0;
    int chosen_end = 0;
    int chosen_width = -1;
    float chosen_distance = 0.0f;
    for (int x = 0; x <= width; ++x) {
        const bool dark = x < width && lf_is_dark_in_band(
            luma, width, height, x, row,
            LF_VISION_CENTERLINE_ROW_HALFHEIGHT_PX, threshold_px
        );
        if (dark && !inside) {
            start = x;
            inside = true;
        }
        if (!dark && inside) {
            const int end = x;
            const int width_px = end - start;
            if (width_px >= LF_VISION_MIN_RUN_WIDTH_PX) {
                const float center = ((float)start + (float)end - 1.0f) * 0.5f;
                const float distance = fabsf(center - anchor);
                const bool choose = !found
                    || (!has_anchor && width_px > chosen_width)
                    || (has_anchor && distance < chosen_distance);
                if (choose) {
                    found = true;
                    chosen_start = start;
                    chosen_end = end;
                    chosen_width = width_px;
                    chosen_distance = distance;
                }
            }
            inside = false;
        }
    }
    if (!found) {
        return false;
    }
    *run_start = chosen_start;
    *run_end = chosen_end;
    return true;
}

static inline bool lf_solve_3x3(double matrix[3][4], double solution[3]) {
    for (int column = 0; column < 3; ++column) {
        int pivot = column;
        for (int row = column + 1; row < 3; ++row) {
            if (fabs(matrix[row][column]) > fabs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (fabs(matrix[pivot][column]) < 1e-12) {
            return false;
        }
        if (pivot != column) {
            for (int entry = column; entry < 4; ++entry) {
                const double temporary = matrix[column][entry];
                matrix[column][entry] = matrix[pivot][entry];
                matrix[pivot][entry] = temporary;
            }
        }
        const double scale = matrix[column][column];
        for (int entry = column; entry < 4; ++entry) {
            matrix[column][entry] /= scale;
        }
        for (int row = 0; row < 3; ++row) {
            if (row == column) {
                continue;
            }
            const double factor = matrix[row][column];
            for (int entry = column; entry < 4; ++entry) {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
        }
    }
    for (int row = 0; row < 3; ++row) {
        solution[row] = matrix[row][3];
    }
    return true;
}

static inline bool lf_fit_centerline(
    const float *rows,
    const float *centers,
    int count,
    float near_row,
    float far_row,
    float *near_x,
    float *far_x
) {
    if (count < 2) {
        return false;
    }
    if (count >= 4) {
        double sum_0 = 0.0;
        double sum_1 = 0.0;
        double sum_2 = 0.0;
        double sum_3 = 0.0;
        double sum_4 = 0.0;
        double sum_x = 0.0;
        double sum_rx = 0.0;
        double sum_r2x = 0.0;
        for (int index = 0; index < count; ++index) {
            const double row = rows[index];
            const double center = centers[index];
            const double row2 = row * row;
            sum_0 += 1.0;
            sum_1 += row;
            sum_2 += row2;
            sum_3 += row2 * row;
            sum_4 += row2 * row2;
            sum_x += center;
            sum_rx += row * center;
            sum_r2x += row2 * center;
        }
        double matrix[3][4] = {
            {sum_4, sum_3, sum_2, sum_r2x},
            {sum_3, sum_2, sum_1, sum_rx},
            {sum_2, sum_1, sum_0, sum_x},
        };
        double coefficients[3] = {0.0, 0.0, 0.0};
        if (!lf_solve_3x3(matrix, coefficients)) {
            return false;
        }
        *near_x = (float)(coefficients[0] * near_row * near_row + coefficients[1] * near_row + coefficients[2]);
        *far_x = (float)(coefficients[0] * far_row * far_row + coefficients[1] * far_row + coefficients[2]);
        return true;
    }

    double sum_0 = 0.0;
    double sum_1 = 0.0;
    double sum_2 = 0.0;
    double sum_x = 0.0;
    double sum_rx = 0.0;
    for (int index = 0; index < count; ++index) {
        const double row = rows[index];
        const double center = centers[index];
        sum_0 += 1.0;
        sum_1 += row;
        sum_2 += row * row;
        sum_x += center;
        sum_rx += row * center;
    }
    const double determinant = sum_2 * sum_0 - sum_1 * sum_1;
    if (fabs(determinant) < 1e-12) {
        return false;
    }
    const double slope = (sum_rx * sum_0 - sum_x * sum_1) / determinant;
    const double intercept = (sum_2 * sum_x - sum_1 * sum_rx) / determinant;
    *near_x = (float)(slope * near_row + intercept);
    *far_x = (float)(slope * far_row + intercept);
    return true;
}

static inline lf_classic_perception_t lf_classic_perceive(
    const uint8_t *luma,
    size_t luma_bytes,
    int width,
    int height
) {
    lf_classic_perception_t result;
    result.valid = false;
    result.e_y = 0.0f;
    result.e_theta_rad = 0.0f;
    result.threshold_px = 0;
    result.trace_points = 0;
    result.state = LF_CLASSIC_PERCEPTION_FRAME_SHAPE;
    if (luma == NULL || width != LF_CAMERA_WIDTH_PX || height != LF_CAMERA_HEIGHT_PX
        || luma_bytes < (size_t)width * (size_t)height) {
        return result;
    }

    uint8_t threshold_px = (uint8_t)LF_VISION_BLACK_THRESHOLD_PX;
#if LF_VISION_USE_ADAPTIVE_OTSU
    result.state = lf_adaptive_otsu_threshold(luma, (size_t)width * (size_t)height, &threshold_px);
    if (result.state != LF_CLASSIC_PERCEPTION_ACCEPTED) {
        return result;
    }
#else
    result.state = LF_CLASSIC_PERCEPTION_ACCEPTED;
#endif

    float rows[LF_VISION_CENTERLINE_ROW_COUNT] = {0.0f};
    float centers[LF_VISION_CENTERLINE_ROW_COUNT] = {0.0f};
    int count = 0;
    bool has_anchor = false;
    float anchor = 0.0f;
    for (int index = LF_VISION_CENTERLINE_ROW_COUNT - 1; index >= 0; --index) {
        const int row = (int)lroundf(
            LF_VISION_CENTERLINE_ROWS_FRACTION[index] * (float)(height - 1)
        );
        int run_start = 0;
        int run_end = 0;
        if (!lf_find_tape_run(
                luma, width, height, row, threshold_px, has_anchor, anchor, &run_start, &run_end
            )) {
            if (count > 0) {
                break;
            }
            continue;
        }
        anchor = ((float)run_start + (float)run_end - 1.0f) * 0.5f;
        has_anchor = true;
        rows[count] = (float)row;
        centers[count] = anchor;
        ++count;
    }
    result.threshold_px = threshold_px;
    result.trace_points = (uint8_t)count;
    if (count < LF_VISION_MIN_CENTERLINE_POINTS) {
        result.state = LF_CLASSIC_PERCEPTION_CENTERLINE_GAP;
        return result;
    }

    const float nearest = rows[0];
    const float farthest = rows[count - 1];
    const float near_row = lf_perception_clip(
        LF_VISION_ROI_NEAR_FRACTION * (float)(height - 1), farthest, nearest
    );
    const float far_row = lf_perception_clip(
        LF_VISION_LOOKAHEAD_ROW_FRACTION * (float)(height - 1), farthest, nearest
    );
    float near_x = 0.0f;
    float far_x = 0.0f;
    if (!lf_fit_centerline(rows, centers, count, near_row, far_row, &near_x, &far_x)) {
        result.state = LF_CLASSIC_PERCEPTION_FIT_FAILED;
        return result;
    }
    const float half_width = ((float)width - 1.0f) * 0.5f;
    result.e_y = lf_perception_clip((far_x - half_width) / half_width, -1.0f, 1.0f);
    result.e_theta_rad = atan2f(far_x - near_x, fmaxf(1.0f, near_row - far_row));
    result.valid = true;
    result.state = LF_CLASSIC_PERCEPTION_ACCEPTED;
    return result;
}

#endif  /* CLASSIC_LINE_PERCEPTION_H */
