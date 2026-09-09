/*
 * ESP32-S3 camera line-following firmware.
 *
 * The camera, motor and encoder definitions below are the hardware interface
 * for this project. It follows the frozen Sim2Real contract:
 *
 *   grayscale camera -> [e_y, e_theta, confidence] -> generated policy
 *   -> signed left/right duty -> H-bridge PWM
 *
 * There is deliberately no wheel-speed PID.  Encoder RPM is measured for the
 * frozen ABI and calibration telemetry only; it is never converted into PWM.
 */

#include <Arduino.h>
#include <stdio.h>
#include <stdlib.h>
#include <esp_camera.h>
#include <esp_timer.h>
#include <driver/ledc.h>

#if __has_include("../generated/line_following_policy.h")
#include "../generated/line_following_policy.h"
#else
#error "No learner policy found. Complete docs/TRAINING.md and export to firmware/generated before compiling."
#endif
#include "classic_line_perception.h"

/* OV2640 parallel-camera mapping for the target ESP32-S3 board. */
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 15
#define SIOD_GPIO_NUM 4
#define SIOC_GPIO_NUM 5
#define Y9_GPIO_NUM 16
#define Y8_GPIO_NUM 17
#define Y7_GPIO_NUM 18
#define Y6_GPIO_NUM 12
#define Y5_GPIO_NUM 10
#define Y4_GPIO_NUM 8
#define Y3_GPIO_NUM 9
#define Y2_GPIO_NUM 11
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM 7
#define PCLK_GPIO_NUM 13

/* Front-encoder inputs and shared left/right motor-driver outputs. */
#define ENC_R_A 1
#define ENC_R_B 2
#define MOTOR_R_DIR 41
#define MOTOR_R_PWM 42
#define ENC_L_A 3
#define ENC_L_B 46
#define MOTOR_L_DIR 48
#define MOTOR_L_PWM 47

/* Verified against the physical OV2640 module the robot carries.  The sim
 * renders an upright, unmirrored frame, so a board that disagrees feeds the
 * policy a mirrored e_y and steers the wrong way. */
#ifndef LF_CAMERA_HMIRROR
#define LF_CAMERA_HMIRROR 1
#endif

#ifndef LF_CAMERA_VFLIP
#define LF_CAMERA_VFLIP 1
#endif

/*
 * Normal deployment is command-free: after boot the controller waits for a
 * few valid camera frames and then starts the policy.  It never applies the
 * zero-confidence hunt action during this boot wait.
 */
/*
 * Bench build.  Off for deployment, so the flashed robot stays camera-only and
 * cannot be driven by a stray serial byte.  Turn it on to measure the plant:
 * the duty at which loaded wheels first turn, straight and pivoting both ways.
 * Those three curves are what replace the assumed scrub constants in
 * isaac_sim/config/default.json, and they cannot be taken with the wheels
 * lifted -- the whole point is the load.
 */
#ifndef LF_BENCH_BUILD
#define LF_BENCH_BUILD 0
#endif

#ifndef LF_AUTOSTART_CAMERA_POLICY
#if LF_BENCH_BUILD
/* A bench robot must never drive itself while somebody is holding it. */
#define LF_AUTOSTART_CAMERA_POLICY 0
#else
#define LF_AUTOSTART_CAMERA_POLICY 1
#endif
#endif

/* Change only after a forward-drive telemetry check shows an inverted RPM. */
#ifndef LF_ENCODER_LEFT_SIGN
#define LF_ENCODER_LEFT_SIGN 1.0f
#endif

#ifndef LF_ENCODER_RIGHT_SIGN
#define LF_ENCODER_RIGHT_SIGN 1.0f
#endif

static constexpr uint32_t kSerialBaud = 115200;
static constexpr uint32_t kPwmFrequencyHz = 20000;
static constexpr uint8_t kPwmResolutionBits = 8;
static constexpr uint32_t kPwmMaximum = (1u << kPwmResolutionBits) - 1u;
/* Camera XCLK owns LEDC channel/timer 0.  Never use Arduino's auto allocator
 * for motor PWM: it can claim channel 0 before esp_camera initializes. */
static constexpr ledc_mode_t kMotorPwmMode = LEDC_LOW_SPEED_MODE;
static constexpr ledc_timer_t kMotorPwmTimer = LEDC_TIMER_1;
static constexpr ledc_channel_t kMotorRightPwmChannel = LEDC_CHANNEL_2;
static constexpr ledc_channel_t kMotorLeftPwmChannel = LEDC_CHANNEL_3;
static constexpr uint8_t kAutoArmValidCameraFrames = 3;
/*
 * Arming precondition. Seeing the tape is not the same as starting from a safe
 * pose. A large lateral error at standstill can request an immediate pivot, so
 * refuse to start until the operator places the line roughly in the centre of
 * the image. Mid-run errors remain unrestricted once the policy is armed.
 */
#ifndef LF_ARM_MAX_ABS_E_Y
#define LF_ARM_MAX_ABS_E_Y 0.25f
#endif
/*
 * Motor stall watchdog.  A skid-steer pivot that sits under the loaded
 * breakaway duty holds the encoders at exactly zero, and because the policy is
 * a pure function of an observation that then stops changing, it re-issues the
 * same dead command forever.  Nothing in the line-loss path catches this: the
 * tape is still perfectly visible, so confidence stays at 1 and
 * LF_MAX_LINE_LOSS_S never elapses.  The robot simply sits there pushing
 * locked-rotor current through the H-bridge.
 *
 * This is a hardware-protection stop, not a control law: it never modifies a
 * duty the policy asked for, it only takes the drive away.
 */
#if LF_BENCH_BUILD
/* Dead-man timer.  A bench command drives for this long and then stops on its
 * own, so a stalled pivot cannot be left pushing locked-rotor current while the
 * operator is reading a meter.  Re-send the same command to continue. */
static constexpr int64_t kBenchCommandTimeoutUs = 4000000;
#endif
static constexpr float kStallCommandDeadband = 0.05f;
static constexpr int64_t kStallTimeoutUs = 500000;
static constexpr uint8_t kStallRecoveryLimit = 3;
static constexpr int64_t kCameraPreviewPeriodUs = 2000000;
static constexpr int kCameraPreviewColumns = 40;
static constexpr int kCameraPreviewRows = 18;
static constexpr size_t kCameraPreviewBufferBytes = 112
    + (size_t)kCameraPreviewRows * (size_t)(kCameraPreviewColumns + 1);

static_assert(LF_ENCODER_COUNTS_PER_WHEEL_REV > 0, "Encoder counts per wheel revolution must be positive.");

typedef enum {
    LF_MODE_SAFE = 0,
    LF_MODE_WAIT_FOR_LINE,
    LF_MODE_CAMERA_POLICY,
#if LF_BENCH_BUILD
    LF_MODE_BENCH,
#endif
} lf_control_mode_t;

typedef struct {
    lf_control_mode_t mode;
    bool have_camera_observation;
    float camera_e_y;
    float camera_e_theta_rad;
    float camera_line_loss_s;
    uint32_t camera_samples;
    uint32_t valid_camera_samples;
    uint8_t camera_valid_streak;
    uint8_t camera_threshold_px;
    uint8_t camera_trace_points;
    lf_classic_perception_state_t camera_state;
    float rpm_left;
    float rpm_right;
    float duty_left;
    float duty_right;
    /* What the actor asked for before the generated envelope and slew limit.
     * Logging only the applied duty makes an envelope clamp indistinguishable
     * from a policy that wanted that value. */
    float target_duty_left;
    float target_duty_right;
    bool stalled;
    uint8_t stall_events;
#if LF_BENCH_BUILD
    /* Commanded straight from the serial line: no policy, no envelope, no slew.
     * A plant measurement has to see the duty the operator asked for, not a
     * duty some safety rule reshaped on the way through. */
    float bench_duty_left;
    float bench_duty_right;
    int64_t bench_expires_us;
#endif
} lf_shared_state_t;

static portMUX_TYPE g_encoder_mux = portMUX_INITIALIZER_UNLOCKED;
static portMUX_TYPE g_state_mux = portMUX_INITIALIZER_UNLOCKED;
static volatile int32_t g_encoder_right_count = 0;
static volatile int32_t g_encoder_left_count = 0;
static volatile uint8_t g_encoder_right_state = 0;
static volatile uint8_t g_encoder_left_state = 0;
static volatile uint32_t g_encoder_epoch = 0;
static lf_shared_state_t g_state = {};
static bool g_camera_ready = false;
static bool g_motor_pwm_ready = false;
static volatile bool g_motor_pwm_fault = false;
static int64_t g_next_camera_us = 0;
static uint32_t g_camera_remainder_us = 0;
static int64_t g_next_camera_preview_us = 0;
/* Preview is a serial-only diagnostic.  It stays off for battery deployment. */
static bool g_camera_preview_enabled = false;
static char g_camera_preview[kCameraPreviewBufferBytes] = {};
static size_t g_camera_preview_length = 0;
static size_t g_camera_preview_offset = 0;

static inline float clip_duty(float duty) {
    return fminf(LF_MAX_DUTY, fmaxf(-LF_MAX_DUTY, duty));
}

static uint8_t read_encoder_state(int pin_a, int pin_b) {
    return (uint8_t)((digitalRead(pin_a) ? 2 : 0) | (digitalRead(pin_b) ? 1 : 0));
}

static void IRAM_ATTR update_quadrature(
    volatile int32_t *count,
    volatile uint8_t *previous_state,
    int pin_a,
    int pin_b
) {
    /* 00 -> 01 -> 11 -> 10 is positive; polarity can be flipped in telemetry. */
    static const int8_t transition_delta[16] = {
        0, 1, -1, 0,
        -1, 0, 0, 1,
        1, 0, 0, -1,
        0, -1, 1, 0,
    };
    portENTER_CRITICAL_ISR(&g_encoder_mux);
    const uint8_t current_state = read_encoder_state(pin_a, pin_b);
    const uint8_t table_index = (uint8_t)((*previous_state << 2) | current_state);
    *count += transition_delta[table_index];
    *previous_state = current_state;
    portEXIT_CRITICAL_ISR(&g_encoder_mux);
}

void IRAM_ATTR encoder_right_isr() {
    update_quadrature(&g_encoder_right_count, &g_encoder_right_state, ENC_R_A, ENC_R_B);
}

void IRAM_ATTR encoder_left_isr() {
    update_quadrature(&g_encoder_left_count, &g_encoder_left_state, ENC_L_A, ENC_L_B);
}

static void encoder_snapshot(int32_t *left, int32_t *right, uint32_t *epoch) {
    portENTER_CRITICAL(&g_encoder_mux);
    *left = g_encoder_left_count;
    *right = g_encoder_right_count;
    *epoch = g_encoder_epoch;
    portEXIT_CRITICAL(&g_encoder_mux);
}

static bool configure_motor_pwm_channel(int pin, ledc_channel_t channel) {
    ledc_channel_config_t config = {};
    config.gpio_num = pin;
    config.speed_mode = kMotorPwmMode;
    config.channel = channel;
    config.intr_type = LEDC_INTR_DISABLE;
    config.timer_sel = kMotorPwmTimer;
    config.duty = 0;
    config.hpoint = 0;
    return ledc_channel_config(&config) == ESP_OK;
}

static bool configure_motor_pwm() {
    ledc_timer_config_t timer = {};
    timer.speed_mode = kMotorPwmMode;
    timer.duty_resolution = LEDC_TIMER_8_BIT;
    timer.timer_num = kMotorPwmTimer;
    timer.freq_hz = kPwmFrequencyHz;
    timer.clk_cfg = LEDC_AUTO_CLK;
    if (ledc_timer_config(&timer) != ESP_OK) {
        return false;
    }
    return configure_motor_pwm_channel(MOTOR_R_PWM, kMotorRightPwmChannel)
        && configure_motor_pwm_channel(MOTOR_L_PWM, kMotorLeftPwmChannel);
}

static void write_motor_pwm(ledc_channel_t channel, uint32_t pwm) {
    if (ledc_set_duty(kMotorPwmMode, channel, pwm) != ESP_OK
        || ledc_update_duty(kMotorPwmMode, channel) != ESP_OK) {
        g_motor_pwm_fault = true;
    }
}

static void set_motor_right_duty(float duty) {
    duty = clip_duty(duty);
    const uint32_t pwm = (uint32_t)lroundf(fabsf(duty) * (float)kPwmMaximum);
    if (duty > 0.0f) {
        digitalWrite(MOTOR_R_DIR, HIGH);
    } else if (duty < 0.0f) {
        digitalWrite(MOTOR_R_DIR, LOW);
    }
    write_motor_pwm(kMotorRightPwmChannel, pwm);
}

static void set_motor_left_duty(float duty) {
    duty = clip_duty(duty);
    const uint32_t pwm = (uint32_t)lroundf(fabsf(duty) * (float)kPwmMaximum);
    if (duty > 0.0f) {
        /* Left motor is mounted as a mirror of the right motor. */
        digitalWrite(MOTOR_L_DIR, LOW);
    } else if (duty < 0.0f) {
        digitalWrite(MOTOR_L_DIR, HIGH);
    }
    write_motor_pwm(kMotorLeftPwmChannel, pwm);
}

static void apply_motor_duty(float left, float right) {
    set_motor_left_duty(left);
    set_motor_right_duty(right);
}

static bool configure_camera() {
    camera_config_t config = {};
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = psramFound() ? 2 : 1;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;

    const esp_err_t status = esp_camera_init(&config);
    if (status != ESP_OK) {
        Serial.printf("CAMERA_ERROR:%X\n", status);
        return false;
    }
    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor != NULL) {
        sensor->set_hmirror(sensor, LF_CAMERA_HMIRROR);
        sensor->set_vflip(sensor, LF_CAMERA_VFLIP);
    }
    Serial.printf(
        "CAMERA_OK grayscale=%dx%d HFOV=%.2f\n",
        LF_CAMERA_WIDTH_PX, LF_CAMERA_HEIGHT_PX, LF_CAMERA_HFOV_DEG
    );
    return true;
}

static void append_camera_preview_char(char value) {
    if (g_camera_preview_length + 1 < sizeof(g_camera_preview)) {
        g_camera_preview[g_camera_preview_length++] = value;
    }
}

static char luma_to_preview_char(uint8_t luma) {
    /* Dark tape is dense; a bright floor is whitespace in the serial preview. */
    static constexpr char kLumaRamp[] = "@%#*+=-:. ";
    const size_t index = (size_t)luma * (sizeof(kLumaRamp) - 2) / 255u;
    return kLumaRamp[index];
}

static void queue_camera_preview(
    const uint8_t *luma,
    int width,
    int height,
    const lf_classic_perception_t &perception
) {
    if (!g_camera_preview_enabled || luma == NULL || width <= 0 || height <= 0) {
        return;
    }
    const int64_t now_us = esp_timer_get_time();
    if (now_us < g_next_camera_preview_us || g_camera_preview_offset < g_camera_preview_length) {
        return;
    }
    g_next_camera_preview_us = now_us + kCameraPreviewPeriodUs;
    g_camera_preview_length = 0;
    g_camera_preview_offset = 0;
    const int header = snprintf(
        g_camera_preview, sizeof(g_camera_preview),
        "CAM raw-luma valid=%u state=%u trace=%u thr=%u (dark=@, bright=space)\n",
        (unsigned)perception.valid, (unsigned)perception.state,
        (unsigned)perception.trace_points, (unsigned)perception.threshold_px
    );
    if (header < 0 || (size_t)header >= sizeof(g_camera_preview)) {
        g_camera_preview_length = 0;
        return;
    }
    g_camera_preview_length = (size_t)header;
    for (int preview_y = 0; preview_y < kCameraPreviewRows; ++preview_y) {
        const int first_y = preview_y * height / kCameraPreviewRows;
        const int last_y = (preview_y + 1) * height / kCameraPreviewRows;
        for (int preview_x = 0; preview_x < kCameraPreviewColumns; ++preview_x) {
            const int first_x = preview_x * width / kCameraPreviewColumns;
            const int last_x = (preview_x + 1) * width / kCameraPreviewColumns;
            uint32_t sum = 0;
            uint32_t count = 0;
            for (int y = first_y; y < last_y; ++y) {
                for (int x = first_x; x < last_x; ++x) {
                    sum += luma[(size_t)y * (size_t)width + (size_t)x];
                    ++count;
                }
            }
            const uint8_t average = count == 0 ? 0 : (uint8_t)(sum / count);
            append_camera_preview_char(luma_to_preview_char(average));
        }
        append_camera_preview_char('\n');
    }
    append_camera_preview_char('\n');
}

static void flush_camera_preview() {
    if (g_camera_preview_offset >= g_camera_preview_length) {
        return;
    }
    const int writable = Serial.availableForWrite();
    if (writable <= 0) {
        return;
    }
    const size_t remaining = g_camera_preview_length - g_camera_preview_offset;
    const size_t count = remaining < (size_t)writable ? remaining : (size_t)writable;
    Serial.write((const uint8_t *)(g_camera_preview + g_camera_preview_offset), count);
    g_camera_preview_offset += count;
}

static void update_camera_observation(
    const lf_classic_perception_t &perception,
    uint32_t elapsed_camera_ticks
) {
    const float elapsed_s = (float)elapsed_camera_ticks / LF_CAMERA_HZ;
    bool auto_armed = false;
    bool report_off_centre = false;
    float off_centre_e_y = 0.0f;
    portENTER_CRITICAL(&g_state_mux);
    g_state.camera_samples += elapsed_camera_ticks;
    g_state.camera_state = perception.state;
    g_state.camera_threshold_px = perception.threshold_px;
    g_state.camera_trace_points = perception.trace_points;
    if (perception.valid) {
        g_state.have_camera_observation = true;
        g_state.camera_e_y = perception.e_y;
        g_state.camera_e_theta_rad = perception.e_theta_rad;
        g_state.camera_line_loss_s = 0.0f;
        ++g_state.valid_camera_samples;
        if (g_state.camera_valid_streak < UINT8_MAX) {
            ++g_state.camera_valid_streak;
        }
#if LF_AUTOSTART_CAMERA_POLICY
        if (g_state.mode == LF_MODE_WAIT_FOR_LINE
            && g_state.camera_valid_streak >= kAutoArmValidCameraFrames) {
            if (fabsf(perception.e_y) <= LF_ARM_MAX_ABS_E_Y) {
                g_state.mode = LF_MODE_CAMERA_POLICY;
                g_state.duty_left = 0.0f;
                g_state.duty_right = 0.0f;
                auto_armed = true;
            } else {
                /* Hold the streak so a nudge onto the line arms immediately. */
                g_state.camera_valid_streak = kAutoArmValidCameraFrames;
                report_off_centre = true;
                off_centre_e_y = perception.e_y;
            }
        }
#endif
    } else {
        /* Advance on every missed camera sample, including before first lock. */
        g_state.camera_line_loss_s += elapsed_s;
        g_state.camera_valid_streak = 0;
    }
    portEXIT_CRITICAL(&g_state_mux);
    if (auto_armed) {
        Serial.println("AUTO_ARMED_VALID_LINE");
    } else if (report_off_centre) {
        Serial.printf("WAIT_LINE_OFF_CENTRE ey=%.3f limit=%.2f\n",
                      off_centre_e_y, (double)LF_ARM_MAX_ABS_E_Y);
    }
}

static void process_camera_frame(uint32_t elapsed_camera_ticks) {
    lf_classic_perception_t perception;
    perception.valid = false;
    perception.e_y = 0.0f;
    perception.e_theta_rad = 0.0f;
    perception.threshold_px = 0;
    perception.trace_points = 0;
    perception.state = LF_CLASSIC_PERCEPTION_FRAME_SHAPE;
    if (!g_camera_ready) {
        update_camera_observation(perception, elapsed_camera_ticks);
        return;
    }
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame == NULL) {
        update_camera_observation(perception, elapsed_camera_ticks);
        return;
    }
    perception = lf_classic_perceive(frame->buf, frame->len, frame->width, frame->height);
    if (g_camera_preview_enabled) {
        queue_camera_preview(frame->buf, frame->width, frame->height, perception);
    }
    esp_camera_fb_return(frame);
    update_camera_observation(perception, elapsed_camera_ticks);
}

static uint32_t next_period_us(uint32_t hz, uint32_t *remainder) {
    uint32_t period_us = 1000000u / hz;
    *remainder += 1000000u % hz;
    if (*remainder >= hz) {
        ++period_us;
        *remainder -= hz;
    }
    return period_us;
}

static void policy_task(void *) {
    int32_t prior_left_count = 0;
    int32_t prior_right_count = 0;
    uint32_t prior_encoder_epoch = 0;
    uint32_t period_remainder = 0;
    int64_t prior_tick_us = esp_timer_get_time();
    int64_t deadline_us = prior_tick_us;
    int32_t stall_reference_left = 0;
    int32_t stall_reference_right = 0;
    int64_t stall_since_us = prior_tick_us;
    encoder_snapshot(&prior_left_count, &prior_right_count, &prior_encoder_epoch);
    for (;;) {
        const int64_t now_us = esp_timer_get_time();
        int32_t left_count = 0;
        int32_t right_count = 0;
        uint32_t encoder_epoch = 0;
        encoder_snapshot(&left_count, &right_count, &encoder_epoch);
        int64_t elapsed_us = now_us - prior_tick_us;
        if (elapsed_us < 1) {
            elapsed_us = 1;
        }
        float rpm_left = 0.0f;
        float rpm_right = 0.0f;
        if (encoder_epoch == prior_encoder_epoch) {
            rpm_left = LF_ENCODER_LEFT_SIGN * (float)(left_count - prior_left_count)
                * 60.0f * 1000000.0f / ((float)LF_ENCODER_COUNTS_PER_WHEEL_REV * (float)elapsed_us);
            rpm_right = LF_ENCODER_RIGHT_SIGN * (float)(right_count - prior_right_count)
                * 60.0f * 1000000.0f / ((float)LF_ENCODER_COUNTS_PER_WHEEL_REV * (float)elapsed_us);
        }
        prior_left_count = left_count;
        prior_right_count = right_count;
        prior_encoder_epoch = encoder_epoch;
        prior_tick_us = now_us;

        lf_shared_state_t local;
        portENTER_CRITICAL(&g_state_mux);
        local = g_state;
        portEXIT_CRITICAL(&g_state_mux);

        float duty_left = 0.0f;
        float duty_right = 0.0f;
        float target_left = 0.0f;
        float target_right = 0.0f;
        const bool hardware_fault = !g_motor_pwm_ready || g_motor_pwm_fault;
        bool wait_for_line = false;
#if LF_BENCH_BUILD
        bool bench_expired = false;
        if (!hardware_fault && local.mode == LF_MODE_BENCH) {
            if (now_us < local.bench_expires_us) {
                /* Deliberately bypasses lf_policy_step: the envelope would lift
                 * a small duty to the loaded minimum and hide the very number
                 * this measurement exists to find. */
                duty_left = local.bench_duty_left;
                duty_right = local.bench_duty_right;
            } else {
                bench_expired = true;
            }
        } else if (!hardware_fault && local.mode == LF_MODE_CAMERA_POLICY) {
#else
        if (!hardware_fault && local.mode == LF_MODE_CAMERA_POLICY) {
#endif
            line_following_observation_t observation;
            observation.rpm_left = rpm_left;
            observation.rpm_right = rpm_right;
            observation.duty_prev_left = local.duty_left;
            observation.duty_prev_right = local.duty_right;
            if (!local.have_camera_observation) {
                /* Deployment never drives without a real camera observation. */
                wait_for_line = true;
            } else if (local.camera_line_loss_s > LF_MAX_LINE_LOSS_S) {
                /* No USB command is available in battery deployment. Stop and
                 * automatically require three fresh valid frames before the
                 * policy may run again. A motor/PWM fault remains latched safe. */
                wait_for_line = true;
            } else {
                observation.e_y = local.camera_e_y;
                observation.e_theta_rad = local.camera_e_theta_rad;
                observation.line_confidence = lf_clip(
                    1.0f - local.camera_line_loss_s / LF_MAX_LINE_LOSS_S, 0.0f, 1.0f
                );
                /* Raw actor request, before the envelope, purely for telemetry. */
                const line_following_action_t request = lf_policy_target(&observation);
                target_left = request.duty_left;
                target_right = request.duty_right;
                const line_following_action_t action = lf_policy_step(&observation);
                duty_left = action.duty_left;
                duty_right = action.duty_right;
            }
        }
        duty_left = clip_duty(duty_left);
        duty_right = clip_duty(duty_right);

        /*
         * Stall watchdog.  "Commanding drive but neither encoder has moved"
         * is the only condition here; it deliberately does not look at which
         * way the policy is steering, because a stuck robot is a stuck robot.
         */
        const bool commanding_drive =
            fabsf(duty_left) > kStallCommandDeadband || fabsf(duty_right) > kStallCommandDeadband;
        const bool encoders_moved = left_count != stall_reference_left
            || right_count != stall_reference_right;
        bool stall_tripped = false;
        const bool healthy_motion = commanding_drive && encoders_moved;
        if (!commanding_drive || encoders_moved) {
            stall_reference_left = left_count;
            stall_reference_right = right_count;
            stall_since_us = now_us;
        } else if (now_us - stall_since_us >= kStallTimeoutUs) {
            stall_tripped = true;
            stall_reference_left = left_count;
            stall_reference_right = right_count;
            stall_since_us = now_us;
        }
#if LF_BENCH_BUILD
        /* Finding the duty at which the wheels do NOT turn is the measurement,
         * so the watchdog reports here instead of intervening.  The dead-man
         * timer above is what protects the motors on a bench build. */
        if (local.mode == LF_MODE_BENCH) {
            stall_tripped = false;
        }
#endif
        if (stall_tripped) {
            /* Cut the drive first, decide the mode afterwards. */
            duty_left = 0.0f;
            duty_right = 0.0f;
            wait_for_line = true;
        }
        apply_motor_duty(duty_left, duty_right);

        portENTER_CRITICAL(&g_state_mux);
        g_state.rpm_left = rpm_left;
        g_state.rpm_right = rpm_right;
        g_state.duty_left = duty_left;
        g_state.duty_right = duty_right;
        g_state.target_duty_left = target_left;
        g_state.target_duty_right = target_right;
        g_state.stalled = stall_tripped;
        bool latch_safe = false;
        if (healthy_motion) {
            g_state.stall_events = 0;
        }
        if (stall_tripped) {
            if (g_state.stall_events < UINT8_MAX) {
                ++g_state.stall_events;
            }
            /* A robot that stalls again as soon as it re-arms is mechanically
             * stuck.  Re-arming forever would keep cooking the motors, so give
             * up and stay off until a human intervenes. */
            latch_safe = g_state.stall_events >= kStallRecoveryLimit;
        }
#if LF_BENCH_BUILD
        if (bench_expired && g_state.mode == local.mode) {
            g_state.mode = LF_MODE_SAFE;
            g_state.bench_duty_left = 0.0f;
            g_state.bench_duty_right = 0.0f;
        }
#endif
        if ((hardware_fault || latch_safe) && g_state.mode == local.mode) {
            g_state.mode = LF_MODE_SAFE;
            g_state.duty_left = 0.0f;
            g_state.duty_right = 0.0f;
        } else if (wait_for_line && g_state.mode == local.mode) {
            g_state.mode = LF_MODE_WAIT_FOR_LINE;
            g_state.have_camera_observation = false;
            g_state.camera_line_loss_s = 0.0f;
            g_state.camera_valid_streak = 0;
        }
        portEXIT_CRITICAL(&g_state_mux);

        deadline_us += next_period_us((uint32_t)LF_POLICY_HZ, &period_remainder);
        for (;;) {
            const int64_t remaining_us = deadline_us - esp_timer_get_time();
            if (remaining_us <= 0) {
                break;
            }
            if (remaining_us > 1500) {
                vTaskDelay((TickType_t)(remaining_us / 1000));
            } else {
                delayMicroseconds((uint32_t)remaining_us);
            }
        }
        if (esp_timer_get_time() - deadline_us > 2 * (1000000 / (uint32_t)LF_POLICY_HZ)) {
            deadline_us = esp_timer_get_time();
        }
    }
}

static const char *mode_name(lf_control_mode_t mode) {
    switch (mode) {
        case LF_MODE_WAIT_FOR_LINE:
            return "wait_line";
        case LF_MODE_CAMERA_POLICY:
            return "camera";
#if LF_BENCH_BUILD
        case LF_MODE_BENCH:
            return "bench";
#endif
        default:
            return "safe";
    }
}

static void arm_wait_for_valid_line() {
    portENTER_CRITICAL(&g_state_mux);
    g_state.mode = LF_MODE_WAIT_FOR_LINE;
    g_state.have_camera_observation = false;
    g_state.camera_line_loss_s = 0.0f;
    g_state.camera_valid_streak = 0;
    g_state.duty_left = 0.0f;
    g_state.duty_right = 0.0f;
    g_state.target_duty_left = 0.0f;
    g_state.target_duty_right = 0.0f;
    g_state.stalled = false;
    portEXIT_CRITICAL(&g_state_mux);
}

static void stop_safely() {
    portENTER_CRITICAL(&g_state_mux);
    g_state.mode = LF_MODE_SAFE;
    g_state.camera_valid_streak = 0;
    g_state.duty_left = 0.0f;
    g_state.duty_right = 0.0f;
    g_state.target_duty_left = 0.0f;
    g_state.target_duty_right = 0.0f;
    g_state.stalled = false;
    portEXIT_CRITICAL(&g_state_mux);
}

#if LF_BENCH_BUILD
/* Bench commands.  Present only in a bench build, so a deployed robot cannot be
 * driven by a stray serial byte.
 *
 *   D <left> <right>   drive this duty pair for kBenchCommandTimeoutUs
 *   S                  stop now
 *   Z                  zero the encoder counters
 *
 * Sweep d over 0.30..1.00 and record the first d that turns the wheels, plus
 * the settled rpm, in all three loaded configurations:
 *   straight       D d d
 *   pivot left     D -d d
 *   pivot right    D d -d
 * The robot must be on the floor carrying its own weight; lifted wheels
 * measure a motor, not this vehicle.
 */
static bool handle_bench_command(char *line) {
    const char verb = *line;
    if (verb == 'S' || verb == 's') {
        stop_safely();
        Serial.println("BENCH_STOP");
        return true;
    }
    if (verb == 'Z' || verb == 'z') {
        portENTER_CRITICAL(&g_encoder_mux);
        g_encoder_left_count = 0;
        g_encoder_right_count = 0;
        ++g_encoder_epoch;
        portEXIT_CRITICAL(&g_encoder_mux);
        Serial.println("BENCH_ENCODER_ZERO");
        return true;
    }
    if (verb != 'D' && verb != 'd') {
        return false;
    }
    char *cursor = line + 1;
    char *end = NULL;
    const float left = strtof(cursor, &end);
    if (end == cursor) {
        Serial.println("BENCH_ERROR expected: D <left> <right>");
        return true;
    }
    cursor = end;
    const float right = strtof(cursor, &end);
    if (end == cursor) {
        Serial.println("BENCH_ERROR expected: D <left> <right>");
        return true;
    }
    if (!g_motor_pwm_ready || g_motor_pwm_fault) {
        Serial.println("BENCH_ERROR motor pwm unavailable");
        return true;
    }
    const float clipped_left = clip_duty(left);
    const float clipped_right = clip_duty(right);
    portENTER_CRITICAL(&g_state_mux);
    g_state.mode = LF_MODE_BENCH;
    g_state.bench_duty_left = clipped_left;
    g_state.bench_duty_right = clipped_right;
    g_state.bench_expires_us = esp_timer_get_time() + kBenchCommandTimeoutUs;
    g_state.stall_events = 0;
    portEXIT_CRITICAL(&g_state_mux);
    Serial.printf("BENCH_DRIVE left=%.3f right=%.3f for=%lldms\n",
                  clipped_left, clipped_right, (long long)(kBenchCommandTimeoutUs / 1000));
    return true;
}
#endif

static void handle_serial_line(char *line) {
    while (*line == ' ' || *line == '\t') {
        ++line;
    }
#if LF_BENCH_BUILD
    if (handle_bench_command(line)) {
        return;
    }
#endif
    if (*line != 'V' && *line != 'v') {
        return;
    }
    ++line;
    while (*line == ' ' || *line == '\t') {
        ++line;
    }
    if (*line != '\0') {
        return;
    }
    g_camera_preview_enabled = !g_camera_preview_enabled;
    g_camera_preview_length = 0;
    g_camera_preview_offset = 0;
    g_next_camera_preview_us = 0;
    Serial.println(g_camera_preview_enabled ? "CAMERA_PREVIEW_ON" : "CAMERA_PREVIEW_OFF");
}

static void poll_serial() {
    static char command_buffer[96] = {};
    static size_t length = 0;
    while (Serial.available() > 0) {
        const char value = (char)Serial.read();
        if (value == '\r') {
            continue;
        }
        if (value == '\n') {
            command_buffer[length] = '\0';
            handle_serial_line(command_buffer);
            length = 0;
            continue;
        }
        if (length + 1 < sizeof(command_buffer)) {
            command_buffer[length++] = value;
        } else {
            length = 0;
        }
    }
}

static void print_telemetry() {
    static int64_t next_telemetry_us = 0;
    static bool have_encoder_baseline = false;
    static int64_t prior_encoder_sample_us = 0;
    static int32_t prior_left_count = 0;
    static int32_t prior_right_count = 0;
    static uint32_t prior_encoder_epoch = 0;
    const int64_t now_us = esp_timer_get_time();
    if (g_camera_preview_offset < g_camera_preview_length || now_us < next_telemetry_us) {
        return;
    }
    next_telemetry_us = now_us + 500000;
    lf_shared_state_t local;
    portENTER_CRITICAL(&g_state_mux);
    local = g_state;
    portEXIT_CRITICAL(&g_state_mux);
    int32_t left_count = 0;
    int32_t right_count = 0;
    uint32_t encoder_epoch = 0;
    encoder_snapshot(&left_count, &right_count, &encoder_epoch);

    float mean_rpm_left = 0.0f;
    float mean_rpm_right = 0.0f;
    /* A bench Z command resets the count origin.  It is not a real reverse
     * wheel step, so discard that telemetry interval rather than reporting a
     * large artificial RPM spike. */
    if (have_encoder_baseline && encoder_epoch == prior_encoder_epoch) {
        const int64_t elapsed_us = now_us - prior_encoder_sample_us;
        if (elapsed_us > 0) {
            const float rpm_scale = 60.0f * 1000000.0f
                / ((float)LF_ENCODER_COUNTS_PER_WHEEL_REV * (float)elapsed_us);
            mean_rpm_left = LF_ENCODER_LEFT_SIGN * (float)(left_count - prior_left_count) * rpm_scale;
            mean_rpm_right = LF_ENCODER_RIGHT_SIGN * (float)(right_count - prior_right_count) * rpm_scale;
        }
    }
    prior_encoder_sample_us = now_us;
    prior_left_count = left_count;
    prior_right_count = right_count;
    prior_encoder_epoch = encoder_epoch;
    have_encoder_baseline = true;

    Serial.printf(
        "T mode=%s ey=%.3f eth=%.3f loss=%.3f tgtL=%.3f tgtR=%.3f dutyL=%.3f dutyR=%.3f "
        "rpmTickL=%.1f rpmTickR=%.1f rpmAvgL=%.1f rpmAvgR=%.1f "
        "encL=%ld encR=%ld cam=%lu/%lu trace=%u thr=%u state=%u stall=%u/%u\n",
        mode_name(local.mode), local.camera_e_y, local.camera_e_theta_rad, local.camera_line_loss_s,
        local.target_duty_left, local.target_duty_right,
        local.duty_left, local.duty_right, local.rpm_left, local.rpm_right,
        mean_rpm_left, mean_rpm_right,
        (long)left_count, (long)right_count,
        (unsigned long)local.valid_camera_samples, (unsigned long)local.camera_samples,
        (unsigned)local.camera_trace_points, (unsigned)local.camera_threshold_px,
        (unsigned)local.camera_state,
        (unsigned)(local.stalled ? 1u : 0u), (unsigned)local.stall_events
    );
}

void setup() {
    Serial.begin(kSerialBaud);
    Serial.setTimeout(5);
    delay(500);
    Serial.println();
    Serial.println("ESP32-S3 Sim2Real line-following baseline");
    Serial.printf("policy=%.0fHz camera=%.0fHz encoder=%d counts/rev\n",
                  LF_POLICY_HZ, LF_CAMERA_HZ, LF_ENCODER_COUNTS_PER_WHEEL_REV);
#if LF_BENCH_BUILD
    Serial.println("BENCH BUILD: autostart disabled; D <left> <right> / S / Z available");
#endif
    Serial.printf("policy_id=%s config_id=%s mirror=%d flip=%d\n",
                  LF_POLICY_ID, LF_POLICY_CONFIG_ID, LF_CAMERA_HMIRROR, LF_CAMERA_VFLIP);

    /* Initialize the camera first: it reserves LEDC channel/timer 0 for XCLK. */
    g_camera_ready = configure_camera();

    pinMode(MOTOR_R_DIR, OUTPUT);
    pinMode(MOTOR_L_DIR, OUTPUT);
    g_motor_pwm_ready = configure_motor_pwm();
    if (g_motor_pwm_ready) {
        apply_motor_duty(0.0f, 0.0f);
        Serial.println("MOTOR_PWM_OK timer=1 channels=2,3");
    } else {
        Serial.println("MOTOR_PWM_ERROR");
    }

    pinMode(ENC_R_A, INPUT_PULLUP);
    pinMode(ENC_R_B, INPUT_PULLUP);
    pinMode(ENC_L_A, INPUT_PULLUP);
    pinMode(ENC_L_B, INPUT_PULLUP);
    g_encoder_right_state = read_encoder_state(ENC_R_A, ENC_R_B);
    g_encoder_left_state = read_encoder_state(ENC_L_A, ENC_L_B);
    attachInterrupt(digitalPinToInterrupt(ENC_R_A), encoder_right_isr, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_R_B), encoder_right_isr, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_L_A), encoder_left_isr, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_L_B), encoder_left_isr, CHANGE);

    g_next_camera_us = esp_timer_get_time();

#if CONFIG_FREERTOS_UNICORE
    const BaseType_t task_created = xTaskCreate(policy_task, "line_policy", 8192, NULL, 2, NULL);
#else
    const BaseType_t task_created = xTaskCreatePinnedToCore(policy_task, "line_policy", 8192, NULL, 2, NULL, 0);
#endif
    if (task_created != pdPASS) {
        Serial.println("POLICY_TASK_ERROR");
        stop_safely();
    } else {
#if LF_AUTOSTART_CAMERA_POLICY
        if (g_camera_ready && g_motor_pwm_ready && !g_motor_pwm_fault) {
            arm_wait_for_valid_line();
            Serial.println("AUTOSTART_WAIT_LINE valid_frames=3");
        } else {
            stop_safely();
            Serial.println("AUTOSTART_DISABLED_NOT_READY");
        }
#endif
    }
    Serial.println("Serial diagnostic: V toggles the ASCII camera preview.");
}

void loop() {
    poll_serial();
    const int64_t now_us = esp_timer_get_time();
    if (now_us >= g_next_camera_us) {
        uint32_t elapsed_camera_ticks = 0;
        do {
            g_next_camera_us += next_period_us((uint32_t)LF_CAMERA_HZ, &g_camera_remainder_us);
            ++elapsed_camera_ticks;
        } while (now_us >= g_next_camera_us);
        process_camera_frame(elapsed_camera_ticks);
    }
    if (g_camera_preview_enabled) {
        flush_camera_preview();
    }
    print_telemetry();
    delay(1);
}
