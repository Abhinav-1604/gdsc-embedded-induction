//Main code for ESP32 deployment

#include <driver/i2s.h> //I2S driver library
#include <LiquidCrystal.h> //LCD library
//Edge Impulse libraries
#include "dsp/speechpy/speechpy.hpp"
#include "dsp/config.hpp"

//TensorFlowLite
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

//Our model
#include "kws_model.h"

//LED Pin 
#define LED_PIN 4

//Microphone Pins
#define I2S_WS 25
#define I2S_SD 33
#define I2S_SCK 32

//LCD Pins
#define LCD_RS 23
#define LCD_E 22
#define LCD_D4 21
#define LCD_D5 19
#define LCD_D6 18
#define LCD_D7 5

LiquidCrystal lcd(LCD_RS, LCD_E, LCD_D4, LCD_D5, LCD_D6, LCD_D7);

//I2S Processor Selection
#define I2S_PORT I2S_NUM_0

//Audio Settings
#define SAMPLE_RATE 16000
#define AUDIO_LEN 16000
int16_t audio_buffer[AUDIO_LEN];

//MFCC Features
#define NUM_FRAMES 49
#define NUM_MFCC 13
float mfcc features[NUM_FRAMES][NUM_MFCC];

static mfcc_config_t mfcc_config = {
  .num filters = 40,
  .num_cepstral  = NUM_MFCC,
  .frame_length  = 480,   // 30 ms
  .frame_stride  = 320,   // 20 ms
  .low_frequency = 20,
  .high_frequency= 8000,
  .fft_length    = 512,
  .pre_emphasis  = 0.98,
  .window_type   = MFCC_WINDOW_HAMMING,
  .log_scale     = true
};

//TFLite Setup
constexpr int kTensorArenaSize = 70*1024;
uint8_t tensor_arena[kTensorArenaSize];

namespace{
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;
}

//Class Labels
#define ON 0
#define OFF 1
#define UNKNOWN 2

//I2S Setup
void SetupI2S(){
 i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 512,
    .use_apll = false,
  };
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
 i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num  = I2S_WS,
    .data_out_num = -1,
    .data_in_num  = I2S_SD
  };
  i2s_set_pin(I2S_PORT, &pin_config);
}
void setup() {
  // put your setup code here, to run once:

}

void loop() {
  // put your main code here, to run repeatedly:

}
