//Main code for ESP32 deployment
#include <driver/i2s.h> //I2S driver library
#include <LiquidCrystal.h> //LCD library

//TensorFlowLite
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"
#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "recognize_commands.h"
#include "micro_features_generator.h"
#include "micro_model_settings.h"

//Our model
#include "model.h"

//LED Pin 
#define LED_PIN 2
uint8_t led_state = 0;

//LCD Pins
#define LCD_RS 13
#define LCD_E 32
#define LCD_D4 14
#define LCD_D5 27
#define LCD_D6 26
#define LCD_D7 33

LiquidCrystal lcd(LCD_RS, LCD_E, LCD_D4, LCD_D5, LCD_D6, LCD_D7);

namespace{
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  FeatureProvider* feature_provider = nullptr;
  RecognizeCommands* recognizer = nullptr;
  int32_t previous_time = 0;
  constexpr int kTensorArenaSize = 35*1024;
  uint8_t tensor_arena[kTensorArenaSize];
  int8_t feature_buffer[kFeatureElementCount];
  int8_t* model_input_buffer = nullptr;
}

//Class Labels
#define ON 2
#define OFF 3
#define UNKNOWN 1
int last_result = -1;

//Setting up audio functionalities
void setupAudio(){
 static FeatureProvider static_feature_provider(kFeatureElementCount, feature_buffer);
 feature_provider = &static_feature_provider;
 static RecognizeCommands static_recognizer;
 recognizer = &static_recognizer;
 previous_time = 0;
}

//Computing the MFCC for model input
void computeMFCC(){
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    MicroPrintf( "Feature generation failed");
    return;
  }
  previous_time = current_time;
  if (how_many_new_slices == 0) {
    return;
  }
}

//Setting up the model
void setupModel(){
  model = tflite::GetModel(g_model);
  static tflite::MicroMutableOpResolver<5> resolver;
  if (resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  if (resolver.AddConv2D() != kTfLiteOk) {
    return;
  }
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (resolver.AddReshape() != kTfLiteOk) {
    return;
  }

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
  MicroPrintf("AllocateTensors FAILED");
  while (1) {}
}

MicroPrintf("AllocateTensors OK");

input = interpreter->input(0);
if ((input->dims->size != 2) || (input->dims->data[0] != 1) || (input->dims->data[1] != (kFeatureCount * kFeatureSize)) || (input->type != kTfLiteInt8)) {
    MicroPrintf("Bad input tensor parameters in model");
    return;
  }
 model_input_buffer = tflite::GetTensorData<int8_t>(input);
}

//Giving audio to input tensors
void fillInputTensors() {
 for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }
}
//Give input to the model and get output
int runModel() {
  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke FAILED");
    return UNKNOWN;
  }
  output = interpreter->output(0);
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;
  int max_idx = 0;
  float max_result = 0.0;
  // Dequantize output values and find the max
  for (int i = 0; i < kCategoryCount; i++) {
    float current_result = (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point)*output_scale;
    if (current_result > max_result) {
      max_result = current_result; // update max result
      max_idx = i; // update category
    }
  }
  if (max_result > 0.8f) {
    MicroPrintf("Detected %7s, score: %.2f", kCategoryLabels[max_idx], static_cast<double>(max_result));
  }
  return max_idx;
}

void setup() {
 Serial.begin(115200);
 pinMode(LED_PIN, OUTPUT);
 setupAudio();
 setupModel();
 lcd.begin(16,2);
 lcd.print("KEYWORD SPOTTING");
}

void loop() {
Serial.println("STEP 1: mfcc");
computeMFCC();

Serial.println("STEP 2: fill");
fillInputTensors();

Serial.println("STEP 3: invoke");
int result = runModel();

Serial.println("STEP 4: led");
  if(result == ON){
   if(led_state == 0){
    led_state = 1;
   }
   else{
    led_state = 1;
   }
  }
  else if(result == OFF){
   if(led_state == 1){
    led_state = 0;
   }
   else{
    led_state = 0;
   }
  }
  if(led_state == 1){
    Serial.println("LED is ON!");
    lcd.setCursor(0,1);
    lcd.print("LED is ON!");
  }
  else if(led_state == 0){
    Serial.println("LED is OFF!");
    lcd.setCursor(0,1);
    lcd.print("LED is OFF!");
  }
  digitalWrite(LED_PIN, led_state);
}


