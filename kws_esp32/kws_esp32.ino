//Main code for ESP32 deployment

#include <driver/i2s.h> //I2S driver library
#include <LiquidCrystal.h> //LCD library

//TensorFlowLite
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"
#include "micro_features_generator.h"
#include "micro_model_settings.h"

//Our model
#include "kws_model.h"

//LED Pin 
#define LED_PIN 13

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
static int16_t audio_buffer[kAudioSampleFrequency];

//MFCC Features
static int8_t mfcc_buffer[kFeatureCount][kFeatureSize];

//TFLite Setup
constexpr int kTensorArenaSize = 60*1024;
uint8_t tensor_arena[kTensorArenaSize];

namespace{
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
}

//Class Labels
#define ON 0
#define OFF 1
#define UNKNOWN 2
int last_result = -1;

//I2S Setup
void setupI2S(){
 i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = kAudioSampleFrequency,
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

//Recording Audio
void recordAudio(){
  size_t bytes_read;
  int i = 0;
  while(i<kAudioSampleFrequency){
    i2s_read(I2S_PORT, &audio_buffer[i], (kAudioSampleFrequency - i)*sizeof(int16_t), &bytes_read, portMAX_DELAY);
    i+=(bytes_read/sizeof(int16_t));
  }
}

//Extracting MFCC from recorded audio
void computeMFCC(){
 size_t num_samples;
 GenerateFeatures(audio_buffer, kAudioSampleFrequency, &mfcc_buffer);
}

//Setting up the model
void setupModel(){
  model = tflite::GetModel(kws_model_tflite);
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize);

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
  MicroPrintf("AllocateTensors FAILED");
  while (1) {}
}

MicroPrintf("AllocateTensors OK");

  input = interpreter->input(0);
  output = interpreter->output(0);
}

//Giving audio to input tensors
void fillInputTensors(){
  for(int i=0; i<kFeatureCount; i++){
    for(int j=0; j<kFeatureSize; j++){
     input->data.int8[i] = mfcc_buffer[i][j];
    }
  }
}

//Give input to the model and get output
int runModel(){
  interpreter->Invoke();
  int best = 0;
  int8_t max_val = output->data.int8[0];
  for(int i=1;i<3;i++){
    if(output->data.int8[i]>max_val){
      max_val = output->data.int8[i];
      best = i;
    }
  }
  return best;
}

//LCD functionality
void updateLCD(int result){
  if(result == last_result){
    return;
  }
lcd.clear();
lcd.setCursor(0,0);
lcd.print("STATUS: ");
lcd.setCursor(0,1);
if(result == ON){
  lcd.print("LED is ON now!");
}
else if(result == OFF){
  lcd.print("LED is OFF now!");
}
else{
  lcd.print("UNKNOWN AUDIO");
}
last_result = result;
}

void setup() {
 Serial.begin(115200);
 pinMode(LED_PIN, OUTPUT);
 lcd.begin(16,2);
 lcd.print("Keyword Spotting On ESP32");
 setupI2S();
 setupModel();
}

void loop() {
  recordAudio();
  computeMFCC();
  fillInputTensors();
  int result = runModel();
  updateLCD(result);
  if(result == ON){
    digitalWrite(LED_PIN, HIGH);
    Serial.println("LED is ON now!");
  }
  else{
    digitalWrite(LED_PIN, LOW);
    Serial.println("LED is OFF now!");
  }

}
