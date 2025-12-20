//main code for esp32 deployment of the trained model

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
