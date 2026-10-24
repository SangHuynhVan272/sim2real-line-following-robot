// Arduino ESP32 core 3.x; ESP32-S3 N16R8; native USB CDC.
// USB CDC On Boot=Enabled; USB Mode=Hardware CDC and JTAG; PSRAM=OPI.
// Protocol: G starts stream; S stops stream and motors;
// D <sequence> <left> <right> replies to a frame. Only cameraTask writes USB.
#include <Arduino.h>
#include <esp_camera.h>
#include <esp_timer.h>
#include <math.h>
#include <string.h>
#if !ARDUINO_USB_CDC_ON_BOOT
#error "Enable USB CDC On Boot"
#endif

#define ENC_R_A 1
#define ENC_R_B 2
#define ENC_L_A 3
#define ENC_L_B 46
#define MOTOR_R_IN1 41
#define MOTOR_R_IN2 40
#define MOTOR_R_PWM 42
#define MOTOR_L_IN1 48
#define MOTOR_L_IN2 39
#define MOTOR_L_PWM 47
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

static constexpr float COUNTS_PER_REV=940.0f; // Measure on your wheels.
static constexpr float ENCODER_LEFT_SIGN=1.0f, ENCODER_RIGHT_SIGN=1.0f;
static constexpr bool LEFT_FORWARD_HIGH=false, RIGHT_FORWARD_HIGH=true;
static constexpr int HMIRROR=1, VFLIP=1;
static constexpr int64_t WATCHDOG_US=250000, REPLY_LIMIT_US=200000;
static constexpr int64_t FRAME_PERIOD_US=100000;
static portMUX_TYPE encoderMux=portMUX_INITIALIZER_UNLOCKED;
static portMUX_TYPE stateMux=portMUX_INITIALIZER_UNLOCKED;
static volatile uint32_t countL=0,countR=0;
static volatile uint8_t previousL=0,previousR=0;
static float rpmL=0,rpmR=0,dutyL=0,dutyR=0;
static uint32_t sequence=0,pendingSequence=0;
static int64_t requestUs=0,lastCommandUs=0;
static bool pending=false,streaming=false,cameraReady=false;
static bool pwmLeftOK=false,pwmRightOK=false,psramOK=false;
static esp_err_t cameraError=ESP_OK;
static char commandBuffer[96];
static size_t commandLength=0;
static bool commandOverflow=false;

static uint8_t encoderState(int a,int b) {
  return (digitalRead(a)?2:0)|(digitalRead(b)?1:0);
}
static void IRAM_ATTR updateEncoder(volatile uint32_t *count,volatile uint8_t *old,int a,int b) {
  static const int8_t table[16]={0,1,-1,0,-1,0,0,1,1,0,0,-1,0,-1,1,0};
  portENTER_CRITICAL_ISR(&encoderMux);
  uint8_t state=encoderState(a,b);
  *count+=table[(*old<<2)|state]; *old=state;
  portEXIT_CRITICAL_ISR(&encoderMux);
}
void IRAM_ATTR encoderLeftISR() { updateEncoder(&countL,&previousL,ENC_L_A,ENC_L_B); }
void IRAM_ATTR encoderRightISR() { updateEncoder(&countR,&previousR,ENC_R_A,ENC_R_B); }

static void writeMotor(int in1,int in2,int pwm,float duty,bool forwardHigh,bool attached) {
  if(!isfinite(duty)) duty=0;
  duty=constrain(duty,-1.0f,1.0f);
  if(attached) ledcWrite(pwm,0);
  if(!attached || duty==0) {
    digitalWrite(in1,LOW);digitalWrite(in2,LOW);return;
  }
  bool firstHigh=(duty>0)==forwardHigh;
  digitalWrite(in1,firstHigh?HIGH:LOW);
  digitalWrite(in2,firstHigh?LOW:HIGH);
  ledcWrite(pwm,(uint32_t)lroundf(fabsf(duty)*255));
}
static void applyMotors(float left,float right) {
  left=isfinite(left)?constrain(left,-1.0f,1.0f):0;
  right=isfinite(right)?constrain(right,-1.0f,1.0f):0;
  writeMotor(MOTOR_L_IN1,MOTOR_L_IN2,MOTOR_L_PWM,left,LEFT_FORWARD_HIGH,pwmLeftOK);
  writeMotor(MOTOR_R_IN1,MOTOR_R_IN2,MOTOR_R_PWM,right,RIGHT_FORWARD_HIGH,pwmRightOK);
  portENTER_CRITICAL(&stateMux);
  dutyL=pwmLeftOK?left:0; dutyR=pwmRightOK?right:0;
  portEXIT_CRITICAL(&stateMux);
}
static bool initCamera() {
  camera_config_t c={};
  c.ledc_channel=LEDC_CHANNEL_0; c.ledc_timer=LEDC_TIMER_0;
  c.pin_d0=Y2_GPIO_NUM;c.pin_d1=Y3_GPIO_NUM;c.pin_d2=Y4_GPIO_NUM;c.pin_d3=Y5_GPIO_NUM;
  c.pin_d4=Y6_GPIO_NUM;c.pin_d5=Y7_GPIO_NUM;c.pin_d6=Y8_GPIO_NUM;c.pin_d7=Y9_GPIO_NUM;
  c.pin_xclk=XCLK_GPIO_NUM;c.pin_pclk=PCLK_GPIO_NUM;
  c.pin_vsync=VSYNC_GPIO_NUM;c.pin_href=HREF_GPIO_NUM;
  c.pin_sccb_sda=SIOD_GPIO_NUM;c.pin_sccb_scl=SIOC_GPIO_NUM;
  c.pin_pwdn=PWDN_GPIO_NUM;c.pin_reset=RESET_GPIO_NUM;
  c.xclk_freq_hz=20000000;c.pixel_format=PIXFORMAT_GRAYSCALE;
  c.frame_size=FRAMESIZE_QVGA;c.jpeg_quality=12;
  c.fb_count=2;c.fb_location=CAMERA_FB_IN_PSRAM;c.grab_mode=CAMERA_GRAB_LATEST;
  cameraError=esp_camera_init(&c);
  if(cameraError!=ESP_OK) return false;
  sensor_t *s=esp_camera_sensor_get();
  if(s) {s->set_hmirror(s,HMIRROR);s->set_vflip(s,VFLIP);}
  return true;
}
static uint32_t crc32(const uint8_t *data,size_t n) {
  uint32_t crc=0xffffffff;
  for(size_t i=0;i<n;i++) {
    crc^=data[i];
    for(int b=0;b<8;b++) crc=(crc>>1)^((crc&1)?0xedb88320:0);
  }
  return ~crc;
}
static void put32(uint8_t *p,uint32_t v) {p[0]=v;p[1]=v>>8;p[2]=v>>16;p[3]=v>>24;}
static void put16(uint8_t *p,uint16_t v) {p[0]=v;p[1]=v>>8;}
static void putFloat(uint8_t *p,float v) {uint32_t b;memcpy(&b,&v,4);put32(p,b);}
static bool writeAll(const uint8_t *p,size_t n) {
  int64_t start=esp_timer_get_time();
  while(n) {
    if(!Serial || esp_timer_get_time()-start>150000) return false;
    int room=Serial.availableForWrite();
    if(room<=0) {vTaskDelay(1);continue;}
    size_t chunk=n;
    if(chunk>512) chunk=512;
    if(chunk>(size_t)room) chunk=room;
    size_t wrote=Serial.write(p,chunk);
    p+=wrote;n-=wrote;
    if(!wrote) vTaskDelay(1);
  }
  return true;
}
static void cameraTask(void *) {
  int64_t lastStatus=0;
  for(;;) {
    if(!Serial) {vTaskDelay(pdMS_TO_TICKS(10));continue;}
    int64_t now=esp_timer_get_time();bool active,busy;
    portENTER_CRITICAL(&stateMux);
    active=streaming;busy=pending && now-requestUs<REPLY_LIMIT_US;
    portEXIT_CRITICAL(&stateMux);
    if(!cameraReady || !active) {
      if(now-lastStatus>=1000000) {
        lastStatus=now;
        Serial.printf("!LF STATUS ready=%d PWM_L=%d PWM_R=%d PSRAM=%d bytes=%u camera=0x%X (%s) streaming=%d\n",
          cameraReady,pwmLeftOK,pwmRightOK,psramOK,(unsigned)ESP.getPsramSize(),
          (unsigned)cameraError,esp_err_to_name(cameraError),active);
      }
      vTaskDelay(pdMS_TO_TICKS(10));continue;
    }
    if(busy) {vTaskDelay(pdMS_TO_TICKS(2));continue;}
    camera_fb_t *f=esp_camera_fb_get();
    if(!f) {Serial.println("!LF ERROR framebuffer unavailable");vTaskDelay(pdMS_TO_TICKS(100));continue;}
    if(f->format!=PIXFORMAT_GRAYSCALE || f->width!=320 || f->height!=240 || f->len!=76800) {
      Serial.printf("!LF ERROR frame width=%u height=%u len=%u format=%d\n",
        (unsigned)f->width,(unsigned)f->height,(unsigned)f->len,(int)f->format);
      esp_camera_fb_return(f);vTaskDelay(pdMS_TO_TICKS(100));continue;
    }
    uint8_t header[40]={'L','F','P','1'};
    now=esp_timer_get_time();
    portENTER_CRITICAL(&stateMux);
    uint32_t seq=++sequence;pendingSequence=seq;pending=true;requestUs=now;
    float l=rpmL,r=rpmR,dl=dutyL,dr=dutyR;
    portEXIT_CRITICAL(&stateMux);
    put32(header+4,seq);put32(header+8,(uint32_t)now);
    put16(header+12,f->width);put16(header+14,f->height);
    putFloat(header+16,l);putFloat(header+20,r);putFloat(header+24,dl);putFloat(header+28,dr);
    put32(header+32,f->len);put32(header+36,crc32(f->buf,f->len));
    bool sent=writeAll(header,40) && writeAll(f->buf,f->len);
    esp_camera_fb_return(f);
    if(!sent) {
      portENTER_CRITICAL(&stateMux);pending=false;portEXIT_CRITICAL(&stateMux);
      // Do not insert text into a partially sent payload. Host resynchronizes by CRC.
    }
    int64_t remaining=FRAME_PERIOD_US-(esp_timer_get_time()-now);
    if(remaining>0) vTaskDelay(pdMS_TO_TICKS((uint32_t)(remaining/1000)+1));
  }
}
static void handleCommand(const char *line) {
  if(!strcmp(line,"S") || !strcmp(line,"G")) {
    applyMotors(0,0);
    portENTER_CRITICAL(&stateMux);
    lastCommandUs=0;pending=false;streaming=!strcmp(line,"G");
    portEXIT_CRITICAL(&stateMux);return;
  }
  unsigned long seq;float l,r;char extra;
  if(sscanf(line,"D %lu %f %f %c",&seq,&l,&r,&extra)!=3 ||
    !isfinite(l)||!isfinite(r)||fabsf(l)>1||fabsf(r)>1) return;
  int64_t now=esp_timer_get_time();bool accept;
  portENTER_CRITICAL(&stateMux);
  accept=cameraReady && streaming && pending && seq==pendingSequence && now-requestUs<REPLY_LIMIT_US;
  if(accept) {pending=false;lastCommandUs=now;}
  portEXIT_CRITICAL(&stateMux);
  if(accept) applyMotors(l,r);
}
void setup() {
  pinMode(MOTOR_L_IN1,OUTPUT);pinMode(MOTOR_L_IN2,OUTPUT);
  pinMode(MOTOR_R_IN1,OUTPUT);pinMode(MOTOR_R_IN2,OUTPUT);
  pinMode(MOTOR_L_PWM,OUTPUT);pinMode(MOTOR_R_PWM,OUTPUT);
  digitalWrite(MOTOR_L_IN1,LOW);digitalWrite(MOTOR_L_IN2,LOW);
  digitalWrite(MOTOR_R_IN1,LOW);digitalWrite(MOTOR_R_IN2,LOW);
  digitalWrite(MOTOR_L_PWM,LOW);digitalWrite(MOTOR_R_PWM,LOW);
  Serial.begin(921600);
  // Channels 2/3 share motor timer 1; camera uses channel 0 / timer 0.
  pwmLeftOK=ledcAttachChannel(MOTOR_L_PWM,20000,8,2);
  pwmRightOK=ledcAttachChannel(MOTOR_R_PWM,20000,8,3);
  applyMotors(0,0);psramOK=psramFound();
  pinMode(ENC_L_A,INPUT_PULLUP);pinMode(ENC_L_B,INPUT_PULLUP);
  pinMode(ENC_R_A,INPUT_PULLUP);pinMode(ENC_R_B,INPUT_PULLUP);
  previousL=encoderState(ENC_L_A,ENC_L_B);previousR=encoderState(ENC_R_A,ENC_R_B);
  attachInterrupt(ENC_L_A,encoderLeftISR,CHANGE);attachInterrupt(ENC_L_B,encoderLeftISR,CHANGE);
  attachInterrupt(ENC_R_A,encoderRightISR,CHANGE);attachInterrupt(ENC_R_B,encoderRightISR,CHANGE);
  if(pwmLeftOK && pwmRightOK && psramOK) cameraReady=initCamera();
  if(xTaskCreatePinnedToCore(cameraTask,"cameraUSB",8192,nullptr,1,nullptr,0)!=pdPASS) {
    cameraReady=false;
    for(;;) {applyMotors(0,0);if(Serial) Serial.println("!LF ERROR task creation failed");delay(1000);}
  }
}
void loop() {
  static uint32_t oldL=0,oldR=0;static int64_t oldUs=esp_timer_get_time();
  int64_t now=esp_timer_get_time();
  if(now-oldUs>=20000) {
    uint32_t l,r;
    portENTER_CRITICAL(&encoderMux);l=countL;r=countR;portEXIT_CRITICAL(&encoderMux);
    float dt=(now-oldUs)*1e-6f;
    float rl=ENCODER_LEFT_SIGN*(int32_t)(l-oldL)*60/(COUNTS_PER_REV*dt);
    float rr=ENCODER_RIGHT_SIGN*(int32_t)(r-oldR)*60/(COUNTS_PER_REV*dt);
    portENTER_CRITICAL(&stateMux);rpmL=rl;rpmR=rr;portEXIT_CRITICAL(&stateMux);
    oldL=l;oldR=r;oldUs=now;
  }
  while(Serial.available()) {
    char c=Serial.read();if(c=='\r') continue;
    if(c=='\n') {
      if(!commandOverflow) {commandBuffer[commandLength]=0;handleCommand(commandBuffer);}
      commandLength=0;commandOverflow=false;
    } else if(!commandOverflow && commandLength<sizeof(commandBuffer)-1) commandBuffer[commandLength++]=c;
    else commandOverflow=true;
  }
  int64_t last;bool active;
  portENTER_CRITICAL(&stateMux);last=lastCommandUs;active=streaming;portEXIT_CRITICAL(&stateMux);
  if(!cameraReady || !active || !Serial || !last || esp_timer_get_time()-last>WATCHDOG_US) applyMotors(0,0);
  delay(1);
}
