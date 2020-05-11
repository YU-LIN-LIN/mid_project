#include "accelerometer_handler.h"

#include "config.h"

#include "magic_wand_model_data.h"

#include "mbed.h"

#include "uLCD_4DGL.h"

#include <cmath>
#include "DA7212.h"


#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "tensorflow/lite/micro/micro_interpreter.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/version.h"

#define bufferLength 32
#define signalLength 196


DA7212 audio;

Serial pc(USBTX, USBRX);

int16_t waveform[kAudioTxBufferSize];
uLCD_4DGL uLCD(D1, D0, D2);
DigitalIn sw2(SW2);
InterruptIn button(SW3);
DigitalIn sw3(SW3);

// The gesture index of the prediction

int gesture_index;

//EventQueue queue(32 * EVENTS_EVENT_SIZE);
Thread t(osPriorityNormal, 120 * 1024 /*120K stack size*/);    //  expand the size of thread stack
Thread t1;

EventQueue queue(32 * EVENTS_EVENT_SIZE);

int song0[42];/* = {
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261};*/

int song1[32];/* = {
  261, 294, 330, 261, 261, 294, 330, 261,
  330, 349, 392, 330, 349, 392,
  392, 440, 392, 349, 330, 261, 
  392, 440, 392, 349, 330, 261,
  261, 196, 261, 261, 196, 261};*/

int song2[24];/* = {
  392, 330, 330, 349, 294, 294,
  261, 293, 329, 349, 392, 392, 392,
  392, 330, 330, 349, 294, 294, 
  261, 330, 392, 392, 261};*/

int noteLength0[42];/* = {
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2};*/

int noteLength1[32];/* = {
  2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 4, 2, 2, 4,
  1, 1, 1, 1, 2, 2,
  1, 1, 1, 1, 2, 2,
  2, 2, 4, 2, 2, 4};*/

int noteLength2[24];/* = {
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 4};*/

// Return the result of the last prediction
int PredictGesture(float* output) {

  // How many times the most recent gesture has been matched in a row

  static int continuous_count = 0;

  // The result of the last prediction

  static int last_predict = -1;


  // Find whichever output has a probability > 0.8 (they sum to 1)

  int this_predict = -1;

  for (int i = 0; i < label_num; i++) {

    if (output[i] > 0.8) this_predict = i;

  }


  // No gesture was detected above the threshold

  if (this_predict == -1) {

    continuous_count = 0;

    last_predict = label_num;

    return label_num;

  }


  if (last_predict == this_predict) {

    continuous_count += 1;

  } else {

    continuous_count = 0;

  }

  last_predict = this_predict;


  // If we haven't yet had enough consecutive matches for this gesture,

  // report a negative result

  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {

    return label_num;

  }

  // Otherwise, we've seen a positive result, so clear all our variables

  // and report it

  continuous_count = 0;

  last_predict = -1;


  return this_predict;

}

//bool got_data, bool should_clear_buffer, int gesture_index, tflite::ErrorReporter* error_reporter, 
// TfLiteTensor* model_input, int input_length, tflite::MicroInterpreter* interpreter
void DNN() 
{
  // Create an area of memory to use for input, output, and intermediate arrays.

  // The size of this will depend on the model you're using, and may need to be

  // determined by experimentation.

  constexpr int kTensorArenaSize = 60 * 1024;

  uint8_t tensor_arena[kTensorArenaSize];


  // Whether we should clear the buffer next time we fetch data

  bool should_clear_buffer = false;

  bool got_data = false;


  


  // Set up logging.

  static tflite::MicroErrorReporter micro_error_reporter;

  tflite::ErrorReporter* error_reporter = &micro_error_reporter;


  // Map the model into a usable data structure. This doesn't involve any

  // copying or parsing, it's a very lightweight operation.

  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {

    error_reporter->Report(

        "Model provided is schema version %d not equal "

        "to supported version %d.",

        model->version(), TFLITE_SCHEMA_VERSION);

    return -1;

  }


  // Pull in only the operation implementations we need.

  // This relies on a complete list of all the ops needed by this graph.

  // An easier approach is to just use the AllOpsResolver, but this will

  // incur some penalty in code space for op implementations that are not

  // needed by this graph.

  static tflite::MicroOpResolver<6> micro_op_resolver;

  micro_op_resolver.AddBuiltin(

      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,

      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,

                               tflite::ops::micro::Register_MAX_POOL_2D());
micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,

                               tflite::ops::micro::Register_RESHAPE());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,

                               tflite::ops::micro::Register_CONV_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,

                               tflite::ops::micro::Register_FULLY_CONNECTED());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,

                               tflite::ops::micro::Register_SOFTMAX());


  // Build an interpreter to run the model with

  static tflite::MicroInterpreter static_interpreter(

      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);

  tflite::MicroInterpreter* interpreter = &static_interpreter;


  // Allocate memory from the tensor_arena for the model's tensors

  interpreter->AllocateTensors();


  // Obtain pointer to the model's input tensor

  TfLiteTensor* model_input = interpreter->input(0);

  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||

      (model_input->dims->data[1] != config.seq_length) ||

      (model_input->dims->data[2] != kChannelNumber) ||

      (model_input->type != kTfLiteFloat32)) {

    error_reporter->Report("Bad input tensor parameters in model");

    return -1;

  }


  int input_length = model_input->bytes / sizeof(float);


  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);

  if (setup_status != kTfLiteOk) {

    error_reporter->Report("Set up failed\n");

    return -1;

  }


  error_reporter->Report("Set up successful...\n");

  int execute_once = 1;

  while (true) {


    // Attempt to read new data from the accelerometer

    got_data = ReadAccelerometer(error_reporter, model_input->data.f,

                                 input_length, should_clear_buffer);


    // If there was no new data,

    // don't try to clear the buffer again and wait until next time

    if (!got_data) {

      should_clear_buffer = false;

      continue;

    }


    // Run inference, and report any error

    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk) {

      error_reporter->Report("Invoke failed on index: %d\n", begin_index);

      continue;

    }


    // Analyze the results to obtain a prediction

    gesture_index = PredictGesture(interpreter->output(0)->data.f);


    // Clear the buffer next time we read data

    should_clear_buffer = gesture_index < label_num;
    execute_once = 1;
  
    // Produce an output
    // gesture_index = 0 : Ring
    // gesture_index = 1 : Slope
    // gesture_index = 2 : N

    if (gesture_index < label_num) {

      error_reporter->Report(config.output_message[gesture_index]);
      execute_once--;
    }

  }
}

void playNote(int freq)
{
  for(int i = 0; i < kAudioTxBufferSize; i++)
  {
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  audio.spk.play(waveform, kAudioTxBufferSize);
}


int main(int argc, char* argv[]) 
{
  int song_th = 0;                    // decide which song
  t.start(DNN);
  bool change = 0;    
  int mode = 0;                        // 0 : previos; 1 : next; 2 : select
  bool mp3_or_Taiko = 0;               // 0 : mp3; 1 : Taiko
  int n;                               // for Taiko wait
  int score = 0;                       // score for Taiko
  
  float signal[signalLength];
  char serialInBuffer[bufferLength];
  int serialCount = 0;



  uLCD.cls();
  uLCD.text_width(2); //2X size text
  uLCD.text_height(2);
  uLCD.printf("\n\n\nLoading..");
  int i = 0;
  serialCount = 0;
  //audio.spk.pause();
  while(i < signalLength/2)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 3)
      {
        serialInBuffer[serialCount] = '\0';
        signal[i] = (int) atoi(serialInBuffer);
        serialCount = 0;
        i++;
      }
    }
  }
  while(i < signalLength)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 1)
      {
        serialInBuffer[serialCount] = '\0';
        signal[i] = (int) atoi(serialInBuffer);
        serialCount = 0;
        i++;
      }
    }
  }
  for (i = 0; i < 42; i++) {
    song0[i] = signal[i];
  }
  for (i = 42; i < 74; i++) {
    song1[i-42] = signal[i];
  }
  for (i = 74; i < 98; i++) {
    song2[i - 74] = signal[i];
  }
  for (i = 98; i < 140; i++) {
    noteLength0[i - 98] = signal[i];
  }
  for (i = 140; i < 172; i++) {
    noteLength1[i - 140] = signal[i];
  }
  for (i = 172; i < 196; i++) {
    noteLength2[i - 172] = signal[i];
  }
  uLCD.printf("\nEnd :)");
  wait(1.0);

  while(1) {
    song_th = 0;
    change = 0;
    mode = 0;
    uLCD.cls();
    uLCD.text_width(3); //3X size text
    uLCD.text_height(3);
    uLCD.printf("=Menu=");
    uLCD.text_width(2); //2X size text
    uLCD.text_height(2);
    uLCD.printf("\nsw2-->mp3");
    uLCD.printf("\nsw3->Game");
    //uLCD.printf("\n\nstate : 0\n");
    while((sw2 && sw3) || (!change)) {
      if (sw2 == 0) {
        mp3_or_Taiko = 0;
        change = 1;
      }
      else if (sw3 == 0) {
        mp3_or_Taiko = 1;
        change = 1;
      }
    }
    if (mp3_or_Taiko == 0) {
      uLCD.cls();
      uLCD.printf("\n\n");
      uLCD.text_width(2); //2X size text
      uLCD.text_height(2);
      uLCD.printf("Plz use \ngestures");
      uLCD.printf("\nchanging\nmodes");
      uLCD.text_width(1); //2X size text
      uLCD.text_height(1);
      uLCD.locate(0,14);
      //uLCD.printf("press sw2 to \ncontinue");
      //uLCD.printf("\n\nstate : 1\n");
      change = 0;
      while(sw2 || !change) {
        if (gesture_index == 0) {
          uLCD.cls();
          uLCD.printf("\n");
          uLCD.text_width(2); //2X size text
          uLCD.text_height(2);
          uLCD.printf("Choose \nMode:");
          uLCD.printf("\n\nprevious");
          uLCD.text_width(1); //2X size text
          uLCD.text_height(1);
          uLCD.locate(0,14);
          uLCD.printf("press sw2 to \ncontinue");
          change = 1;
          mode = 0;
        }
        else if (gesture_index == 1) {
          uLCD.cls();
          uLCD.printf("\n");
          uLCD.text_width(2); //2X size text
          uLCD.text_height(2);
          uLCD.printf("Choose \nMode:");
          uLCD.printf("\n\nnext");
          uLCD.text_width(1); //2X size text
          uLCD.text_height(1);
          uLCD.locate(0,14);
          uLCD.printf("press sw2 to \ncontinue");
          change = 1;
          mode = 1;
        }
        else if (gesture_index == 2) {
          uLCD.cls();
          uLCD.printf("\n");
          uLCD.text_width(2); //2X size text
          uLCD.text_height(2);
          uLCD.printf("Choose \nMode:");
          uLCD.printf("\n\nselect");
          uLCD.text_width(1); //2X size text
          uLCD.text_height(1);
          uLCD.locate(0,14);
          uLCD.printf("press sw2 to \ncontinue");
          change = 1;
          mode = 2;
        }
        gesture_index = 10;
      }
      uLCD.cls();
      uLCD.printf("\n\n");
      uLCD.text_width(2); //2X size text
      uLCD.text_height(2);
      uLCD.printf("Plz use \ngestures");
      uLCD.printf("\nchanging\nsongs");
      uLCD.text_width(1); //2X size text
      uLCD.text_height(1);
      change = 0;
      while(sw2 || !change) {
        if ((mode == 0) && (gesture_index == 0)) {
          if (song_th == 0) {
            song_th = 2;
            uLCD.cls();
            uLCD.text_width(2); //2X size text
            uLCD.text_height(2);
            uLCD.printf("\n");
            uLCD.printf("Bees");
            uLCD.printf("\n\n\n");
            uLCD.text_width(1); //2X size text
            uLCD.text_height(1);
            uLCD.locate(0,14);
            uLCD.printf("press sw2 to play");
            change = 1;
          }
          else if(song_th == 1){
            song_th = 0;
            uLCD.cls();
            uLCD.text_width(2); //2X size text
            uLCD.text_height(2);
            uLCD.printf("\n");
            uLCD.printf("Twinkle\nStar");
            uLCD.printf("\n\n");
            uLCD.text_width(1); //2X size text
            uLCD.text_height(1);
            uLCD.locate(0,14);
            uLCD.printf("press sw2 to play");
            change = 1;
          }
          else {
            song_th = 1;
            uLCD.cls();
            uLCD.text_width(2); //2X size text
            uLCD.text_height(2);
            uLCD.printf("\n");
            uLCD.printf("Two\nTigers");
            uLCD.printf("\n\n");
            uLCD.text_width(1); //2X size text
            uLCD.text_height(1);
            uLCD.locate(0,14);
            uLCD.printf("press sw2 to play");
            change = 1;
          }
          gesture_index = 10;
        } 
          else if ((mode == 1) && (gesture_index == 1)) {
          if (song_th == 0) {
            song_th = 1;
            uLCD.cls();
            uLCD.text_width(2); //2X size text
            uLCD.text_height(2);
            uLCD.printf("\n");
            uLCD.printf("Two\nTigers");
            uLCD.printf("\n\n");
            uLCD.text_width(1); //2X size text
            uLCD.text_height(1);
            uLCD.locate(0,14);
            uLCD.printf("press sw2 to play");
            change = 1;
          }
          else if(song_th == 1){
            song_th = 2;
            uLCD.cls();
            uLCD.text_width(2); //2X size text
            uLCD.text_height(2);
            uLCD.printf("\n");
            uLCD.printf("Bees");
            uLCD.printf("\n\n\n");
            uLCD.text_width(1); //2X size text
            uLCD.text_height(1);
            uLCD.locate(0,14);
            uLCD.printf("press sw2 to play");
            change = 1;
          }
          else {
            song_th = 0;
            uLCD.cls();
            uLCD.text_width(2); //2X size text
            uLCD.text_height(2);
            uLCD.printf("\n");
            uLCD.printf("Twinkle\nStar");
            uLCD.printf("\n\n");
            uLCD.text_width(1); //2X size text
            uLCD.text_height(1);
            uLCD.locate(0,14);
            uLCD.printf("press sw2 to play");
            change = 1;
          }
          gesture_index = 10;
        } 
        else if ((mode == 2) && (gesture_index == 0)) {
          if (song_th == 0) {
            song_th = 2;
            uLCD.cls();
            uLCD.text_width(2); //2X size text
            uLCD.text_height(2);
            uLCD.printf("\n");
            uLCD.printf("Bees");
            uLCD.printf("\n\n\n");
            uLCD.text_width(1); //2X size text
            uLCD.text_height(1);
            uLCD.locate(0,14);
            uLCD.printf("press sw2 to play");
            change = 1;
          }
          else if(song_th == 1){
            song_th = 0;
            uLCD.cls();
            uLCD.text_width(2); //2X size text
            uLCD.text_height(2);
            uLCD.printf("\n");
            uLCD.printf("Twinkle\nStar");
            uLCD.printf("\n\n");
            uLCD.text_width(1); //2X size text
            uLCD.text_height(1);
            uLCD.locate(0,14);
            uLCD.printf("press sw2 to play");
            change = 1;
          }else {
            song_th = 1;
            uLCD.cls();
            uLCD.text_width(2); //2X size text
            uLCD.text_height(2);
            uLCD.printf("\n");
            uLCD.printf("Two\nTigers");
            uLCD.printf("\n\n");
            uLCD.text_width(1); //2X size text
            uLCD.text_height(1);
            uLCD.locate(0,14);
            uLCD.printf("press sw2 to play");
            change = 1;
          }
          gesture_index = 10;
        } 
        else if ((mode == 2) && (gesture_index == 1)) {
          if (song_th == 0) {
            song_th = 1;
            uLCD.cls();
            uLCD.text_width(2); //2X size text
            uLCD.text_height(2);
            uLCD.printf("\n");
            uLCD.printf("Two\nTigers");
            uLCD.printf("\n\n");
            uLCD.text_width(1); //2X size text
            uLCD.text_height(1);
            uLCD.locate(0,14);
            uLCD.printf("press sw2 to play");
            change = 1;
          }
          else if(song_th == 1){
            song_th = 2;
            uLCD.cls();
            uLCD.text_width(2); //2X size text
            uLCD.text_height(2);
            uLCD.printf("\n");
            uLCD.printf("Bees");
            uLCD.printf("\n\n\n");
            uLCD.text_width(1); //2X size text
            uLCD.text_height(1);
            uLCD.locate(0,14);
            uLCD.printf("press sw2 to play");
            change = 1;
          }else {
            song_th = 0;
            uLCD.cls();
            uLCD.text_width(2); //2X size text
            uLCD.text_height(2);
            uLCD.printf("\n");
            uLCD.printf("Twinkle\nStar");
            uLCD.printf("\n\n");
            uLCD.text_width(1); //2X size text
            uLCD.text_height(1);
            uLCD.locate(0,14);
            uLCD.printf("press sw2 to play");
            change = 1;
          }
          gesture_index = 10;
        }      
      }
      
    
    /*uLCD.cls();
    uLCD.printf("\nTwinkle Star\n");
    uLCD.printf("\n\nstate : 1\n");
    audio.spk.status(true);
  */
      if (song_th == 0) {
        change = 0;
        for(int i = 0; i < 42 && (sw3 || (!change)); i++)
        {
          change = 1;
          int length = noteLength0[i];
          while(length-- && (sw3 || (!change)))
          {
            // the loop below will play the note for the duration of 1s
            for(int j = 0; (j < kAudioSampleFrequency / kAudioTxBufferSize/4) && (sw3 || (!change)); ++j)
            {
              playNote(song0[i]);
            }
            if(length<1) {
              wait(0.2);
              playNote(0);
              wait(0.1);
            }
            else {
              wait(0.3);
            }
          }
        }
      }
      else if (song_th == 1) {
        change = 0;
        for(int i = 0; i < 32 && (sw3 || (!change)); i++)
        {
          change = 1;
          int length = noteLength1[i];
          while(length-- && (sw3 || (!change)))
          {
            // the loop below will play the note for the duration of 1s
            for(int j = 0; (j < kAudioSampleFrequency / kAudioTxBufferSize/4) && (sw3 || (!change)); ++j)
            {
              playNote(song1[i]);
            }
           if(length<1) {
              wait(0.2);
              playNote(0);
              wait(0.1);
            }
            else {
              wait(0.3);
            }
          }
        }
      }
      else {
        change = 0;
        for(int i = 0; i < 24 && (sw3 || (!change)); i++)
        {
          change = 1;
          int length = noteLength2[i];
          while(length-- && (sw3 || (!change)))
          {
            // the loop below will play the note for the duration of 1s
            for(int j = 0; (j < kAudioSampleFrequency / kAudioTxBufferSize/4) && (sw3 || (!change)); ++j)
            {
              playNote(song2[i]);
            }
            if(length<1) {
              wait(0.2);
              playNote(0);
              wait(0.1);
            }
            else {
              wait(0.3);
            }
          }
        }
      }
      playNote(0); 
    }
    else {
      score = 0;
      uLCD.cls();
      uLCD.text_width(2); //2X size text
      uLCD.text_height(2);
      uLCD.printf("Welcome\nTaiko\n");
      uLCD.printf("\nPlz Push\nSW2 \nTo Play\n");
      while(sw2);
      uLCD.cls();
      uLCD.locate(3,3);  // address of font file
      uLCD.set_font(FONT_7X8);  // back to built-in system font
      uLCD.text_width(10); //10X size text
      uLCD.text_height(10);
      uLCD.printf("3");
      
      wait(1.0);
      uLCD.cls();
      uLCD.locate(3,3);  // address of font file
      uLCD.set_font(FONT_7X8);  // back to built-in system font
      uLCD.text_width(10); //10X size text
      uLCD.text_height(10);
      uLCD.printf("2");
      
      wait(1.0);
      uLCD.cls();
      uLCD.locate(3,3);  // address of font file
      uLCD.set_font(FONT_7X8);  // back to built-in system font
      uLCD.text_width(10); //10X size text
      uLCD.text_height(10);
      uLCD.printf("1");
      
      wait(1.0);
      uLCD.cls();
      uLCD.locate(0,3);  // address of font file
      uLCD.set_font(FONT_7X8);  // back to built-in system font
      uLCD.text_width(8); //10X size text
      uLCD.text_height(8);
      uLCD.printf("GO");
      
      wait(1.0);
      change = 0;
      for(int i = 0; i < 24 && (sw3 || (!change)); i++) {
        int once = 1;
        change = 1;
        int length = noteLength2[i];
        int length_initial = noteLength2[i];
        if (length == 1) {
          uLCD.cls();
          uLCD.locate(3,3);  // address of font file
          uLCD.set_font(FONT_7X8);  // back to built-in system font
          uLCD.text_width(10); //10X size text
          uLCD.text_height(10);
          uLCD.printf("O");
        }
        else {
          uLCD.cls();
          uLCD.locate(3,3);  // address of font file
          uLCD.set_font(FONT_7X8);  // back to built-in system font
          uLCD.text_width(8); //10X size text
          uLCD.text_height(8);
          uLCD.printf("/");
          uLCD.locate(3,3);  // address of font file
          uLCD.set_font(FONT_7X8);  // back to built-in system font
          uLCD.text_width(8); //10X size text
          uLCD.text_height(8);
          uLCD.printf("_");
        }
        while(length-- && (sw3 || (!change)))
        {
          // the loop below will play the note for the duration of 1s
          for(int j = 0; (j < kAudioSampleFrequency / kAudioTxBufferSize/4) && (sw3 || (!change)); ++j)
          {
            playNote(song2[i]);
          }
          
          if(length<1) {
            n = 150;
            while(n) {
              if (length_initial == 1 && once) {
                if (gesture_index == 0) {
                  uLCD.cls();
                  uLCD.locate(3,3);  // address of font file
                  uLCD.set_font(FONT_7X8);  // back to built-in system font
                  uLCD.text_width(2); //10X size text
                  uLCD.text_height(2);
                  uLCD.printf("Well\nDone");
                  score += 20;
                  once--;
                  gesture_index = 10;
                }
              }
              else {
                if (gesture_index == 1 && once) {
                  uLCD.cls();
                  uLCD.locate(3,3);  // address of font file
                  uLCD.set_font(FONT_7X8);  // back to built-in system font
                  uLCD.text_width(2); //10X size text
                  uLCD.text_height(2);
                  uLCD.printf("Well\nDone");
                  score += 20;
                  once--;
                  gesture_index = 10;
                }
              }
              wait(0.001);
              n--;
            }
            playNote(0);
            wait(0.05);
          }
          else {
            n = 200;
            while(n) {
              if (length_initial == 1 && once) {
                if (gesture_index == 0) {
                  uLCD.cls();
                  uLCD.locate(3,3);  // address of font file
                  uLCD.set_font(FONT_7X8);  // back to built-in system font
                  uLCD.text_width(2); //10X size text
                  uLCD.text_height(2);
                  uLCD.printf("Well\nDone");
                  score += 20;
                  once--;
                  gesture_index = 10;
                }
              }
              else {
                if (gesture_index == 1 && once) {
                  uLCD.cls();
                  uLCD.locate(3,3);  // address of font file
                  uLCD.set_font(FONT_7X8);  // back to built-in system font
                  uLCD.text_width(2); //10X size text
                  uLCD.text_height(2);
                  uLCD.printf("Well\nDone");
                  score += 20;
                  once--;
                  gesture_index = 10;
                }
              }
              wait(0.001);
              n--;
            }
          }
        }
      }
      playNote(0); 
      uLCD.cls();
      uLCD.locate(0,3);  // address of font file
      uLCD.set_font(FONT_7X8);  // back to built-in system font
      uLCD.text_width(2); //10X size text
      uLCD.text_height(2);
      uLCD.printf("Score :\n  %D", score);
      wait(1.0);
      uLCD.text_width(1); //10X size text
      uLCD.text_height(1);
      uLCD.locate(0,15);  // address of font file
      uLCD.printf("press sw3 to end");
      while (sw3);
      uLCD.cls();
      wait(1.0);
    }
  }

  
}