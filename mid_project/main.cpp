#include "mbed.h"
#include "uLCD_4DGL.h"
#include <cmath>
#include "DA7212.h"

InterruptIn sw2(SW2);
DA7212 audio;
int16_t waveform[kAudioTxBufferSize];
EventQueue queue(32 * EVENTS_EVENT_SIZE);
Thread t;

uLCD_4DGL uLCD(D1, D0, D2);

int song[42] = {
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261};

int noteLength[42] = {
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2};

void playNote(int freq)
{
  for (int i = 0; i < kAudioTxBufferSize; i++)
  {
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  // the loop below will play the note for the duration of 1s
  for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
  {
    audio.spk.play(waveform, kAudioTxBufferSize);
  }
}

void playSong(void) 
{
  uLCD.printf("\nTwinkle, Twinkle, Little Star\n");

  for(int i = 0; i < 42; i++)
    {
      int length = noteLength[i];
      while(length--)
      {
        playNote(song[i]);
        if(length <= 1) wait(1.0);
      }
    }
}

Trig_playSong() {
  queue.call(playSong);
}

int main(void)
{
  t.start(callback(&queue, &EventQueue::dispatch_forever));

  sw2.rise(queue.event(Trig_playSong));
  
}
