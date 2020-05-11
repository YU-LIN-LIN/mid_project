#include "mbed.h"
#include <cmath>
#include "DA7212.h"

DA7212 audio;
int16_t waveform[kAudioTxBufferSize];
EventQueue queue(32 * EVENTS_EVENT_SIZE);
Thread t;

int song1[42] = {
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261};

int song2[32] = {
  261, 294, 330, 261, 261, 294, 330, 261,
  330, 349, 392, 330, 349, 392,
  392, 440, 392, 349, 330, 261, 
  392, 440, 392, 349, 330, 261,
  261, 196, 261, 261, 196, 261};

int noteLength[42] = {
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2};

int noteLength2[32] = {
  1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1,
  1, 1, 2, 1, 1, 2};

void playNote(int freq)
{
  for(int i = 0; i < kAudioTxBufferSize; i++)
  {
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  audio.spk.play(waveform, kAudioTxBufferSize);
}

int main(void)
{
  t.start(callback(&queue, &EventQueue::dispatch_forever));

  for(int i = 0; i < 32; i++)
  {
    int length = noteLength2[i];
    while(length)
    {
      // the loop below will play the note for the duration of 1s
      for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
      {
        queue.call(playNote, song2[i]);
      }
      if(length) wait(0.5);
      length--;
    }
  }
}