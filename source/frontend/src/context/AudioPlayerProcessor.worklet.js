// Audio sample buffer to minimize reallocations
class ExpandableBuffer {
  constructor() {
    // Start with one second's worth of buffered audio capacity before needing to expand
    this.buffer = new Float32Array(24000);
    this.readIndex = 0;
    this.writeIndex = 0;
    this.underflowedSamples = 0;
    this.isInitialBuffering = true;
    this.initialBufferLength = 24000; // One second
    this.lastWriteTime = 0;
  }

  logTimeElapsedSinceLastWrite() {
    const now = Date.now();
    if (this.lastWriteTime !== 0) {
      const elapsed = now - this.lastWriteTime;
      // console.log(`Elapsed time since last audio buffer write: ${elapsed} ms`);
    }
    this.lastWriteTime = now;
  }

  write(samples) {
    this.logTimeElapsedSinceLastWrite();
    if (this.writeIndex + samples.length <= this.buffer.length) {
      // Enough space to append the new samples
    } else {
      // Not enough space ...
      if (samples.length <= this.readIndex) {
        // ... but we can shift samples to the beginning of the buffer
        const subarray = this.buffer.subarray(this.readIndex, this.writeIndex);
        console.log(
          `Shifting the audio buffer of length ${subarray.length} by ${this.readIndex}`
        );
        this.buffer.set(subarray);
      } else {
        // ... and we need to grow the buffer capacity to make room for more audio
        const newLength =
          (samples.length + this.writeIndex - this.readIndex) * 2;
        const newBuffer = new Float32Array(newLength);
        console.log(
          `Expanding the audio buffer from ${this.buffer.length} to ${newLength}`
        );
        newBuffer.set(this.buffer.subarray(this.readIndex, this.writeIndex));
        this.buffer = newBuffer;
      }
      this.writeIndex -= this.readIndex;
      this.readIndex = 0;
    }
    this.buffer.set(samples, this.writeIndex);
    this.writeIndex += samples.length;
    if (this.writeIndex - this.readIndex >= this.initialBufferLength) {
      // Filled the initial buffer length, so we can start playback with some cushion
      this.isInitialBuffering = false;
      // console.log("Initial audio buffer filled");
    }
  }

  read(destination) {
    let copyLength = 0;
    if (!this.isInitialBuffering) {
      // Only start to play audio after we've built up some initial cushion
      copyLength = Math.min(
        destination.length,
        this.writeIndex - this.readIndex
      );
    }
    destination.set(
      this.buffer.subarray(this.readIndex, this.readIndex + copyLength)
    );
    this.readIndex += copyLength;
    if (copyLength > 0 && this.underflowedSamples > 0) {
      console.log(
        `Detected audio buffer underflow of ${this.underflowedSamples} samples`
      );
      this.underflowedSamples = 0;
    }
    if (copyLength < destination.length) {
      // Not enough samples (buffer underflow). Fill the rest with silence.
      destination.fill(0, copyLength);
      this.underflowedSamples += destination.length - copyLength;
    }
    if (copyLength === 0) {
      // Ran out of audio, so refill the buffer to the initial length before playing more
      this.isInitialBuffering = true;
    }
  }

  clearBuffer() {
    this.readIndex = 0;
    this.writeIndex = 0;
  }
}

class AudioPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(0);
    this.writeIndex = 0;
    this.readIndex = 0;
    this.isInitialBuffering = true;
    this.isBargeIn = false;
    this.isForceStop = false;
    this.initialBufferLength = 4096;
    this.sampleRate = 24000;
    
    this.port.onmessage = (event) => {
      const { type, audioData, bufferLength } = event.data;
      
      switch (type) {
        case 'audio':
          if (!audioData || audioData.length === 0) {
            console.log('[AudioPlayerProcessor] Received empty audio data');
            return;
          }
          console.log('[AudioPlayerProcessor] Received audio data, length:', audioData.length);
          
          // Create a new buffer with enough space for the new data
          const newBuffer = new Float32Array(this.buffer.length + audioData.length);
          newBuffer.set(this.buffer);
          newBuffer.set(audioData, this.buffer.length);
          this.buffer = newBuffer;
          
          console.log('[AudioPlayerProcessor] Buffer state after write:', {
            bufferLength: this.buffer.length,
            writeIndex: this.writeIndex,
            readIndex: this.readIndex,
            isInitialBuffering: this.isInitialBuffering
          });
          break;
          
        case 'reset':
          console.log('[AudioPlayerProcessor] Resetting playback');
          this.buffer = new Float32Array(0);
          this.writeIndex = 0;
          this.readIndex = 0;
          this.isInitialBuffering = true;
          this.isBargeIn = false;
          this.isForceStop = false;
          break;
          
        case 'stop':
          console.log('[AudioPlayerProcessor] Stopping playback');
          this.isForceStop = true;
          break;
          
        case 'barge-in':
          console.log('[AudioPlayerProcessor] Barge-in requested');
          this.isBargeIn = true;
          break;
          
        case 'initial-buffer-length':
          if (bufferLength && !isNaN(bufferLength)) {
            console.log('[AudioPlayerProcessor] Setting initial buffer length:', bufferLength);
            this.initialBufferLength = bufferLength;
          }
          break;
      }
    };
  }

  process(inputs, outputs) {
    const output = outputs[0];
    if (!output || !output[0]) {
      console.warn('[AudioPlayerProcessor] No output channel available');
      return true;
    }

    const channel = output[0];
    
    // If force stop is active, output silence
    if (this.isForceStop) {
      channel.fill(0);
      return true;
    }

    // Check if we have enough data to start playback
    if (this.isInitialBuffering && this.buffer.length >= this.initialBufferLength) {
      console.log('[AudioPlayerProcessor] Initial buffering complete, starting playback');
      this.isInitialBuffering = false;
    }

    // If we're still buffering or have no data, output silence
    if (this.isInitialBuffering || this.buffer.length === 0) {
      channel.fill(0);
      return true;
    }

    // Handle barge-in by clearing the buffer
    if (this.isBargeIn) {
      console.log('[AudioPlayerProcessor] Clearing buffer due to barge-in');
      this.buffer = new Float32Array(0);
      this.writeIndex = 0;
      this.readIndex = 0;
      this.isBargeIn = false;
      this.isInitialBuffering = true;
      channel.fill(0);
      return true;
    }

    // Copy data from the buffer to the output
    const availableSamples = this.buffer.length - this.readIndex;
    const samplesToWrite = Math.min(channel.length, availableSamples);
    
    if (samplesToWrite > 0) {
      channel.set(this.buffer.subarray(this.readIndex, this.readIndex + samplesToWrite));
      this.readIndex += samplesToWrite;
      
      // If we've read all the data, reset the buffer
      if (this.readIndex >= this.buffer.length) {
        console.log('[AudioPlayerProcessor] Buffer depleted, resetting');
        this.buffer = new Float32Array(0);
        this.writeIndex = 0;
        this.readIndex = 0;
        this.isInitialBuffering = true;
      }
    } else {
      channel.fill(0);
    }

    return true;
  }
}

registerProcessor('audio-player-processor', AudioPlayerProcessor);