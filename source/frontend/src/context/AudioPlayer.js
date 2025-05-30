export class ObjectExt {
    static exists(obj) {
      return obj !== undefined && obj !== null;
    }
  
    static checkArgument(condition, message) {
      if (!condition) {
        throw TypeError(message);
      }
    }
  
    static checkExists(obj, message) {
      if (ObjectsExt.exists(obj)) {
        throw TypeError(message);
      }
    }
  }

const AudioPlayerWorkletUrl = new URL(
  "./AudioPlayerProcessor.worklet.js",
  import.meta.url
).toString();

export default class AudioPlayer {
  constructor() {
    this.onAudioPlayedListeners = [];
    this.initialized = false;
    console.log('[AudioPlayer] Constructor called');
  }

  addEventListener(event, callback) {
    switch (event) {
      case "onAudioPlayed":
        this.onAudioPlayedListeners.push(callback);
        break;
      default:
        console.error(
          "Listener registered for event type: " +
            event +
            " which is not supported"
        );
    }
  }

  async start() {
    try {
      console.log('[AudioPlayer] Starting initialization');
      
      // Create audio context with correct sample rate
      this.audioContext = new AudioContext({ 
        sampleRate: 24000,
        latencyHint: 'interactive'
      });
      
      // Resume the audio context if it's suspended
      if (this.audioContext.state === 'suspended') {
        console.log('[AudioPlayer] Resuming suspended audio context');
        await this.audioContext.resume();
      }
      
      // Create and configure analyser
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 512;
      this.analyser.smoothingTimeConstant = 0.8;

      // Load the audio worklet
      try {
        await this.audioContext.audioWorklet.addModule(AudioPlayerWorkletUrl);
        console.log('[AudioPlayer] AudioWorklet module loaded successfully');
      } catch (error) {
        console.error('[AudioPlayer] Failed to load AudioWorklet:', error);
        throw error;
      }
      
      // Create and configure the worklet node
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        "audio-player-processor"
      );
      
      // Set up message handling from the worklet
      this.workletNode.port.onmessage = (event) => {
        console.log('[AudioPlayer] Received message from worklet:', event.data);
      };
      
      // Handle worklet errors
      this.workletNode.onprocessorerror = (error) => {
        console.error('[AudioPlayer] Worklet processor error:', error);
      };
      
      console.log('[AudioPlayer] AudioWorkletNode created');
      
      // Connect the audio nodes
      this.workletNode.connect(this.analyser);
      this.analyser.connect(this.audioContext.destination);
      
      // Configure initial buffer length if needed
      this.#maybeOverrideInitialBufferLength();
      
      this.initialized = true;
      console.log('[AudioPlayer] Initialization complete');
    } catch (error) {
      console.error('[AudioPlayer] Initialization failed:', error);
      this.stop();
      throw error;
    }
  }

  bargeIn() {
    if (!this.initialized) {
      console.warn('[AudioPlayer] Cannot barge in - not initialized');
      return;
    }
    console.log('[AudioPlayer] Sending barge-in request');
    this.workletNode.port.postMessage({
      type: "barge-in",
    });
  }

  stop() {
    console.log('[AudioPlayer] Stopping audio player');
    if (ObjectExt.exists(this.audioContext)) {
      this.audioContext.close().catch(error => {
        console.error('[AudioPlayer] Error closing audio context:', error);
      });
    }

    if (ObjectExt.exists(this.analyser)) {
      this.analyser.disconnect();
    }

    if (ObjectExt.exists(this.workletNode)) {
      this.workletNode.disconnect();
    }

    this.initialized = false;
    this.audioContext = null;
    this.analyser = null;
    this.workletNode = null;
  }

  #maybeOverrideInitialBufferLength() {
    const params = new URLSearchParams(window.location.search);
    const value = params.get("audioPlayerInitialBufferLength");
    if (value === null) {
      return;
    }
    const bufferLength = parseInt(value);
    if (isNaN(bufferLength)) {
      console.error("Invalid audioPlayerInitialBufferLength value:", value);
      return;
    }
    console.log('[AudioPlayer] Setting initial buffer length:', bufferLength);
    this.workletNode.port.postMessage({
      type: "initial-buffer-length",
      bufferLength: bufferLength,
    });
  }

  playAudio(samples) {
    if (!this.initialized) {
      console.error(
        "[AudioPlayer] Cannot play audio - not initialized. Call start() first."
      );
      return;
    }
    
    if (!samples || samples.length === 0) {
      console.warn('[AudioPlayer] Received empty audio samples');
      return;
    }
    
    console.log('[AudioPlayer] Playing audio samples, length:', samples.length);
    
    // Ensure audio context is running
    if (this.audioContext.state === 'suspended') {
      console.log('[AudioPlayer] Resuming suspended audio context');
      this.audioContext.resume().catch(error => {
        console.error('[AudioPlayer] Failed to resume audio context:', error);
      });
    }
    
    this.workletNode.port.postMessage({
      type: "audio",
      audioData: samples,
    });
  }

  getSamples() {
    if (!this.initialized) {
      return null;
    }
    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    this.analyser.getByteTimeDomainData(dataArray);
    return [...dataArray].map((e) => e / 128 - 1);
  }

  getVolume() {
    if (!this.initialized) {
      return 0;
    }
    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    this.analyser.getByteTimeDomainData(dataArray);
    let normSamples = [...dataArray].map((e) => e / 128 - 1);
    let sum = 0;
    for (let i = 0; i < normSamples.length; i++) {
      sum += normSamples[i] * normSamples[i];
    }
    return Math.sqrt(sum / normSamples.length);
  }
}