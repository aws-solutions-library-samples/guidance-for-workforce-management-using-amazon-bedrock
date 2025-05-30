declare module "AudioPlayer" {
  export default class AudioPlayer {
    constructor();
    addEventListener(event: string, callback: (samples: Float32Array) => void): void;
    start(): Promise<void>;
    stop(): void;
    bargeIn(): void;
    playAudio(samples: Float32Array): void;
    getSamples(): Float32Array | null;
    getVolume(): number;
  }
} 