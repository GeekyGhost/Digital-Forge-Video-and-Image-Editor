// Minimal WebGPU typings for the spike. The project does not depend on @webgpu/types, and adding
// a dependency for throwaway code is not worth it; Phase 1 adds the real types if WebGPU is chosen.
/* eslint-disable @typescript-eslint/no-explicit-any */
declare type GPU = any;
declare type GPUAdapter = any;
declare type GPUAdapterInfo = any;
declare type GPUDevice = any;
declare type GPUBlendState = any;
declare type GPUCommandEncoder = any;
declare type GPUTexture = any;
declare type GPUBuffer = any;
declare const GPUTextureUsage: any;
declare const GPUBufferUsage: any;
declare const GPUMapMode: any;
