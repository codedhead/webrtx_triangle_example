import 'webrtx';

const RAY_GENERATION_SHADER = `
#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadEXT vec3 payload;
layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(std140, set = 0, binding = 1) buffer PixelBuffer { vec4 pixels[]; }
pixelBuffer;

vec3 calcRayDirection(const vec2 pixel) {
  const float aspectRatio = gl_LaunchSizeEXT.x / gl_LaunchSizeEXT.y;
  const vec2 imagePlaneSize =
      vec2(aspectRatio * 2.0, 2.0);  // vertical fov: 45'
  vec2 ixy = (vec2(-0.5, -0.5) + pixel / gl_LaunchSizeEXT.xy) * imagePlaneSize;
  return normalize(vec3(ixy.x, -ixy.y, 1.0));
}

void main() {
  const vec3 rayOrigin = {0, 0, -1};
  const vec3 rayDirection = calcRayDirection(vec2(gl_LaunchIDEXT));
  const float rayTmin = 1e-3;
  const float rayTmax = 1e5;

  payload = vec3(0.);
  traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, 0, 1, 0, rayOrigin,
              rayTmin, rayDirection, rayTmax, 0);

  const uint pixelIndex = uint(gl_LaunchIDEXT.y) * uint(gl_LaunchSizeEXT.x) +
                          uint(gl_LaunchIDEXT.x);
  pixelBuffer.pixels[pixelIndex] = vec4(payload, 1.0);
}
`;

const RAY_CLOSEST_HIT_SHADER = `
#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec2 barycentricCoord;

void main() {
  payload = vec3(1.0 - barycentricCoord.x - barycentricCoord.y, barycentricCoord.x, barycentricCoord.y);
}
`;

const SCREEN_VERTEX_SHADER = `
@vertex
fn main(@builtin(vertex_index) VertexIndex : u32) -> @builtin(position) vec4<f32> {
  let pos:vec2<f32> = vec2<f32>(f32((VertexIndex << 1u) & 2u), f32(VertexIndex & 2u));
  return vec4<f32>(pos * 2.0 - 1.0, 0.0, 1.0);
}
`;

const SCREEN_FRAG_SHADER = `
struct PixelBuffer {
  pixels: array<vec4<f32>>,
}
@group(0) @binding(0) var<storage, read> pixelBuffer: PixelBuffer;

struct ScreenDimension {
  resolution : vec2<f32>,
}
@group(0) @binding(1) var<uniform> screenDimension: ScreenDimension;

@fragment
fn main(
  @builtin(position) coord : vec4<f32>
)-> @location(0) vec4<f32> {
  let pixelIndex:u32 =
      u32(coord.x) + u32(coord.y) * u32(screenDimension.resolution.x);
  let pixelColor:vec4<f32> = pixelBuffer.pixels[pixelIndex];
  return vec4<f32>(pixelColor.xyz, 1.0);
}
`;

const LITTLE_ENDIAN = true;

function alignTo(x: number, align: number): number {
  return Math.floor((x + align - 1) / align) * align;
}

function createRenderPipeline(device: GPUDevice, swapChainFormat: GPUTextureFormat) {
  return device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({
        code: SCREEN_VERTEX_SHADER,
      }),
      entryPoint: "main",
    },
    fragment: {
      module: device.createShaderModule({
        code: SCREEN_FRAG_SHADER,
      }),
      entryPoint: "main",
      targets: [
        {
          format: swapChainFormat,
        },
      ],
    },
    primitive: {
      topology: 'triangle-list',
      frontFace: "ccw",
      cullMode: "none",
    },
  });
}

function createAndBuildRayTracingAccelerationContainer(device: GPUDevice) {
  const vertices = new Float32Array([
    0, 0.5, 0,
    -0.5, -0.5, 0,
    0.5, -0.5, 0,
  ]);
  const buffer = device.createBuffer({
    size: vertices.byteLength,
    usage: GPUBufferUsageRTX.ACCELERATION_STRUCTURE_BUILD_INPUT_READONLY,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(vertices);
  buffer.unmap();
  const blas: GPURayTracingAccelerationContainerDescriptor_bottom = {
    usage: GPURayTracingAccelerationContainerUsage.NONE,
    level: 'bottom',
    geometries: [{
      usage: GPURayTracingAccelerationGeometryUsage.NONE,
      type: 'triangles',
      vertex: {
        buffer,
        format: 'float32x3',
        offset: 0,
        stride: 3 * 4,
        size: buffer.size,
      },
    }],
  };
  const tlas = device.createRayTracingAccelerationContainer({
    level: 'top',
    usage: GPURayTracingAccelerationContainerUsage.NONE,
    instances: [{
      usage: GPURayTracingAccelerationInstanceUsage.NONE,
      mask: 0xFF,
      transformMatrix: undefined,
      instanceCustomIndex: undefined,
      instanceSBTRecordOffset: 0,
      blas,
    }],
  });
  device.hostBuildRayTracingAccelerationContainer(tlas);
  return tlas;
}

async function createRayTracingPipeline(device: GPUDevice, tlas: GPURayTracingAccelerationContainer_top) {
  const groups: GPURayTracingShaderGroupDescriptor[] = [{
    type: 'general',
    generalIndex: 0,
  }, {
    type: 'triangles-hit-group',
    closestHitIndex: 1,
  }];
  return device.createRayTracingPipeline({
    stages: [{
      stage: GPUShaderStageRTX.RAY_GENERATION,
      entryPoint: 'main',
      glslCode: RAY_GENERATION_SHADER,
    }, {
      stage: GPUShaderStageRTX.RAY_CLOSEST_HIT,
      entryPoint: 'main',
      glslCode: RAY_CLOSEST_HIT_SHADER,
    }],
    groups,
  }, tlas);
}

function createShaderBindingTable(
  device: GPUDevice,
  pipeline: GPURayTracingPipeline,
) {
  const sbt: GPUShaderBindingTable = {
    rayGen: {},
    rayMiss: {},
    rayHit: {},
    callable: {
      start: 0,
      stride: 0,
      size: 0,
    },
  } as GPUShaderBindingTable;
  // always start at 0
  sbt.rayGen.start = 0;
  sbt.rayGen.stride = alignTo(
    device.ShaderGroupHandleSize + /*maxRgenShaderRecordSize=*/0,
    device.ShaderGroupHandleAlignment);
  const numRayGenGroups = 1;
  sbt.rayGen.size = numRayGenGroups * sbt.rayGen.stride;

  // no rmiss
  sbt.rayMiss.start = alignTo(sbt.rayGen.start + sbt.rayGen.size, device.ShaderGroupBaseAlignment);
  sbt.rayMiss.stride = alignTo(
    device.ShaderGroupHandleSize + /*maxRmissShaderRecordSize=*/0,
    device.ShaderGroupHandleAlignment);
  sbt.rayMiss.size = 0;

  sbt.rayHit.start = alignTo(sbt.rayMiss.start + sbt.rayMiss.size, device.ShaderGroupBaseAlignment);
  sbt.rayHit.stride = alignTo(
    device.ShaderGroupHandleSize + /*maxHitGroupShaderRecordByteSize=*/0,
    device.ShaderGroupHandleAlignment);
  // geometries in all instances
  const numAllGeometries = 1;
  const NUM_RAY_TYPES = 1;
  sbt.rayHit.size = numAllGeometries * NUM_RAY_TYPES * sbt.rayHit.stride;

  const alignedSbtSize = alignTo(sbt.rayHit.start + sbt.rayHit.size, device.ShaderGroupBaseAlignment);
  sbt.buffer = device.createBuffer({
    size: alignedSbtSize,
    usage: GPUBufferUsageRTX.SHADER_BINDING_TABLE,
    mappedAtCreation: true,
  });
  {
    // two groups: 1 rgen + 1 hit
    const groupHandles = pipeline.getShaderGroupHandles(0, 2);
    let groupHandleIndex = 0;
    const dvSbtBuffer = new DataView(sbt.buffer.getMappedRange());

    // single rgen group
    {
      const byteOffset = sbt.rayGen.start;
      dvSbtBuffer.setUint32(byteOffset, groupHandles[groupHandleIndex], LITTLE_ENDIAN);
    }

    // no rmiss

    // NUM_RAY_TYPES(=1) * numAllGeometries hitGroups
    const hitGroupId = 0;
    const hitShaderGroupIndex = 0;
    const handle = groupHandles[
      groupHandleIndex /* num rgen + num rmiss */
      + hitShaderGroupIndex];
    // every geometry:ray_type pair occupy one group
    const byteOffset = sbt.rayHit.start + hitGroupId * sbt.rayHit.stride;
    dvSbtBuffer.setUint32(byteOffset, handle, LITTLE_ENDIAN);
  }
  sbt.buffer.unmap();
  return sbt;
}

async function main(canvas: HTMLCanvasElement) {
  if (!navigator.gpu || !navigator.gpu.requestAdapter) {
    throw 'WebGPU is not supported or not enabled, please check chrome://flags/#enable-unsafe-webgpu';
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('failed to requestAdapter')
  }
  const device = await adapter.requestDevice({
    requiredFeatures: ["ray_tracing" as GPUFeatureName],
  });
  if (!device) {
    throw new Error('failed to get gpu device');
  }

  const context = canvas.getContext('webgpu');
  if (!context) {
    throw new Error('failed to get WebGPU context')
  }
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentationFormat,
  });

  // Ray tracing pipeline
  const tlas = createAndBuildRayTracingAccelerationContainer(device);
  const rayTracingPipeline = await createRayTracingPipeline(device, tlas);
  const sbt = createShaderBindingTable(device, rayTracingPipeline);

  // bind groups
  const pixelBuffer = device.createBuffer({
    size: canvas.width * canvas.height * 4 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE
  });
  const resolutionUniformBuffer = device.createBuffer({
    size: 2 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Float32Array(resolutionUniformBuffer.getMappedRange()).set(new Float32Array([
    canvas.width, canvas.height
  ]));
  resolutionUniformBuffer.unmap();
  const rayTracingBindGroup = device.createBindGroup({
    layout: rayTracingPipeline.getBindGroupLayout(0),
    entries: [{
      binding: 0,
      // any ok: acceleration container is a new resource type introduced in WebRTX
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      resource: tlas as any,
    }, {
      binding: 1,
      resource: { buffer: pixelBuffer },
    }]
  });

  // Render pipeline for displaying the pixel buffer.
  const renderPipeline = createRenderPipeline(device, presentationFormat);
  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [{
      binding: 0,
      resource: { buffer: pixelBuffer },
    }, {
      binding: 1,
      resource: { buffer: resolutionUniformBuffer },
    }]
  });

  function frame() {
    const commandEncoder = device.createCommandEncoder();
    const textureView = context!.getCurrentTexture().createView();

    // ray tracing pass
    {
      const passEncoder = commandEncoder.beginRayTracingPass();
      passEncoder.setPipeline(rayTracingPipeline);
      passEncoder.setBindGroup(
        0,
        rayTracingBindGroup);
      passEncoder.traceRays(
        device,
        sbt,
        canvas.width,
        canvas.height,
      );
      passEncoder.end();
    }
    // rasterization pass
    {
      const passEncoder = commandEncoder.beginRenderPass({
        colorAttachments: [{
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
          view: textureView,
        }]
      });
      passEncoder.setPipeline(renderPipeline);
      passEncoder.setBindGroup(0, renderBindGroup);
      passEncoder.draw(3, 1, 0, 0);
      passEncoder.end();
    }
    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
};

document.addEventListener('DOMContentLoaded', async () => {
  try {
    await main(document.getElementById('canvas') as HTMLCanvasElement);
  } catch (e) {
    alert(e)
  }
});
