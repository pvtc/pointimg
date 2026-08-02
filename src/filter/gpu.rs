//! Optional wgpu compute path for the local variance pass of the density map.
//!
//! Enable with `cargo build --features gpu`. If no compatible adapter is
//! available, or if device creation/readback fails, callers transparently use
//! the existing CPU SAT implementation.

use bytemuck::{Pod, Zeroable};
use image::RgbImage;
use std::sync::Mutex;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

// Certains drivers natifs ne supportent pas la création simultanée de plusieurs
// devices/instances dans des threads différents. Les tests Rayon/ Rust lancent
// plusieurs calculs en parallèle : sérialiser la durée de vie du device évite un
// crash driver tout en gardant le parallélisme à l'intérieur du shader.
static GPU_LOCK: Mutex<()> = Mutex::new(());

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    radius: i32,
    sensitivity: f32,
}

/// Computes the raw local variance values on the GPU and normalizes them using
/// the same curve as the CPU implementation.
pub(crate) fn compute_density_map(src: &RgbImage, sensitivity: f32) -> Option<Vec<f32>> {
    // Opt-in explicite tant que la couverture des drivers GPU n'est pas testée
    // dans la CI. Sans cette variable, le chemin CPU/SAT historique reste exact.
    std::env::var_os("POINTIMG_GPU")?;
    let _gpu_guard = GPU_LOCK.lock().ok()?;
    let raw = compute_raw_variance(src)?;
    if std::env::var_os("POINTIMG_GPU_DEBUG").is_some() {
        let min = raw.iter().copied().fold(f32::INFINITY, f32::min);
        let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("gpu density raw min={min} max={max} len={}", raw.len());
    }
    let max_var = raw.iter().copied().fold(0.0_f32, f32::max).max(1e-6);
    Some(
        raw.into_iter()
            .map(|v| {
                let norm = (v / max_var).sqrt();
                1.0 - sensitivity * (1.0 - norm)
            })
            .collect(),
    )
}

fn compute_raw_variance(src: &RgbImage) -> Option<Vec<f32>> {
    let (width, height) = src.dimensions();
    if width == 0 || height == 0 {
        return Some(Vec::new());
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .ok()?;

    let packed: Vec<u32> = src
        .pixels()
        .map(|p| u32::from(p[0]) | (u32::from(p[1]) << 8) | (u32::from(p[2]) << 16))
        .collect();
    let output_size = (width as u64) * (height as u64) * 4;
    let params = Params {
        width,
        height,
        radius: 4,
        sensitivity: 0.0,
    };

    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pointimg density input"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pointimg density output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pointimg density readback"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pointimg density params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pointimg density shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pointimg density bindings"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pointimg density bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pointimg density pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pointimg density pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pointimg density encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pointimg density pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let (sender, receiver) = mpsc::sync_channel(1);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    instance.poll_all(true);
    receiver.recv().ok()?.ok()?;
    let view = staging.slice(..).get_mapped_range();
    let result = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    staging.unmap();
    Some(result)
}
