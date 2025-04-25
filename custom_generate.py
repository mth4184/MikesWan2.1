import os
import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

def setup_model(model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers"):
    # Initialize VAE with float32 for better precision
    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.float32
    )
    
    # Setup scheduler with flow prediction for 720p
    flow_shift = 5.0  # 5.0 for 720P
    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=flow_shift
    )
    
    # Initialize pipeline with bfloat16 for memory efficiency
    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    )
    pipe.scheduler = scheduler
    
    # Enable FSDP with Ring strategy for 8 GPUs
    pipe.enable_fsdp(
        ring_size=8,  # Use Ring strategy across all 8 GPUs
        sharding_strategy="FULL_SHARD",  # Full sharding for maximum memory efficiency
        mixed_precision=True,  # Enable mixed precision
        device_map="auto"  # Automatically distribute across available GPUs
    )
    
    return pipe

def generate_video(
    pipe,
    prompt,
    negative_prompt=None,
    height=720,
    width=1280,
    num_frames=81,
    guidance_scale=5.0,
    output_path="output.mp4"
):
    if negative_prompt is None:
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, \
            style, works, paintings, images, static, overall gray, worst quality, low quality, \
            JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, \
            poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, \
            still picture, messy background, three legs, many people in the background, \
            walking backwards"
    
    # Generate frames with optimized memory settings
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        ).frames[0]
    
    # Export to video
    export_to_video(output, output_path, fps=16)
    return output_path

def main():
    # Setup the model using the Diffusers model ID
    pipe = setup_model()
    
    # Example prompt
    prompt = "A majestic mountain landscape with snow-capped peaks and a flowing river in the valley below, captured during golden hour"
    
    # Generate video
    output_path = generate_video(
        pipe=pipe,
        prompt=prompt,
        height=720,
        width=1280,
        num_frames=81,  # Approximately 5 seconds at 16 fps
        guidance_scale=5.0
    )
    
    print(f"Video generated successfully at: {output_path}")

if __name__ == "__main__":
    # Set environment variables for optimal performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Use all 8 GPUs
    
    main()
