#!/usr/bin/env python3

import torch
from diffusers import AutoencoderKL

def test_vae_loading():
    """Test loading the VAE model to ensure it works correctly."""
    
    print("Testing VAE model loading...")
    
    # Test the original sd-1-5 VAE (this should fail)
    print("\n1. Testing original sd-1-5 VAE (should fail):")
    try:
        vae = AutoencoderKL.from_pretrained("pretrained_models/StableDiffusion/sd-1-5", subfolder="vae")
        print("✓ Original VAE loaded successfully")
    except Exception as e:
        print(f"✗ Original VAE failed to load: {e}")
    
    # Test the sd-vae-ft-mse VAE (this should work)
    print("\n2. Testing sd-vae-ft-mse VAE (should work):")
    try:
        vae = AutoencoderKL.from_pretrained("pretrained_models/StableDiffusion/sd-vae-ft-mse")
        print("✓ sd-vae-ft-mse VAE loaded successfully")
        
        # Test with the specific subfolder that stage2 uses
        vae = AutoencoderKL.from_pretrained("pretrained_models/StableDiffusion/sd-vae-ft-mse", subfolder="sd-vae-ft-mse")
        print("✓ sd-vae-ft-mse VAE with subfolder loaded successfully")
        
    except Exception as e:
        print(f"✗ sd-vae-ft-mse VAE failed to load: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_vae_loading() 