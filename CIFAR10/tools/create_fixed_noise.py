import argparse
from pathlib import Path
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Create a fixed noise tensor for deterministic sampling.")
    parser.add_argument("--output", required=True, help="Path to save the noise tensor (.pt).")
    parser.add_argument("--num_images", type=int, default=128, help="Number of images (noise samples).")
    parser.add_argument("--channels", type=int, default=3, help="Number of latent channels.")
    parser.add_argument("--height", type=int, default=32, help="Latent height.")
    parser.add_argument("--width", type=int, default=32, help="Latent width.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    noise = torch.randn(
        (args.num_images, args.channels, args.height, args.width),
        generator=gen,
        dtype=torch.float32,
    )
    torch.save(noise, output_path)
    print(f"Saved noise tensor: {output_path}")


if __name__ == "__main__":
    main()
