#!/usr/bin/env python3
"""
Command-line interface for synthetic data generation
"""

import argparse
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from synthetic import SyntheticDataGenerator


def main():
    """Main entry point for the synthetic data generator CLI"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic sensor data from motion plans",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-p", "--plan", 
        required=True, 
        help="Path to the JSON plan file"
    )
    
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Output MCAP file path"
    )
    
    parser.add_argument(
        "-q", "--quiet", 
        action="store_true", 
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.plan):
        print(f"Error: Plan file '{args.plan}' does not exist", file=sys.stderr)
        return 1
    
    try:
        generator = SyntheticDataGenerator()
        generator.generate(args.plan, args.output, verbose=not args.quiet)
        
        if not args.quiet:
            file_size = os.path.getsize(args.output)
            print(f"Successfully generated: {args.output} ({file_size:,} bytes)")
        
    except Exception as e:
        print(f"Error generating synthetic data: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
