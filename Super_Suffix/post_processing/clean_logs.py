#!/usr/bin/env python3

import os
import sys
import re
import argparse
from collections import defaultdict
from datetime import datetime

class LogConfig:
    """Class to store and compare log configurations"""
    
    def __init__(self):
        self.model = None
        self.pg_model = None
        self.prompt = None
        self.device = None
        self.data_type = None
        self.probe_sampling = None
        self.num_optimization_steps = None
        self.target_direction_path = None
        self.target_layer = None
        
        # GCG Configuration
        self.batch_size = None
        self.search_width = None
        self.num_steps = None
        self.tokens_to_replace = None
        self.top_k = None
        self.buffer_size = None
        self.direction_weight = None
        self.use_last_token = None
        self.maximize_similarity = None
        self.layer_for_direction = None
        
        # Metadata
        self.timestamp = None
        self.is_complete = False
        self.file_path = None
        self.file_size = 0
    
    def get_config_tuple(self):
        """Return a tuple of all config values for comparison"""
        return (
            self.model,
            self.pg_model,
            self.prompt,
            self.device,
            self.data_type,
            self.probe_sampling,
            self.num_optimization_steps,
            self.target_direction_path,
            self.target_layer,
            self.batch_size,
            self.search_width,
            self.num_steps,
            self.tokens_to_replace,
            self.top_k,
            self.buffer_size,
            self.direction_weight,
            self.use_last_token,
            self.maximize_similarity,
            self.layer_for_direction
        )
    
    def __hash__(self):
        return hash(self.get_config_tuple())
    
    def __eq__(self, other):
        return self.get_config_tuple() == other.get_config_tuple()

def parse_log_config(log_path):
    """Parse configuration from a log file"""
    config = LogConfig()
    config.file_path = log_path
    config.file_size = os.path.getsize(log_path)
    
    # Regex patterns for extracting configuration
    patterns = {
        'timestamp': r'Timestamp: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
        'model': r'Model: (.+?)$',
        'pg_model': r'PG Model: (.+?)$',
        'prompt': r'Input prompt: (.+?)$',
        'device': r'Device: (.+?)$',
        'data_type': r'Data type: (.+?)$',
        'probe_sampling': r'Probe sampling: (.+?)$',
        'num_optimization_steps': r'Number of optimization steps: (\d+)',
        'target_direction_path': r'Target direction path: (.+?)$',
        'target_layer': r'Target layer: (\d+)',
        'batch_size': r'Batch size: (\d+)',
        'search_width': r'Search width: (\d+)',
        'num_steps': r'Number of steps: (\d+)',
        'tokens_to_replace': r'Tokens to replace per step: (\d+)',
        'top_k': r'Top-k: (\d+)',
        'buffer_size': r'Buffer size: (\d+)',
        'direction_weight': r'Direction weight: ([\d.]+)',
        'use_last_token': r'Use last token: (.+?)$',
        'maximize_similarity': r'Maximize similarity: (.+?)$',
        'layer_for_direction': r'Layer for direction: (\d+)'
    }
    
    try:
        with open(log_path, 'rb') as f:
            # Read first 10KB for configuration (should be at the beginning)
            content = f.read(10240).decode('utf-8', errors='ignore')
            
            # Extract all configuration values
            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    # Convert numeric values
                    if key in ['target_layer', 'batch_size', 'search_width', 'num_steps', 
                              'tokens_to_replace', 'top_k', 'buffer_size', 'layer_for_direction',
                              'num_optimization_steps']:
                        value = int(value)
                    elif key == 'direction_weight':
                        value = float(value)
                    elif key in ['probe_sampling', 'use_last_token', 'maximize_similarity']:
                        value = value.lower() == 'true'
                    elif key == 'timestamp':
                        value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                    
                    setattr(config, key, value)
            
            # Check if log is complete
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            read_size = min(10240, file_size)
            f.seek(max(0, file_size - read_size))
            last_content = f.read().decode('utf-8', errors='ignore')
            config.is_complete = "EXPERIMENT COMPLETED SUCCESSFULLY" in last_content
            
    except Exception as e:
        print(f"Warning: Error parsing {log_path}: {e}")
    
    return config

def find_duplicates_and_incomplete(logs_dir, verbose=False):
    """Find duplicate and incomplete logs in the directory"""
    
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory not found: {logs_dir}")
        return None, None
    
    configs = []
    config_groups = defaultdict(list)
    
    # Parse all log files
    log_files = sorted([f for f in os.listdir(logs_dir) if f.endswith('.log')])
    
    for filename in log_files:
        log_path = os.path.join(logs_dir, filename)
        
        if verbose:
            print(f"Parsing: {filename}...", end=" ")
        
        config = parse_log_config(log_path)
        configs.append(config)
        
        # Group by configuration
        config_groups[config.get_config_tuple()].append(config)
        
        if verbose:
            status = "COMPLETE" if config.is_complete else "INCOMPLETE"
            print(f"{status}")
    
    # Identify duplicates and incomplete logs
    duplicates_to_delete = []
    incomplete_to_delete = []
    
    for config_tuple, group in config_groups.items():
        if len(group) > 1:
            # Multiple logs with same configuration - keep the best one
            # Priority: 1) Complete logs, 2) Newest/largest log
            group.sort(key=lambda c: (c.is_complete, c.timestamp or datetime.min, c.file_size), reverse=True)
            
            # Keep the first (best) one, mark others for deletion
            for config in group[1:]:
                duplicates_to_delete.append(config.file_path)
        
        # Also check for incomplete logs in this group
        for config in group:
            if not config.is_complete and config.file_path not in duplicates_to_delete:
                incomplete_to_delete.append(config.file_path)
    
    return duplicates_to_delete, incomplete_to_delete

def clean_logs(logs_dir, dry_run=True, verbose=False, clean_incomplete=True, clean_duplicates=True):
    """
    Clean incomplete and/or duplicate logs from the logs directory
    
    Args:
        logs_dir: Path to logs directory
        dry_run: If True, only show what would be deleted without actually deleting
        verbose: If True, show all files being checked
        clean_incomplete: If True, remove incomplete logs
        clean_duplicates: If True, remove duplicate logs
    """
    
    print(f"Analyzing logs in {logs_dir}...")
    print(f"Options: Clean incomplete={clean_incomplete}, Clean duplicates={clean_duplicates}\n")
    
    # Find duplicates and incomplete logs
    duplicates, incomplete = find_duplicates_and_incomplete(logs_dir, verbose)
    
    if duplicates is None:
        return 1
    
    # Build list of files to delete based on options
    files_to_delete = set()
    
    if clean_duplicates:
        files_to_delete.update(duplicates)
    
    if clean_incomplete:
        files_to_delete.update(incomplete)
    
    # Remove files that are both duplicate and incomplete from the incomplete list 
    # (they're already being deleted as duplicates)
    duplicate_set = set(duplicates)
    incomplete_only = [f for f in incomplete if f not in duplicate_set]
    
    # Calculate total size
    total_size = sum(os.path.getsize(f) for f in files_to_delete if os.path.exists(f))
    
    # Report findings
    print(f"\nSummary:")
    print(f"  Duplicate logs found: {len(duplicates)}")
    print(f"  Incomplete logs found: {len(incomplete)}")
    print(f"  Incomplete (not duplicates): {len(incomplete_only)}")
    print(f"  Total files to delete: {len(files_to_delete)}")
    
    if files_to_delete:
        print(f"  Space to be freed: {total_size / (1024*1024):.2f} MB")
        
        if dry_run:
            print("\n--- DRY RUN MODE ---")
            print("The following files would be deleted:")
            
            if clean_duplicates and duplicates:
                print("\nDuplicates:")
                for path in sorted(duplicates):
                    print(f"  {os.path.basename(path)}")
            
            if clean_incomplete and incomplete_only:
                print("\nIncomplete (not duplicates):")
                for path in sorted(incomplete_only):
                    print(f"  {os.path.basename(path)}")
            
            print("\nTo actually delete these files, run with --delete flag")
        else:
            print("\nDeleting files...")
            deleted_count = 0
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"  Deleted: {os.path.basename(file_path)}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  Error deleting {file_path}: {e}")
            
            print(f"\nDeleted {deleted_count} files")
    else:
        print("\nNo files to clean!")
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Clean incomplete and duplicate experiment logs from the logs directory"
    )
    parser.add_argument(
        "--logs-dir", 
        type=str, 
        default="../logs",
        help="Path to logs directory (default: ../logs)"
    )
    parser.add_argument(
        "--delete", 
        action="store_true",
        help="Actually delete files (without this flag, only shows what would be deleted)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all files being checked"
    )
    parser.add_argument(
        "--no-incomplete",
        action="store_true",
        help="Don't remove incomplete logs, only duplicates"
    )
    parser.add_argument(
        "--no-duplicates",
        action="store_true",
        help="Don't remove duplicates, only incomplete logs"
    )
    
    args = parser.parse_args()
    
    # Determine what to clean
    clean_incomplete = not args.no_incomplete
    clean_duplicates = not args.no_duplicates
    
    if not clean_incomplete and not clean_duplicates:
        print("Error: Both --no-incomplete and --no-duplicates specified. Nothing to do!")
        return 1
    
    # Confirm deletion if --delete is used
    if args.delete:
        print(f"WARNING: This will permanently delete files from {args.logs_dir}")
        
        action_desc = []
        if clean_incomplete:
            action_desc.append("incomplete logs")
        if clean_duplicates:
            action_desc.append("duplicate logs")
        print(f"Will delete: {' and '.join(action_desc)}")
        
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return 1
    
    # Run the cleaning
    return clean_logs(
        args.logs_dir, 
        dry_run=not args.delete, 
        verbose=args.verbose,
        clean_incomplete=clean_incomplete,
        clean_duplicates=clean_duplicates
    )

if __name__ == "__main__":
    sys.exit(main())