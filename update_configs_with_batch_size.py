import os
import yaml
import glob

def update_config_with_batch_size(config_path, batch_size=1024):
    """Update a YAML config file with batch size settings."""
    try:
        # Read the existing config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add batch size settings if they don't exist
        config['gpu_batch_size'] = batch_size
        config['correlation_batch_size'] = batch_size
        
        # Make sure n_jobs is set
        if 'n_jobs' not in config:
            config['n_jobs'] = -1
        
        # Write the updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Updated {config_path} with batch size {batch_size}")
        return True
    
    except Exception as e:
        print(f"Error updating {config_path}: {e}")
        return False

def main():
    """Update all config files in the configs directory."""
    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    config_files = glob.glob(os.path.join(configs_dir, '*.yaml'))
    
    if not config_files:
        print(f"No config files found in {configs_dir}")
        return
    
    print(f"Found {len(config_files)} config files")
    
    # Update each config file
    for config_file in config_files:
        # Skip the template file
        if os.path.basename(config_file) == 'batch_size_template.yaml':
            continue
            
        update_config_with_batch_size(config_file)
    
    print("\nAll configs updated successfully.")
    print("You can now use the following settings in your experiments:")
    print("  - gpu_batch_size: 1024")
    print("  - correlation_batch_size: 1024")

if __name__ == "__main__":
    main() 