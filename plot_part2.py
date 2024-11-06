import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_log_file(file_path):
    """Parse the log file and extract epoch, train PSNR, and valid PSNR."""
    epochs = []
    train_psnr = []
    valid_psnr = []
    
    pattern = r'Epoch \[ *(\d+)/\d+\].*psnr: ([\d.]+)\(([\d.]+)\)'
    
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                train = float(match.group(2))
                valid = float(match.group(3))
                
                epochs.append(epoch)
                train_psnr.append(train)
                valid_psnr.append(valid)
    
    return pd.DataFrame({
        'epoch': epochs,
        'train_psnr': train_psnr,
        'valid_psnr': valid_psnr
    })

def plot_multiple_experiments():
    """Create a single plot showing PSNR values for all experiments."""
    plt.figure(figsize=(12, 8))
    
    # copy and save the logs obtained while training the model 
    experiments = {
        'Original': {'file': 'loss_log_rmse.txt', 'color': 'blue', 'linestyle': '-'},
        'Improved': {'file': 'loss_log_mix.txt', 'color': 'green', 'linestyle': '-'}
    }
    
    for name, config in experiments.items():
        df = parse_log_file(config['file'])
        
        # Plot training data with solid lines
        plt.plot(df['epoch'], df['train_psnr'], 
                color=config['color'], 
                linestyle=config['linestyle'],
                label=f'{name} (Train)',
                marker='o', 
                markersize=4,
                markevery=5)
        
        # Plot validation data with dashed lines
        plt.plot(df['epoch'], df['valid_psnr'], 
                color=config['color'], 
                linestyle='--',
                label=f'{name} (Valid)',
                marker='s',
                markersize=4,
                markevery=5,
                alpha=0.7)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('PSNR', fontsize=12)
    plt.title('PSNR Values Over Training - All Experiments', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_multiple_experiments()
