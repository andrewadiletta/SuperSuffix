# Loss Functions Visualization

This folder contains code for generating loss function graphs as shown in the paper.

## Overview

The `plot_individual_log.py` script parses GCG optimization experiment logs and generates publication-quality graphs showing the optimization progress over time. These graphs visualize:

- Prompt Guard 2 Loss
- Text Generation Loss
- Prompt Guard 2 Benign Score

## Usage

To generate a loss graph:

1. **Obtain a log file** from a Super Suffix generation experiment (located in `Super_Suffix/logs/`)

2. **Copy the log file** to this directory:
   ```bash
   cp ../Super_Suffix/logs/your_experiment_log.log .
   ```

3. **Run the plotting script**:
   ```bash
   python plot_individual_log.py your_experiment_log.log
   ```

The script will generate a PDF visualization of the optimization progress.

## Customizing the Output

You can modify the graph appearance by editing the configuration variables at the top of `plot_individual_log.py`:

```python
# Plot order configuration (0 = top, 1 = middle, 2 = bottom)
PLOT_ORDER = {
    'pg_loss': 0,           # Prompt Guard 2 Loss position
    'text_gen_loss': 1,     # Text Generation Loss position
    'pg_score': 2           # Prompt Guard 2 Benign position (bottom)
}

# Background colors for objectives
OBJECTIVE_COLORS = {
    'text_gen': "#fbc1c1",  # Grey for text generation mode
    'pg': "#c1dfff"         # Light blue for PG mode
}

# Font sizes for publication
FONT_CONFIG = {
    'base_size': 20,
    'title_size': 22,
    'label_size': 20,
    'tick_size': 18,
    'legend_size': 18
}

# Figure dimensions
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 14
```

## Example

After running a Super Suffix experiment with Google Gemma 2B, you might have a log file like:
```
Super_Suffix/logs/gcg_direction_opt_20251024_151126_prompt_0.log
```

Copy it and run:
```bash
cp ../Super_Suffix/logs/gcg_direction_opt_20251024_151126_prompt_0.log .
python plot_individual_log.py gcg_direction_opt_20251024_151126_prompt_0.log
```

This will generate a graph showing how the losses evolved during the joint optimization process for generating a Super Suffix.
