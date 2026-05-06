import os
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter

def aggregate_tensorboard_runs(parent_dir, output_dir, tag_to_average):
    
    search_path = os.path.join(parent_dir, "*", "tensorboard", "events.out.tfevents.*")
    event_files = glob.glob(search_path)
    
    if not event_files:
        return

    step_to_values = {}
    for ef in event_files:
        ea = EventAccumulator(ef)
        ea.Reload()
        
        if tag_to_average in ea.Tags()['scalars']:
            events = ea.Scalars(tag_to_average)
            for e in events:
                if e.step not in step_to_values:
                    step_to_values[e.step] = []
                step_to_values[e.step].append(e.value)

    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir)
    
    for step in sorted(step_to_values.keys()):
        values = step_to_values[step]
        mean_val = np.mean(values)
        writer.add_scalar(f"{tag_to_average}_Mean", mean_val, step)
        std_val = np.std(values)
        writer.add_scalar(f"{tag_to_average}_StdDev", std_val, step)

    writer.close()

if __name__ == '__main__':
    EXPERIMENTS_DIR = "output/experiments_alien_0"
    AVERAGE_OUTPUT_DIR = "output/experiments_alien_0/averaged_results/tensorboard"
    TARGET_TAG = "agent/episode_mean_rewards"
    aggregate_tensorboard_runs(EXPERIMENTS_DIR, AVERAGE_OUTPUT_DIR, TARGET_TAG)