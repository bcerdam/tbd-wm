import os
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter

def aggregate_tensorboard_runs(parent_dir, target_tag):
    search_path = os.path.join(parent_dir, "*", "tensorboard", "events.out.tfevents.*")
    event_files = [f for f in glob.glob(search_path) if "AVERAGED_METRICS" not in f]
    
    if not event_files:
        return

    step_to_values = {}
    
    for ef in event_files:
        ea = EventAccumulator(ef, size_guidance={'images': 0, 'audio': 0, 'histograms': 0, 'tensors': 0, 'scalars': 0})
        ea.Reload()
        
        if target_tag in ea.Tags().get('scalars', []):
            events = ea.Scalars(target_tag)
            for e in events:
                if e.step not in step_to_values:
                    step_to_values[e.step] = []
                step_to_values[e.step].append(e.value)

    output_dir = os.path.join(parent_dir, "AVERAGED_METRICS", "tensorboard")
    os.makedirs(output_dir, exist_ok=True)
    avg_writer = SummaryWriter(log_dir=output_dir)
    
    for step in sorted(step_to_values.keys()):
        values = step_to_values[step]
        mean_val = np.mean(values)
        avg_writer.add_scalar(f"{target_tag}_Mean", mean_val, step)

    avg_writer.close()

if __name__ == '__main__':
    EXPERIMENTS_DIR = "output/experiments_alien_0"
    TARGET_TAG = "agent/episode_mean_rewards"
    
    aggregate_tensorboard_runs(EXPERIMENTS_DIR, TARGET_TAG)