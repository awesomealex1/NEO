import wandb
import time
import torch
from utils.cli_utils import AverageMeter, ProgressMeter, accuracy, StoreMeter
from calibration_library.metrics import ECELoss
from utils.utils import get_device, calculate_entropy

def _log_and_update_metrics(output, target, batch_size, meters, phase, batch_idx, corruption_name):
    """
    Helper function to calculate, update, and log metrics for a single batch.
    (This function remains unchanged as it's already well-structured)
    """
    # 1. Calculate accuracy
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    meters['top1'].update(acc1[0], batch_size)
    meters['top5'].update(acc5[0], batch_size)

    # 2. Calculate entropy
    batch_entropy = calculate_entropy(output)
    mean_entropy = batch_entropy.mean()
    meters['entropy'].update(mean_entropy.item(), batch_size)

    # 3. Log metrics to wandb in a structured way
    log_payload = {
        f"{phase}/{corruption_name}/acc1": acc1[0],
        f"{phase}/{corruption_name}/acc5": acc5[0],
        f"{phase}/{corruption_name}/entropy": mean_entropy.item(),
        f"{phase}/{corruption_name}/acc1_avg": meters['top1'].avg,
        f"{phase}/{corruption_name}/acc5_avg": meters['top5'].avg,
        f"{phase}/{corruption_name}/entropy_avg": meters['entropy'].avg,
    }
    wandb.log(log_payload, step=batch_idx)

def run(loader, model, args, adapt):
    """
    Runs the adaptation phase on a specified number of batches from the provided loader.
    The model's internal state is updated during this phase.
    """
    print(f"--- Starting Run for {len(loader.dataset)} samples with adapt set to {str(adapt)} ---")
    batch_time = AverageMeter('Time', ':6.3f')
    meters = {
        'top1': StoreMeter('Acc@1', ':6.2f'),
        'top5': StoreMeter('Acc@5', ':6.2f'),
        'entropy': StoreMeter('Entropy', ':6.4f'),
    }
    # Use the length of the passed loader for the ProgressMeter
    progress = ProgressMeter(len(loader), 
                             [batch_time, meters['top1'], meters['top5'], meters['entropy']], 
                             prefix=f'Adapt ({args.algorithm} - {args.corruption}): ')
    
    outputs_list, targets_list = [], []

    with torch.no_grad():
        end = time.time()
        # Iterate over the passed loader
        for i, (images, target) in enumerate(loader):
            images = images.to(get_device())
            target = target.to(get_device())

            # Forward pass with adaptation enabled
            output = model.forward(x=images, adapt=adapt)
            if hasattr(output, "logits"):
                output = output.logits

            _log_and_update_metrics(output, target, images.size(0), meters, 'adapt' if adapt else 'validate', i, args.corruption)

            outputs_list.append(output.cpu())
            targets_list.append(target.cpu())

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 5 == 0:
                print(progress.display(i))
    
    outputs_list = torch.cat(outputs_list, dim=0).numpy()
    targets_list = torch.cat(targets_list, dim=0).numpy()
    logits = args.algorithm != 'lame'

    ece_avg = ECELoss().loss(outputs_list, targets_list, logits=logits)
    meters["ece"] = StoreMeter('ECE', ':6.4f')
    meters["ece"].update(ece_avg, n=1)

    print(f"--- Run Finished ---")
    return meters['top1'], meters['top5'], meters['entropy'], meters["ece"]